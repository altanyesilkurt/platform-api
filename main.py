from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from supabase import create_client, Client
from openai import OpenAI
from dotenv import load_dotenv
from promts.commit_analysis_system_prompt import COMMIT_ANALYSIS_SYSTEM_PROMPT
from promts.general_query_system_prompt import GENERAL_QUERY_SYSTEM_PROMPT
from promts.pr_analysis_system_prompt import PR_ANALYSIS_SYSTEM_PROMPT
from promts.follow_up_conversation_prompt import FOLLOW_UP_CONVERSATION_PROMPT
import os
import json
import re
import httpx

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="AI PR Assistant API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://localhost:9090"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


class ChatCreate(BaseModel):
    title: str = Field(default="New Chat", max_length=200)


class ChatUpdate(BaseModel):
    title: str = Field(..., max_length=200)


class MessageCreate(BaseModel):
    chat_id: str
    message: str = Field(..., min_length=1)


# Patterns
PR_URL_PATTERN = r'https://github\.com/([^/]+)/([^/]+)/pull/(\d+)'
COMMIT_URL_PATTERN = r'https://github\.com/([^/]+)/([^/]+)/commit/([a-fA-F0-9]+)'

def extract_pr_url(message: str) -> Optional[tuple]:
    match = re.search(PR_URL_PATTERN, message)
    if match:
        return (match.group(1), match.group(2), int(match.group(3)))
    return None


def extract_commit_url(message: str) -> Optional[tuple]:
    match = re.search(COMMIT_URL_PATTERN, message)
    if match:
        return (match.group(1), match.group(2), match.group(3))
    return None


async def fetch_github_pr(owner: str, repo: str, pr_number: int) -> Dict[str, Any]:
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "PR-Assistant"
    }
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    else:
        print("WARNING: No GITHUB_TOKEN set. API rate limits will be very restrictive.")

    async with httpx.AsyncClient(timeout=30.0) as client:
        pr_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}"
        pr_response = await client.get(pr_url, headers=headers)

        if pr_response.status_code == 404:
            raise HTTPException(status_code=404, detail=f"PR #{pr_number} not found in {owner}/{repo}")
        elif pr_response.status_code == 403:
            raise HTTPException(status_code=403, detail="GitHub API rate limit or access denied")
        elif pr_response.status_code != 200:
            raise HTTPException(status_code=pr_response.status_code, detail=f"GitHub API error")

        pr_data = pr_response.json()

        # Fetch diff
        diff_content = ""
        try:
            diff_headers = {**headers, "Accept": "application/vnd.github.v3.diff"}
            diff_response = await client.get(pr_url, headers=diff_headers)
            if diff_response.status_code == 200:
                diff_content = diff_response.text or ""
        except:
            pass

        # Fetch files
        files_data = []
        try:
            files_response = await client.get(f"{pr_url}/files", headers=headers)
            if files_response.status_code == 200:
                files_data = files_response.json() or []
        except:
            pass

        # Fetch commits
        commits_data = []
        try:
            commits_response = await client.get(f"{pr_url}/commits", headers=headers)
            if commits_response.status_code == 200:
                commits_data = commits_response.json() or []
        except:
            pass

        # Fetch comments
        comments_data = []
        try:
            comments_response = await client.get(f"{pr_url}/comments", headers=headers)
            if comments_response.status_code == 200:
                comments_data = comments_response.json() or []
        except:
            pass

        return {
            "pr": pr_data,
            "diff": diff_content[:50000] if diff_content else "",
            "files": files_data[:100] if files_data else [],
            "commits": commits_data[:50] if commits_data else [],
            "comments": comments_data[:50] if comments_data else [],
            "url": f"https://github.com/{owner}/{repo}/pull/{pr_number}"
        }


async def fetch_github_commit(owner: str, repo: str, commit_sha: str) -> Dict[str, Any]:
    """Fetch commit details from GitHub API."""
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "PR-Assistant"
    }
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"

    async with httpx.AsyncClient(timeout=30.0) as client:
        commit_url = f"https://api.github.com/repos/{owner}/{repo}/commits/{commit_sha}"
        response = await client.get(commit_url, headers=headers)

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Failed to fetch commit")

        return {"commit": response.json(), "url": f"https://github.com/{owner}/{repo}/commit/{commit_sha}"}


def format_pr_for_conversation(pr_data: Dict[str, Any]) -> str:
    """Format PR data in a way that's useful for conversation."""
    pr = pr_data.get("pr", {})
    files = pr_data.get("files", [])
    diff = pr_data.get("diff", "")
    user = pr.get("user", {}) or {}
    base = pr.get("base", {}) or {}
    head = pr.get("head", {}) or {}

    # Group files by type/directory for better context
    files_by_dir = {}
    for f in files:
        path = f.get("filename", "")
        dir_name = "/".join(path.split("/")[:-1]) or "root"
        if dir_name not in files_by_dir:
            files_by_dir[dir_name] = []
        files_by_dir[dir_name].append({
            "name": path.split("/")[-1],
            "full_path": path,
            "status": f.get("status", ""),
            "additions": f.get("additions", 0),
            "deletions": f.get("deletions", 0),
            "patch": f.get("patch", "")[:3000]
        })

    files_summary = ""
    for dir_name, dir_files in files_by_dir.items():
        files_summary += f"\n**{dir_name}/**\n"
        for f in dir_files:
            files_summary += f"  - `{f['name']}` ({f['status']}, +{f['additions']}/-{f['deletions']})\n"

    return f"""
## PR Context for Conversation

**Title:** {pr.get('title', 'Unknown')}
**Author:** {user.get('login', 'Unknown')}
**Branch:** {head.get('ref', '?')} → {base.get('ref', '?')}
**State:** {pr.get('state', 'unknown')} | Merged: {pr.get('merged', False)}
**URL:** {pr_data.get('url', 'N/A')}

### Description
{(pr.get('body') or 'No description provided')[:1500]}

### Stats
- **{len(files)} files** changed
- **+{pr.get('additions', 0)}** additions / **-{pr.get('deletions', 0)}** deletions
- **{pr.get('commits', 0)}** commits

### Files Changed (grouped by directory)
{files_summary}

### Code Changes (Diff)
```diff
{diff[:30000]}
```

---
Use this context to have a natural conversation about the PR. Reference specific files and code when answering questions.
"""


def format_commit_for_conversation(commit_data: Dict[str, Any]) -> str:
    """Format commit data for conversation."""
    commit = commit_data.get("commit", {})
    info = commit.get("commit", {})
    author = info.get("author", {})
    stats = commit.get("stats", {})
    files = commit.get("files", [])

    files_summary = "\n".join([
        f"- `{f.get('filename')}` ({f.get('status')}, +{f.get('additions', 0)}/-{f.get('deletions', 0)})"
        for f in (files or [])[:20]
    ])

    patches = ""
    for f in (files or [])[:8]:
        if f.get("patch"):
            patches += f"\n### `{f.get('filename')}`\n```diff\n{f.get('patch', '')[:2000]}\n```\n"

    return f"""
## Commit Context for Conversation

**Message:** {info.get('message', 'No message')}
**Author:** {author.get('name', 'Unknown')} ({author.get('email', '')})
**Date:** {author.get('date', 'N/A')}
**SHA:** {commit.get('sha', '')[:10]}
**URL:** {commit_data.get('url', 'N/A')}

### Stats
- **{len(files) if files else 0}** files changed
- **+{stats.get('additions', 0)}** / **-{stats.get('deletions', 0)}**

### Files Changed
{files_summary}

### Code Changes
{patches if patches else 'No patch data available'}

---
Use this context to discuss the commit naturally.
"""


def get_chat_context(chat_id: str) -> Dict[str, Any]:
    """Retrieve stored context for a chat (PR/commit being discussed)."""
    try:
        response = supabase.table("chat_contexts").select("*").eq("chat_id", chat_id).single().execute()
        return response.data if response.data else {}
    except:
        return {}


def save_chat_context(chat_id: str, context_type: str, context_data: Dict[str, Any]):
    """Save or update context for a chat."""
    try:
        existing = supabase.table("chat_contexts").select("id").eq("chat_id", chat_id).execute()
        if existing.data:
            supabase.table("chat_contexts").update({
                "context_type": context_type,
                "context_data": json.dumps(context_data),
                "updated_at": "now()"
            }).eq("chat_id", chat_id).execute()
        else:
            supabase.table("chat_contexts").insert({
                "chat_id": chat_id,
                "context_type": context_type,
                "context_data": json.dumps(context_data)
            }).execute()
    except Exception as e:
        print(f"Warning: Could not save chat context: {e}")


def generate_chat_title(user_message: str, ai_response: str) -> str:
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system",
                 "content": "Generate a short, conversational title (3-6 words) for this chat. Return ONLY the title."},
                {"role": "user", "content": f"User: {user_message[:200]}\nAssistant: {ai_response[:300]}"}
            ],
            max_tokens=20
        )
        return response.choices[0].message.content.strip().strip('"\'')[:50]
    except:
        return "Code Review Chat"


def is_first_exchange(chat_id: str) -> bool:
    response = supabase.table("messages").select("id", count="exact").eq("chat_id", chat_id).execute()
    return response.count <= 1


@app.get("/chats")
async def get_chats():
    response = supabase.table("chats").select("*").order("updated_at", desc=True).execute()
    return response.data


@app.post("/chats")
async def create_chat(chat: ChatCreate):
    response = supabase.table("chats").insert({"title": chat.title}).execute()
    return response.data[0]


@app.get("/chats/{chat_id}")
async def get_chat(chat_id: str):
    response = supabase.table("chats").select("*").eq("id", chat_id).single().execute()
    return response.data


@app.put("/chats/{chat_id}")
async def update_chat(chat_id: str, chat_update: ChatUpdate):
    response = supabase.table("chats").update({"title": chat_update.title}).eq("id", chat_id).execute()
    return response.data[0] if response.data else None


@app.delete("/chats/{chat_id}")
async def delete_chat(chat_id: str):
    supabase.table("messages").delete().eq("chat_id", chat_id).execute()
    supabase.table("chat_contexts").delete().eq("chat_id", chat_id).execute()
    supabase.table("chats").delete().eq("id", chat_id).execute()
    return {"success": True}


@app.get("/chats/{chat_id}/messages")
async def get_messages(chat_id: str):
    response = supabase.table("messages").select("*").eq("chat_id", chat_id).order("created_at").execute()
    return response.data


@app.post("/chat/stream")
async def send_message_stream(message_data: MessageCreate):
    user_message = message_data.message
    chat_id = message_data.chat_id
    first_exchange = is_first_exchange(chat_id)

    # Save user message
    supabase.table("messages").insert({
        "chat_id": chat_id, "role": "user", "content": user_message
    }).execute()

    # Check for URLs in message
    pr_info = extract_pr_url(user_message)
    commit_info = extract_commit_url(user_message)

    # Get existing context for this chat
    existing_context = get_chat_context(chat_id)

    pr_context = None
    commit_context = None
    metadata = None
    fetch_error = None

    # Fetch new PR if URL provided
    if pr_info:
        owner, repo, pr_number = pr_info
        try:
            pr_data = await fetch_github_pr(owner, repo, pr_number)
            pr = pr_data.get("pr", {})
            pr_state = pr.get("state", "unknown")
            pr_merged = pr.get("merged", False)

            # Build commits list for metadata
            commits = []
            for c in pr_data.get("commits", [])[:10]:
                c_info = c.get("commit", {})
                commits.append({
                    "sha": c.get("sha", "")[:7],
                    "message": c_info.get("message", "").split("\n")[0][:100],
                    "author": c_info.get("author", {}).get("name", "Unknown")
                })

            # Build files list with patches for metadata
            files = []
            for f in pr_data.get("files", [])[:20]:
                files.append({
                    "filename": f.get("filename", ""),
                    "status": f.get("status", ""),
                    "additions": f.get("additions", 0),
                    "deletions": f.get("deletions", 0),
                    "patch": f.get("patch", "")[:5000] if f.get("patch") else ""
                })

            # Build full metadata (same as original)
            metadata = {
                "type": "pr",
                "pr_url": pr_data.get("url", f"https://github.com/{owner}/{repo}/pull/{pr_number}"),
                "pr_title": pr.get("title", "Unknown PR"),
                "pr_body": (pr.get("body") or "No description provided")[:500],
                "pr_state": pr_state,
                "pr_merged": pr_merged,
                "pr_author": (pr.get("user") or {}).get("login", "Unknown"),
                "files_changed": len(pr_data.get("files", [])),
                "additions": pr.get("additions", 0),
                "deletions": pr.get("deletions", 0),
                "commits": commits,
                "files": files
            }

            # Check if closed/merged
            if pr_merged or pr_state == "closed":
                merged_by = (pr.get("merged_by") or {}).get("login", "")
                if pr_merged:
                    status_message = f"This PR has already been merged{' by ' + merged_by if merged_by else ''}. I can still discuss the changes if you'd like!"
                else:
                    status_message = "This PR has been closed without merging. I can still discuss the changes if you'd like!"
                metadata["status_message"] = status_message

                # Still save context so user can ask follow-up questions about closed/merged PRs
                pr_context_for_storage = format_pr_for_conversation(pr_data)
                save_chat_context(chat_id, "pr", {
                    "url": pr_data["url"],
                    "title": pr.get("title"),
                    "formatted_context": pr_context_for_storage,
                    "files": [f.get("filename") for f in pr_data.get("files", [])]
                })
            else:
                # Only set pr_context for open PRs (for AI analysis)
                pr_context = format_pr_for_conversation(pr_data)
                # Save context for follow-ups
                save_chat_context(chat_id, "pr", {
                    "url": pr_data["url"],
                    "title": pr.get("title"),
                    "formatted_context": pr_context,
                    "files": [f.get("filename") for f in pr_data.get("files", [])]
                })

            print(f"Successfully fetched PR: {metadata['pr_title']} (state: {pr_state}, merged: {pr_merged})")

        except HTTPException as e:
            fetch_error = e.detail
        except Exception as e:
            fetch_error = str(e)

    # Fetch new commit if URL provided
    elif commit_info:
        owner, repo, commit_sha = commit_info
        try:
            commit_data = await fetch_github_commit(owner, repo, commit_sha)
            commit = commit_data.get("commit", {})
            commit_info_data = commit.get("commit", {})
            author = commit_info_data.get("author", {})
            stats = commit.get("stats", {})
            files = commit.get("files", [])

            # Build files list with patches for metadata
            files_list = []
            for f in (files or [])[:20]:
                files_list.append({
                    "filename": f.get("filename", ""),
                    "status": f.get("status", ""),
                    "additions": f.get("additions", 0),
                    "deletions": f.get("deletions", 0),
                    "patch": f.get("patch", "")[:5000] if f.get("patch") else ""
                })

            # Build full commit metadata (same as original)
            metadata = {
                "type": "commit",
                "commit_url": commit_data.get("url"),
                "commit_sha": commit.get("sha", "")[:7],
                "commit_sha_full": commit.get("sha", ""),
                "commit_message": commit_info_data.get("message", ""),
                "commit_author": author.get("name", "Unknown"),
                "commit_author_email": author.get("email", ""),
                "commit_date": author.get("date", ""),
                "files_changed": len(files) if files else 0,
                "additions": stats.get("additions", 0),
                "deletions": stats.get("deletions", 0),
                "total_changes": stats.get("total", 0),
                "files": files_list
            }

            commit_context = format_commit_for_conversation(commit_data)
            save_chat_context(chat_id, "commit", {
                "url": commit_data["url"],
                "sha": commit.get("sha", "")[:10],
                "message": commit_info_data.get("message", ""),
                "formatted_context": commit_context
            })

            print(f"Successfully fetched Commit: {metadata['commit_sha']} - {metadata['commit_message'][:50]}")
        except Exception as e:
            fetch_error = str(e)

    # Use existing context for follow-ups
    elif existing_context:
        context_data = json.loads(existing_context.get("context_data", "{}"))
        if existing_context.get("context_type") == "pr":
            pr_context = context_data.get("formatted_context", "")
        elif existing_context.get("context_type") == "commit":
            commit_context = context_data.get("formatted_context", "")

    # Get conversation history
    history = supabase.table("messages").select("role, content").eq("chat_id", chat_id).order("created_at").execute()

    if pr_context:
        if pr_info:  # NEW PR URL in this message → Structured analysis
            system_prompt = PR_ANALYSIS_SYSTEM_PROMPT
            user_content = f"{pr_context}\n\n---\n\nUser Request: {user_message}"
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
        else:  # Follow-up on existing PR → Conversational
            system_prompt = FOLLOW_UP_CONVERSATION_PROMPT
            user_content = f"""## PR Context (from earlier in conversation)
{pr_context}

---

**User's follow-up question:** {user_message}

Provide a detailed, conversational response. If they're asking about a specific class, model, file, or code pattern, show the actual code and explain it thoroughly."""
            messages = [{"role": "system", "content": system_prompt}]
            # Include conversation history for context
            for m in history.data[-6:]:
                messages.append({"role": m["role"], "content": m["content"]})
            messages.append({"role": "user", "content": user_content})

    elif commit_context:
        if commit_info:  # NEW commit URL → Structured analysis
            system_prompt = COMMIT_ANALYSIS_SYSTEM_PROMPT
            user_content = f"{commit_context}\n\n---\n\nUser Request: Analyze this commit and provide insights."
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
        else:  # Follow-up on existing commit → Conversational
            system_prompt = FOLLOW_UP_CONVERSATION_PROMPT
            user_content = f"""## Commit Context (from earlier in conversation)
{commit_context}

---

**User's follow-up question:** {user_message}

Provide a detailed, conversational response."""
            messages = [{"role": "system", "content": system_prompt}]
            for m in history.data[-6:]:
                messages.append({"role": m["role"], "content": m["content"]})
            messages.append({"role": "user", "content": user_content})

    else:
        # General conversation (no PR/commit context)
        system_prompt = GENERAL_QUERY_SYSTEM_PROMPT
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend([{"role": m["role"], "content": m["content"]} for m in history.data])

    async def generate():
        full_response = ""
        try:
            if fetch_error:
                error_msg = f"Hmm, I ran into an issue fetching that from GitHub: {fetch_error}\n\nCould you double-check the URL? Or if it's a private repo, make sure the GitHub token has access."
                yield f"data: {json.dumps({'content': error_msg})}\n\n"
                result = supabase.table("messages").insert({
                    "chat_id": chat_id, "role": "assistant", "content": error_msg
                }).execute()
                yield f"data: {json.dumps({'done': True, 'id': result.data[0]['id']})}\n\n"
                return

            # Send metadata first (for PR/Commit card in UI)
            if metadata:
                # Use pr_metadata or commit_metadata key based on type for frontend compatibility
                if metadata.get("type") == "pr":
                    yield f"data: {json.dumps({'pr_metadata': metadata})}\n\n"
                else:
                    yield f"data: {json.dumps({'commit_metadata': metadata})}\n\n"

            # Handle closed/merged PR - just show status message, no AI analysis
            if metadata and metadata.get("status_message"):
                status_msg = metadata["status_message"]
                yield f"data: {json.dumps({'content': status_msg})}\n\n"
                result = supabase.table("messages").insert({
                    "chat_id": chat_id, "role": "assistant", "content": status_msg
                }).execute()
                yield f"data: {json.dumps({'done': True, 'id': result.data[0]['id']})}\n\n"
                return

            stream = openai_client.chat.completions.create(
                model="gpt-4o", messages=messages, max_tokens=2000, stream=True, temperature=0.7
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield f"data: {json.dumps({'content': content})}\n\n"

            result = supabase.table("messages").insert({
                "chat_id": chat_id, "role": "assistant", "content": full_response
            }).execute()

            new_title = None
            if first_exchange:
                new_title = generate_chat_title(user_message, full_response)
                supabase.table("chats").update({"title": new_title}).eq("id", chat_id).execute()

            yield f"data: {json.dumps({'done': True, 'id': result.data[0]['id'], 'new_title': new_title})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})


class PRReviewRequest(BaseModel):
    pr_url: str
    review_type: str = Field(..., pattern="^(COMMENT|APPROVE|REQUEST_CHANGES)$")
    body: str = Field(default="")


@app.post("/pr/review")
async def submit_pr_review(review: PRReviewRequest):
    """Submit a review to a GitHub PR (Comment, Approve, or Request Changes)."""
    pr_info = extract_pr_url(review.pr_url)
    if not pr_info:
        raise HTTPException(status_code=400, detail="Invalid GitHub PR URL")

    if not GITHUB_TOKEN:
        raise HTTPException(status_code=401, detail="GitHub token not configured. Add GITHUB_TOKEN to your .env file.")

    owner, repo, pr_number = pr_info

    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {GITHUB_TOKEN}",
        "User-Agent": "PR-Assistant"
    }

    # GitHub API requires a body for REQUEST_CHANGES
    if review.review_type == "REQUEST_CHANGES" and not review.body.strip():
        raise HTTPException(status_code=400, detail="A comment is required when requesting changes.")

    review_data = {
        "event": review.review_type,
    }

    # Only include body if provided
    if review.body.strip():
        review_data["body"] = review.body

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/reviews"
            response = await client.post(url, headers=headers, json=review_data)

            if response.status_code in [200, 201]:
                result = response.json()
                return {
                    "success": True,
                    "review_id": result.get("id"),
                    "state": result.get("state"),
                    "message": f"Review submitted successfully as '{review.review_type}'"
                }
            elif response.status_code == 422:
                error_data = response.json()
                error_msg = error_data.get("message", "Validation failed")
                if "Can not approve your own pull request" in str(error_data):
                    raise HTTPException(status_code=422, detail="You cannot approve your own pull request.")
                raise HTTPException(status_code=422, detail=f"GitHub validation error: {error_msg}")
            elif response.status_code == 404:
                raise HTTPException(status_code=404, detail="PR not found or you don't have access.")
            elif response.status_code == 403:
                raise HTTPException(status_code=403,
                                    detail="Permission denied. Your token may not have write access to this repository.")
            else:
                raise HTTPException(status_code=response.status_code, detail=f"GitHub API error: {response.text}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit review: {str(e)}")


@app.post("/pr/comment")
async def add_pr_comment(pr_url: str, body: str):
    """Add a general comment to a PR (not a review)."""
    pr_info = extract_pr_url(pr_url)
    if not pr_info:
        raise HTTPException(status_code=400, detail="Invalid GitHub PR URL")

    if not GITHUB_TOKEN:
        raise HTTPException(status_code=401, detail="GitHub token not configured.")

    if not body.strip():
        raise HTTPException(status_code=400, detail="Comment body is required.")

    owner, repo, pr_number = pr_info

    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {GITHUB_TOKEN}",
        "User-Agent": "PR-Assistant"
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            url = f"https://api.github.com/repos/{owner}/{repo}/issues/{pr_number}/comments"
            response = await client.post(url, headers=headers, json={"body": body})

            if response.status_code == 201:
                result = response.json()
                return {
                    "success": True,
                    "comment_id": result.get("id"),
                    "html_url": result.get("html_url"),
                    "message": "Comment added successfully"
                }
            else:
                raise HTTPException(status_code=response.status_code, detail=f"GitHub API error: {response.text}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add comment: {str(e)}")


@app.post("/analyze-pr")
async def analyze_pr(pr_url: str, analysis_type: str = "full"):
    """Direct endpoint for PR analysis without chat context."""
    pr_info = extract_pr_url(pr_url)
    if not pr_info:
        raise HTTPException(status_code=400,
                            detail="Invalid GitHub PR URL. Expected format: https://github.com/owner/repo/pull/123")

    owner, repo, pr_number = pr_info
    pr_data = await fetch_github_pr(owner, repo, pr_number)
    pr_context = format_pr_for_conversation(pr_data)
    pr = pr_data.get("pr", {})

    query_map = {
        "full": "Provide a comprehensive analysis of this PR including summary, risks, and suggestions.",
        "summary": "Summarize what changed in this PR.",
        "risks": "Analyze the risks and potential breaking changes in this PR.",
        "review": "Perform a detailed code review of this PR as a senior developer."
    }

    query = query_map.get(analysis_type, query_map["full"])

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": PR_ANALYSIS_SYSTEM_PROMPT},
            {"role": "user", "content": f"{pr_context}\n\n---\n\nUser Request: {query}"}
        ],
        max_tokens=2000,
        temperature=0.3
    )
    analysis = response.choices[0].message.content

    try:
        parsed = json.loads(analysis)
    except:
        parsed = {"response": analysis}

    user = pr.get("user", {}) or {}

    return {
        "pr_url": pr_data.get("url", pr_url),
        "pr_title": pr.get("title", "Unknown"),
        "author": user.get("login", "Unknown"),
        "stats": {
            "files_changed": len(pr_data.get("files", [])),
            "additions": pr.get("additions", 0),
            "deletions": pr.get("deletions", 0)
        },
        "analysis": parsed
    }


@app.post("/analyze-commit")
async def analyze_commit(commit_url: str):
    """Direct endpoint for commit analysis without chat context."""
    commit_info = extract_commit_url(commit_url)
    if not commit_info:
        raise HTTPException(status_code=400,
                            detail="Invalid GitHub Commit URL. Expected format: https://github.com/owner/repo/commit/sha")

    owner, repo, commit_sha = commit_info
    commit_data = await fetch_github_commit(owner, repo, commit_sha)
    commit_context = format_commit_for_conversation(commit_data)
    commit = commit_data.get("commit", {})
    commit_info_data = commit.get("commit", {})
    author = commit_info_data.get("author", {})
    stats = commit.get("stats", {})

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": COMMIT_ANALYSIS_SYSTEM_PROMPT},
            {"role": "user", "content": f"{commit_context}\n\n---\n\nAnalyze this commit and provide insights."}
        ],
        max_tokens=2000,
        temperature=0.3
    )
    analysis = response.choices[0].message.content

    return {
        "commit_url": commit_data.get("url", commit_url),
        "commit_sha": commit.get("sha", "")[:7],
        "commit_message": commit_info_data.get("message", "").split("\n")[0],
        "author": author.get("name", "Unknown"),
        "date": author.get("date", ""),
        "stats": {
            "files_changed": len(commit.get("files", [])),
            "additions": stats.get("additions", 0),
            "deletions": stats.get("deletions", 0)
        },
        "analysis": {"response": analysis}
    }


@app.get("/health")
async def health_check():
    return {"status": "ok", "database": "supabase", "ai": "openai",
            "github": "configured" if GITHUB_TOKEN else "anonymous"}


# Debug endpoint to test GitHub connectivity
@app.get("/test-github/{owner}/{repo}/{pr_number}")
async def test_github_pr(owner: str, repo: str, pr_number: int):
    """Test endpoint to verify GitHub API connectivity."""
    try:
        pr_data = await fetch_github_pr(owner, repo, pr_number)
        pr = pr_data.get("pr", {})

        commits = []
        for commit in pr_data.get("commits", [])[:10]:
            commit_info = commit.get("commit", {})
            commits.append({
                "sha": commit.get("sha", "")[:7],
                "message": commit_info.get("message", "").split("\n")[0][:100],
            })

        files = []
        for f in pr_data.get("files", [])[:20]:
            files.append({
                "filename": f.get("filename", ""),
                "status": f.get("status", ""),
                "additions": f.get("additions", 0),
                "deletions": f.get("deletions", 0)
            })

        return {
            "success": True,
            "pr_title": pr.get("title"),
            "pr_state": pr.get("state"),
            "pr_merged": pr.get("merged"),
            "pr_body": (pr.get("body") or "")[:200],
            "pr_author": (pr.get("user") or {}).get("login"),
            "files_count": len(pr_data.get("files", [])),
            "commits_count": len(pr_data.get("commits", [])),
            "commits": commits,
            "files": files,
            "github_token_configured": bool(GITHUB_TOKEN)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "github_token_configured": bool(GITHUB_TOKEN)
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)