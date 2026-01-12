from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from supabase import create_client, Client
from openai import OpenAI
from dotenv import load_dotenv
from promts.chat_with_pr_contect_prompt import CHAT_WITH_PR_CONTEXT_PROMPT
from promts.pr_analysis_system_prompt import PR_ANALYSIS_SYSTEM_PROMPT

import os
import json
import re
import httpx

load_dotenv()

# Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")  # Add this to your .env

# Initialize clients
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


# Pydantic Models
class ChatCreate(BaseModel):
    title: str = Field(default="New Chat", max_length=200)


class ChatUpdate(BaseModel):
    title: str = Field(..., max_length=200)


class MessageCreate(BaseModel):
    chat_id: str
    message: str = Field(..., min_length=1)


class PRAnalysisResponse(BaseModel):
    summary: str
    risk_level: str  # low, medium, high, critical
    risk_details: List[str]
    key_changes: List[str]
    suggestions: List[str]
    breaking_changes: List[str]
    files_affected: int
    lines_added: int
    lines_deleted: int


# GitHub PR URL pattern
PR_URL_PATTERN = r'https://github\.com/([^/]+)/([^/]+)/pull/(\d+)'


async def fetch_github_pr(owner: str, repo: str, pr_number: int) -> Dict[str, Any]:
    """Fetch PR details from GitHub API."""
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "PR-Assistant"
    }
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    else:
        print("WARNING: No GITHUB_TOKEN set. API rate limits will be very restrictive.")

    print(f"Fetching PR: {owner}/{repo}#{pr_number}")

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Fetch PR details
            pr_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}"
            pr_response = await client.get(pr_url, headers=headers)

            print(f"GitHub PR response status: {pr_response.status_code}")

            if pr_response.status_code == 404:
                raise HTTPException(status_code=404,
                                    detail=f"PR #{pr_number} not found in {owner}/{repo}. It may not exist or you don't have access to this repository.")
            elif pr_response.status_code == 403:
                error_data = pr_response.json() if pr_response.text else {}
                if "rate limit" in error_data.get("message", "").lower():
                    raise HTTPException(status_code=403,
                                        detail="GitHub API rate limit exceeded. Please add a GITHUB_TOKEN to your .env file.")
                raise HTTPException(status_code=403,
                                    detail=f"Access denied to {owner}/{repo}. For private repos, ensure your GITHUB_TOKEN has 'repo' scope.")
            elif pr_response.status_code != 200:
                raise HTTPException(status_code=pr_response.status_code,
                                    detail=f"GitHub API error (status {pr_response.status_code}): {pr_response.text[:200]}")

            pr_data = pr_response.json()

            if not pr_data:
                raise HTTPException(status_code=500, detail="GitHub returned empty PR data")

            print(f"PR Title: {pr_data.get('title', 'N/A')}, State: {pr_data.get('state', 'N/A')}")

            # Fetch PR diff
            diff_content = ""
            try:
                diff_headers = {**headers, "Accept": "application/vnd.github.v3.diff"}
                diff_response = await client.get(pr_url, headers=diff_headers)
                if diff_response.status_code == 200:
                    diff_content = diff_response.text or ""
            except Exception as e:
                print(f"Warning: Failed to fetch diff: {e}")

            # Fetch PR files
            files_data = []
            try:
                files_url = f"{pr_url}/files"
                files_response = await client.get(files_url, headers=headers)
                if files_response.status_code == 200:
                    files_data = files_response.json() or []
            except Exception as e:
                print(f"Warning: Failed to fetch files: {e}")

            # Fetch PR commits
            commits_data = []
            try:
                commits_url = f"{pr_url}/commits"
                commits_response = await client.get(commits_url, headers=headers)
                if commits_response.status_code == 200:
                    commits_data = commits_response.json() or []
            except Exception as e:
                print(f"Warning: Failed to fetch commits: {e}")

            # Fetch PR comments
            comments_data = []
            try:
                comments_url = f"{pr_url}/comments"
                comments_response = await client.get(comments_url, headers=headers)
                if comments_response.status_code == 200:
                    comments_data = comments_response.json() or []
            except Exception as e:
                print(f"Warning: Failed to fetch comments: {e}")

            return {
                "pr": pr_data,
                "diff": diff_content[:50000] if diff_content else "",
                "files": files_data[:100] if files_data else [],
                "commits": commits_data[:50] if commits_data else [],
                "comments": comments_data[:50] if comments_data else [],
                "url": f"https://github.com/{owner}/{repo}/pull/{pr_number}"
            }
    except HTTPException:
        raise
    except httpx.TimeoutException:
        raise HTTPException(status_code=504,
                            detail=f"Request to GitHub timed out. The repository {owner}/{repo} may be slow to respond.")
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Network error connecting to GitHub: {str(e)}")
    except Exception as e:
        print(f"Unexpected error in fetch_github_pr: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error fetching PR: {type(e).__name__}: {str(e)}")


def extract_pr_url(message: str) -> Optional[tuple]:
    """Extract GitHub PR URL from message."""
    match = re.search(PR_URL_PATTERN, message)
    if match:
        result = (match.group(1), match.group(2), int(match.group(3)))
        print(f"Extracted PR URL: owner={result[0]}, repo={result[1]}, pr={result[2]}")
        return result
    print(f"No PR URL found in message: {message[:100]}...")
    return None


def format_pr_context(pr_data: Dict[str, Any]) -> str:
    """Format PR data for AI context."""
    pr = pr_data.get("pr", {})
    files = pr_data.get("files", [])
    diff = pr_data.get("diff", "")

    # Safely get nested values
    user = pr.get("user", {}) or {}
    base = pr.get("base", {}) or {}
    head = pr.get("head", {}) or {}

    files_summary = "\n".join([
        f"- {f.get('filename', 'unknown')} (+{f.get('additions', 0)}/-{f.get('deletions', 0)}) [{f.get('status', 'unknown')}]"
        for f in (files or [])[:30]
    ]) or "No files information available"

    return f"""
## Pull Request: {pr.get('title', 'Unknown Title')}
**URL:** {pr_data.get('url', 'N/A')}
**Author:** {user.get('login', 'Unknown')}
**State:** {pr.get('state', 'unknown')}
**Base Branch:** {base.get('ref', 'unknown')} ← **Head Branch:** {head.get('ref', 'unknown')}
**Created:** {pr.get('created_at', 'N/A')}
**Updated:** {pr.get('updated_at', 'N/A')}

### Description:
{(pr.get('body') or 'No description provided')[:2000]}

### Statistics:
- Files Changed: {len(files) if files else 0}
- Additions: {pr.get('additions', 0)}
- Deletions: {pr.get('deletions', 0)}
- Commits: {pr.get('commits', 0)}

### Files Changed:
{files_summary}

### Code Diff (truncated):
```diff
{diff[:30000] if diff else 'No diff available'}
```
"""


def generate_chat_title(user_message: str, ai_response: str) -> str:
    """Generate a short title for the chat."""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system",
                 "content": "Generate a short title (3-6 words max) for this conversation. Return ONLY the title, no quotes."},
                {"role": "user",
                 "content": f"User asked: {user_message[:200]}\n\nAssistant replied: {ai_response[:300]}"}
            ],
            max_tokens=20,
            temperature=0.7
        )
        title = response.choices[0].message.content.strip().strip('"\'')
        return title[:50] if len(title) > 50 else title
    except:
        return "PR Review Chat"


def is_first_exchange(chat_id: str) -> bool:
    response = supabase.table("messages").select("id", count="exact").eq("chat_id", chat_id).execute()
    return response.count <= 1


async def analyze_pr_with_ai(pr_context: str, user_query: str, is_structured: bool = True) -> str:
    """Analyze PR using OpenAI."""
    system_prompt = PR_ANALYSIS_SYSTEM_PROMPT if is_structured else CHAT_WITH_PR_CONTEXT_PROMPT

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{pr_context}\n\n---\n\nUser Request: {user_query}"}
        ],
        max_tokens=2000,
        temperature=0.3
    )
    return response.choices[0].message.content


def detect_query_type(message: str) -> str:
    """Detect the type of PR-related query."""
    message_lower = message.lower()
    if any(word in message_lower for word in ['summarize', 'summary', 'what changed', 'overview']):
        return 'summarize'
    elif any(word in message_lower for word in ['review', 'code review', 'feedback']):
        return 'review'
    elif any(word in message_lower for word in ['risk', 'risky', 'dangerous', 'breaking', 'break']):
        return 'risk_analysis'
    elif any(word in message_lower for word in ['suggest', 'improve', 'issues', 'follow-up']):
        return 'suggestions'
    return 'general'


# Existing endpoints
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
    if not response.data:
        raise HTTPException(status_code=404, detail="Chat not found")
    return response.data[0]


@app.delete("/chats/{chat_id}")
async def delete_chat(chat_id: str):
    supabase.table("messages").delete().eq("chat_id", chat_id).execute()
    supabase.table("chats").delete().eq("id", chat_id).execute()
    return {"success": True, "deleted_id": chat_id}


@app.get("/chats/{chat_id}/messages")
async def get_messages(chat_id: str):
    response = supabase.table("messages").select("*").eq("chat_id", chat_id).order("created_at").execute()
    return response.data


# Enhanced chat endpoint with PR intelligence
@app.post("/chat")
async def send_message(message_data: MessageCreate):
    user_message = message_data.message
    chat_id = message_data.chat_id
    first_exchange = is_first_exchange(chat_id)

    # Save user message
    supabase.table("messages").insert({
        "chat_id": chat_id, "role": "user", "content": user_message
    }).execute()

    # Check for PR URL
    pr_info = extract_pr_url(user_message)
    pr_context = None
    pr_metadata = None
    pr_fetch_error = None

    if pr_info:
        owner, repo, pr_number = pr_info
        try:
            pr_data = await fetch_github_pr(owner, repo, pr_number)
            pr_context = format_pr_context(pr_data)
            pr_metadata = {
                "pr_url": pr_data["url"],
                "pr_title": pr_data["pr"]["title"],
                "files_changed": len(pr_data["files"]),
                "additions": pr_data["pr"]["additions"],
                "deletions": pr_data["pr"]["deletions"]
            }
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"GitHub API error: {str(e)}")

    # Get conversation history
    history = supabase.table("messages").select("role, content").eq("chat_id", chat_id).order("created_at").execute()

    # Build messages
    if pr_context:
        query_type = detect_query_type(user_message)
        is_structured = query_type in ['summarize', 'review', 'risk_analysis']
        ai_response = await analyze_pr_with_ai(pr_context, user_message, is_structured)
    else:
        messages = [{"role": "system", "content": CHAT_WITH_PR_CONTEXT_PROMPT}]
        messages.extend([{"role": m["role"], "content": m["content"]} for m in history.data])

        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini", messages=messages, max_tokens=1500
            )
            ai_response = response.choices[0].message.content
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"AI error: {str(e)}")

    # Save assistant response
    assistant_msg = supabase.table("messages").insert({
        "chat_id": chat_id, "role": "assistant", "content": ai_response
    }).execute()

    # Generate title if first exchange
    new_title = None
    if first_exchange:
        new_title = generate_chat_title(user_message, ai_response)
        supabase.table("chats").update({"title": new_title}).eq("id", chat_id).execute()

    return {
        "id": assistant_msg.data[0]["id"],
        "role": "assistant",
        "content": ai_response,
        "new_title": new_title,
        "pr_metadata": pr_metadata
    }


# Streaming endpoint with PR intelligence
@app.post("/chat/stream")
async def send_message_stream(message_data: MessageCreate):
    user_message = message_data.message
    chat_id = message_data.chat_id
    first_exchange = is_first_exchange(chat_id)

    supabase.table("messages").insert({
        "chat_id": chat_id, "role": "user", "content": user_message
    }).execute()

    # Check for PR URL
    pr_info = extract_pr_url(user_message)
    pr_context = None
    pr_metadata = None
    pr_fetch_error = None

    if pr_info:
        owner, repo, pr_number = pr_info
        try:
            pr_data = await fetch_github_pr(owner, repo, pr_number)
            pr_context = format_pr_context(pr_data)
            pr = pr_data.get("pr", {})
            pr_state = pr.get("state", "unknown")
            pr_merged = pr.get("merged", False)

            # Extract commits
            commits = []
            for commit in pr_data.get("commits", [])[:10]:
                commit_info = commit.get("commit", {})
                commits.append({
                    "sha": commit.get("sha", "")[:7],
                    "message": commit_info.get("message", "").split("\n")[0][:100],
                    "author": commit_info.get("author", {}).get("name", "Unknown")
                })

            # Extract files
            files = []
            for f in pr_data.get("files", [])[:20]:
                files.append({
                    "filename": f.get("filename", ""),
                    "status": f.get("status", ""),
                    "additions": f.get("additions", 0),
                    "deletions": f.get("deletions", 0)
                })

            pr_metadata = {
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

            # Check if PR is closed or merged - skip analysis
            if pr_state == "closed" or pr_merged:
                merged_by = (pr.get("merged_by") or {}).get("login", "")
                if pr_merged:
                    status_message = f"This PR has already been merged{' by ' + merged_by if merged_by else ''}. No analysis needed."
                else:
                    status_message = "This PR has been closed without merging. No analysis needed."
                pr_metadata["status_message"] = status_message
                pr_context = None  # Skip analysis

            print(f"Successfully fetched PR: {pr_metadata['pr_title']} (state: {pr_state}, merged: {pr_merged})")
        except HTTPException as e:
            print(f"HTTP ERROR fetching PR: {e.detail}")
            pr_fetch_error = e.detail
        except Exception as e:
            print(f"ERROR fetching PR: {str(e)}")
            pr_fetch_error = str(e)

    history = supabase.table("messages").select("role, content").eq("chat_id", chat_id).order("created_at").execute()

    if pr_context:
        system_prompt = CHAT_WITH_PR_CONTEXT_PROMPT
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{pr_context}\n\n---\n\nUser Request: {user_message}"}
        ]
    else:
        messages = [{"role": "system", "content": CHAT_WITH_PR_CONTEXT_PROMPT}]
        messages.extend([{"role": m["role"], "content": m["content"]} for m in history.data])

    async def generate():
        full_response = ""
        try:
            # Handle PR fetch error
            if pr_info and pr_fetch_error:
                error_msg = f"Unable to fetch the PR from GitHub. {pr_fetch_error}"
                yield f"data: {json.dumps({'content': error_msg})}\n\n"

                result = supabase.table("messages").insert({
                    "chat_id": chat_id, "role": "assistant", "content": error_msg
                }).execute()
                yield f"data: {json.dumps({'done': True, 'id': result.data[0]['id'], 'new_title': None})}\n\n"
                return

            # Handle closed/merged PR - just show status, no analysis
            if pr_metadata and pr_metadata.get("status_message"):
                status_msg = pr_metadata["status_message"]
                yield f"data: {json.dumps({'pr_metadata': pr_metadata})}\n\n"
                yield f"data: {json.dumps({'content': status_msg})}\n\n"

                result = supabase.table("messages").insert({
                    "chat_id": chat_id, "role": "assistant", "content": status_msg
                }).execute()
                yield f"data: {json.dumps({'done': True, 'id': result.data[0]['id'], 'new_title': None})}\n\n"
                return

            # Send PR metadata first if available
            if pr_metadata:
                yield f"data: {json.dumps({'pr_metadata': pr_metadata})}\n\n"

            stream = openai_client.chat.completions.create(
                model="gpt-4o", messages=messages, max_tokens=2000, stream=True
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

    return StreamingResponse(
        generate(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
    )


# Direct PR Analysis endpoint
@app.post("/analyze-pr")
async def analyze_pr(pr_url: str, analysis_type: str = "full"):
    """Direct endpoint for PR analysis without chat context."""
    pr_info = extract_pr_url(pr_url)
    if not pr_info:
        raise HTTPException(status_code=400,
                            detail="Invalid GitHub PR URL. Expected format: https://github.com/owner/repo/pull/123")

    owner, repo, pr_number = pr_info
    pr_data = await fetch_github_pr(owner, repo, pr_number)
    pr_context = format_pr_context(pr_data)
    pr = pr_data.get("pr", {})

    query_map = {
        "full": "Provide a comprehensive analysis of this PR including summary, risks, and suggestions.",
        "summary": "Summarize what changed in this PR.",
        "risks": "Analyze the risks and potential breaking changes in this PR.",
        "review": "Perform a detailed code review of this PR as a senior developer."
    }

    query = query_map.get(analysis_type, query_map["full"])
    analysis = await analyze_pr_with_ai(pr_context, query, is_structured=True)

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

        # Build the same metadata as streaming endpoint
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