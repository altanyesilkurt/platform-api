from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from supabase import create_client, Client
from openai import OpenAI
from dotenv import load_dotenv
from promts.chat_with_pr_context_promt import CHAT_WITH_PR_CONTEXT_PROMPT
from promts.commit_analysis_system_prompt import COMMIT_ANALYSIS_SYSTEM_PROMPT
from promts.general_query_system_prompt import GENERAL_QUERY_SYSTEM_PROMPT
from promts.pr_analysis_system_prompt import PR_ANALYSIS_SYSTEM_PROMPT
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
    analyze_merged: bool = Field(default=True, description="Whether to analyze merged/closed PRs")


# GitHub URL patterns
PR_URL_PATTERN = r'https://github\.com/([^/]+)/([^/]+)/pull/(\d+)'
COMMIT_URL_PATTERN = r'https://github\.com/([^/]+)/([^/]+)/commit/([a-fA-F0-9]+)'

PR_RELATED_KEYWORDS = [
    'pull request', 'pr', 'code review', 'github', 'merge', 'commit', 'branch',
    'diff', 'repository', 'repo', 'git', 'review this', 'analyze this pr',
    'check this pr', 'look at this pr', 'examine this', 'code changes',
    'file changes', 'breaking changes', 'risk assessment', 'code analysis',
    '/pull/', '/commit/', 'github.com', 'approve', 'request changes', 'mergeable',
    'conflicts', 'base branch', 'head branch', 'squash', 'rebase',
    'analyze commit', 'review commit', 'commit changes', 'sha',
    'merged pr', 'what was merged', 'merge history', 'merged changes'
]

# Updated system prompt to handle merged PRs
MERGED_PR_ANALYSIS_PROMPT = """You are a Senior Software Engineer reviewing a MERGED Pull Request. 
This PR has already been merged into the codebase. Your analysis should focus on:

1. **Understanding what was merged** - Summarize the changes that are now in the codebase
2. **Post-merge review** - Identify any concerns that might need follow-up issues
3. **Documentation** - Help understand the historical context of these changes
4. **Learning** - Extract lessons or patterns from this merged code

Since this PR is already merged, focus on documentation and understanding rather than approval/rejection.

FORMATTING RULES:
- Use ## for section headers
- Use *italics* for class/variable names
- Use `backticks` for file names
- Use **bold** for important terms
- Show code in fenced blocks with language specified
- Leave blank lines between sections"""


def is_pr_related_query(message: str, chat_history: List[Dict] = None) -> bool:
    """Determine if the query is related to GitHub PR/code review/commit."""
    message_lower = message.lower()
    if re.search(PR_URL_PATTERN, message) or re.search(COMMIT_URL_PATTERN, message):
        return True
    for keyword in PR_RELATED_KEYWORDS:
        if keyword in message_lower:
            return True
    if chat_history:
        for msg in chat_history[-5:]:
            content = msg.get("content", "").lower()
            if re.search(PR_URL_PATTERN, content) or re.search(COMMIT_URL_PATTERN, content):
                return True
            if any(kw in content for kw in ['pull request', 'github.com/pull', 'pr #', 'this pr', 'this commit']):
                return True
    return False


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
    """Fetch PR details including merge information."""
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "PR-Assistant"
    }
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
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

            diff_content = ""
            try:
                diff_headers = {**headers, "Accept": "application/vnd.github.v3.diff"}
                diff_response = await client.get(pr_url, headers=diff_headers)
                if diff_response.status_code == 200:
                    diff_content = diff_response.text or ""
            except Exception as e:
                print(f"Warning: Failed to fetch diff: {e}")

            files_data = []
            try:
                files_response = await client.get(f"{pr_url}/files", headers=headers)
                if files_response.status_code == 200:
                    files_data = files_response.json() or []
            except Exception:
                pass

            # Fetch commits
            commits_data = []
            try:
                commits_response = await client.get(f"{pr_url}/commits", headers=headers)
                if commits_response.status_code == 200:
                    commits_data = commits_response.json() or []
            except Exception:
                pass

            # Fetch comments
            comments_data = []
            try:
                comments_response = await client.get(f"{pr_url}/comments", headers=headers)
                if comments_response.status_code == 200:
                    comments_data = comments_response.json() or []
            except Exception:
                pass

            # Fetch review comments (inline code comments)
            review_comments = []
            try:
                review_comments_response = await client.get(f"{pr_url}/comments", headers=headers)
                if review_comments_response.status_code == 200:
                    review_comments = review_comments_response.json() or []
            except Exception:
                pass

            # Fetch reviews (approvals, change requests)
            reviews = []
            try:
                reviews_response = await client.get(f"{pr_url}/reviews", headers=headers)
                if reviews_response.status_code == 200:
                    reviews = reviews_response.json() or []
            except Exception:
                pass

            return {
                "pr": pr_data,
                "diff": diff_content[:50000] if diff_content else "",
                "files": files_data[:100] if files_data else [],
                "commits": commits_data[:50] if commits_data else [],
                "comments": comments_data[:50] if comments_data else [],
                "review_comments": review_comments[:50] if review_comments else [],
                "reviews": reviews[:20] if reviews else [],
                "url": f"https://github.com/{owner}/{repo}/pull/{pr_number}"
            }
    except HTTPException:
        raise
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Request to GitHub timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


async def fetch_github_commit(owner: str, repo: str, commit_sha: str) -> Dict[str, Any]:
    """Fetch commit details from GitHub API."""
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "PR-Assistant"
    }
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            commit_url = f"https://api.github.com/repos/{owner}/{repo}/commits/{commit_sha}"
            response = await client.get(commit_url, headers=headers)

            if response.status_code == 404:
                raise HTTPException(status_code=404, detail=f"Commit {commit_sha[:7]} not found")
            elif response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail="GitHub API error")

            return {
                "commit": response.json(),
                "url": f"https://github.com/{owner}/{repo}/commit/{commit_sha}"
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


def format_pr_context(pr_data: Dict[str, Any], include_merge_info: bool = True) -> str:
    """Format PR data for AI context, including merge information."""
    pr = pr_data.get("pr", {})
    files = pr_data.get("files", [])
    diff = pr_data.get("diff", "")
    commits = pr_data.get("commits", [])
    reviews = pr_data.get("reviews", [])

    user = pr.get("user", {}) or {}
    base = pr.get("base", {}) or {}
    head = pr.get("head", {}) or {}

    # Merge information
    merge_info = ""
    if pr.get("merged"):
        merged_by = (pr.get("merged_by") or {}).get("login", "Unknown")
        merge_info = f"""
### Merge Information:
- **Status:** ✅ MERGED
- **Merged By:** {merged_by}
- **Merged At:** {pr.get('merged_at', 'N/A')}
- **Merge Commit SHA:** {pr.get('merge_commit_sha', 'N/A')[:7] if pr.get('merge_commit_sha') else 'N/A'}
"""
    elif pr.get("state") == "closed":
        merge_info = "\n### Status: ❌ CLOSED (Not Merged)\n"

    files_summary = "\n".join([
        f"- `{f.get('filename', 'unknown')}` (+{f.get('additions', 0)}/-{f.get('deletions', 0)}) [{f.get('status', 'unknown')}]"
        for f in (files or [])[:30]
    ]) or "No files information available"

    # Commits summary
    commits_summary = "\n".join([
        f"- `{c.get('sha', '')[:7]}` - {c.get('commit', {}).get('message', '').split(chr(10))[0][:80]}"
        for c in (commits or [])[:15]
    ]) or "No commits information"

    # Reviews summary
    reviews_summary = ""
    if reviews:
        reviews_summary = "\n### Reviews:\n" + "\n".join([
            f"- **{r.get('user', {}).get('login', 'Unknown')}**: {r.get('state', 'UNKNOWN')}"
            for r in reviews[:10]
        ])

    return f"""
## Pull Request: {pr.get('title', 'Unknown Title')}
**URL:** {pr_data.get('url', 'N/A')}
**Author:** {user.get('login', 'Unknown')}
**State:** {pr.get('state', 'unknown')}
**Base Branch:** {base.get('ref', 'unknown')} ← **Head Branch:** {head.get('ref', 'unknown')}
**Created:** {pr.get('created_at', 'N/A')}
**Updated:** {pr.get('updated_at', 'N/A')}
{merge_info if include_merge_info else ''}
### Description:
{(pr.get('body') or 'No description provided')[:2000]}

### Statistics:
- Files Changed: {len(files) if files else 0}
- Additions: {pr.get('additions', 0)}
- Deletions: {pr.get('deletions', 0)}
- Total Commits: {pr.get('commits', 0)}

### Commits:
{commits_summary}

### Files Changed:
{files_summary}
{reviews_summary}
### Code Diff (truncated):
```diff
{diff[:30000] if diff else 'No diff available'}
```
"""


def format_commit_context(commit_data: Dict[str, Any]) -> str:
    """Format commit data for AI context."""
    commit = commit_data.get("commit", {})
    commit_info = commit.get("commit", {})
    author = commit_info.get("author", {})
    stats = commit.get("stats", {})
    files = commit.get("files", [])

    files_summary = "\n".join([
        f"- `{f.get('filename', 'unknown')}` (+{f.get('additions', 0)}/-{f.get('deletions', 0)})"
        for f in (files or [])[:30]
    ]) or "No files information"

    patches = ""
    for f in (files or [])[:10]:
        if f.get("patch"):
            patches += f"\n### `{f.get('filename')}`\n```diff\n{f['patch'][:3000]}\n```\n"

    return f"""
## Commit: {commit_info.get('message', 'No message')[:200]}
**URL:** {commit_data.get('url', 'N/A')}
**SHA:** {commit.get('sha', 'unknown')}
**Author:** {author.get('name', 'Unknown')}
**Date:** {author.get('date', 'N/A')}

### Statistics:
- Files: {len(files) if files else 0}
- Additions: {stats.get('additions', 0)}
- Deletions: {stats.get('deletions', 0)}

### Files:
{files_summary}

### Changes:
{patches if patches else 'No patch data'}
"""


def generate_chat_title(user_message: str, ai_response: str) -> str:
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Generate a short title (3-6 words). Return ONLY the title."},
                {"role": "user", "content": f"User: {user_message[:200]}\nAssistant: {ai_response[:300]}"}
            ],
            max_tokens=20
        )
        return response.choices[0].message.content.strip().strip('"\'')[:50]
    except:
        return "PR Review Chat"


def is_first_exchange(chat_id: str) -> bool:
    response = supabase.table("messages").select("id", count="exact").eq("chat_id", chat_id).execute()
    return response.count <= 1


# ============ ENDPOINTS ============

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


@app.post("/chat/stream")
async def send_message_stream(message_data: MessageCreate):
    user_message = message_data.message
    chat_id = message_data.chat_id
    analyze_merged = message_data.analyze_merged
    first_exchange = is_first_exchange(chat_id)

    supabase.table("messages").insert({
        "chat_id": chat_id, "role": "user", "content": user_message
    }).execute()

    pr_info = extract_pr_url(user_message)
    commit_info = extract_commit_url(user_message)

    pr_context = None
    commit_context = None
    pr_metadata = None
    commit_metadata = None
    fetch_error = None

    # Handle PR URL
    if pr_info:
        owner, repo, pr_number = pr_info
        try:
            pr_data = await fetch_github_pr(owner, repo, pr_number)
            pr = pr_data.get("pr", {})
            pr_state = pr.get("state", "unknown")
            pr_merged = pr.get("merged", False)

            # Always format context (including for merged PRs)
            pr_context = format_pr_context(pr_data, include_merge_info=True)

            # Extract commits
            commits = [{
                "sha": c.get("sha", "")[:7],
                "sha_full": c.get("sha", ""),
                "message": c.get("commit", {}).get("message", "").split("\n")[0][:100],
                "author": c.get("commit", {}).get("author", {}).get("name", "Unknown"),
                "date": c.get("commit", {}).get("author", {}).get("date", "")
            } for c in pr_data.get("commits", [])[:20]]

            # Extract files with patches
            files = [{
                "filename": f.get("filename", ""),
                "status": f.get("status", ""),
                "additions": f.get("additions", 0),
                "deletions": f.get("deletions", 0),
                "patch": f.get("patch", "")[:5000] if f.get("patch") else ""
            } for f in pr_data.get("files", [])[:30]]

            # Extract reviews
            reviews = [{
                "user": r.get("user", {}).get("login", "Unknown"),
                "state": r.get("state", ""),
                "body": r.get("body", "")[:500] if r.get("body") else "",
                "submitted_at": r.get("submitted_at", "")
            } for r in pr_data.get("reviews", [])[:10]]

            pr_metadata = {
                "pr_url": pr_data.get("url"),
                "pr_title": pr.get("title", "Unknown PR"),
                "pr_body": (pr.get("body") or "")[:500],
                "pr_state": pr_state,
                "pr_merged": pr_merged,
                "pr_author": (pr.get("user") or {}).get("login", "Unknown"),
                "base_branch": (pr.get("base") or {}).get("ref", "unknown"),
                "head_branch": (pr.get("head") or {}).get("ref", "unknown"),
                "files_changed": len(pr_data.get("files", [])),
                "additions": pr.get("additions", 0),
                "deletions": pr.get("deletions", 0),
                "commits_count": pr.get("commits", 0),
                "commits": commits,
                "files": files,
                "reviews": reviews,
                "created_at": pr.get("created_at", ""),
                "updated_at": pr.get("updated_at", ""),
                # Merge-specific fields
                "merged_at": pr.get("merged_at") if pr_merged else None,
                "merged_by": (pr.get("merged_by") or {}).get("login") if pr_merged else None,
                "merge_commit_sha": pr.get("merge_commit_sha") if pr_merged else None,
            }

            # Only skip analysis if explicitly requested AND PR is merged/closed
            if not analyze_merged and (pr_state == "closed" or pr_merged):
                status = "merged" if pr_merged else "closed"
                pr_metadata["skip_analysis"] = True
                pr_metadata["status_message"] = f"This PR has been {status}. Analysis skipped (analyze_merged=false)."
                pr_context = None

        except HTTPException as e:
            fetch_error = e.detail
        except Exception as e:
            fetch_error = str(e)

    # Handle Commit URL
    elif commit_info:
        owner, repo, commit_sha = commit_info
        try:
            commit_data = await fetch_github_commit(owner, repo, commit_sha)
            commit_context = format_commit_context(commit_data)
            commit = commit_data.get("commit", {})
            commit_info_data = commit.get("commit", {})
            author = commit_info_data.get("author", {})
            stats = commit.get("stats", {})
            files = commit.get("files", [])

            commit_metadata = {
                "type": "commit",
                "commit_url": commit_data.get("url"),
                "commit_sha": commit.get("sha", "")[:7],
                "commit_sha_full": commit.get("sha", ""),
                "commit_message": commit_info_data.get("message", ""),
                "commit_author": author.get("name", "Unknown"),
                "commit_date": author.get("date", ""),
                "files_changed": len(files) if files else 0,
                "additions": stats.get("additions", 0),
                "deletions": stats.get("deletions", 0),
                "files": [{
                    "filename": f.get("filename", ""),
                    "status": f.get("status", ""),
                    "additions": f.get("additions", 0),
                    "deletions": f.get("deletions", 0),
                    "patch": f.get("patch", "")[:5000] if f.get("patch") else ""
                } for f in (files or [])[:20]]
            }
        except HTTPException as e:
            fetch_error = e.detail
        except Exception as e:
            fetch_error = str(e)

    history = supabase.table("messages").select("role, content").eq("chat_id", chat_id).order("created_at").execute()

    # Determine system prompt based on context
    if pr_context:
        # Use merged PR prompt if PR is merged
        is_merged = pr_metadata and pr_metadata.get("pr_merged")
        system_prompt = MERGED_PR_ANALYSIS_PROMPT if is_merged else CHAT_WITH_PR_CONTEXT_PROMPT
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{pr_context}\n\n---\n\nUser Request: {user_message}"}
        ]
        max_tokens = 2500
    elif commit_context:
        messages = [
            {"role": "system", "content": COMMIT_ANALYSIS_SYSTEM_PROMPT},
            {"role": "user", "content": f"{commit_context}\n\n---\n\nAnalyze this commit."}
        ]
        max_tokens = 2000
    else:
        is_pr_query = is_pr_related_query(user_message, history.data)
        system_prompt = CHAT_WITH_PR_CONTEXT_PROMPT if is_pr_query else GENERAL_QUERY_SYSTEM_PROMPT
        max_tokens = 2000 if is_pr_query else 4000
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend([{"role": m["role"], "content": m["content"]} for m in history.data])

    async def generate():
        full_response = ""
        try:
            # Handle fetch error
            if (pr_info or commit_info) and fetch_error:
                error_msg = f"Unable to fetch from GitHub: {fetch_error}"
                yield f"data: {json.dumps({'content': error_msg})}\n\n"
                result = supabase.table("messages").insert({
                    "chat_id": chat_id, "role": "assistant", "content": error_msg
                }).execute()
                yield f"data: {json.dumps({'done': True, 'id': result.data[0]['id']})}\n\n"
                return

            # Handle skipped analysis (only when analyze_merged=false)
            if pr_metadata and pr_metadata.get("skip_analysis"):
                yield f"data: {json.dumps({'pr_metadata': pr_metadata})}\n\n"
                msg = pr_metadata["status_message"]
                yield f"data: {json.dumps({'content': msg})}\n\n"
                result = supabase.table("messages").insert({
                    "chat_id": chat_id, "role": "assistant", "content": msg
                }).execute()
                yield f"data: {json.dumps({'done': True, 'id': result.data[0]['id']})}\n\n"
                return

            # Send metadata first
            if pr_metadata:
                yield f"data: {json.dumps({'pr_metadata': pr_metadata})}\n\n"
            elif commit_metadata:
                yield f"data: {json.dumps({'commit_metadata': commit_metadata})}\n\n"

            # Stream AI response
            stream = openai_client.chat.completions.create(
                model="gpt-4o", messages=messages, max_tokens=max_tokens, stream=True
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


@app.get("/pr/{owner}/{repo}/{pr_number}")
async def get_pr_details(owner: str, repo: str, pr_number: int):
    """Get detailed PR information including merge status."""
    try:
        pr_data = await fetch_github_pr(owner, repo, pr_number)
        pr = pr_data.get("pr", {})

        return {
            "success": True,
            "pr_url": pr_data.get("url"),
            "title": pr.get("title"),
            "state": pr.get("state"),
            "merged": pr.get("merged", False),
            "merged_at": pr.get("merged_at"),
            "merged_by": (pr.get("merged_by") or {}).get("login") if pr.get("merged") else None,
            "merge_commit_sha": pr.get("merge_commit_sha"),
            "author": (pr.get("user") or {}).get("login"),
            "base_branch": (pr.get("base") or {}).get("ref"),
            "head_branch": (pr.get("head") or {}).get("ref"),
            "created_at": pr.get("created_at"),
            "updated_at": pr.get("updated_at"),
            "stats": {
                "files_changed": len(pr_data.get("files", [])),
                "additions": pr.get("additions", 0),
                "deletions": pr.get("deletions", 0),
                "commits": pr.get("commits", 0)
            },
            "commits": [{
                "sha": c.get("sha", "")[:7],
                "message": c.get("commit", {}).get("message", "").split("\n")[0][:100],
                "author": c.get("commit", {}).get("author", {}).get("name", "Unknown")
            } for c in pr_data.get("commits", [])[:20]],
            "files": [{
                "filename": f.get("filename"),
                "status": f.get("status"),
                "additions": f.get("additions", 0),
                "deletions": f.get("deletions", 0)
            } for f in pr_data.get("files", [])[:50]],
            "reviews": [{
                "user": r.get("user", {}).get("login"),
                "state": r.get("state"),
                "submitted_at": r.get("submitted_at")
            } for r in pr_data.get("reviews", [])[:10]]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/pr/{owner}/{repo}/{pr_number}/commits")
async def get_pr_commits(owner: str, repo: str, pr_number: int):
    """Get all commits from a PR (including merged PRs)."""
    try:
        pr_data = await fetch_github_pr(owner, repo, pr_number)
        commits = pr_data.get("commits", [])

        return {
            "pr_url": pr_data.get("url"),
            "pr_merged": pr_data.get("pr", {}).get("merged", False),
            "total_commits": len(commits),
            "commits": [{
                "sha": c.get("sha"),
                "sha_short": c.get("sha", "")[:7],
                "message": c.get("commit", {}).get("message", ""),
                "author": {
                    "name": c.get("commit", {}).get("author", {}).get("name"),
                    "email": c.get("commit", {}).get("author", {}).get("email"),
                    "date": c.get("commit", {}).get("author", {}).get("date")
                },
                "url": c.get("html_url")
            } for c in commits]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze-pr")
async def analyze_pr(
        pr_url: str,
        analysis_type: str = "full",
        include_merged: bool = Query(True, description="Analyze even if PR is merged")
):
    """Direct endpoint for PR analysis (supports merged PRs)."""
    pr_info = extract_pr_url(pr_url)
    if not pr_info:
        raise HTTPException(status_code=400, detail="Invalid GitHub PR URL")

    owner, repo, pr_number = pr_info
    pr_data = await fetch_github_pr(owner, repo, pr_number)
    pr = pr_data.get("pr", {})
    is_merged = pr.get("merged", False)

    if not include_merged and is_merged:
        return {
            "pr_url": pr_data.get("url"),
            "status": "merged",
            "message": "PR is already merged. Set include_merged=true to analyze.",
            "merged_at": pr.get("merged_at"),
            "merged_by": (pr.get("merged_by") or {}).get("login")
        }

    pr_context = format_pr_context(pr_data)

    query_map = {
        "full": "Provide a comprehensive analysis including summary, risks, and suggestions.",
        "summary": "Summarize what changed in this PR.",
        "risks": "Analyze the risks and potential breaking changes.",
        "review": "Perform a detailed code review."
    }
    query = query_map.get(analysis_type, query_map["full"])

    system_prompt = MERGED_PR_ANALYSIS_PROMPT if is_merged else PR_ANALYSIS_SYSTEM_PROMPT

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{pr_context}\n\n---\n\n{query}"}
        ],
        max_tokens=2500
    )

    return {
        "pr_url": pr_data.get("url"),
        "pr_title": pr.get("title"),
        "author": (pr.get("user") or {}).get("login"),
        "state": pr.get("state"),
        "merged": is_merged,
        "merged_at": pr.get("merged_at") if is_merged else None,
        "merged_by": (pr.get("merged_by") or {}).get("login") if is_merged else None,
        "stats": {
            "files_changed": len(pr_data.get("files", [])),
            "additions": pr.get("additions", 0),
            "deletions": pr.get("deletions", 0)
        },
        "analysis": response.choices[0].message.content
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