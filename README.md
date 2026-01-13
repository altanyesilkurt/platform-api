# Chat Platform API
A FastAPI-powered backend service that uses AI to analyze GitHub Pull Requests and Commits, providing intelligent code review insights, risk assessments, and actionable suggestions.

## Features

- **PR Analysis**: Comprehensive analysis of GitHub Pull Requests including summary, risk assessment, key changes, and suggestions
- **Commit Analysis**: Detailed analysis of individual commits with code quality evaluation
- **Chat Interface**: Conversational interface with persistent chat history
- **Streaming Responses**: Real-time streaming of AI responses for better UX
- **PR Actions**: Submit reviews (Approve, Request Changes, Comment) directly through the API
- **Auto-Generated Titles**: Intelligent chat title generation based on conversation content
- **Context-Aware Responses**: Distinguishes between PR-related and general queries

## Tech Stack

- **Framework**: FastAPI
- **Database**: Supabase
- **AI Model**: OpenAI GPT-4o
- **HTTP Client**: httpx (async)
- **Validation**: Pydantic

## Prerequisites

- Python 3.8+
- Supabase account and project
- OpenAI API key
- GitHub Personal Access Token (for private repos and higher rate limits)

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd platform-api
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```env
   SUPABASE_URL=your_supabase_project_url
   SUPABASE_KEY=your_supabase_anon_key
   OPENAI_API_KEY=your_openai_api_key
   GITHUB_TOKEN=your_github_personal_access_token
   ```

5. **Set up Supabase tables**
   
   Create the following tables in your Supabase project:

   **chats**
   ```sql
   CREATE TABLE chats (
       id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
       title VARCHAR(200) DEFAULT 'New Chat',
       created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
       updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
   );
   ```

   **messages**
   ```sql
   CREATE TABLE messages (
       id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
       chat_id UUID REFERENCES chats(id) ON DELETE CASCADE,
       role VARCHAR(20) NOT NULL,
       content TEXT NOT NULL,
       created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
   );
   ```

## Running the Server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

## API Endpoints

### Chat Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/chats` | List all chats |
| POST | `/chats` | Create a new chat |
| GET | `/chats/{chat_id}` | Get chat details |
| PUT | `/chats/{chat_id}` | Update chat title |
| DELETE | `/chats/{chat_id}` | Delete chat and messages |
| GET | `/chats/{chat_id}/messages` | Get all messages in a chat |

### Messaging

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/chat` | Send message (non-streaming) |
| POST | `/chat/stream` | Send message (streaming SSE) |

### PR Analysis

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/analyze-pr` | Direct PR analysis |
| POST | `/analyze-commit` | Direct commit analysis |
| POST | `/pr/review` | Submit PR review |
| POST | `/pr/comment` | Add comment to PR |

### Utility

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/test-github/{owner}/{repo}/{pr_number}` | Test GitHub connectivity |

## Usage Examples

### Chat Stream Endpoint (POST /chat/stream)

This endpoint provides real-time streaming responses via Server-Sent Events (SSE).

**Request**

```bash
curl -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "chat_id": "your-chat-uuid",
    "message": "Please review https://github.com/owner/repo/pull/123"
  }'
```

**Request Body**

```json
{
  "chat_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Please review https://github.com/owner/repo/pull/123"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| chat_id | string (UUID) | Yes | The ID of the chat session |
| message | string | Yes | User message (min 1 character) |

**Response (SSE Stream)**

The response is a stream of Server-Sent Events. Each event is prefixed with `data: ` and contains JSON.

```
data: {"pr_metadata": {"pr_url": "https://github.com/owner/repo/pull/123", "pr_title": "Add new feature", "pr_body": "This PR adds...", "pr_state": "open", "pr_merged": false, "pr_author": "username", "files_changed": 5, "additions": 120, "deletions": 30, "commits": [{"sha": "abc1234", "message": "Initial commit", "author": "username"}], "files": [{"filename": "src/main.py", "status": "modified", "additions": 50, "deletions": 10, "patch": "@@ -1,5 +1,10 @@..."}]}}

data: {"content": "## Summary\n\n"}

data: {"content": "This PR introduces a new feature that..."}

data: {"content": " allows users to..."}

data: {"done": true, "id": "message-uuid", "new_title": "Feature Review Discussion"}
```

**Stream Event Types**

| Event | Description |
|-------|-------------|
| `pr_metadata` | PR details (sent first when PR URL detected) |
| `commit_metadata` | Commit details (sent first when commit URL detected) |
| `content` | Streamed AI response chunks |
| `done` | Final event with message ID and optional new chat title |
| `error` | Error message if something fails |

**PR Metadata Object**

```json
{
  "pr_url": "https://github.com/owner/repo/pull/123",
  "pr_title": "Add new feature",
  "pr_body": "Description of the PR...",
  "pr_state": "open",
  "pr_merged": false,
  "pr_author": "username",
  "files_changed": 5,
  "additions": 120,
  "deletions": 30,
  "commits": [
    {
      "sha": "abc1234",
      "message": "Initial commit",
      "author": "username"
    }
  ],
  "files": [
    {
      "filename": "src/main.py",
      "status": "modified",
      "additions": 50,
      "deletions": 10,
      "patch": "@@ -1,5 +1,10 @@..."
    }
  ]
}
```

**Commit Metadata Object**

```json
{
  "type": "commit",
  "commit_url": "https://github.com/owner/repo/commit/abc123",
  "commit_sha": "abc1234",
  "commit_sha_full": "abc1234567890abcdef",
  "commit_message": "Fix bug in authentication",
  "commit_author": "username",
  "commit_author_email": "user@example.com",
  "commit_date": "2024-01-15T10:30:00Z",
  "files_changed": 3,
  "additions": 45,
  "deletions": 12,
  "total_changes": 57,
  "files": [
    {
      "filename": "src/auth.py",
      "status": "modified",
      "additions": 30,
      "deletions": 10,
      "patch": "@@ -10,5 +10,25 @@..."
    }
  ]
}
```


### Direct PR Analysis

```bash
curl -X POST "http://localhost:8000/analyze-pr?pr_url=https://github.com/owner/repo/pull/123&analysis_type=full"
```

### Submit a PR Review

```bash
curl -X POST http://localhost:8000/pr/review \
  -H "Content-Type: application/json" \
  -d '{
    "pr_url": "https://github.com/owner/repo/pull/123",
    "review_type": "APPROVE",
    "body": "LGTM! Great work."
  }'
```

### Analyze a Commit

```bash
curl -X POST "http://localhost:8000/analyze-commit?commit_url=https://github.com/owner/repo/commit/abc123"
```

## GitHub Token Permissions

For full functionality, your GitHub Personal Access Token needs:

- `repo` scope (for private repositories)
- `public_repo` scope (for public repositories only)

Without a token, you'll be limited to 60 requests per hour. With a token, this increases to 5,000 requests per hour.

## CORS Configuration

The API is configured to accept requests from:
- `http://localhost:5173`
- `http://localhost:3000`
- `http://localhost:9090`

Modify the `allow_origins` list in `main.py` to add additional origins.

## Response Format

PR analysis responses include:

- **Summary**: Concise overview of the PR
- **Key Changes**: Main modifications with code snippets
- **Risk Assessment**: Risk level (Low/Medium/High/Critical) with details
- **Suggestions**: Actionable improvement recommendations
- **Breaking Changes**: Identified breaking changes

## Error Handling

The API returns appropriate HTTP status codes:

- `400` - Bad request (invalid URL format, missing required fields)
- `401` - Unauthorized (missing GitHub token for protected operations)
- `403` - Forbidden (rate limit exceeded, insufficient permissions)
- `404` - Not found (PR/commit doesn't exist)
- `422` - Validation error (e.g., can't approve own PR)
- `500` - Internal server error
- `502` - Bad gateway (GitHub/OpenAI API errors)
- `504` - Gateway timeout

## License

MIT License
