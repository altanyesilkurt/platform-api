from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import AsyncGenerator
from supabase import create_client, Client
from openai import OpenAI
from dotenv import load_dotenv
import os
import json

# Load environment variables from .env file
load_dotenv()

# Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize clients
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Create FastAPI app
app = FastAPI(title="Chat API", version="1.0.0")

# CORS middleware
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


def generate_chat_title(user_message: str, ai_response: str) -> str:
    """Generate a short title for the chat based on the conversation."""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Generate a very short title (3-6 words max) for this conversation. Return ONLY the title, no quotes, no punctuation at the end."
                },
                {
                    "role": "user",
                    "content": f"User asked: {user_message}\n\nAssistant replied: {ai_response[:500]}"
                }
            ],
            max_tokens=20,
            temperature=0.7
        )
        title = response.choices[0].message.content.strip()
        # Clean up the title
        title = title.strip('"\'')
        # Limit length
        if len(title) > 50:
            title = title[:47] + "..."
        return title
    except Exception as e:
        print(f"Error generating title: {e}")
        return "New Chat"


def is_first_exchange(chat_id: str) -> bool:
    """Check if this is the first message exchange in the chat."""
    response = supabase.table("messages").select("id", count="exact").eq("chat_id", chat_id).execute()
    # If there's only 1 message (the user message we just inserted), this is the first exchange
    return response.count <= 1


# Get all chats
@app.get("/chats")
async def get_chats():
    response = supabase.table("chats").select("*").order("updated_at", desc=True).execute()
    return response.data


# Create a new chat
@app.post("/chats")
async def create_chat(chat: ChatCreate):
    response = supabase.table("chats").insert({"title": chat.title}).execute()
    return response.data[0]


# Get a specific chat
@app.get("/chats/{chat_id}")
async def get_chat(chat_id: str):
    response = supabase.table("chats").select("*").eq("id", chat_id).single().execute()
    return response.data


# Update chat title
@app.put("/chats/{chat_id}")
async def update_chat(chat_id: str, chat_update: ChatUpdate):
    response = supabase.table("chats").update({"title": chat_update.title}).eq("id", chat_id).execute()
    if not response.data:
        raise HTTPException(status_code=404, detail="Chat not found")
    return response.data[0]


# Delete chat
@app.delete("/chats/{chat_id}")
async def delete_chat(chat_id: str):
    # Delete messages first
    supabase.table("messages").delete().eq("chat_id", chat_id).execute()
    # Then delete the chat
    response = supabase.table("chats").delete().eq("id", chat_id).execute()
    return {"success": True, "deleted_id": chat_id}


# Get messages for a chat
@app.get("/chats/{chat_id}/messages")
async def get_messages(chat_id: str):
    response = supabase.table("messages").select("*").eq("chat_id", chat_id).order("created_at").execute()
    return response.data


# Send message and get AI response (non-streaming)
@app.post("/chat")
async def send_message(message_data: MessageCreate):
    user_message = message_data.message

    # Check if this is the first exchange before inserting
    first_exchange = is_first_exchange(message_data.chat_id)

    # Save user message
    supabase.table("messages").insert({
        "chat_id": message_data.chat_id,
        "role": "user",
        "content": user_message
    }).execute()

    # Get conversation history
    history = supabase.table("messages").select("role, content").eq(
        "chat_id", message_data.chat_id
    ).order("created_at").execute()

    # Build messages for OpenAI
    messages = [{"role": "system", "content": "You are a helpful AI assistant."}]
    messages.extend([{"role": m["role"], "content": m["content"]} for m in history.data])

    # Call OpenAI
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=1000
        )
        ai_response = response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"AI error: {str(e)}")

    # Save assistant response
    assistant_msg = supabase.table("messages").insert({
        "chat_id": message_data.chat_id,
        "role": "assistant",
        "content": ai_response
    }).execute()

    # Generate and update title if first exchange
    new_title = None
    if first_exchange:
        new_title = generate_chat_title(user_message, ai_response)
        supabase.table("chats").update({"title": new_title}).eq("id", message_data.chat_id).execute()

    return {
        "id": assistant_msg.data[0]["id"],
        "role": "assistant",
        "content": ai_response,
        "new_title": new_title
    }


# Streaming endpoint
@app.post("/chat/stream")
async def send_message_stream(message_data: MessageCreate):
    user_message = message_data.message
    chat_id = message_data.chat_id

    # Check if this is the first exchange before inserting
    first_exchange = is_first_exchange(chat_id)

    # Save user message
    supabase.table("messages").insert({
        "chat_id": chat_id,
        "role": "user",
        "content": user_message
    }).execute()

    # Get conversation history
    history = supabase.table("messages").select("role, content").eq(
        "chat_id", chat_id
    ).order("created_at").execute()

    messages = [{"role": "system", "content": "You are a helpful AI assistant."}]
    messages.extend([{"role": m["role"], "content": m["content"]} for m in history.data])

    def generate():
        full_response = ""
        try:
            stream = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=1000,
                stream=True
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield f"data: {json.dumps({'content': content})}\n\n"

            # Save complete response
            result = supabase.table("messages").insert({
                "chat_id": chat_id,
                "role": "assistant",
                "content": full_response
            }).execute()

            # Generate and update title if first exchange
            new_title = None
            if first_exchange:
                new_title = generate_chat_title(user_message, full_response)
                supabase.table("chats").update({"title": new_title}).eq("id", chat_id).execute()

            yield f"data: {json.dumps({'done': True, 'id': result.data[0]['id'], 'new_title': new_title})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "X-Accel-Buffering": "no",
        }
    )


# Health check
@app.get("/health")
async def health_check():
    return {"status": "ok", "database": "supabase", "ai": "openai"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)