import logging
import time
from typing import List, Dict

from fastapi import FastAPI
from pydantic import BaseModel

from app.services.llm import generate_llm_reply
from app.services.rag import init_rag, retrieve_context, build_rag_prompt

logger = logging.getLogger("ai_rag_app")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# --------- Simple in-memory short-term memory (per user) --------- #

# Store chat history per user_id: list of {"role": "user"/"assistant", "content": str}
USER_CONVERSATIONS: Dict[str, List[Dict[str, str]]] = {}

# How many past messages (pairs of turns) to include in context
MAX_HISTORY_MESSAGES = 10

SYSTEM_PROMPT = (
    "You are a helpful AI assistant. "
    "You DO have access to the previous messages in the conversation "
    "because they are included in the chat history. "
    "If the user asks about previous questions, use the chat history provided."
)


app = FastAPI(
    title="AI RAG Backend",
    version="0.1.0",
    description="Backend service for RAG + LLM chatbot"
)


class ChatRequest(BaseModel):
    user_id: str | None = None
    message: str
    


class ChatResponse(BaseModel):
    reply: str
    used_context: list[str] | None = None
    latency_ms: float | None = None


@app.on_event("startup")
async def startup_event():
    # Initialize the RAG index with demo docs
    init_rag()


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.get("/")
async def root():
    return {"message": "AI RAG Backend is running"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint with:
      - RAG (Qdrant retrieval)
      - Short-term conversational memory (per user)
      - Basic latency logging
    """

    start_time = time.perf_counter()

    # Determine user id (fallback if not provided)
    user_id = request.user_id or "anonymous"

    # 1) Retrieve context from vector DB for THIS question
    context_chunks = retrieve_context(request.message, top_k=3)

    # 2) Build RAG prompt for the current user message
    rag_prompt = build_rag_prompt(request.message, context_chunks)

    # 3) Load short-term history for this user (last N messages)
    history = USER_CONVERSATIONS.get(user_id, [])
    recent_history = history[-MAX_HISTORY_MESSAGES:]

    # 4) Build messages for the LLM
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(recent_history)
    messages.append({"role": "user", "content": rag_prompt})

    # 5) Call LLM
    reply_text = await generate_llm_reply(messages)

    latency_ms = (time.perf_counter() - start_time) * 1000

    # 6) Update in-memory conversation for this user
    #    We store the "natural" user message, not the rag_prompt,
    #    because that’s what the user actually said.
    updated_history = history + [
        {"role": "user", "content": request.message},
        {"role": "assistant", "content": reply_text},
    ]
    USER_CONVERSATIONS[user_id] = updated_history

    # 7) Log metrics
    logger.info(
        "chat_request",
        extra={
            "user_id": user_id,
            "message_length": len(request.message),
            "num_context_chunks": len(context_chunks),
            "latency_ms": round(latency_ms, 2),
        },
    )

    return ChatResponse(
        reply=reply_text,
        used_context=context_chunks,
        latency_ms=round(latency_ms, 2),
    )
