from typing import AsyncIterable, Optional
import fastapi
from fastapi.responses import StreamingResponse
import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure API key and assistant ID exist
API_KEY = os.getenv("OPENAI_API_KEY")
ASSISTANT_ID = os.getenv("OPENAI_ASSISTANT_ID")

# Ensure API key and assistant ID exist

if not API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in environment variables.")
if not ASSISTANT_ID:
    raise ValueError("Missing OPENAI_ASSISTANT_ID in environment variables.")

# Initialize FastAPI app
app = fastapi.FastAPI()

# OpenAI client
client = openai.AsyncOpenAI(api_key=API_KEY)


async def get_assistant_response(
    raw_messages: list[dict], 
    custom_session_id: Optional[str] = None
) -> AsyncIterable[str]:
    """Generate responses using OpenAI's Assistants API and format for Hume EVI."""
    
    try:
        # Create a thread for this session
        thread = await client.beta.threads.create()
        thread_id = thread.id  # Unique thread ID
        
        # Send messages to the assistant thread
        for message in raw_messages:
            await client.beta.threads.messages.create(
                thread_id=thread_id,
                role=message["role"],
                content=message["content"]
            )

        # Run the assistant with the given thread
        run = await client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=ASSISTANT_ID,
            stream=True  # Enable streaming
        )

        # Stream response
        async for event in run:
            # Check if the event contains the assistant's response
            if event.event == "thread.message.delta":
                for content in event.data.delta.content:
                    if content.type == "text":
                        # Format the response for Hume EVI
                        response = {"text": content.text.value}
                        yield f"data: {response}\n\n"

    except Exception as e:
        yield f"data: {{'error': '{str(e)}'}}\n\n"

    yield "data: [DONE]\n\n"


@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.post("/chat/completions", response_class=StreamingResponse)
async def chat_completions(request: fastapi.Request):
    """Chat completion endpoint using OpenAI Assistant, formatted for Hume EVI."""
    
    try:
        request_json = await request.json()
        messages = request_json.get("messages", [])

        if not messages:
            raise fastapi.HTTPException(status_code=400, detail="Missing 'messages' in request body")
        
        custom_session_id = request.query_params.get("custom_session_id")

        return StreamingResponse(
            get_assistant_response(messages, custom_session_id=custom_session_id),
            media_type="text/event-stream",
        )

    except fastapi.HTTPException as e:
        raise e  # Preserve HTTP status codes
    except Exception as e:
        raise fastapi.HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")