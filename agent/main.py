# Import necessary libraries for FastAPI web server and AI agent functionality
from fastapi import FastAPI
from fastapi.responses import StreamingResponse  # For streaming responses
from typing import Any
import os
import uvicorn  # ASGI server for running FastAPI applications
from pydantic import ValidationError
from pydantic_ai import Agent
from pydantic_ai.ag_ui import run_ag_ui, SSE_CONTENT_TYPE, StateDeps, handle_ag_ui_request
# from fastapi.requests import Request  # Commented out - not used
from http import HTTPStatus
from fastapi.responses import Response, StreamingResponse
from stock import AgentState, agent  # Import the AI agent and its state from stock module
from dotenv import load_dotenv  # For loading environment variables from .env file
from starlette.requests import Request
from starlette.responses import Response

# Import the AG UI request handler (duplicate import - could be cleaned up)
from pydantic_ai.ag_ui import handle_ag_ui_request

# Load environment variables from .env file (must be called before using env vars)
load_dotenv()

# Create FastAPI application instance
app = FastAPI()

@app.post('/pydantic-agent')
async def run_agent(request: Request) -> Response:
    """
    Handle POST requests to the /pydantic-agent endpoint.
    
    This endpoint processes requests using the pydantic AI agent with AG UI integration.
    It creates a new agent state for each request and delegates processing to the
    AG UI request handler.
    
    Args:
        request: The incoming HTTP request containing the user's input
        
    Returns:
        Response: The agent's response, typically streamed back to the client
    """
    return await handle_ag_ui_request(agent = agent, deps = StateDeps(AgentState()), request=request)

def main():
    """
    Run the uvicorn ASGI server to serve the FastAPI application.
    
    This function starts the development server with hot reload enabled.
    The server configuration:
    - Host: 0.0.0.0 (accepts connections from any IP address)
    - Port: Configurable via PORT environment variable (defaults to 8000)
    - Reload: True (automatically restarts server when code changes)
    """
    port = int(os.getenv("PORT", "8000"))  # Get port from environment or use default
    uvicorn.run(
        "main:app",  # Module and app instance to run
        host="0.0.0.0",  # Listen on all network interfaces
        port=port,
        reload=True,  # Enable auto-reload for development
    )


if __name__ == "__main__":
    # Entry point: run the server when this script is executed directly
    main()