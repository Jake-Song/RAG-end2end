# server/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag_hyde_or_multi.rag_hyde_or_multi import app as app_hyde_or_multi
from rag_single_or_decom.rag_single_or_decom import router_app as router_app

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Agent API")

class ChatRequest(BaseModel):
    messages: list[dict]
    thread_id: str = "default_thread"

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    input_state = {
        "messages": req.messages
    }
    config = {"configurable": {"thread_id": req.thread_id}}
    
    try:
        # Invoke the graph. If it hits an interrupt, it will pause.
        # With a checkpointer, we can inspect the state after invoke returns.
        result = router_app.invoke(input_state, config=config)
        
        # Check if we are in an interrupted state
        if "__interrupt__" in result:
            return {"status": "interrupted", "value": result["__interrupt__"][0].value}
        else:
            # If finished, extract the final answer
            if "messages" in result and result["messages"]:
                last_message = result["messages"][-1]
                return {"status": "finished", "response": last_message.content}
            else:
                return {"status": "finished", "response": "No response generated."}
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
