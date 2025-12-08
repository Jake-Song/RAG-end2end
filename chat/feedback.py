# server/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_feedback.rag_hyde_or_multi_feedback import graph as app_hyde_or_multi_feedback
from rag_feedback.rag_single_or_decom_feedback import graph as app_single_or_decom_feedback
from rag_feedback.rag_filter_feedback import graph as app_filter_feedback
from langgraph.types import Command
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Agent API")

class ChatRequest(BaseModel):
    messages: list[dict]
    thread_id: str = "default_thread"

class FeedbackRequest(BaseModel):
    thread_id: str
    action: str  # "approve", "edit", "reject"
    query: str
    filter: list[str]
    step: str

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    input_state = {
        "messages": req.messages
    }
    config = {"configurable": {"thread_id": req.thread_id}}
    
    try:
        # Invoke the graph. If it hits an interrupt, it will pause.
        # With a checkpointer, we can inspect the state after invoke returns.
        result = app_hyde_or_multi_feedback.invoke(input_state, config=config)
        
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

@app.post("/chat/feedback")
async def feedback_endpoint(req: FeedbackRequest):
    config = {"configurable": {"thread_id": req.thread_id}}
    
    try:
        resume_value = {"action": req.action, "query": req.query, "filter": req.filter, "step": req.step}
        # We invoke with Command(resume=value)
        result = app_hyde_or_multi_feedback.invoke(Command(resume=resume_value), config=config)
        
        # Check again if we are interrupted (unlikely for this specific graph but good practice)
        if "__interrupt__" in result:
            return {"status": "interrupted", "value": result["__interrupt__"][0].value}
        else:
            if "messages" in result and result["messages"]:
                last_message = result["messages"][-1]
                return {"status": "finished", "response": last_message.content}
            else:
                return {"status": "finished", "response": "No response generated."}

    except Exception as e:
        logger.error(f"Error in resume endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
