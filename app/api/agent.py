from fastapi import APIRouter
from app.schemas.agent import AgentRequest
from app.agents.research_graph import research_graph

router = APIRouter(prefix="/agent", tags=["agent"])

@router.post("/run")
async def run_agent(req: AgentRequest):
    result = await research_graph.ainvoke({
        "question": req.question,
        "filename": req.filename,
    })
    return result