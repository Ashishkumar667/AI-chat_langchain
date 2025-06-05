from fastapi import FastAPI, Request
from pydantic import BaseModel
from agent import get_user_agent
from fastapi import File, UploadFile
from utils.pdf_loader import extract_text_from_pdf_bytes
from vectorstore import store_pdf_in_vectorstore
from fastapi import Form


app = FastAPI()
user_agents = {}  # In-memory dict for user sessions (for demo)

class ChatRequest(BaseModel):
    user_id: str
    message: str

@app.post("/chat")
async def chat(req: ChatRequest):
    user_id = req.user_id
    message = req.message
    
    if user_id not in user_agents:
        user_agents[user_id] = get_user_agent(user_id)
    
    agent = user_agents[user_id]
    try:
        result = agent.invoke({"input": message})
        return {"success": True, "reply": result}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/upload-pdf")
async def upload_pdf(user_id: str = Form(...), file: UploadFile = File(...)):
    contents = await file.read()
    with open(file.filename, "wb") as f:
        f.write(contents)
    
    text = extract_text_from_pdf(file.filename)
    store_pdf_in_vectorstore(text, user_id)
    return {"success": True, "message": "PDF uploaded and processed"}
