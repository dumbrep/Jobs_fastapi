# main.py
from fastapi import FastAPI, File, UploadFile, Form, WebSocket, WebSocketDisconnect, HTTPException
import openai
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import io
import asyncio
import base64
from PIL import Image
import pdf2image
import shutil
import requests
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from database.mongodb import saveSummary
from upstash_redis import Redis
import json
import google.generativeai as genai
from openai import OpenAI

load_dotenv()

# ----------------- Config -----------------

client = OpenAI(api_key=os.getenv("OPENAI_KEY"))
genai_api_key = os.getenv("GOOGLE_API_KEY")
UPSTASH_URL = os.getenv("UPSTASH_REDIS_REST_URL")
UPSTASH_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN")
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
RAPIDAPI_HOST = os.getenv("RAPIDAPI_HOST", "jsearch.p.rapidapi.com")
model_local = genai.GenerativeModel("gemini-2.0-flash")

if not (UPSTASH_URL and UPSTASH_TOKEN):
    raise RuntimeError("Please set UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN in .env")

# configure genai client

genai.configure(api_key=genai_api_key)

# Upstash Redis client (async-capable)
redis = Redis(url=UPSTASH_URL, token=UPSTASH_TOKEN)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,   
    allow_methods=["*"],
    allow_headers=["*"], 
)
 
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ----------------- Models -----------------
class ResumeData(BaseModel):
    resume_dt: str
    job_description: str
    jobType: str
    role: str
    experience: int
    interview_type: str

class QAHighlight(BaseModel):
    question: str = Field(description="System's question")
    candidate_key_points: str = Field(description="Summary of candidate's answer")
    assessment: str = Field(description="Evaluation (Good / Average / Needs Improvement)")

class InterviewSummary(BaseModel):
    overall_performance: str
    question_response_highlights: List[QAHighlight]
    strengths: List[str]
    areas_of_improvement: List[str]

class SummaryRequest(BaseModel):
    email: str

# ----------------- Helpers -----------------
def session_key(session_id: str, name: str) -> str:
    return f"session:{session_id}:{name}"

async def set_session_value(session_id: str, name: str, value, ttl: int = 3600):
    key = session_key(session_id, name)
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    # upstash_redis client's set returns coroutine
    redis.set(key, value)
    redis.expire(key, ttl)

async def get_session_value(session_id: str, name: str):
    key = session_key(session_id, name)
    v = redis.get(key)
    if v is None:
        return None
    # try parse JSON
    try:
        return json.loads(v)
    except Exception:
        return v

async def append_interaction(session_id: str, question: str, answer: str):
    key = session_key(session_id, "interactions")
    entry = json.dumps({"question": question, "answer": answer}, ensure_ascii=False)
    redis.rpush(key, entry)
    redis.expire(key, 3600)

async def get_interactions(session_id: str):
    key = session_key(session_id, "interactions")
    items =  redis.lrange(key, 0, -1)
    parsed = []
    for it in items:
        try:
            parsed.append(json.loads(it))
        except Exception:
            parsed.append({"raw": it})
    return parsed

async def clear_session(session_id: str):
    keys = [
        "resume", "jobDescription", "jobType", "role",
        "experience", "interviewType", "interactions", "page_parts",
        "ATSJobdescription", "ats_prompt"
    ]
    for name in keys:
        redis.delete(session_key(session_id, name))

# ----------------- Prompts (same as original) -----------------
input_prompt1 = """
You are an experienced Technical Human Resource Manager,your task is to review the provided resume against the job description. 
Please share your professional evaluation on whether the candidate's profile aligns with the role. 
Highlight the strengths and weaknesses of the applicant in relation to the specified job requirements.
Make sure that do **NOT** include any heading or introduction.
Do not include initial introduction but return actual data only. Generate Well structured and detailed response.
"""

input_prompt3 = """
You are an skilled ATS (Applicant Tracking System) scanner with a deep understanding of data science and ATS functionality, 
your task is to evaluate the resume against the provided job description. give me the percentage of match if the resume matches
the job description. First the output should come as percentage and then keywords missing and last final thoughts.
Do not include initial introduction but return actual data only.
"""

# ----------------- LLM wrappers that use session data -----------------
def _system_content_for_question(resume, role, job_type, experience, interview_type, interactions):
    prev = json.dumps(interactions, ensure_ascii=False)
    return f"""You are an interviewer for the role of {role} {job_type}. He is having {experience} years of experience.
The previous questions and candidate responses are : {prev}
- You have to generate only one question to ask.
- Analyse previous interaction and generate appropriate question.
- NOTE THAT INTERVIEW SHOULD BE {interview_type}. SO ONLY {interview_type} QUESTIONS SHOULD BE PRESENT.
- Maintain the flow of question answering.
- Your first question should be "Tell me about yourself." If there are no previous responses, ask it.
- Do not include introductions. Add both technical and non-technical questions aligned with the resume and job description.
**Candidate resume**: {resume}
**Job description**: {job_type}

-- NOTE : You have to generate only question, not anything extra other than that.
"""

async def generate_question(session_id: str) -> str:
    resume = await get_session_value(session_id, "resume") or ""
    role = await get_session_value(session_id, "role") or ""
    job_type = await get_session_value(session_id, "jobType") or ""
    experience = await get_session_value(session_id, "experience") or 0
    interview_type = await get_session_value(session_id, "interviewType") or ""
    interactions = await get_interactions(session_id)

    system_content = _system_content_for_question(resume, role, job_type, experience, interview_type, interactions)

    resp = model_local.generate_content(system_content)

    return resp.text.strip()

async def analyze_response(session_id: str, question: str, answer: str) -> str:
    resume = await get_session_value(session_id, "resume") or ""
    job_desc = await get_session_value(session_id, "jobDescription") or ""
    prompt = f"""
        You have given the question and its corresponding response by candidate
        Question : {question}
        Response : {answer}
        Generate best feedback to be given to the candidate
        - Generate response in a tone which feets like you are actually talking with candidate.
        - Only generate response, do not ask question
        - If candidate asks for ideal response, provide it only when explicitly requested.
        - Generate feedback in paragraph format, short and sweet.
        - Do not add introduction.
        **Candidate resume** : {resume}
        **Job descriptions** : {job_desc}
        """
    resp = model_local.generate_content(prompt)

    return resp.text.strip()


def generateInterviewSummary(interactions):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
    summaryTemplate = f"""
You are given the transcript of an interview. The transcript contains system-generated questions and the candidate's responses. Your task is to generate a structured interview summary in valid JSON format.

Instructions:
 - Be concise and objective.
 - Do not copy entire responses, only capture key points.
 - Use professional language.
 - Always return valid JSON only (no extra text).

Transcript:
{json.dumps(interactions, ensure_ascii=False)}
"""
    summary_model = llm.with_structured_output(InterviewSummary)
    response = summary_model.invoke(summaryTemplate)
    return response.model_dump()

# ----------------- Endpoints -----------------
from fastapi import Form

@app.post("/resume")
async def post_resume(
    session_id: str = Form(...),
    resume_dt: str = Form(...),
    job_description: str = Form(...),
    jobType: str = Form(...),
    role: str = Form(...),
    experience: str = Form(...),
    interview_type: str = Form(...)
):
    await set_session_value(session_id, "resume", resume_dt)
    await set_session_value(session_id, "jobDescription", job_description)
    await set_session_value(session_id, "jobType", jobType)
    await set_session_value(session_id, "role", role)
    await set_session_value(session_id, "experience", experience)
    await set_session_value(session_id, "interviewType", interview_type)

    redis.delete(session_key(session_id, "interactions"))
    print(f"resume : {resume_dt}")

    return {"message": "Session data saved", "session_id": session_id}


@app.post("/get_resume_file")
async def upload_resume_file(ATSdescription: str = Form(...), prompt_number: int = Form(...),
                             file: UploadFile = File(...), session_id: str = Form(...)):
    """
    Upload PDF resume for ATS processing. Store page parts in Redis under session.
    """
    file_path = f"{UPLOAD_DIR}/{session_id}_{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Convert PDF pages to base64 images
    page_parts = []
    with open(file_path, "rb") as f:
        images = pdf2image.convert_from_bytes(f.read())
    for image in images:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="JPEG")
        page_parts.append({
            "mime_type": "image/jpeg",
            "data": base64.b64encode(img_byte_arr.getvalue()).decode()
        })
    # cleanup file
    os.remove(file_path)

    await set_session_value(session_id, "page_parts", page_parts)
    await set_session_value(session_id, "ATSJobdescription", ATSdescription)
    if prompt_number == 1:
        await set_session_value(session_id, "ats_prompt", input_prompt1)
    else:
        await set_session_value(session_id, "ats_prompt", input_prompt3)
    return {"message": "Uploaded and stored for ATS", "session_id": session_id}

@app.post("/ats_response")
async def sendAtsData(session_id: str = Form(...)):
    """
    Use stored page_parts + ats_prompt to call Gemini and return result.
    """
    page_parts = await get_session_value(session_id, "page_parts")
    ats_prompt = await get_session_value(session_id, "ats_prompt")
    ATSJobdescription = await get_session_value(session_id, "ATSJobdescription") or ""

    if not page_parts or not ats_prompt:
        raise HTTPException(status_code=400, detail="No uploaded resume/page parts or prompt for this session.")

    try:
        model_local = genai.GenerativeModel('gemini-2.0-flash')
        response = model_local.generate_content([ats_prompt] + page_parts + [ATSJobdescription])
        return {"result": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/jobSearch")
async def getJobs(resume: UploadFile = File(...), session_id: str = Form(...)):
    """
    Analyze uploaded resume (PDF) and call RapidAPI job search.
    """
    file_path = f"{UPLOAD_DIR}/{session_id}_{resume.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(resume.file, buffer)

    loader = PyPDFLoader(file_path)
    documents = loader.load()
    os.remove(file_path)

    prompt = f"""
You are the skilled resume analyzer. Your task is to analyze the given resume and identify the job roles which are matching with them.
Output: Jobs for <role1 , role2 , .....> Roles in India.
DO NOT ADD ANY HEADING OR INTRODUCTION.
Resume:
{documents[0].page_content}
"""
    try:
        model_local = genai.GenerativeModel('gemini-2.0-flash')
        response = model_local.generate_content(prompt)
        query_text = response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini error: {e}")

    url = f"https://{RAPIDAPI_HOST}/search"
    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": RAPIDAPI_HOST
    }
    params = {"query": query_text, "page": "1"}
    jobs_resp = requests.get(url, headers=headers, params=params)
    try:
        jobs_json = jobs_resp.json()
        return {"jobs": jobs_json.get("data", []), "query": query_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summary")
async def generateSummary(  session_id: str = Form(...),
    email: str = Form(...),):
    """
    Generate interview summary from stored interactions and persist using saveSummary.
    """
    interactions = await get_interactions(session_id)
    if not interactions:
        return {"error": "No interactions for this session."}
    try:
        summary = generateInterviewSummary(interactions)
        saveSummary(email, summary)
        await clear_session(session_id)
        return {"message": "Summary generated and saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------- WebSocket interview -----------------
@app.websocket("/interview")
async def interview_ws(websocket: WebSocket):
    session_id = websocket.query_params.get("session_id")

    if not session_id:
     
        await websocket.accept()
        await websocket.send_text(json.dumps({
            "type": "error",
            "text": "Missing session_id in query params"
        }))
        await websocket.close()
        return

  
    await websocket.accept()
    print(f"WebSocket connected for session: {session_id}")

    try:
        while True:
            
            question = await generate_question(session_id)
            await websocket.send_text(json.dumps({
                "type": "question",
                "text": question
            }))

            answer = await websocket.receive_text()

            await append_interaction(session_id, question, answer)

            
            feedback = await analyze_response(session_id, question, answer)

            await websocket.send_text(json.dumps({
                "type": "feedback",
                "text": feedback
            }))

    except WebSocketDisconnect:
        print(f"Client disconnected from session {session_id}")

    except Exception as e:
        print(f"Interview websocket error for session {session_id}: {e}")
        try:
            await websocket.send_text(json.dumps({"type": "error", "text": str(e)}))
        except:
            pass


@app.post("/clear_session")
async def api_clear_session(session_id: str = Form(...)):
    await clear_session(session_id)
    return {"message": "Session cleared", "session_id": session_id}
