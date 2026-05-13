# main.py - Complete Conversational Interview API

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from pipeline.question_generator import (
    generate_opening_question,
    generate_next_question,
    generate_final_feedback,
    analyze_answer
)

from pipeline.tts_processor import text_to_speech
from pipeline.stt_processor import speech_to_text

import base64
import os
import tempfile
import shutil
import uuid

app = FastAPI(
    title="AI Interviewer API",
    version="4.0"
)

# ── CORS ───────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session storage
# (Use Redis/database in production)
sessions = {}

TOTAL_QUESTIONS = 15


# ── Audio Encoding Helper ──────────────────────────────────────
def encode_audio(text: str) -> str:
    audio_bytes = text_to_speech(text)

    if isinstance(audio_bytes, str):
        audio_bytes = audio_bytes.encode()

    return base64.b64encode(audio_bytes).decode("utf-8")


# ── Root Endpoint ──────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "AI Interviewer API is running",
        "docs": "/docs",
        "health": "/health"
    }


# ── 1. Start Interview ─────────────────────────────────────────
@app.post("/start-interview")
async def start_interview(
    topic: str = Form(...),
    candidate_name: str = Form("Candidate"),
    job_level: str = Form("mid")
):
    session_id = str(uuid.uuid4())

    opening = generate_opening_question(
        topic,
        candidate_name
    )

    sessions[session_id] = {
        "topic": topic,
        "candidate_name": candidate_name,
        "job_level": job_level,
        "history": [],
        "current_question": opening["question"],
        "current_phase": opening.get(
            "phase",
            "introduction"
        ),
        "question_number": 1,
        "scores": [],
        "status": "active"
    }

    greeting = (
        f"Hello {candidate_name}, I'm Alex, and I'll be conducting your "
        f"{topic} interview today. "
        f"This will take about 10 to 15 minutes. "
        f"Feel free to take your time with your answers. "
        f"Let's get started. "
        f"{opening['question']}"
    )

    return JSONResponse({
        "session_id": session_id,
        "question_number": 1,
        "total_questions": TOTAL_QUESTIONS,
        "question": opening["question"],
        "phase": opening.get("phase", "introduction"),
        "type": opening.get("type", "behavioral"),
        "audio_base64": encode_audio(greeting),
        "status": "active"
    })


# ── 2. Submit Answer ───────────────────────────────────────────
@app.post("/submit-answer")
async def submit_answer(
    session_id: str = Form(...),
    audio: UploadFile = File(...)
):
    if session_id not in sessions:
        return JSONResponse(
            {"error": "Session not found"},
            status_code=404
        )

    session = sessions[session_id]

    if session["status"] != "active":
        return JSONResponse(
            {"error": "Interview already completed"},
            status_code=400
        )

    # Save uploaded audio temporarily
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "answer.wav")

        with open(path, "wb") as f:
            shutil.copyfileobj(audio.file, f)

        transcript = speech_to_text(path)

    # Analyze answer
    analysis = analyze_answer(
        question=session["current_question"],
        answer=transcript,
        topic=session["topic"]
    )

    # Store history
    session["history"].append({
        "number": session["question_number"],
        "question": session["current_question"],
        "phase": session["current_phase"],
        "answer": transcript,
        "quality": analysis.get("quality"),
        "score": analysis.get("score", 50)
    })

    session["scores"].append(
        analysis.get("score", 50)
    )

    # Finish interview
    if session["question_number"] >= TOTAL_QUESTIONS:
        session["status"] = "completed"

        feedback = generate_final_feedback(
            topic=session["topic"],
            candidate_name=session["candidate_name"],
            conversation_history=session["history"]
        )

        closing_text = (
            f"Thank you {session['candidate_name']}, "
            f"that concludes our interview. "
            f"You've answered all {TOTAL_QUESTIONS} questions. "
            f"We'll be in touch soon. "
            f"Have a great day."
        )

        return JSONResponse({
            "transcript": transcript,
            "answer_quality": analysis.get("quality"),
            "status": "completed",
            "feedback": feedback,
            "closing_audio": encode_audio(closing_text)
        })

    # Generate next question
    next_q = generate_next_question(
        topic=session["topic"],
        conversation_history=session["history"],
        last_answer_analysis=analysis,
        question_number=session["question_number"] + 1,
        total_questions=TOTAL_QUESTIONS
    )

    session["current_question"] = next_q["question"]

    session["current_phase"] = next_q.get(
        "phase",
        "technical"
    )

    session["question_number"] += 1

    quality = analysis.get("quality", "adequate")

    if quality in ["weak", "unclear"]:
        bridge = "I see. Let me ask you this — "
    elif quality == "strong":
        bridge = "That's helpful context. "
    else:
        bridge = "Got it. "

    spoken = bridge + next_q["question"]

    return JSONResponse({
        "transcript": transcript,
        "answer_quality": quality,
        "answer_score": analysis.get("score"),
        "question_number": session["question_number"],
        "total_questions": TOTAL_QUESTIONS,
        "question": next_q["question"],
        "phase": next_q.get("phase"),
        "type": next_q.get("type"),
        "audio_base64": encode_audio(spoken),
        "status": "active",
        "progress": round(
            session["question_number"] /
            TOTAL_QUESTIONS * 100
        )
    })


# ── 3. Session Info ────────────────────────────────────────────
@app.get("/session/{session_id}")
def get_session(session_id: str):

    if session_id not in sessions:
        return JSONResponse(
            {"error": "Not found"},
            status_code=404
        )

    s = sessions[session_id]

    avg = (
        sum(s["scores"]) / len(s["scores"])
        if s["scores"]
        else 0
    )

    return JSONResponse({
        "topic": s["topic"],
        "candidate": s["candidate_name"],
        "question_number": s["question_number"],
        "total_questions": TOTAL_QUESTIONS,
        "status": s["status"],
        "current_phase": s["current_phase"],
        "average_score_so_far": round(avg)
    })


# ── 4. End Interview ───────────────────────────────────────────
@app.post("/end-interview/{session_id}")
def end_interview(session_id: str):

    if session_id not in sessions:
        return JSONResponse(
            {"error": "Not found"},
            status_code=404
        )

    s = sessions[session_id]

    s["status"] = "completed"

    feedback = generate_final_feedback(
        s["topic"],
        s["candidate_name"],
        s["history"]
    )

    return JSONResponse({
        "feedback": feedback,
        "status": "completed"
    })


# ── 5. Health Check ────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "4.0",
        "interviewer": "Groq/Llama-3.3-70b",
        "stt": "Whisper",
        "tts": "SpeechT5"
    }