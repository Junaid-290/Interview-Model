# main.py - Complete All-in-One API- Mercor-level Conversational Interview API v4.0
# Run: uvicorn main:app --reload
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, Response
from pipeline.question_generator import (
    generate_opening_question,
    generate_next_question,
    generate_final_feedback,
    analyze_answer
)
from pipeline.tts_processor import text_to_speech
from pipeline.stt_processor import speech_to_text
import base64, os, tempfile, shutil, uuid

app = FastAPI(title="AI Interviewer API", version="4.0")

# Session storage (use Redis in production)
sessions = {}

TOTAL_QUESTIONS = 15  # adjust as needed


def encode_audio(text: str) -> str:
    audio = text_to_speech(text)
    return base64.b64encode(audio).decode()


# ── 1. Start Interview ─────────────────────────────────────────
@app.post("/start-interview")
async def start_interview(
    topic: str = Form(...),
    candidate_name: str = Form("Candidate"),
    job_level: str = Form("mid")  # junior/mid/senior
):
    session_id = str(uuid.uuid4())

    # Generate opening question
    opening = generate_opening_question(topic, candidate_name)

    sessions[session_id] = {
        "topic": topic,
        "candidate_name": candidate_name,
        "job_level": job_level,
        "history": [],
        "current_question": opening["question"],
        "current_phase": opening.get("phase", "introduction"),
        "question_number": 1,
        "scores": [],
        "status": "active"
    }

    # Greeting + first question
    greeting = (
        f"Hello {candidate_name}, I'm Alex, and I'll be conducting your "
        f"{topic} interview today. This will take about 10 to 15 minutes. "
        f"Feel free to take your time with your answers. Let's get started. "
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
            {"error": "Session not found"}, status_code=404)

    session = sessions[session_id]

    if session["status"] != "active":
        return JSONResponse(
            {"error": "Interview already completed"}, status_code=400)

    # Transcribe
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "answer.wav")
        with open(path, "wb") as f:
            shutil.copyfileobj(audio.file, f)
        transcript = speech_to_text(path)

    # Analyze answer quality
    analysis = analyze_answer(
        question=session["current_question"],
        answer=transcript,
        topic=session["topic"]
    )

    # Store turn
    session["history"].append({
        "number": session["question_number"],
        "question": session["current_question"],
        "phase": session["current_phase"],
        "answer": transcript,
        "quality": analysis.get("quality"),
        "score": analysis.get("score", 50)
    })
    session["scores"].append(analysis.get("score", 50))

    # Check if interview is done
    if session["question_number"] >= TOTAL_QUESTIONS:
        session["status"] = "completed"

        feedback = generate_final_feedback(
            topic=session["topic"],
            candidate_name=session["candidate_name"],
            conversation_history=session["history"]
        )

        closing_text = (
            f"Thank you {session['candidate_name']}, that concludes our "
            f"interview. You've answered all {TOTAL_QUESTIONS} questions. "
            f"We'll be in touch soon. Have a great day."
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
    session["current_phase"] = next_q.get("phase", "technical")
    session["question_number"] += 1

    # Brief acknowledgment before next question
    quality = analysis.get("quality", "adequate")
    if quality == "weak" or quality == "unclear":
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
            session["question_number"] / TOTAL_QUESTIONS * 100)
    })


# ── 3. Get Session ─────────────────────────────────────────────
@app.get("/session/{session_id}")
def get_session(session_id: str):
    if session_id not in sessions:
        return JSONResponse({"error": "Not found"}, status_code=404)
    s = sessions[session_id]
    avg = sum(s["scores"]) / len(s["scores"]) if s["scores"] else 0
    return JSONResponse({
        "topic": s["topic"],
        "candidate": s["candidate_name"],
        "question_number": s["question_number"],
        "total_questions": TOTAL_QUESTIONS,
        "status": s["status"],
        "current_phase": s["current_phase"],
        "average_score_so_far": round(avg)
    })


# ── 4. End Early ───────────────────────────────────────────────
@app.post("/end-interview/{session_id}")
def end_interview(session_id: str):
    if session_id not in sessions:
        return JSONResponse({"error": "Not found"}, status_code=404)
    s = sessions[session_id]
    s["status"] = "completed"
    feedback = generate_final_feedback(
        s["topic"], s["candidate_name"], s["history"])
    return JSONResponse({"feedback": feedback, "status": "completed"})


# ── 5. Health ──────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "4.0 - Mercor-level",
        "interviewer": "Groq/Llama-3.3-70b",
        "stt": "Whisper",
        "tts": "SpeechT5",
        "endpoints": [
            "POST /start-interview",
            "POST /submit-answer",
            "GET  /session/{id}",
            "POST /end-interview/{id}"
        ]
    }