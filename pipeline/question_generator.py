# pipeline/question_generator.py
# Uses Groq (free) with Llama 3.3 as conversational interviewer
# Mercor-level conversational interviewer using Groq (free)
from dotenv import load_dotenv
load_dotenv()
from groq import Groq
import json
import os

client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "llama-3.3-70b-versatile"

# ── Core Interviewer Personality ───────────────────────────────
INTERVIEWER_SYSTEM = """You are Alex, a senior technical interviewer at a top tech company.
You have 10+ years of hiring experience. You are sharp, professional, and conversational.

YOUR BEHAVIOR:
- You listen carefully to answers and probe weak spots
- If an answer is vague, you ask for specifics: "Can you walk me through exactly how you did that?"
- If an answer is strong, you go deeper: "Interesting — how would you handle X edge case?"
- If a candidate says "I don't know", you help them think: "Let's approach it differently — what do you know about..."
- You never repeat questions already asked
- You vary question types: technical depth, real experience, problem solving, behavioral
- You detect when candidates are bluffing and probe gently
- You sound human — use phrases like "Got it", "Interesting", "That makes sense"

CONVERSATION PHASES:
1. introduction (1-2 questions): warm up, background
2. technical (3-4 questions): core skills, depth
3. behavioral (1-2 questions): teamwork, challenges
4. closing (1 question): candidate's questions or reflection

RESPONSE FORMAT — always return valid JSON only, no markdown, no extra text:
{
  "question": "your question",
  "type": "technical|behavioral|situational|followup|clarification",
  "phase": "introduction|technical|behavioral|closing",
  "reasoning": "why you asked this (internal, not shown to candidate)",
  "answer_quality": "strong|adequate|weak|bluffing|unclear"
}

answer_quality only applies when analyzing an answer. For opening questions set it to null."""


# ── Answer Quality Analyzer ────────────────────────────────────
ANALYZER_SYSTEM = """You are evaluating a candidate's interview answer.
Be strict but fair. Look for:
- Specificity (did they give real examples or vague generalities?)
- Technical accuracy (is what they said actually correct?)
- Depth (do they understand WHY not just WHAT?)
- Communication clarity

Return ONLY valid JSON:
{
  "quality": "strong|adequate|weak|bluffing|unclear",
  "score": <0-100>,
  "issues": ["issue1", "issue2"],
  "follow_up_angle": "what to probe next based on this answer"
}"""


def _call_groq(messages: list, system: str = None, 
               max_tokens: int = 600) -> str:
    all_messages = []
    if system:
        all_messages.append({"role": "system", "content": system})
    all_messages.extend(messages)

    response = client.chat.completions.create(
        model=MODEL,
        messages=all_messages,
        max_tokens=max_tokens,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()


def _safe_parse(raw: str) -> dict:
    # Strip markdown
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1])
    # Find JSON object
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start != -1 and end > start:
        raw = raw[start:end]
    return json.loads(raw)


def _safe_parse_list(raw: str) -> list:
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1])
    start = raw.find("[")
    end = raw.rfind("]") + 1
    if start != -1 and end > start:
        raw = raw[start:end]
    return json.loads(raw)


def analyze_answer(
    question: str,
    answer: str,
    topic: str
) -> dict:
    """Analyze quality of candidate's answer"""
    raw = _call_groq(
        system=ANALYZER_SYSTEM,
        messages=[{
            "role": "user",
            "content": f"""Topic: {topic}
Question asked: {question}
Candidate answered: {answer}

Analyze this answer."""
        }],
        max_tokens=400
    )
    try:
        return _safe_parse(raw)
    except Exception:
        return {
            "quality": "unclear",
            "score": 50,
            "issues": [],
            "follow_up_angle": "ask for more detail"
        }


def generate_opening_question(topic: str, candidate_name: str) -> dict:
    """Generate the very first interview question"""
    raw = _call_groq(
        system=INTERVIEWER_SYSTEM,
        messages=[{
            "role": "user",
            "content": f"""Start a new interview.
Topic/Role: {topic}
Candidate name: {candidate_name}

Generate a warm, professional opening question to start the interview.
This should be an introduction phase question — get them talking about 
their background and experience with {topic}.
Return JSON only."""
        }],
        max_tokens=400
    )
    try:
        result = _safe_parse(raw)
        result["phase"] = "introduction"
        return result
    except Exception:
        return {
            "question": f"Tell me about your experience with {topic} and what you've built.",
            "type": "behavioral",
            "phase": "introduction",
            "reasoning": "fallback opening",
            "answer_quality": None
        }


def generate_next_question(
    topic: str,
    conversation_history: list,
    last_answer_analysis: dict,
    question_number: int,
    total_questions: int
) -> dict:
    """Generate next question based on conversation context"""

    # Build conversation context
    history_text = ""
    for turn in conversation_history[-8:]:
        history_text += f"[Q{turn['number']}] Interviewer: {turn['question']}\n"
        history_text += f"Candidate: {turn['answer']}\n"
        if turn.get("quality"):
            history_text += f"(Answer quality: {turn['quality']})\n"
        history_text += "\n"

    # Determine phase
    progress = question_number / total_questions
    if progress < 0.2:
        current_phase = "introduction"
    elif progress < 0.7:
        current_phase = "technical"
    elif progress < 0.9:
        current_phase = "behavioral"
    else:
        current_phase = "closing"

    follow_up_angle = last_answer_analysis.get(
        "follow_up_angle", "continue naturally")
    answer_quality = last_answer_analysis.get("quality", "adequate")

    raw = _call_groq(
        system=INTERVIEWER_SYSTEM,
        messages=[{
            "role": "user",
            "content": f"""Interview context:
Topic/Role: {topic}
Question {question_number} of {total_questions}
Current phase: {current_phase}
Last answer quality: {answer_quality}
Suggested follow-up angle: {follow_up_angle}

Conversation so far:
{history_text}

Generate question {question_number}. 
{"Probe deeper on their last answer — it was weak or vague." if answer_quality in ["weak", "bluffing", "unclear"] else ""}
{"Their last answer was strong — go deeper or move to next topic." if answer_quality == "strong" else ""}
{"We are in closing phase — wrap up professionally." if current_phase == "closing" else ""}

Return JSON only."""
        }],
        max_tokens=500
    )

    try:
        return _safe_parse(raw)
    except Exception:
        return {
            "question": f"Can you tell me more about your experience with {topic}?",
            "type": "followup",
            "phase": current_phase,
            "reasoning": "fallback",
            "answer_quality": None
        }


def generate_final_feedback(
    topic: str,
    candidate_name: str,
    conversation_history: list
) -> dict:
    """Generate detailed final feedback report"""

    history_text = ""
    for turn in conversation_history:
        history_text += f"Q{turn['number']}: {turn['question']}\n"
        history_text += f"Answer: {turn['answer']}\n"
        history_text += f"Quality: {turn.get('quality', 'unknown')}\n\n"

    raw = _call_groq(
        messages=[{
            "role": "user",
            "content": f"""You are a senior hiring manager reviewing an interview.
Candidate: {candidate_name}
Role: {topic}

Full interview transcript:
{history_text}

Generate a comprehensive hiring report. Return ONLY this JSON:
{{
  "overall_score": <0-100>,
  "verdict": "Strong Hire" or "Hire" or "No Hire" or "Strong No Hire",
  "confidence": "high" or "medium" or "low",
  "scores": {{
    "technical_depth": <0-100>,
    "communication": <0-100>,
    "problem_solving": <0-100>,
    "experience": <0-100>,
    "behavioral": <0-100>
  }},
  "strengths": ["specific strength with evidence", "..."],
  "weaknesses": ["specific weakness with evidence", "..."],
  "red_flags": ["any concerns", "..."],
  "standout_moments": ["best answers or moments", "..."],
  "summary": "3-4 sentence executive summary",
  "recommendation": "detailed hiring recommendation paragraph"
}}"""
        }],
        max_tokens=1200
    )

    try:
        return _safe_parse(raw)
    except Exception:
        return {
            "overall_score": 50,
            "verdict": "No Hire",
            "summary": "Unable to generate feedback.",
            "scores": {}
        }