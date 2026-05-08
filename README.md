# AI Interview Model 🎙️

An intelligent conversational interview platform powered by Groq (Llama 3.3 70B), Whisper STT, and SpeechT5 TTS. Conducts real adaptive interviews like Mercor.ai — asks follow-up questions based on your answers, detects weak responses, and generates scored feedback reports.

---

## Features

- 🤖 Conversational AI interviewer (adapts to your answers)
- 🎤 Speech-to-text (Whisper)
- 🔊 Text-to-speech AI voice (SpeechT5)
- 📊 Automatic answer quality scoring
- 📝 Full feedback report on completion
- 🔄 Session-aware multi-turn interviews
- 🆓 100% free APIs (Groq)

---

## Project Structure

```
Interview Model/
├── main.py                  ← FastAPI server (your API)
├── train.py                 ← Fine-tuning pipeline
├── setup_models.py          ← Download models locally
├── setup_on_deploy.py       ← Download models on server
├── requirements.txt
├── Procfile                 ← Railway deployment config
├── railway.json             ← Railway build config
├── labels.json              ← Training data labels
│
├── pipeline/
│   ├── question_generator.py  ← Groq/Llama interviewer brain
│   ├── stt_processor.py       ← Whisper speech-to-text
│   ├── tts_processor.py       ← SpeechT5 text-to-speech
│   ├── audio_processor.py     ← Audio feature extraction
│   ├── video_processor.py     ← Video/pose analysis
│   ├── text_processor.py      ← Text embedding
│   └── fusion_model.py        ← Score fusion neural net
│
├── models/
│   ├── whisper_model/         ← STT model
│   ├── tts_model/             ← TTS model + vocoder
│   ├── text_model/            ← Sentence transformer
│   └── question_model/        ← Flan-T5 (fallback)
│
└── data/
    ├── audio/                 ← Training audio files
    ├── video/                 ← Training video files
    └── text/                  ← Training text/PDF files
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/start-interview` | Begin new interview session |
| POST | `/submit-answer` | Send audio answer, get next question |
| GET | `/session/{id}` | Check session status |
| POST | `/end-interview/{id}` | End early and get feedback |
| GET | `/health` | Server health check |
| GET | `/docs` | Interactive API docs (Swagger UI) |

---

## Quickstart (Local)

### 1. Clone

```bash
git clone https://github.com/Junaid-290/Interview-Model.git
cd Interview-Model
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download models

```bash
python setup_models.py
```

### 4. Set environment variable

**Windows:**
```powershell
$env:GROQ_API_KEY="your_groq_key_here"
```

**Mac/Linux:**
```bash
export GROQ_API_KEY="your_groq_key_here"
```

Get your free key at: [console.groq.com](https://console.groq.com)

### 5. Run

```bash
uvicorn main:app --reload
```

Open: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## How to Use the API

### Start an interview

```bash
curl -X POST http://localhost:8000/start-interview \
  -F "topic=Flutter Developer" \
  -F "candidate_name=Ali" \
  -F "job_level=mid"
```

Response:
```json
{
  "session_id": "abc-123",
  "question": "Tell me about your experience with Flutter...",
  "audio_base64": "...",
  "phase": "introduction",
  "status": "active"
}
```

### Submit an answer

```bash
curl -X POST http://localhost:8000/submit-answer \
  -F "session_id=abc-123" \
  -F "audio=@answer.wav"
```

Response:
```json
{
  "transcript": "I have 2 years of Flutter experience...",
  "answer_quality": "adequate",
  "question": "Can you walk me through a specific project you built?",
  "audio_base64": "...",
  "progress": 25,
  "status": "active"
}
```

---

## Deploy on Railway

### 1. Install Railway CLI

```bash
npm install -g @railway/cli
```

### 2. Login and deploy

```bash
railway login
railway init
railway up
```

### 3. Add environment variable in Railway dashboard

```
GROQ_API_KEY = your_groq_key_here
```

### 4. Generate domain

Railway dashboard → Settings → Domains → Generate Domain

Your API will be live at:
```
https://your-app.up.railway.app/docs
```

---

## Flutter Integration

```dart
// Start interview
final response = await http.post(
  Uri.parse('https://your-app.up.railway.app/start-interview'),
  body: {'topic': 'Flutter Developer', 'candidate_name': 'Ali'},
);
final data = json.decode(response.body);
final sessionId = data['session_id'];
final question = data['question'];

// Submit answer
final request = http.MultipartRequest(
  'POST',
  Uri.parse('https://your-app.up.railway.app/submit-answer'),
);
request.fields['session_id'] = sessionId;
request.files.add(await http.MultipartFile.fromPath('audio', audioPath));
final result = await request.send();
```

---

## Training Your Own Model

Add matching files to `data/text`, `data/video`, `data/audio`, update `labels.json`, then:

```bash
python train.py
```

The fusion model saves to `models/fusion_model.pt` automatically.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| API Framework | FastAPI |
| Interviewer AI | Groq / Llama 3.3 70B |
| Speech to Text | OpenAI Whisper |
| Text to Speech | Microsoft SpeechT5 |
| Text Embeddings | Sentence Transformers |
| Score Fusion | PyTorch MLP |
| Deployment | Railway |

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes | Free at console.groq.com |

---

## License

MIT License — free to use and modify.

---

Built by [Junaid](https://github.com/Junaid-290)