# 🤖 Kayfa AI Sales Agent

An intelligent, bilingual (Arabic/English) sales agent for **Kayfa** — an Arabic-first e-learning platform specialising in Data Science, AI, SOC, PenTest, and Fullstack Development.

Built with **LangGraph**, **Groq LLM**, **Qdrant**, and **MongoDB**, deployed via a **Streamlit** interface.

---

## ✨ Features

- 🌍 **Bilingual support** — Arabic (Egyptian, Saudi, Syrian, MSA, Mixed dialects) + English
- 🧠 **Multi-node LangGraph pipeline** — modular, traceable, production-ready
- 🎯 **Intent detection** — 9 intent labels with confidence scoring
- 🔍 **Semantic retrieval** — dense vector search across 10+ Qdrant collections
- 💬 **Dialect-aware responses** — Reem adapts her tone to the user's Arabic dialect
- 📊 **Lead scoring** — 0–100 score with cold/warm/hot/enrolled stages
- 🗂️ **CRM auto-logging** — structured tickets saved to MongoDB Atlas
- 🧾 **Enrollment flow** — context-aware data collection when user wants to register
- 📡 **Full observability** — token usage, latency, cost tracking per message
- 📈 **Streamlit dashboard** — chat UI + cost monitoring + trace replay

---

## 🏗️ Architecture

```
START
  │
  ├─ [Arabic] → accent_detector → prompt_enhancer
  └─ [English] → prompt_enhancer
                      │
               intent_detector
                      │
          ┌───────────┼───────────┬──────────────┐
          ▼           ▼           ▼              ▼
  course_retriever  roadmap_  diploma_      faq_retriever
                   retriever  retriever
          └───────────┴───────────┴──────────────┘
                              │
                        lead_scorer
                              │
                    response_generator
                              │
                ┌─────────────┴──────────────┐
                ▼                            ▼
     [enrollment intent]            conversation_ender
     enrollment_collector                   │
                │               ┌───────────┴────────────┐
                └───────────────▼                        ▼
                         crm_field_collector         crm_logger
                                │                       │
                                └───────────────────────┘
                                            │
                                           END
```

### Pipeline Nodes

| Node | Role |
|------|------|
| `accent_detector` | Classifies Arabic dialect (Egyptian / Saudi / Syrian / MSA / Mixed) |
| `prompt_enhancer` | Fixes typos, expands abbreviations, enriches the query for retrieval |
| `intent_detector` | Classifies into 9 intents with confidence score + user goal extraction |
| `course_retriever` | Searches paid tracks, individual courses, and free content collections |
| `roadmap_retriever` | Searches learning roadmaps + local JSON roadmap file |
| `diploma_retriever` | Searches diploma-specific collections (AI, Data Science, Fullstack, PenTest, SOC) |
| `faq_retriever` | Searches policies, FAQ, privacy policy, and company overview |
| `lead_scorer` | Scores lead intent 0–100 and assigns cold/warm/hot/enrolled stage |
| `response_generator` | Generates Reem's dialect-aware, grounded sales response |
| `enrollment_collector` | Triggered on enrollment intent — naturally collects name + contact |
| `conversation_ender` | Detects farewell signals to trigger final CRM collection |
| `crm_field_collector` | Asks for any remaining missing CRM fields before saving |
| `crm_logger` | Saves structured lead ticket to MongoDB with LLM-extracted fields |

---

## 🗃️ Qdrant Collections

| Collection | Content |
|-----------|---------|
| `kayfa_paid_educational_tracks` | Paid course bundles and tracks |
| `kayfa_paid_individual_courses` | Individual paid courses |
| `kayfa_free_educational_content` | Free courses and content |
| `kayfa_knowledge` | General knowledge base |
| `kayfa_policies_and_faqs` | FAQ and platform policies |
| `kayfa_privacy_policy` | Privacy policy content |
| `kayfa_company_overview` | About Kayfa, mission, values |
| `kayfa_ai_diploma` | AI diploma program details |
| `kayfa_data_science_diploma` | Data Science diploma details |
| `kayfa_fullstack_diploma` | Fullstack diploma details |
| `kayfa_pentest_diploma` | PenTest diploma details |
| `kayfa_soc_diploma` | SOC diploma details |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Groq — `llama3-70b-8192` |
| Embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (local) |
| Vector DB | Qdrant Cloud |
| Graph engine | LangGraph `StateGraph` |
| Database | MongoDB Atlas (`kayfa_crm`) |
| Frontend | Streamlit |
| Env management | `python-dotenv` |

---

## 📁 Project Structure

```
kayfa-ai-agent/
│
├── agent.py                  # Main LangGraph pipeline
├── app.py                    # Streamlit chat interface
├── ingest.py                 # Data ingestion into Qdrant
├── kayfa_roadmaps.json       # Local roadmap data
│
├── SystemMessages/           # Prompt files (loaded at runtime)
│   ├── accent_detector.txt
│   └── intent_detector.txt
│
├── pages/                    # Streamlit multi-page app
│   ├── cost_monitoring.py    # Token usage & cost dashboard
│   └── trace_replay.py       # Per-message trace explorer
│
├── .env                      # Environment variables (not committed)
└── requirements.txt
```

---

## ⚙️ Setup

### 1. Clone the repo

```bash
git clone https://github.com/your-username/kayfa-ai-agent.git
cd kayfa-ai-agent
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama3-70b-8192

QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key

MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/
MONGO_DB=kayfa_crm

HF_TOKEN=your_huggingface_token        # optional, for gated models
embedding_model=paraphrase-multilingual-MiniLM-L12-v2
```

### 4. Ingest your data into Qdrant

```bash
python ingest.py
```

### 5. Run the Streamlit app

```bash
streamlit run app.py
```

---

## 💬 Usage

The chat interface supports both **Arabic and English**. Set the language toggle in the sidebar before starting a conversation.

**Example flows:**

| User message | Agent behaviour |
|---|---|
| "ممكن تكلمني عن كورسات الـ AI؟" | Retrieves courses, responds in detected dialect |
| "عايز اسجل في دبلومة الـ AI" | Enrollment mode — collects name + contact |
| "ما الفرق بين Data Science و AI؟" | Comparison retrieval from multiple collections |
| "I want to learn cybersecurity" | SOC/PenTest diploma retrieval in English |
| "شكرا خلاص" | Farewell detected → collects any missing CRM fields before ending |

---

## 📊 Monitoring

The app includes two monitoring pages accessible from the Streamlit sidebar:

### Cost Monitoring (`/pages/cost_monitoring.py`)
- Total spend by day / by node
- Token breakdown (input / output / embedding)
- Average latency per node
- Lead conversion funnel

### Trace Replay (`/pages/trace_replay.py`)
- Full step-by-step replay of any message trace
- LLM call details: prompt snippet, response snippet, tokens, cost, latency
- Tool call details: Qdrant queries, collections searched, scores
- Decision logs: intent, accent, lead score, routing decisions

Traces are stored in MongoDB under the `message_traces` collection.

---

## 🗂️ CRM Ticket Structure

Every qualified lead (score ≥ 60) generates a ticket in MongoDB `leads` collection:

```json
{
  "ticket_id": "LEAD-2026-0628-A3F1",
  "timestamp_display": "2026-06-28 · 14:32",
  "status": "new",
  "lang": "ar",
  "accent": "Egyptian",
  "lead_score": 82,
  "lead_stage": "hot",
  "intent": "enrollment",
  "name": "Ahmed Hassan",
  "contact": "+201012345678",
  "city": "Cairo",
  "country": "Egypt",
  "products": ["AI Diploma", "Data Science Track"],
  "goal": "يريد الانتقال إلى مجال الذكاء الاصطناعي خلال 6 أشهر",
  "buying_signals": "سأل عن السعر وطلب التسجيل مباشرة",
  "objections": "لم تُذكر اعتراضات",
  "summary": "مستخدم مصري مهتم بدبلومة الذكاء الاصطناعي...",
  "next_action": "التواصل عبر الواتساب خلال 24 ساعة",
  "response_preview": "...",
  "conversation_turns": 4
}
```

---

## 💰 Pricing Constants

| Model | Input | Output |
|-------|-------|--------|
| Groq llama3-70b-8192 | $0.59 / 1M tokens | $0.79 / 1M tokens |
| Local embeddings (MiniLM) | $0.00 | — |

Update `_GROQ_INPUT_PRICE_PER_M` and `_GROQ_OUTPUT_PRICE_PER_M` in `agent.py` when rates change.

---

## 🤖 Meet Reem

**ريم** is Kayfa's AI sales consultant persona. She is:
- Warm, knowledgeable, and persuasive — never pushy
- Dialect-aware: switches between Egyptian, Saudi, Syrian, MSA, and Mixed Arabic
- Strictly grounded: never invents course names, prices, or durations not in the knowledge base
- Enrollment-focused: naturally collects lead data when the user shows buying intent

---

## 📄 License

MIT License — see `LICENSE` for details.

---

## 🙏 Acknowledgements

Built on top of [LangGraph](https://github.com/langchain-ai/langgraph), [Groq](https://groq.com), [Qdrant](https://qdrant.tech), and [Streamlit](https://streamlit.io).
