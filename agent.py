"""
Kayfa AI Sales Agent — Full LangGraph Pipeline
Nodes: accent_detector → intent_detector → knowledge_router →
       [course | roadmap | diploma | faq]_retriever → response_generator →
       lead_scorer → (conditional) crm_logger → END

CRM logs ONCE per conversation (not per message).
"""

from __future__ import annotations

import json
import os
import re
import time
import threading
import uuid as _uuid
from datetime import datetime
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pymongo import MongoClient
from qdrant_client import QdrantClient

load_dotenv()

# ──────────────────────────────────────────────
# Pricing constants  (update when rates change)
# ──────────────────────────────────────────────
# Groq — llama3-70b-8192  (per 1M tokens)
_GROQ_INPUT_PRICE_PER_M = float(os.getenv("GROQ_INPUT_PRICE_PER_M", 0.05))
_GROQ_OUTPUT_PRICE_PER_M = float(os.getenv("GROQ_OUTPUT_PRICE_PER_M", 0.08))
_EMBED_PRICE_PER_M = float(os.getenv("EMBED_PRICE_PER_M", 0.0))

def _calc_llm_cost(input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens  / 1_000_000 * _GROQ_INPUT_PRICE_PER_M
        + output_tokens / 1_000_000 * _GROQ_OUTPUT_PRICE_PER_M
    )

def _calc_embed_cost(tokens: int) -> float:
    return tokens / 1_000_000 * _EMBED_PRICE_PER_M


# ──────────────────────────────────────────────
# TraceCollector — one per agent invocation
# ──────────────────────────────────────────────
_trace_local = threading.local()   # isolates collector per thread


class TraceCollector:
    """Accumulates steps, token counts and costs for one agent run."""

    def __init__(self, conversation_id: str, user_id: str = "anonymous"):
        self.run_id          = str(_uuid.uuid4())
        self.conversation_id = conversation_id
        self.user_id         = user_id
        self.started_at      = time.time()
        self.steps: list[dict] = []

        # running totals
        self.total_input_tokens  = 0
        self.total_output_tokens = 0
        self.total_embed_tokens  = 0
        self.total_cost_usd      = 0.0
        self.total_latency_ms    = 0.0

    # ── step helpers ──────────────────────────────────────
    def add_llm_call(
        self,
        node: str,
        purpose: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        prompt_snippet: str = "",
        result_snippet: str = "",
    ):
        cost = _calc_llm_cost(input_tokens, output_tokens)
        self.total_input_tokens  += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost_usd      += cost
        self.total_latency_ms    += latency_ms
        self.steps.append({
            "type":          "llm_call",
            "node":          node,
            "purpose":       purpose,
            "input_tokens":  input_tokens,
            "output_tokens": output_tokens,
            "latency_ms":    round(latency_ms, 1),
            "cost_usd":      round(cost, 8),
            "prompt_snippet":  prompt_snippet[:300],
            "result_snippet":  result_snippet[:600],
        })

    def add_tool_call(
        self,
        node: str,
        tool: str,
        args: dict,
        result_snippet: str,
        embed_tokens: int = 0,
        latency_ms: float = 0.0,
    ):
        cost = _calc_embed_cost(embed_tokens)
        self.total_embed_tokens += embed_tokens
        self.total_cost_usd     += cost
        self.total_latency_ms   += latency_ms
        self.steps.append({
            "type":           "tool_call",
            "node":           node,
            "tool":           tool,
            "args":           args,
            "embed_tokens":   embed_tokens,
            "latency_ms":     round(latency_ms, 1),
            "cost_usd":       round(cost, 8),
            "result_snippet": result_snippet[:600],
        })

    def add_decision(self, node: str, decision: str, detail: str = ""):
        self.steps.append({
            "type":     "decision",
            "node":     node,
            "decision": decision,
            "detail":   detail,
        })

    def finalise(self, user_message: str, final_response: str, agent_state: dict) -> dict:
        elapsed = (time.time() - self.started_at) * 1000
        doc = {
            "run_id":            self.run_id,
            "conversation_id":   self.conversation_id,
            "user_id":           self.user_id,
            "timestamp":         datetime.utcnow().isoformat(),
            "user_message":      user_message,
            "final_response":    final_response[:600],
            "intent":            agent_state.get("intent", ""),
            "lead_score":        agent_state.get("lead_score", 0),
            "lead_stage":        agent_state.get("lead_stage", "cold"),
            "lang":              agent_state.get("lang", "en"),
            "accent":            agent_state.get("accent", ""),
            "source_collection": agent_state.get("source_collection", ""),
            # cost
            "input_tokens":      self.total_input_tokens,
            "output_tokens":     self.total_output_tokens,
            "embed_tokens":      self.total_embed_tokens,
            "cost_usd":          round(self.total_cost_usd, 8),
            "latency_ms":        round(elapsed, 1),
            # trace
            "steps":             self.steps,
        }
        return doc


def get_trace() -> TraceCollector | None:
    """Return the TraceCollector bound to this thread (None if not set)."""
    return getattr(_trace_local, "collector", None)


def set_trace(tc: TraceCollector):
    _trace_local.collector = tc


# ──────────────────────────────────────────────
# MongoDB — cost / trace collection
# ──────────────────────────────────────────────
def _get_monitor_collection():
    """Returns the 'message_traces' collection in the same DB as CRM."""
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = MongoClient(os.getenv("MONGODB_URI", "mongodb://localhost:27017"))
    db = _mongo_client[os.getenv("MONGO_DB", "kayfa_crm")]
    return db["message_traces"]


def save_trace(doc: dict):
    """Persist a finished trace document; silently swallows errors."""
    try:
        _get_monitor_collection().insert_one(doc)
    except Exception as exc:
        print(f"[monitor] Failed to save trace: {exc}")


# ──────────────────────────────────────────────
# LLM
# ──────────────────────────────────────────────
llm = ChatGroq(
    model=os.getenv("GROQ_MODEL", "llama3-70b-8192"),
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3,
)

# ──────────────────────────────────────────────
# Qdrant
# ──────────────────────────────────────────────
qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

# ── Dense embedding via sentence-transformers (local) ──────────────────────
from sentence_transformers import SentenceTransformer as _SentenceTransformer

_embedder = _SentenceTransformer(os.getenv("embedding_model", "paraphrase-multilingual-MiniLM-L12-v2"),
                                token=os.getenv("HF_TOKEN", None))


def _get_dense_vector(text: str) -> tuple[list[float], int]:
    """Encode text locally → (384-dim dense vector, approx_tokens)."""
    vec = _embedder.encode(text, convert_to_numpy=True).tolist()
    approx_tokens = max(1, len(text.split()))
    return vec, approx_tokens


def _qdrant_search(collection: str, query: str, top_k: int = 5, _node: str = "retriever") -> str:
    """Search a Qdrant collection with a dense vector and return concatenated payloads."""
    tc = get_trace()
    t0 = time.time()
    try:
        vector, embed_tokens = _get_dense_vector(query)
        try:
            response = qdrant.query_points(
                collection_name=collection,
                query=vector,
                limit=top_k,
                with_payload=True,
            )
            results = response.points
            print(results)
        except AttributeError:
            results = qdrant.search(
                collection_name=collection,
                query_vector=vector,
                limit=top_k,
                with_payload=True,
            )

        if not results:
            if tc:
                tc.add_tool_call(
                    node=_node, tool="qdrant_search",
                    args={"collection": collection, "query": query[:100], "top_k": top_k},
                    result_snippet="(no results)",
                    embed_tokens=embed_tokens,
                    latency_ms=(time.time() - t0) * 1000,
                )
            return ""

        MIN_SCORE = 0.05
        chunks = []
        for r in results:
            print(r.score)
            if getattr(r, "score", 1.0) < MIN_SCORE:
                continue
            payload = r.payload
            text = payload.get("text") or payload.get("content") or ""
            if text:
                chunks.append(text)

        result_text = "\n\n".join(chunks)
        if tc:
            sources = [r.payload.get("source", "?") for r in results[:3]]
            tc.add_tool_call(
                node=_node, tool="qdrant_search",
                args={"collection": collection, "query": query[:100], "top_k": top_k},
                result_snippet=f"sources: {sources}\n\n{result_text[:400]}",
                embed_tokens=embed_tokens,
                latency_ms=(time.time() - t0) * 1000,
            )
        return result_text
    except Exception as exc:
        print(f"[Qdrant] Error querying {collection}: {exc}")
        return ""


# ──────────────────────────────────────────────
# MongoDB CRM
# ──────────────────────────────────────────────
_mongo_client: MongoClient | None = None


def _get_crm_collection():
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = MongoClient(os.getenv("MONGODB_URI", "mongodb://localhost:27017"))
    db = _mongo_client[os.getenv("MONGO_DB", "kayfa_crm")]
    return db["leads"]


# ──────────────────────────────────────────────
# Agent State
# ──────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

    lang: str           # "ar" | "en"
    accent: str         # Egyptian, Saudian, Syrian, MSA, Mixed

    # prompt enhancer
    original_query:  str   # raw user message before enhancement
    enhanced_query:  str   # cleaned / expanded version
    enhancer_hint:   str   # coarse intent hint from enhancer

    intent: str         # one of 9 intent labels
    intent_confidence: float

    user_goal: str      # natural-language distillation of what the user wants

    knowledge: str      # retrieved context to ground the response
    source_collection: str

    lead_score: int     # 0–100
    lead_stage: str     # cold / warm / hot / enrolled

    crm_required: bool
    crm_ticket: dict

    # ── NEW: conversation-level CRM guard ─────────────
    # True once a lead has been persisted for this conversation.
    # Prevents duplicate CRM entries across multiple message turns.
    crm_logged: bool

    response: str       # final message to return to the user


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def _read_prompt(filename: str) -> str:
    path = os.path.join("SystemMessages", filename)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""


def _strip_json(raw: str) -> str:
    """Remove markdown fences and leading/trailing whitespace."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


def _last_human_message(state: AgentState) -> str:
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


# ──────────────────────────────────────────────
# Node 0 — prompt_enhancer
# ──────────────────────────────────────────────
_ENHANCER_PROMPT = """
You are a query enhancer for Kayfa, an Arabic e-learning platform.

Your job is to take the user's raw message and return a cleaner, richer version
that will help a retrieval system find the most relevant courses and information.

Rules:
1. Fix spelling mistakes and typos (Arabic and English).
2. Expand abbreviations (e.g. "BI" → "Business Intelligence", "AI" → "Artificial Intelligence").
3. If the message is vague (e.g. "ابغى اتعلم"), add context based on Kayfa's domain
   (Data Science, AI, SOC, PenTest, Fullstack, Power BI).
4. Keep the same language and dialect as the original — do NOT translate.
5. Do NOT answer the question. Only rewrite it.
6. If the message is already clear, return it as-is with minimal changes.
7. Keep the enhanced query under 120 words.

Reply ONLY with valid JSON:
{{
  "enhanced": "<the improved query>",
  "original": "<the original message unchanged>",
  "hint":     "<one keyword that best describes what the user wants, e.g. 'pricing', 'enrollment', 'course_info', 'roadmap', 'general'>"
}}

User message:
{user_message}
"""

_HINT_TO_INTENT = {
    "pricing":     "pricing",
    "enrollment":  "enrollment",
    "course_info": "course_search",
    "roadmap":     "roadmap",
    "general":     "general",
    "diploma":     "diploma",
    "faq":         "faq",
}


def prompt_enhancer(state: AgentState) -> dict:
    user_text = _last_human_message(state)

    if len(user_text.strip()) < 8:
        print(f"[prompt_enhancer] skipped (too short)")
        return {"enhanced_query": user_text, "original_query": user_text}

    prompt = _ENHANCER_PROMPT.format(user_message=user_text)
    t0 = time.time()
    response = llm.invoke([HumanMessage(content=prompt)])
    latency = (time.time() - t0) * 1000

    usage   = getattr(response, "usage_metadata", None) or {}
    in_tok  = usage.get("input_tokens",  len(prompt.split()))
    out_tok = usage.get("output_tokens", 30)

    raw = _strip_json(response.content)
    try:
        data     = json.loads(raw)
        enhanced = data.get("enhanced", user_text)
        original = data.get("original", user_text)
        hint     = data.get("hint", "general").lower()
    except (json.JSONDecodeError, ValueError):
        print(f"[prompt_enhancer] Bad JSON: {repr(raw[:200])}")
        enhanced = user_text
        original = user_text
        hint     = "general"

    tc = get_trace()
    if tc:
        tc.add_llm_call(
            node="prompt_enhancer", purpose="Enhance & clarify user query",
            input_tokens=in_tok, output_tokens=out_tok, latency_ms=latency,
            prompt_snippet=user_text[:200],
            result_snippet=enhanced[:300],
        )
        tc.add_decision(
            node="prompt_enhancer",
            decision=f"hint={hint}",
            detail=f"original: {original[:80]} → enhanced: {enhanced[:80]}",
        )

    print(f"[prompt_enhancer] hint={hint}")
    print(f"[prompt_enhancer] original : {original[:100]}")
    print(f"[prompt_enhancer] enhanced : {enhanced[:100]}")

    updated_messages = []
    replaced = False
    for msg in reversed(state["messages"]):
        if not replaced and isinstance(msg, HumanMessage):
            updated_messages.insert(0, HumanMessage(content=enhanced))
            replaced = True
        else:
            updated_messages.insert(0, msg)

    return {
        "messages":       updated_messages,
        "enhanced_query": enhanced,
        "original_query": original,
        "enhancer_hint":  _HINT_TO_INTENT.get(hint, "general"),
    }


# ──────────────────────────────────────────────
# Node 1 — accent_detector
# ──────────────────────────────────────────────
def accent_detector(state: AgentState) -> dict:
    if state.get("lang") != "ar":
        return {}

    system_prompt = _read_prompt("accent_detector.txt") or (
        "You are an expert Arabic dialect classification system.\n\n"
        "Your task is to analyze a user's message and identify the spoken Arabic dialect (accent).\n\n"
        "You must return ONLY one label and nothing else.\n\n"
        "## Allowed Output Labels:\n"
        "- Egyptian\n"
        "- Saudian\n"
        "- Syrian\n"
        "## Instructions:\n"
        "1. Focus on vocabulary, phrasing, and expressions (not spelling mistakes).\n"
        "2. Ignore typos and informal writing styles.\n"
        "3. Do NOT explain your answer.\n"
        "4. Do NOT add punctuation, extra words, or formatting.\n"
        "5. If the dialect is unclear, choose MSA.\n"
        "6. If multiple dialects are present, choose Mixed.\n"
        "7. Be conservative: avoid guessing if unsure.\n\n"
        "## Examples:\n"
        "Input: ازيك عامل ايه يا باشا → Output: Egyptian\n"
        "Input: كيفك يا صديقي → Output: Syrian\n"
        "Input: شنو حوالك اخي الكريم → Output: Saudian\n\n"
        "Output ONLY one label. No explanations. No extra text. No JSON. No formatting."
    )

    user_text = _last_human_message(state)
    t0 = time.time()
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_text),
    ])
    latency = (time.time() - t0) * 1000

    usage = getattr(response, "usage_metadata", None) or {}
    in_tok  = usage.get("input_tokens",  len(system_prompt.split()) + len(user_text.split()))
    out_tok = usage.get("output_tokens", 5)

    tc = get_trace()
    if tc:
        tc.add_llm_call(
            node="accent_detector", purpose="Classify Arabic dialect",
            input_tokens=in_tok, output_tokens=out_tok, latency_ms=latency,
            prompt_snippet=user_text[:200],
            result_snippet=response.content.strip(),
        )

    raw_accent = response.content.strip().split()[0].strip(".,!؟") if response.content.strip() else "MSA"
    _ACCENT_MAP = {
        "egyptian": "Egyptian",
        "مصري":    "Egyptian",
        "مصرية":   "Egyptian",
        "saudian":  "Saudian",
        "saudi":    "Saudian",
        "سعودي":   "Saudian",
        "خليجي":   "Saudian",
        "syrian":   "Syrian",
        "سوري":    "Syrian",
        "شامي":    "Syrian",
        "levantine":"Syrian",
    }
    accent = _ACCENT_MAP.get(raw_accent.lower(), raw_accent.title())
    print(f"[accent_detector] detected={accent}")
    return {"accent": accent}


# ──────────────────────────────────────────────
# Node 2 — intent_detector
# ──────────────────────────────────────────────
INTENT_LABELS = Literal[
    "course_search",
    "course_details",
    "roadmap",
    "diploma",
    "faq",
    "pricing",
    "comparison",
    "enrollment",
    "general",
]

_INTENT_JSON_INSTRUCTION = """
IMPORTANT — respond ONLY with a valid JSON object, nothing else:
{
  "intent": "<one of: course_search | course_details | roadmap | diploma | faq | pricing | comparison | enrollment | general>",
  "confidence": <0.0 to 1.0>,
  "user_goal": "<one sentence describing what the user wants>"
}
No markdown, no explanation, no extra keys.
"""


def intent_detector(state: AgentState) -> dict:
    
    # ── Context Lock ──────────────────────────────────────
    # لو ريم سألت عن بيانات التواصل في آخر رسالة،
    # والمستخدم رد بموافقة → lock على enrollment
    _AGREEMENT_SIGNALS = [
        "ماشي", "تمام", "اوكي", "اوك", "نعم", "آه", "أه", "يلا",
        "ok", "okay", "sure", "yes", "go ahead", "يتواصلوا", "تتواصلوا"
    ]
    _DATA_REQUEST_SIGNALS = [
        "اسم", "واتساب", "إيميل", "بيانات", "تتواصل", "يتواصل",
        "name", "whatsapp", "email", "contact", "reach out"
    ]
    
    last_ai_message = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage):
            last_ai_message = msg.content.lower()
            break
    
    user_text_lower = _last_human_message(state).lower()
    
    ai_asked_for_data = any(sig in last_ai_message for sig in _DATA_REQUEST_SIGNALS)
    user_agreed = any(sig in user_text_lower for sig in _AGREEMENT_SIGNALS)
    
    if ai_asked_for_data and user_agreed:
        print(f"[intent_detector] context lock → enrollment (user agreed to share data)")
        return {
            "intent":            "enrollment",
            "intent_confidence": 0.95,
            "user_goal":         "المستخدم وافق على مشاركة بياناته للتواصل",
        }
    # ── End Context Lock ───────────────────────────────────

    # ... باقي الكود زي ما هو (LLM classification)

# ──────────────────────────────────────────────
# Conditional edge — accent routing (START → node)
# ──────────────────────────────────────────────
def accent_routing(state: AgentState) -> str:
    return "accent_detector" if state.get("lang") == "ar" else "prompt_enhancer"


# ──────────────────────────────────────────────
# Conditional edge — knowledge routing
# ──────────────────────────────────────────────
def knowledge_routing(state: AgentState) -> str:
    intent        = state.get("intent", "general")
    user_goal     = (state.get("user_goal") or "").lower()
    enhanced      = (state.get("enhanced_query") or "").lower()
    print("="*50)
    print("intent:", state.get("intent"))
    print("user_goal:", state.get("user_goal"))
    print("enhanced:", state.get("enhanced_query"))
    print("="*50)
    # نجمع الاتنين عشان نكون sure
    combined = user_goal + " " + enhanced

    # SOC / PenTest / Diploma keywords
    diploma_keywords = ["soc", "pentest", "diploma", "دبلوم", "اختراق", "أمن سيبراني"]
    if any(kw in combined for kw in diploma_keywords):
        return "diploma_retriever"

    # Track / course bundle keywords → course_retriever
    track_keywords = ["track", "تراك", "bundle", "program", "برنامج", "مسار"]
    if any(kw in combined for kw in track_keywords):
        return "course_retriever"

    mapping = {
    "course_search":  "course_retriever",
    "course_details": "course_retriever",
    "roadmap":        "roadmap_retriever",
    "diploma":        "diploma_retriever",
    "pricing":        "course_retriever",
    "comparison":     "course_retriever",
    "enrollment":     "faq_retriever",   # ← كان diploma_retriever، غيّره لـ faq
    }
    return mapping.get(intent, "faq_retriever")

# ──────────────────────────────────────────────
# Node 3a — course_retriever
# ──────────────────────────────────────────────
def course_retriever(state: AgentState) -> dict:
    query  = state.get("user_goal") or _last_human_message(state)
    intent = state.get("intent", "")
    enhanced = (state.get("enhanced_query") or "").lower()
    
    # لو الكلام عن track → زود النتايج من tracks collection
    is_track = "track" in enhanced or "تراك" in enhanced or "مسار" in enhanced
    top_k    = 6 if (intent == "pricing" or is_track) else 4

    tracks  = _qdrant_search("kayfa_paid_educational_tracks", query, top_k=top_k, _node="course_retriever")
    courses = _qdrant_search("kayfa_paid_individual_courses",  query, top_k=top_k, _node="course_retriever")
    free    = _qdrant_search("kayfa_free_educational_content", query, top_k=2,     _node="course_retriever")

    knowledge = "\n\n---\n\n".join(filter(None, [tracks, courses, free]))
    print(f"[course_retriever] knowledge length={len(knowledge)}")
    return {
        "knowledge":         knowledge,
        "source_collection": "paid_courses+tracks",
    }

# ──────────────────────────────────────────────
# Node 3b — roadmap_retriever
# ──────────────────────────────────────────────
def roadmap_retriever(state: AgentState) -> dict:
    query = state.get("user_goal") or _last_human_message(state)

    qdrant_knowledge = _qdrant_search("kayfa_knowledge", query, top_k=4, _node="roadmap_retriever")

    json_knowledge = ""
    try:
        with open("kayfa_roadmaps.json", "r", encoding="utf-8") as f:
            roadmaps = json.load(f)
        q_lower = query.lower()
        relevant = [
            json.dumps(rm, ensure_ascii=False)
            for rm in (roadmaps if isinstance(roadmaps, list) else [roadmaps])
            if any(kw in json.dumps(rm).lower() for kw in q_lower.split())
        ]
        json_knowledge = "\n\n".join(relevant[:3])
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    knowledge = "\n\n---\n\n".join(filter(None, [qdrant_knowledge, json_knowledge]))
    print(f"[roadmap_retriever] knowledge length={len(knowledge)}")
    return {
        "knowledge":         knowledge,
        "source_collection": "roadmaps",
    }


# ──────────────────────────────────────────────
# Node 3c — diploma_retriever
# ──────────────────────────────────────────────
DIPLOMA_COLLECTIONS = [
    "kayfa_ai_diploma",
    "kayfa_data_science_diploma",
    "kayfa_fullstack_diploma",
    "kayfa_pentest_diploma",
    "kayfa_soc_diploma",
]


def diploma_retriever(state: AgentState) -> dict:
    query = state.get("user_goal") or _last_human_message(state)

    q_lower = query.lower()
    keyword_map = {
        "kayfa_ai_diploma":           ["ai", "artificial", "machine learning", "ml", "ذكاء"],
        "kayfa_data_science_diploma": ["data science", "data", "بيانات", "تحليل"],
        "kayfa_fullstack_diploma":    ["fullstack", "full stack", "web", "frontend", "backend"],
        "kayfa_pentest_diploma":      ["pentest", "penetration", "اختراق", "security"],
        "kayfa_soc_diploma":          ["soc", "security operations", "أمن"],
    }

    scored = sorted(
        DIPLOMA_COLLECTIONS,
        key=lambda c: sum(1 for kw in keyword_map.get(c, []) if kw in q_lower),
        reverse=True,
    )
    top_collections = scored[:2] if scored else DIPLOMA_COLLECTIONS[:2]

    chunks = [_qdrant_search(col, query, top_k=4, _node="diploma_retriever") for col in top_collections]
    knowledge = "\n\n---\n\n".join(filter(None, chunks))
    print(f"[diploma_retriever] knowledge length={len(knowledge)}")
    return {
        "knowledge":         knowledge,
        "source_collection": "+".join(top_collections),
    }


# ──────────────────────────────────────────────
# Node 3d — faq_retriever
# ──────────────────────────────────────────────
def faq_retriever(state: AgentState) -> dict:
    query = state.get("user_goal") or _last_human_message(state)

    faq     = _qdrant_search("kayfa_policies_and_faqs",  query, top_k=5, _node="faq_retriever")
    privacy = _qdrant_search("kayfa_privacy_policy",      query, top_k=2, _node="faq_retriever")
    overview= _qdrant_search("kayfa_company_overview",    query, top_k=2, _node="faq_retriever")

    knowledge = "\n\n---\n\n".join(filter(None, [faq, privacy, overview]))
    return {
        "knowledge":         knowledge,
        "source_collection": "faq+policies+overview",
    }


# ──────────────────────────────────────────────
# Node 4 — response_generator
# ──────────────────────────────────────────────
_DIALECT_INSTRUCTIONS = {
    "Egyptian": (
        "Speak in Egyptian Arabic (عامية مصرية). "
        "Use Egyptian expressions naturally: إيه، عامل إيه، يعني، بقى، دلوقتي، كمان، أهو، طب، معلش، خليني، هنا. "
        "Keep the tone warm and casual, like talking to a friend from Cairo. "
        "Avoid formal Modern Standard Arabic phrasing — it should feel like a real Egyptian conversation."
    ),
    "Saudian": (
        "Speak in Saudi/Gulf Arabic (عامية سعودية/خليجية). "
        "Use Gulf expressions naturally: شنو، وش، كيف حالك، إيش، ودي، أبغى، حق، زين، تعال، لا بأس، عاد. "
        "Keep the tone respectful but warm, fitting for a professional conversation in the Gulf region."
    ),
    "Syrian": (
        "Speak in Syrian/Levantine Arabic (عامية شامية). "
        "Use Levantine expressions naturally: كيفك، شو، هلق، كتير، بدي، لازم، منيح، ماشي، يلا، شو رأيك. "
        "Keep the tone friendly and conversational, like a natural chat from Damascus or Beirut."
    ),
    "Mixed": (
        "The user mixes dialects. Speak in clear Modern Standard Arabic (فصحى مبسّطة) "
        "that is comfortable for all Arab audiences — simple, warm, and easy to follow."
    ),
    "MSA": (
        "Speak in simplified Modern Standard Arabic (فصحى مبسّطة). "
        "Keep sentences clear and natural — avoid overly formal or academic phrasing."
    ),
}

_RESPONSE_SYSTEM = """
You are Reem (ريم), the AI sales consultant for Kayfa — an Arabic e-learning platform
specialising in Data Science, AI, SOC, PenTest, and Fullstack Development.

Your personality: warm, knowledgeable, persuasive but never pushy or dishonest.

Language instruction: {lang_instruction}

{dialect_instruction}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚨 STRICT GROUNDING RULES — NEVER BREAK THESE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. If the user asks about a specific product (e.g. 'SOC course') and KNOWLEDGE
    contains only loosely related items, say: 
    'الكورسات اللي عندي دلوقتي في المجال ده هي: [list what IS in KNOWLEDGE].
    للتفاصيل الكاملة عن برنامج SOC، https://kayfa.io' 
    NEVER invent a duration, price, or course name not in KNOWLEDGE.
"2. For pricing questions, ALWAYS list the prices found in KNOWLEDGE.
   Never say you don't have pricing info if prices appear in KNOWLEDGE.
3. NEVER invent course names, prices, durations, or instructor names.
4. NEVER list courses not present in the KNOWLEDGE section.
5. NEVER mention a price more than once per conversation turn — pick the first
   price found in KNOWLEDGE and stick to it. Never give two different prices.
6. If the retrieved knowledge contains one or more URLs relevant to the user's request,
    ALWAYS include the most relevant URL in your response.

    Never replace an existing URL with generic text like
    "the Kayfa team will contact you".

    If the knowledge contains a course URL or the main Kayfa website,
    show it explicitly.
7. If you are unsure, say you are unsure. Honesty builds trust.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Sales behaviours:
1. Map the user's goal to a real Kayfa product found in KNOWLEDGE.
2. Frame value honestly using only facts from KNOWLEDGE.
3. Handle objections (price, time, prerequisites) with empathy, then redirect.
4. End EVERY response with ONE clear next step.
5. Answer the user's question completely using KNOWLEDGE.

KNOWLEDGE:
{knowledge}

REMINDER: If the above KNOWLEDGE is empty or irrelevant, do NOT invent information.
"""


def response_generator(state: AgentState) -> dict:
    lang      = state.get("lang", "en")
    accent    = state.get("accent", "")
    intent    = state.get("intent", "")          # ← أضف السطر ده
    goal      = state.get("user_goal", _last_human_message(state))
    knowledge = state.get("knowledge", "")
    is_ar     = lang == "ar"

    # ── Enrollment override ──────────────────────────────────
    # لو intent = enrollment، خلي ريم تجمع البيانات على طول
    enrollment_instruction = ""
    if intent == "enrollment":
        if is_ar:
            enrollment_instruction = """
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    🎯 ENROLLMENT MODE
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    المستخدم وافق على التواصل أو طلب التسجيل.

    افعل هذا فقط:
    1. اشكره على موافقته بجملة واحدة دافئة.
    2. اطلب منه البيانات التالية بشكل طبيعي:
    - الاسم الكريم
    - رقم الواتساب أو الإيميل
    - المدينة (اختياري)
    3. أكد له إن الفريق هيتواصل معاه خلال 24 ساعة.

    🚫 لا تذكر أي كورسات أو معلومات جديدة.
    🚫 لا تسأل أي أسئلة غير البيانات المطلوبة.
    🚫 لا تستخدم الـ KNOWLEDGE في الرد ده خالص.
    """
        else:
            enrollment_instruction = """
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    🎯 ENROLLMENT MODE
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    The user agreed to be contacted or requested enrollment.

    Do ONLY this:
    1. Thank them warmly in one sentence.
    2. Ask for:
    - Full name
    - WhatsApp number or email
    - City (optional)
    3. Confirm the team will reach out within 24 hours.

    🚫 Do NOT mention any courses or new information.
    🚫 Do NOT use the KNOWLEDGE section in this response.
    """
# ────────────────────────────────────────────────────────

    if not knowledge or not knowledge.strip():
        knowledge = (
            "لا توجد معلومات كافية..." if is_ar else "NO RELEVANT KNOWLEDGE FOUND..."
        )

    if lang == "ar":
        lang_instruction    = f"Arabic — {accent} dialect" if accent else "Arabic (Modern Standard)"
        dialect_instruction = _DIALECT_INSTRUCTIONS.get(accent, _DIALECT_INSTRUCTIONS["MSA"])
    else:
        lang_instruction    = "English"
        dialect_instruction = "Respond in clear, friendly English."

    system_content = _RESPONSE_SYSTEM.format(
        lang_instruction    = lang_instruction,
        dialect_instruction = dialect_instruction,
        knowledge           = knowledge,
    )

    # أضف الـ enrollment instruction في الآخر لو موجود
    if enrollment_instruction:
        system_content += enrollment_instruction

    history = [msg for msg in state["messages"] if not isinstance(msg, AIMessage)]
    messages_to_send = [SystemMessage(content=system_content)] + history

    t0 = time.time()
    response = llm.invoke(messages_to_send)
    latency = (time.time() - t0) * 1000

    usage   = getattr(response, "usage_metadata", None) or {}
    in_tok  = usage.get("input_tokens",  len(system_content.split()) + sum(len(m.content.split()) for m in history))
    out_tok = usage.get("output_tokens", len(response.content.split()))

    tc = get_trace()
    if tc:
        tc.add_llm_call(
            node="response_generator", purpose=f"Generate sales response — {accent or 'en'}",
            input_tokens=in_tok, output_tokens=out_tok, latency_ms=latency,
            prompt_snippet=goal[:200],
            result_snippet=response.content[:400],
        )
    print(f"[response_generator] response = (response.content)")
    return {"response": response.content}


# ──────────────────────────────────────────────
# Node 5 — lead_scorer
# ──────────────────────────────────────────────
_LEAD_SCORE_PROMPT = """
You are a B2C sales lead scorer for an e-learning platform.

Score the conversation below from 0 to 100:
- 0–30: cold (browsing, no buying signals)
- 31–59: warm (interest shown, comparing options)
- 60–79: hot (asked about price, enrollment, certificates)
- 80–100: ready to enroll (explicit intent, urgency, payment questions)

Also assign a stage: cold | warm | hot | enrolled
And set crm_required to true if score ≥ 60.

Reply ONLY with valid JSON:
{{
  "lead_score": <int 0-100>,
  "lead_stage": "<cold|warm|hot|enrolled>",
  "crm_required": <true|false>
}}

Conversation:
{conversation}
"""


def lead_scorer(state: AgentState) -> dict:
    lines = []
    for msg in state["messages"]:
        role = "User" if isinstance(msg, HumanMessage) else "Agent"
        lines.append(f"{role}: {msg.content}")
    conversation = "\n".join(lines)

    high_intent = state.get("intent") in ("enrollment", "pricing", "diploma")
    if high_intent:
        conversation += "\n[System note: high-purchase-intent detected by intent classifier]"

    prompt = _LEAD_SCORE_PROMPT.format(conversation=conversation)
    t0 = time.time()
    response = llm.invoke([HumanMessage(content=prompt)])
    latency = (time.time() - t0) * 1000

    usage = getattr(response, "usage_metadata", None) or {}
    in_tok  = usage.get("input_tokens",  len(prompt.split()))
    out_tok = usage.get("output_tokens", 20)

    raw = _strip_json(response.content)
    try:
        data = json.loads(raw)
        score   = int(data.get("lead_score", 0))
        stage   = data.get("lead_stage", "cold")
        crm_req = bool(data.get("crm_required", False))
    except (json.JSONDecodeError, ValueError):
        print(f"[lead_scorer] Bad JSON: {repr(raw)}")
        score, stage, crm_req = 0, "cold", False

    tc = get_trace()
    if tc:
        tc.add_llm_call(
            node="lead_scorer", purpose="Score lead intent",
            input_tokens=in_tok, output_tokens=out_tok, latency_ms=latency,
            prompt_snippet=conversation[-300:],
            result_snippet=raw[:200],
        )
        tc.add_decision(
            node="lead_scorer",
            decision=f"score={score} stage={stage} crm_required={crm_req}",
        )

    print(f"[lead_scorer] score={score} stage={stage} crm_required={crm_req}")
    return {
        "lead_score":  score,
        "lead_stage":  stage,
        "crm_required": crm_req,
    }


# ──────────────────────────────────────────────
# Conditional edge — after response_generator
# ──────────────────────────────────────────────
def crm_routing(state: AgentState) -> str:
    if not state.get("crm_required"):
        return END

    if state.get("crm_logged"):
        return END

    # لو enrollment، سجّل من أول تيرن
    if state.get("intent") == "enrollment":
        return "crm_logger"

    # غير كده، استنى تيرنين
    human_turns = len([m for m in state["messages"] if isinstance(m, HumanMessage)])
    if human_turns < 2:
        return END

    return "crm_logger"

# ──────────────────────────────────────────────
# Node 6 — crm_logger
# ──────────────────────────────────────────────
_CRM_EXTRACT_PROMPT = """
You are a CRM data extractor for Kayfa, an Arabic e-learning platform.

Read the conversation below and extract structured lead data.
Reply ONLY with valid JSON — no markdown, no explanation.

CRITICAL LANGUAGE RULES:
- The "summary" field MUST be written in Modern Standard Arabic (فصحى) only.
- The "goal", "buying_signals", "objections", "next_action" fields MUST be in Arabic only.
- Never mix languages. Never use Russian, French, or any non-Arabic language.
- Names and emails stay as-is (Latin characters are fine for those fields only).

Extract:
{{
  "name":           "<full name if mentioned, else null>",
  "contact":        "<phone/whatsapp/email if mentioned, else null>",
  "city":           "<city if mentioned, else null>",
  "country":        "<country if mentioned or inferable from dialect, else null>",
  "products":       ["<list of Kayfa courses/tracks/diplomas the user asked about>"],
  "goal":           "<user's career or learning goal in one Arabic sentence>",
  "buying_signals": "<buying signals in Arabic — payment questions, enrollment intent, etc.>",
  "objections":     "<objections in Arabic — price, time, prerequisites — and how handled>",
  "summary":        "<2-3 sentence summary in Modern Standard Arabic (فصحى) ONLY>",
  "next_action":    "<recommended next step for the sales team, in Arabic>"
}}

Conversation:
{conversation}
"""


def crm_logger(state: AgentState) -> dict:
    import uuid as _u

    lang       = state.get("lang", "en")
    accent     = state.get("accent", "")
    lead_score = state.get("lead_score", 0)
    lead_stage = state.get("lead_stage", "cold")
    intent     = state.get("intent", "general")
    is_ar      = lang == "ar"

    # ── Build conversation string for LLM extraction ──
    lines = []
    for msg in state["messages"]:
        role = "User" if isinstance(msg, HumanMessage) else "Agent"
        lines.append(f"{role}: {msg.content}")
    conversation = "\n".join(lines)

    # ── Extract structured fields with LLM ────────────
    extracted = {}
    try:
        extract_response = llm.invoke([
            HumanMessage(content=_CRM_EXTRACT_PROMPT.format(conversation=conversation))
        ])
        raw = _strip_json(extract_response.content)
        extracted = json.loads(raw)
    except Exception as exc:
        print(f"[crm_logger] extraction failed: {exc}")

    # ── Stage label (bilingual) ────────────────────────
    _STAGE_AR = {
        "cold":     "بارد",
        "warm":     "دافئ",
        "hot":      "ساخن",
        "enrolled": "مسجّل",
    }
    stage_label_ar = _STAGE_AR.get(lead_stage, lead_stage)
    stage_label_en = lead_stage.title()

    # ── Dialect label (bilingual) ──────────────────────
    _DIALECT_AR = {
        "Egyptian": "اللهجة المصرية",
        "Saudian":  "اللهجة السعودية",
        "Syrian":   "اللهجة السورية",
        "MSA":      "الفصحى",
        "Mixed":    "لهجة مختلطة",
        "":         "الفصحى",
    }
    dialect_ar    = _DIALECT_AR.get(accent, accent if accent else "الفصحى")
    lang_label_ar = f"العربية — {dialect_ar}" if is_ar else "الإنجليزية"
    lang_label_en = f"Arabic — {accent}" if (is_ar and accent) else ("Arabic" if is_ar else "English")

    # ── Ticket ID ──────────────────────────────────────
    ticket_id = f"LEAD-{datetime.utcnow().strftime('%Y-%m%d')}-{str(_u.uuid4())[:4].upper()}"

    # ── Timestamp (Cairo UTC+3) ────────────────────────
    from datetime import timezone, timedelta
    cairo_tz = timezone(timedelta(hours=3))
    ts_cairo = datetime.now(cairo_tz).strftime("%Y-%m-%d · %H:%M")

    # ── Build the rich ticket document ────────────────
    ticket = {
        # meta
        "ticket_id":         ticket_id,
        "timestamp":         datetime.utcnow().isoformat(),
        "timestamp_display": ts_cairo,
        "status":            "new",

        # classification
        "lang":              lang,
        "accent":            accent,
        "lang_label_ar":     lang_label_ar,
        "lang_label_en":     lang_label_en,
        "lead_score":        lead_score,
        "lead_stage":        lead_stage,
        "stage_label_ar":    stage_label_ar,
        "stage_label_en":    stage_label_en,
        "intent":            intent,

        # extracted fields
        "name":              extracted.get("name"),
        "contact":           extracted.get("contact"),
        "city":              extracted.get("city"),
        "country":           extracted.get("country"),
        "products":          extracted.get("products", []),
        "goal":              extracted.get("goal") or state.get("user_goal", ""),
        "buying_signals":    extracted.get("buying_signals", ""),
        "objections":        extracted.get("objections", ""),
        "summary":           extracted.get("summary", ""),
        "next_action":       extracted.get("next_action", ""),

        # agent context
        "source_collection":  state.get("source_collection", ""),
        "response_preview":   (state.get("response") or "")[:300],
        "conversation_turns": len([m for m in state["messages"] if isinstance(m, HumanMessage)]),
    }

    try:
        col    = _get_crm_collection()
        result = col.insert_one(ticket)
        ticket["_id"] = str(result.inserted_id)
        print(f"[crm_logger] Lead saved: {ticket_id} score={lead_score} stage={lead_stage}")
    except Exception as exc:
        print(f"[crm_logger] MongoDB error: {exc}")

    # ── Mark conversation as logged so crm_routing skips on future turns ──
    return {"crm_ticket": ticket, "crm_logged": True}

# ──────────────────────────────────────────────────────────────────
# Node 4.5 — conversation_ender
# Detects if the user is wrapping up the conversation.
# If yes AND crm_required AND not yet logged → trigger field collection.
# ──────────────────────────────────────────────────────────────────

_FAREWELL_SIGNALS_AR = [
    "شكرا", "شكراً", "شكرا جزيلا", "تسلم", "تسلمي", "مع السلامة",
    "باي", "بيه", "وداعا", "الله يسلمك", "يسعدك", "ربنا يوفقك",
    "اوكي كده", "خلاص", "كفاية", "تمام خلاص", "يلا باي",
]
_FAREWELL_SIGNALS_EN = [
    "bye", "goodbye", "see you", "thanks", "thank you", "that's all",
    "no more questions", "i'm good", "i'm done", "that'll be all",
    "cheers", "ok thanks", "alright thanks", "take care",
]

_ENDER_PROMPT = """
You are a conversation-end detector for Kayfa, an Arabic e-learning chatbot.

Read the LAST USER MESSAGE and decide: is the user clearly ending or wrapping up the conversation?

Signals of ending: farewells, thank-yous with no follow-up question, expressions of satisfaction (خلاص / that's all / ok thanks / bye).
NOT ending: objections, new questions, asking for more info, or silence.

Reply ONLY with valid JSON:
{{
  "is_ending": <true|false>,
  "confidence": <0.0–1.0>,
  "reason": "<one short phrase>"
}}

Last user message:
{user_message}
"""


def conversation_ender(state: AgentState) -> dict:
    """
    Detects whether the user is wrapping up.
    Sets state key 'conversation_ending' = True/False.
    Does NOT modify the response — just signals downstream nodes.
    """
    user_text = _last_human_message(state).strip().lower()

    # Fast keyword check first (cheap, no LLM needed)
    farewell_hit = any(sig in user_text for sig in _FAREWELL_SIGNALS_AR + _FAREWELL_SIGNALS_EN)

    if farewell_hit:
        print(f"[conversation_ender] farewell keyword matched → ending=True")
        return {"conversation_ending": True}

    # For ambiguous cases, use LLM
    prompt = _ENDER_PROMPT.format(user_message=_last_human_message(state))
    t0 = time.time()
    response = llm.invoke([HumanMessage(content=prompt)])
    latency = (time.time() - t0) * 1000

    usage   = getattr(response, "usage_metadata", None) or {}
    in_tok  = usage.get("input_tokens",  len(prompt.split()))
    out_tok = usage.get("output_tokens", 15)

    raw = _strip_json(response.content)
    try:
        data       = json.loads(raw)
        is_ending  = bool(data.get("is_ending", False))
        confidence = float(data.get("confidence", 0.0))
        reason     = data.get("reason", "")
    except (json.JSONDecodeError, ValueError):
        is_ending, confidence, reason = False, 0.0, "parse error"

    tc = get_trace()
    if tc:
        tc.add_llm_call(
            node="conversation_ender", purpose="Detect farewell / conversation end",
            input_tokens=in_tok, output_tokens=out_tok, latency_ms=latency,
            prompt_snippet=user_text[:200],
            result_snippet=f"is_ending={is_ending} ({confidence:.0%}) — {reason}",
        )
        tc.add_decision(
            node="conversation_ender",
            decision=f"is_ending={is_ending}",
            detail=reason,
        )

    print(f"[conversation_ender] is_ending={is_ending} confidence={confidence:.2f} reason={reason}")
    return {"conversation_ending": is_ending}


# ──────────────────────────────────────────────────────────────────
# Node 4.6 — crm_field_collector
# If CRM fields are incomplete, Reem asks for them naturally
# before the ticket is saved.
# Required fields: name, contact, city, products
# ──────────────────────────────────────────────────────────────────

_FIELD_COLLECTOR_PROMPT = """
You are Reem (ريم), the warm AI sales consultant for Kayfa — an Arabic e-learning platform.

The conversation is wrapping up. Before saying goodbye, you need to gently collect
the missing information below so our team can follow up with the user.

MISSING FIELDS:
{missing_fields}

CONVERSATION HISTORY:
{conversation}

LANGUAGE: {lang_instruction}
{dialect_instruction}

Rules:
1. Sound completely natural — this is a warm, friendly closing, NOT a form.
2. Ask for ALL missing fields in ONE short message — do not send multiple messages.
3. Frame it as "so our team can follow up with you personally".
4. Keep it under 3 sentences. Be warm, not pushy.
5. If the user has already given their name, DO NOT ask for it again.
6. End with a warm closing phrase appropriate to the dialect.

Write ONLY the message to send to the user — nothing else.
"""

# Fields we want in every CRM ticket
_CRM_REQUIRED_FIELDS = {
    "name":    "الاسم الكريم / your name",
    "contact": "رقم الواتساب أو الإيميل / WhatsApp or email",
    "city":    "مدينتك / your city",
    "products": "الكورس أو البرنامج اللي بتفكر فيه / course or program of interest",
}


def _extract_known_fields(state: AgentState) -> dict:
    """
    Pull what we already know from the existing crm_ticket (if any)
    and from the conversation itself via a quick LLM pass.
    Returns dict of field → value (None if unknown).
    """
    ticket = state.get("crm_ticket") or {}
    known = {
        "name":     ticket.get("name"),
        "contact":  ticket.get("contact"),
        "city":     ticket.get("city"),
        "products": ticket.get("products") or [],
    }

    # Also scan messages for quick wins
    conversation = " ".join(
        msg.content for msg in state["messages"] if isinstance(msg, HumanMessage)
    ).lower()

    # Products: check if intent gives us a clue
    if not known["products"] and state.get("source_collection"):
        known["products"] = [state.get("source_collection")]

    return known


def crm_field_collector(state: AgentState) -> dict:
    """
    Checks which CRM fields are still missing.
    If any are missing, generates a natural closing message asking for them.
    Updates state['response'] so the graph returns this message to the user.
    Also updates state['crm_ticket'] with any already-known fields.
    """
    known = _extract_known_fields(state)

    # Determine which fields are genuinely missing
    missing = {}
    for field, label in _CRM_REQUIRED_FIELDS.items():
        val = known.get(field)
        if not val or (isinstance(val, list) and len(val) == 0):
            missing[field] = label

    print(f"[crm_field_collector] known={known}")
    print(f"[crm_field_collector] missing fields={list(missing.keys())}")

    if not missing:
        # All fields present — no message needed, proceed to crm_logger
        print("[crm_field_collector] all fields present, skipping collection")
        return {"fields_collected": True}

    # Build dialect-aware prompt
    lang   = state.get("lang", "en")
    accent = state.get("accent", "")
    is_ar  = lang == "ar"

    if is_ar:
        lang_instruction    = f"Arabic — {accent} dialect" if accent else "Arabic (Modern Standard)"
        dialect_instruction = _DIALECT_INSTRUCTIONS.get(accent, _DIALECT_INSTRUCTIONS["MSA"])
    else:
        lang_instruction    = "English"
        dialect_instruction = "Respond in clear, friendly English."

    # Format missing fields list for the prompt
    missing_list = "\n".join(f"- {label}" for label in missing.values())

    lines = []
    for msg in state["messages"]:
        role = "User" if isinstance(msg, HumanMessage) else "Reem"
        lines.append(f"{role}: {msg.content[:300]}")
    conversation_str = "\n".join(lines[-6:])  # last 3 exchanges

    prompt = _FIELD_COLLECTOR_PROMPT.format(
        missing_fields      = missing_list,
        conversation        = conversation_str,
        lang_instruction    = lang_instruction,
        dialect_instruction = dialect_instruction,
    )

    t0 = time.time()
    response = llm.invoke([HumanMessage(content=prompt)])
    latency = (time.time() - t0) * 1000

    usage   = getattr(response, "usage_metadata", None) or {}
    in_tok  = usage.get("input_tokens",  len(prompt.split()))
    out_tok = usage.get("output_tokens", 60)

    tc = get_trace()
    if tc:
        tc.add_llm_call(
            node="crm_field_collector", purpose="Collect missing CRM fields naturally",
            input_tokens=in_tok, output_tokens=out_tok, latency_ms=latency,
            prompt_snippet=f"missing: {list(missing.keys())}",
            result_snippet=response.content[:300],
        )

    print(f"[crm_field_collector] collection message: {response.content[:150]}")

    return {
        "response":        response.content,
        "fields_collected": False,   # still waiting for user to reply
    }
# ──────────────────────────────────────────────
# Graph assembly
# ──────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────
# Add to AgentState TypedDict:
# ──────────────────────────────────────────────────────────────────

# conversation_ending: bool   — True when farewell detected
# fields_collected: bool      — True when all CRM fields are present


# ──────────────────────────────────────────────────────────────────
# Conditional edge — after response_generator (replaces crm_routing)
# ──────────────────────────────────────────────────────────────────

def post_response_routing(state: AgentState) -> str:
    """
    After response_generator:
    1. Always run lead_scorer (unchanged).
    2. Run conversation_ender to check if user is leaving.
    """
    return "lead_scorer"   # lead_scorer already exists, keep it


def after_lead_scorer_routing(state: AgentState) -> str:
    """
    New edge: lead_scorer → conversation_ender OR crm_logger OR END
    """
    if not state.get("crm_required"):
        return END
    if state.get("crm_logged"):
        return END
    # Always run conversation_ender — it's cheap and decides next step
    return "conversation_ender"


def after_ender_routing(state: AgentState) -> str:
    """
    conversation_ender → crm_field_collector OR crm_logger OR END
    """
    if not state.get("conversation_ending"):
        # Not ending yet — log only if we have enough turns
        human_turns = len([m for m in state["messages"] if isinstance(m, HumanMessage)])
        if human_turns >= 2 and state.get("crm_required"):
            return "crm_logger"
        return END

    # Conversation IS ending — collect missing fields first
    known = _extract_known_fields(state)
    has_all = all(
        known.get(f) and (not isinstance(known.get(f), list) or len(known.get(f)) > 0)
        for f in _CRM_REQUIRED_FIELDS
    )
    if has_all:
        return "crm_logger"   # nothing missing, log directly
    return "crm_field_collector"


def after_collector_routing(state: AgentState) -> str:
    """
    crm_field_collector → crm_logger (when all fields filled) OR END (wait for user reply)
    """
    if state.get("fields_collected"):
        return "crm_logger"
    return END   # wait — the user needs to reply with their info


# ──────────────────────────────────────────────────────────────────
# Updated build_graph()
# ──────────────────────────────────────────────────────────────────
_ENROLLMENT_COLLECT_PROMPT = """
أنتِ ريم، مستشارة مبيعات Kayfa الذكية.

المستخدم أعرب عن رغبته في التسجيل أو سأل عن كيفية الاشتراك.

قدّمتِ له المعلومات اللازمة. الآن مهمتك أن تطلبي منه بياناته بشكل طبيعي ودافئ
حتى يتمكن فريقنا من التواصل معه وإتمام التسجيل.

البيانات المطلوبة:
- الاسم
- رقم الواتساب أو الإيميل
- المدينة (اختياري)

اللغة: {lang_instruction}
{dialect_instruction}

قواعد:
1. اجعلي الطلب طبيعياً ودافئاً — مش استمارة.
2. اذكري أن فريق Kayfa سيتواصل معه خلال 24 ساعة.
3. الرسالة قصيرة (3 جمل بحد أقصى).
4. لا تكرري المعلومات اللي قلتيها قبل كده.

اكتبي فقط الرسالة للمستخدم.
"""


def enrollment_collector(state: AgentState) -> dict:
    """
    Triggered when intent == 'enrollment'.
    Appends a natural data-collection message to Reem's response.
    """
    lang   = state.get("lang", "en")
    accent = state.get("accent", "")
    is_ar  = lang == "ar"

    if is_ar:
        lang_instruction    = f"Arabic — {accent} dialect" if accent else "Arabic (Modern Standard)"
        dialect_instruction = _DIALECT_INSTRUCTIONS.get(accent, _DIALECT_INSTRUCTIONS["MSA"])
    else:
        lang_instruction    = "English"
        dialect_instruction = "Respond in clear, friendly English."

    prompt = _ENROLLMENT_COLLECT_PROMPT.format(
        lang_instruction    = lang_instruction,
        dialect_instruction = dialect_instruction,
    )

    t0 = time.time()
    response = llm.invoke([HumanMessage(content=prompt)])
    latency = (time.time() - t0) * 1000

    usage   = getattr(response, "usage_metadata", None) or {}
    in_tok  = usage.get("input_tokens",  len(prompt.split()))
    out_tok = usage.get("output_tokens", 60)

    tc = get_trace()
    if tc:
        tc.add_llm_call(
            node="enrollment_collector", purpose="Collect enrollment info naturally",
            input_tokens=in_tok, output_tokens=out_tok, latency_ms=latency,
            prompt_snippet="enrollment intent detected",
            result_snippet=response.content[:300],
        )

    # Append collection message to the existing response
    existing_response = state.get("response", "")
    combined = f"{existing_response}\n\n{response.content}".strip()

    print(f"[enrollment_collector] appended collection message")
    return {"response": combined}

def after_response_routing(state: AgentState) -> str:
    """
    بعد response_generator:
    - لو enrollment → enrollment_collector أولاً
    - غير كده → conversation_ender (أو END)
    """
    intent     = state.get("intent", "")
    crm_req    = state.get("crm_required", False)
    crm_logged = state.get("crm_logged", False)

    if intent == "enrollment" and not crm_logged:
        return "enrollment_collector"

    if crm_req and not crm_logged:
        return "conversation_ender"

    return END


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("prompt_enhancer",      prompt_enhancer)
    graph.add_node("accent_detector",      accent_detector)
    graph.add_node("intent_detector",      intent_detector)
    graph.add_node("course_retriever",     course_retriever)
    graph.add_node("roadmap_retriever",    roadmap_retriever)
    graph.add_node("diploma_retriever",    diploma_retriever)
    graph.add_node("faq_retriever",        faq_retriever)
    graph.add_node("response_generator",   response_generator)
    graph.add_node("lead_scorer",          lead_scorer)
    graph.add_node("enrollment_collector", enrollment_collector)   # ← NEW
    graph.add_node("conversation_ender",   conversation_ender)
    graph.add_node("crm_field_collector",  crm_field_collector)
    graph.add_node("crm_logger",           crm_logger)

    graph.add_conditional_edges(START, accent_routing, {
        "accent_detector": "accent_detector",
        "prompt_enhancer": "prompt_enhancer",
    })
    graph.add_edge("accent_detector", "prompt_enhancer")
    graph.add_edge("prompt_enhancer", "intent_detector")

    graph.add_conditional_edges("intent_detector", knowledge_routing, {
        "course_retriever":  "course_retriever",
        "roadmap_retriever": "roadmap_retriever",
        "diploma_retriever": "diploma_retriever",
        "faq_retriever":     "faq_retriever",
    })

    for retriever in ("course_retriever", "roadmap_retriever", "diploma_retriever", "faq_retriever"):
        graph.add_edge(retriever, "lead_scorer")

    graph.add_edge("lead_scorer", "response_generator")

    # ← التعديل الجوهري هنا
    graph.add_conditional_edges("response_generator", after_response_routing, {
        "enrollment_collector": "enrollment_collector",
        "conversation_ender":   "conversation_ender",
        END:                    END,
    })

    # enrollment_collector → conversation_ender (عشان نحفظ في CRM بعدين)
    graph.add_edge("enrollment_collector", "conversation_ender")

    graph.add_conditional_edges("conversation_ender", after_ender_routing, {
        "crm_field_collector": "crm_field_collector",
        "crm_logger":          "crm_logger",
        END:                   END,
    })

    graph.add_conditional_edges("crm_field_collector", after_collector_routing, {
        "crm_logger": "crm_logger",
        END:          END,
    })

    graph.add_edge("crm_logger", END)

    return graph.compile()
app=build_graph()
# print(app.get_graph().draw_ascii())
# ──────────────────────────────────────────────
# Public monitoring API (imported by app.py)
# ──────────────────────────────────────────────
__all__ = ["app", "TraceCollector", "set_trace", "get_trace", "save_trace"]


# # ──────────────────────────────────────────────
# # Quick test
# # ──────────────────────────────────────────────
# if __name__ == "__main__":
#     test_cases = [
#         {
#             "messages":   [HumanMessage(content="I want to learn data science, where do I start?")],
#             "lang":       "en",
#             "accent":     "",
#             "crm_logged": False,   # ← always initialise
#         },
#         {
#             "messages":   [HumanMessage(content="عايز أعرف أكتر عن دبلومة الـ AI، كمان عايز أعرف السعر")],
#             "lang":       "ar",
#             "accent":     "",
#             "crm_logged": False,   # ← always initialise
#         },
#     ]

#     for i, state in enumerate(test_cases, 1):
#         print(f"\n{'='*60}")
#         print(f"Test {i}: {state['messages'][0].content[:60]}")
#         print("="*60)
#         result = app.invoke(state)
#         print(f"Intent       : {result.get('intent')}")
#         print(f"Accent       : {result.get('accent')}")
#         print(f"Lead score   : {result.get('lead_score')}")
#         print(f"Lead stage   : {result.get('lead_stage')}")
#         print(f"CRM required : {result.get('crm_required')}")
#         print(f"CRM logged   : {result.get('crm_logged')}")
#         print(f"\nResponse:\n{result.get('response', '')[:400]}")

    # qdrant.close()