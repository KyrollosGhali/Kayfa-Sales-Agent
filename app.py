import os
import streamlit as st
from datetime import datetime, timedelta
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from pymongo import MongoClient
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

load_dotenv()

# ──────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Kayfa AI Sales Agent",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────
# Safe field helper — prevents ALL None/type crashes
# ──────────────────────────────────────────────────────────
def safe(val, fallback="—"):
    """Return val as a non-empty string, or fallback."""
    if val is None:
        return fallback
    s = str(val).strip()
    return s if s else fallback


def safe_list(val):
    """Return val as a non-empty list, or []."""
    if not val:
        return []
    if isinstance(val, list):
        return [str(v) for v in val if v]
    return [str(val)]


# ──────────────────────────────────────────────────────────
# MongoDB helper (cached)
# ──────────────────────────────────────────────────────────
@st.cache_resource
def get_mongo():
    uri = os.getenv("MONGODB_URI") or st.secrets.get("MONGODB_URI", "mongodb://localhost:27017")
    db  = os.getenv("MONGO_DB")    or st.secrets.get("MONGO_DB", "kayfa_crm")
    client = MongoClient(uri)
    return client[db]["leads"]


@st.cache_resource
def get_traces_col():
    uri = os.getenv("MONGODB_URI") or st.secrets.get("MONGODB_URI", "mongodb://localhost:27017")
    db  = os.getenv("MONGO_DB")    or st.secrets.get("MONGO_DB", "kayfa_crm")
    client = MongoClient(uri)
    return client[db]["message_traces"]


# ──────────────────────────────────────────────────────────
# Agent (cached so model loads once)
# ──────────────────────────────────────────────────────────
@st.cache_resource
def get_agent():
    from agent import app
    return app


# ──────────────────────────────────────────────────────────
# RTL / LTR CSS injection
# ──────────────────────────────────────────────────────────
def inject_direction(lang: str):
    if lang == "ar":
        st.markdown("""
        <style>
        .stChatMessage, .stChatInput, .stMarkdown, p, li, h1, h2, h3, label {
            direction: rtl !important;
            text-align: right !important;
            font-family: 'Segoe UI', 'Cairo', 'Noto Sans Arabic', sans-serif !important;
        }
        .stChatInput textarea {
            direction: rtl !important;
            text-align: right !important;
        }
        [data-testid="stChatMessageContent"] {
            direction: rtl;
        }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        .stChatMessage, .stChatInput, .stMarkdown, p, li, h1, h2, h3, label {
            direction: ltr !important;
            text-align: left !important;
        }
        </style>
        """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────
# Shared sidebar
# ──────────────────────────────────────────────────────────
def render_sidebar():
    is_ar      = st.session_state.get("lang") == "ar"
    logged_in  = st.session_state.get("admin_logged_in", False)

    with st.sidebar:
        st.image("logo.png", use_container_width=True)
        st.markdown("---")

        public_pages = ["💬 Chat"]
        admin_pages  = ["📋 CRM Dashboard", "📊 Analytics", "💰 Cost Monitor", "🔍 Trace Viewer"]
        auth_page    = ["🔑 Login"] if not logged_in else []
        all_pages    = public_pages + (admin_pages if logged_in else []) + auth_page

        nav = st.session_state.pop("nav_target", None)
        default_idx = 0
        if nav and nav in all_pages:
            default_idx = all_pages.index(nav)

        page = st.radio(
            "التنقل" if is_ar else "Navigate",
            all_pages,
            index=default_idx,
            key="page_radio",
        )
        st.markdown("---")

        if logged_in:
            st.success(f"{'مرحباً، مشرف' if is_ar else 'Admin'} ✓")
            if st.button("🚪 تسجيل الخروج" if is_ar else "🚪 Logout"):
                st.session_state["admin_logged_in"] = False
                st.session_state["nav_target"] = "💬 Chat"
                st.rerun()
        else:
            st.info("🔒 سجّل دخولك للوحة الإدارة" if is_ar else "🔒 Log in to access admin pages")

        st.markdown("---")

        lang_choice = st.selectbox(
            "Language / اللغة",
            ["English", "العربية"],
            index=0 if st.session_state.get("lang", "en") == "en" else 1,
            key="lang_selector",
        )
        new_lang = "en" if lang_choice == "English" else "ar"
        if new_lang != st.session_state.get("lang"):
            st.session_state["lang"] = new_lang
            st.session_state["messages"] = []
            st.rerun()

    return page


# ──────────────────────────────────────────────────────────
# SESSION STATE INIT
# ──────────────────────────────────────────────────────────
if "lang" not in st.session_state:
    st.session_state["lang"] = "en"
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "agent_state" not in st.session_state:
    st.session_state["agent_state"] = {}
if "lead_score" not in st.session_state:
    st.session_state["lead_score"] = 0
if "lead_stage" not in st.session_state:
    st.session_state["lead_stage"] = "cold"
if "admin_logged_in" not in st.session_state:
    st.session_state["admin_logged_in"] = False
if "nav_target" not in st.session_state:
    st.session_state["nav_target"] = None


# ──────────────────────────────────────────────────────────
# AUTH HELPERS
# ──────────────────────────────────────────────────────────
_ADMIN_USER = os.getenv("ADMIN_USERNAME") or st.secrets.get("ADMIN_USERNAME", "admin")
_ADMIN_PASS = os.getenv("ADMIN_PASSWORD") or st.secrets.get("ADMIN_PASSWORD", "admin")


def _verify_admin(username: str, password: str) -> bool:
    import hmac
    return (
        hmac.compare_digest(username.strip(), _ADMIN_USER) and
        hmac.compare_digest(password.strip(), _ADMIN_PASS)
    )


def _require_admin() -> bool:
    if st.session_state.get("admin_logged_in"):
        return True
    st.warning(
        "🔒 هذه الصفحة للمشرفين فقط — يرجى تسجيل الدخول"
        if st.session_state.get("lang") == "ar"
        else "🔒 This page is for admins only — please log in."
    )
    if st.button(
        "🔑 تسجيل الدخول" if st.session_state.get("lang") == "ar" else "🔑 Go to Login"
    ):
        st.session_state["nav_target"] = "🔑 Login"
        st.rerun()
    return False


# ──────────────────────────────────────────────────────────
# PAGE 1 — CHAT
# ──────────────────────────────────────────────────────────
def page_chat():
    lang  = st.session_state["lang"]
    is_ar = lang == "ar"
    inject_direction(lang)

    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🎓 كايفا — مساعد المبيعات" if is_ar else "🎓 Kayfa — Sales Agent")
        st.caption(
            "اسألني عن الكورسات، الدبلومات، أو خطط التعلم"
            if is_ar else
            "Ask me about courses, diplomas, or learning roadmaps"
        )
    with col2:
        score = st.session_state.get("lead_score", 0)
        stage = st.session_state.get("lead_stage", "cold")
        stage_color = {"cold": "🔵", "warm": "🟡", "hot": "🟠", "enrolled": "🟢"}.get(stage, "⚪")

    st.divider()

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    placeholder = "اكتب رسالتك هنا..." if is_ar else "Type your message here..."
    user_input  = st.chat_input(placeholder)

    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        lc_messages = []
        for m in st.session_state["messages"]:
            if m["role"] == "user":
                lc_messages.append(HumanMessage(content=m["content"]))
            else:
                lc_messages.append(AIMessage(content=m["content"]))

        with st.chat_message("assistant"):
            with st.spinner("جاري التفكير..." if is_ar else "Thinking..."):
                try:
                    from agent import TraceCollector, set_trace, save_trace
                    agent = get_agent()

                    conv_id = st.session_state.get("conversation_id", "")
                    if not conv_id:
                        import uuid
                        conv_id = str(uuid.uuid4())
                        st.session_state["conversation_id"] = conv_id

                    user_id = st.session_state.get("user_id", "visitor")
                    tc = TraceCollector(conversation_id=conv_id, user_id=user_id)
                    set_trace(tc)

                    # Carry forward crm_logged so we never double-log
                    prev_state   = st.session_state.get("agent_state", {})
                    state_input  = {
                        "messages":            lc_messages,
                        "lang":                lang,
                        "accent":              st.session_state.get("accent", ""),
                        "crm_logged":          prev_state.get("crm_logged", False),
                        "conversation_ending": False,
                        "fields_collected":    False,
                    }
                    result = agent.invoke(state_input)

                    response_text = result.get("response", "")
                    st.markdown(response_text)

                    trace_doc = tc.finalise(
                        user_message=user_input,
                        final_response=response_text,
                        agent_state=result,
                    )
                    save_trace(trace_doc)
                    st.session_state["last_trace"] = trace_doc

                    st.session_state["messages"].append({"role": "assistant", "content": response_text})
                    st.session_state["agent_state"] = result
                    st.session_state["lead_score"]  = result.get("lead_score", 0)
                    st.session_state["lead_stage"]  = result.get("lead_stage", "cold")
                    if result.get("accent"):
                        st.session_state["accent"] = result["accent"]

                except Exception as exc:
                    err = f"❌ خطأ: {exc}" if is_ar else f"❌ Error: {exc}"
                    st.error(err)

    if st.session_state.get("agent_state"):
        with st.expander(
            "🔍 Debug — Agent State" if not is_ar else "🔍 تفاصيل — حالة الوكيل",
            expanded=False,
        ):
            s = st.session_state["agent_state"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Intent",      safe(s.get("intent"), "—"))
            c2.metric("Confidence",  f"{s.get('intent_confidence', 0):.0%}")
            c3.metric("Source",      safe(s.get("source_collection"), "—"))
            c1.metric("Lead Score",  s.get("lead_score", 0))
            c2.metric("Lead Stage",  safe(s.get("lead_stage"), "—"))
            c3.metric("CRM Logged",  "✅" if s.get("crm_logged") else "❌")
            if s.get("user_goal"):
                st.info(f"**Goal:** {s['user_goal']}")

    if st.button("🗑️ مسح المحادثة" if is_ar else "🗑️ Clear Chat", key="clear_chat"):
        st.session_state["messages"]       = []
        st.session_state["agent_state"]    = {}
        st.session_state["lead_score"]     = 0
        st.session_state["lead_stage"]     = "cold"
        st.session_state["conversation_id"]= ""
        st.rerun()


# ──────────────────────────────────────────────────────────
# PAGE 2 — CRM DASHBOARD
# ──────────────────────────────────────────────────────────
def page_crm():
    lang  = st.session_state["lang"]
    is_ar = lang == "ar"
    inject_direction(lang)

    st.title("📋 لوحة إدارة العملاء" if is_ar else "📋 CRM Dashboard")

    try:
        col   = get_mongo()
        leads = list(col.find({}).sort("timestamp", -1))
        for lead in leads:
            lead["_id"] = str(lead["_id"])
    except Exception as exc:
        st.error(f"MongoDB error: {exc}")
        return

    if not leads:
        st.info("لا يوجد عملاء بعد." if is_ar else "No leads yet.")
        return

    df = pd.DataFrame(leads)
    df["timestamp"]  = pd.to_datetime(df["timestamp"], errors="coerce")
    df["date"]       = df["timestamp"].dt.date
    df["lead_score"] = pd.to_numeric(df.get("lead_score", 0), errors="coerce").fillna(0)

    # ── KPI row ────────────────────────────────────────────
    total   = len(df)
    hot     = len(df[df["lead_stage"].isin(["hot", "enrolled"])])
    avg_sc  = round(df["lead_score"].mean(), 1)
    today_n = len(df[df["date"] == datetime.utcnow().date()])

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("إجمالي العملاء"  if is_ar else "Total Leads",   total)
    k2.metric("ساخن / مسجّل"   if is_ar else "Hot / Enrolled", hot)
    k3.metric("متوسط النقاط"   if is_ar else "Avg Score",      avg_sc)
    k4.metric("اليوم"           if is_ar else "Today",          today_n)

    st.divider()

    # ── Filters ────────────────────────────────────────────
    fc1, fc2, fc3 = st.columns(3)
    stage_opts  = ["All"] + sorted(df["lead_stage"].dropna().unique().tolist())
    intent_opts = ["All"] + sorted(df["intent"].dropna().unique().tolist())
    accent_opts = ["All"] + sorted(df["accent"].dropna().unique().tolist())

    sel_stage  = fc1.selectbox("المرحلة" if is_ar else "Stage",  stage_opts)
    sel_intent = fc2.selectbox("النية"   if is_ar else "Intent", intent_opts)
    sel_accent = fc3.selectbox("اللهجة" if is_ar else "Accent", accent_opts)

    filtered = df.copy()
    if sel_stage  != "All": filtered = filtered[filtered["lead_stage"] == sel_stage]
    if sel_intent != "All": filtered = filtered[filtered["intent"]     == sel_intent]
    if sel_accent != "All": filtered = filtered[filtered["accent"]     == sel_accent]

    st.caption(f"{'عرض' if is_ar else 'Showing'} {len(filtered)} {'عملاء' if is_ar else 'leads'}")

    # ── Status update helper ───────────────────────────────
    def update_status(lead_id: str, new_status: str):
        from bson import ObjectId
        try:
            get_mongo().update_one(
                {"_id": ObjectId(lead_id)},
                {"$set": {"status": new_status}},
            )
            st.success("تم التحديث!" if is_ar else "Updated!")
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

    _STAGE_COLOR = {"cold": "#60a5fa", "warm": "#fbbf24", "hot": "#f97316", "enrolled": "#22c55e"}
    _STAGE_EMOJI = {"cold": "🔵", "warm": "🟡", "hot": "🟠", "enrolled": "🟢"}
    _STAGE_AR    = {"cold": "بارد", "warm": "دافئ", "hot": "ساخن", "enrolled": "مسجّل"}

    # ── Ticket cards ───────────────────────────────────────
    for _, row in filtered.iterrows():
        stage = safe(row.get("lead_stage"), "cold")
        score = int(row.get("lead_score") or 0)
        emoji = _STAGE_EMOJI.get(stage, "⚪")
        color = _STAGE_COLOR.get(stage, "#888")
        name  = safe(row.get("name"))
        goal  = safe(row.get("goal") or row.get("user_goal"), "")
        tid   = safe(row.get("ticket_id") or row["_id"][:8])
        ts    = safe(row.get("timestamp_display") or str(row.get("timestamp", ""))[:16])

        # ── Products — always a safe joined string ─────────
        products_list = safe_list(row.get("products"))
        products_str  = ", ".join(products_list) if products_list else "—"

        stage_label = _STAGE_AR.get(stage, stage) if is_ar else stage.title()
        header = f"{emoji} {name}  ·  {score}/100  ·  {stage_label}  ·  {tid}"

        with st.expander(header, expanded=False):

            st.markdown(
                f'<div style="display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap;">'
                f'<span style="background:{color};color:#fff;padding:3px 10px;'
                f'border-radius:12px;font-size:13px;">{stage_label}</span>'
                f'<span style="background:#6366f1;color:#fff;padding:3px 10px;'
                f'border-radius:12px;font-size:13px;">عميل محتمل · {stage_label}</span>'
                f'<span style="background:#1e293b;color:#94a3b8;padding:3px 10px;'
                f'border-radius:12px;font-size:12px;">{tid}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            lc, rc = st.columns(2)

            with lc:
                st.markdown(f"**{'الاسم' if is_ar else 'Name'}**")
                st.write(safe(row.get("name")))

                st.markdown(f"**{'رقم التواصل' if is_ar else 'Contact'}**")
                st.write(safe(row.get("contact")))

                st.markdown(f"**{'المدينة' if is_ar else 'City'}**")
                city    = safe(row.get("city"),    "")
                country = safe(row.get("country"), "")
                location = f"{city}{'، ' + country if country and country != '—' else ''}".strip("، ") or "—"
                st.write(location)

                st.markdown(f"**{'اللغة / اللهجة' if is_ar else 'Language / Dialect'}**")
                lang_field = row.get("lang_label_ar") if is_ar else row.get("lang_label_en")
                st.write(safe(lang_field or row.get("accent")))

            with rc:
                st.markdown(f"**{'المنتجات محل الاهتمام' if is_ar else 'Products of Interest'}**")
                st.write(products_str)

                st.markdown(f"**{'الهدف' if is_ar else 'Goal'}**")
                st.write(goal or "—")

                st.markdown(f"**{'نقاط العميل' if is_ar else 'Lead Score'}**")
                st.progress(score / 100)
                st.caption(f"{score} / 100")

                st.markdown(f"**{'التاريخ' if is_ar else 'Date'}**")
                st.write(ts)

            st.divider()

            bs_col, obj_col = st.columns(2)
            with bs_col:
                st.markdown(f"**{'إشارات الشراء' if is_ar else 'Buying Signals'}**")
                st.info(safe(row.get("buying_signals")))
            with obj_col:
                st.markdown(f"**{'الاعتراضات' if is_ar else 'Objections'}**")
                st.warning(safe(row.get("objections")))

            if row.get("summary"):
                st.markdown(f"**{'ملخّص المحادثة' if is_ar else 'Conversation Summary'}**")
                st.markdown(safe(row.get("summary")))

            if row.get("next_action"):
                st.markdown(f"**{'الإجراء التالي' if is_ar else 'Next Action'}**")
                st.success(safe(row.get("next_action")))

            if row.get("response_preview"):
                with st.expander("💬 آخر رد من ريم" if is_ar else "💬 Last agent response"):
                    st.markdown(safe(row.get("response_preview")))

            st.divider()
            a1, a2, a3, _ = st.columns([1, 1, 1, 3])
            if a1.button("✅ تم التواصل" if is_ar else "✅ Contacted", key=f"c_{row['_id']}"):
                update_status(row["_id"], "contacted")
            if a2.button("🎯 مسجّل"      if is_ar else "🎯 Enrolled",  key=f"e_{row['_id']}"):
                update_status(row["_id"], "enrolled")
            if a3.button("🗑️ أرشفة"     if is_ar else "🗑️ Archive",   key=f"a_{row['_id']}"):
                update_status(row["_id"], "archived")


# ──────────────────────────────────────────────────────────
# PAGE 3 — ANALYTICS
# ──────────────────────────────────────────────────────────
def page_analytics():
    lang  = st.session_state["lang"]
    is_ar = lang == "ar"
    inject_direction(lang)

    st.title("📊 تحليلات المبيعات" if is_ar else "📊 Sales Analytics")

    try:
        col   = get_mongo()
        leads = list(col.find({}, {
            "timestamp": 1, "lead_score": 1, "lead_stage": 1,
            "intent": 1, "lang": 1, "accent": 1, "status": 1,
        }))
    except Exception as exc:
        st.error(f"MongoDB error: {exc}")
        return

    if len(leads) < 2:
        st.info("Need more leads for analytics." if not is_ar else "نحتاج لمزيد من البيانات.")
        return

    df = pd.DataFrame(leads)
    df["timestamp"]  = pd.to_datetime(df["timestamp"], errors="coerce")
    df["date"]       = df["timestamp"].dt.date
    df["week"]       = df["timestamp"].dt.to_period("W").astype(str)
    df["lead_score"] = pd.to_numeric(df.get("lead_score", 0), errors="coerce").fillna(0)

    # ── Row 1: Stage funnel + Intent breakdown ─────────────
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        stage_order  = ["cold", "warm", "hot", "enrolled"]
        stage_counts = (
            df["lead_stage"]
            .value_counts()
            .reindex(stage_order, fill_value=0)
            .reset_index()
        )
        stage_counts.columns = ["stage", "count"]
        fig_funnel = go.Figure(go.Funnel(
            y=stage_counts["stage"],
            x=stage_counts["count"],
            textinfo="value+percent initial",
            marker_color=["#60a5fa", "#fbbf24", "#f97316", "#22c55e"],
        ))
        fig_funnel.update_layout(
            title="Lead Funnel" if not is_ar else "قمع العملاء",
            margin=dict(t=40, b=20, l=20, r=20),
            height=320,
        )
        st.plotly_chart(fig_funnel, use_container_width=True)

    with r1c2:
        intent_counts = df["intent"].value_counts().reset_index()
        intent_counts.columns = ["intent", "count"]
        fig_intent = px.pie(
            intent_counts, names="intent", values="count",
            title="Intent Distribution" if not is_ar else "توزيع النوايا",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set3,
        )
        fig_intent.update_layout(height=320, margin=dict(t=40, b=20, l=20, r=20))
        st.plotly_chart(fig_intent, use_container_width=True)

    # ── Row 2: Lead score histogram + daily volume ─────────
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        fig_hist = px.histogram(
            df, x="lead_score", nbins=20, color="lead_stage",
            title="Lead Score Distribution" if not is_ar else "توزيع نقاط العملاء",
            color_discrete_map={
                "cold": "#60a5fa", "warm": "#fbbf24",
                "hot": "#f97316", "enrolled": "#22c55e",
            },
            labels={"lead_score": "Score", "count": "Leads"},
        )
        fig_hist.update_layout(height=320, margin=dict(t=40, b=20, l=20, r=20))
        st.plotly_chart(fig_hist, use_container_width=True)

    with r2c2:
        daily = df.groupby("date").size().reset_index(name="leads")
        daily["date"] = pd.to_datetime(daily["date"])
        fig_daily = px.bar(
            daily, x="date", y="leads",
            title="Daily Lead Volume" if not is_ar else "حجم العملاء اليومي",
            color_discrete_sequence=["#6366f1"],
        )
        fig_daily.update_layout(height=320, margin=dict(t=40, b=20, l=20, r=20))
        st.plotly_chart(fig_daily, use_container_width=True)

    # ── Row 3: Language split + Accent breakdown ───────────
    r3c1, r3c2 = st.columns(2)

    with r3c1:
        lang_counts = df["lang"].value_counts().reset_index()
        lang_counts.columns = ["lang", "count"]
        lang_counts["lang"] = lang_counts["lang"].map(
            {"ar": "Arabic 🇪🇬", "en": "English 🇬🇧"}
        )
        fig_lang = px.pie(
            lang_counts, names="lang", values="count",
            title="Language Split" if not is_ar else "توزيع اللغة",
            hole=0.4,
            color_discrete_sequence=["#6366f1", "#22c55e"],
        )
        fig_lang.update_layout(height=300, margin=dict(t=40, b=20, l=20, r=20))
        st.plotly_chart(fig_lang, use_container_width=True)

    with r3c2:
        ar_df = df[df["lang"] == "ar"]
        if not ar_df.empty and ar_df["accent"].notna().any():
            accent_counts = ar_df["accent"].value_counts().reset_index()
            accent_counts.columns = ["accent", "count"]
            fig_accent = px.bar(
                accent_counts, x="accent", y="count",
                title="Arabic Dialect Breakdown" if not is_ar else "توزيع اللهجات العربية",
                color="accent",
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            fig_accent.update_layout(
                height=300, margin=dict(t=40, b=20, l=20, r=20), showlegend=False,
            )
            st.plotly_chart(fig_accent, use_container_width=True)
        else:
            st.info("No Arabic leads yet." if not is_ar else "لا يوجد عملاء عرب بعد.")

    # ── Row 4: Weekly trend line ───────────────────────────
    weekly = df.groupby("week").agg(
        leads=("lead_score", "count"),
        avg_score=("lead_score", "mean"),
    ).reset_index()
    fig_weekly = go.Figure()
    fig_weekly.add_trace(go.Bar(
        x=weekly["week"], y=weekly["leads"],
        name="Leads", marker_color="#6366f1",
    ))
    fig_weekly.add_trace(go.Scatter(
        x=weekly["week"], y=weekly["avg_score"],
        name="Avg Score", yaxis="y2",
        line=dict(color="#f97316", width=2),
    ))
    fig_weekly.update_layout(
        title="Weekly Leads & Avg Score" if not is_ar else "الأسبوعي: العملاء ومتوسط النقاط",
        yaxis=dict(title="Leads"),
        yaxis2=dict(title="Avg Score", overlaying="y", side="right", range=[0, 100]),
        height=340,
        margin=dict(t=40, b=20, l=40, r=40),
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig_weekly, use_container_width=True)

    # ── Summary table ──────────────────────────────────────
    st.subheader("Summary by Intent" if not is_ar else "ملخص حسب النية")
    summary = df.groupby("intent").agg(
        Leads=("lead_score", "count"),
        Avg_Score=("lead_score", "mean"),
        Hot_Leads=("lead_stage", lambda x: x.isin(["hot", "enrolled"]).sum()),
    ).round(1).reset_index()
    summary.columns = ["Intent", "Leads", "Avg Score", "Hot Leads"]
    st.dataframe(summary, use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────────────────
# PAGE 4 — COST MONITOR
# ──────────────────────────────────────────────────────────
def page_cost_monitor():
    lang  = st.session_state["lang"]
    is_ar = lang == "ar"
    inject_direction(lang)

    st.title("💰 مراقبة التكاليف" if is_ar else "💰 Cost Monitor")
    st.caption(
        "تكلفة كل رسالة · كل محادثة · كل مستخدم — لا تفوّت أي استدعاء"
        if is_ar else
        "Per-message · per-conversation · per-user — every model call counted"
    )

    try:
        traces = list(get_traces_col().find(
            {},
            {
                "run_id": 1, "conversation_id": 1, "user_id": 1, "timestamp": 1,
                "user_message": 1, "intent": 1, "lang": 1,
                "input_tokens": 1, "output_tokens": 1, "embed_tokens": 1,
                "cost_usd": 1, "latency_ms": 1, "lead_score": 1,
            },
        ).sort("timestamp", -1).limit(2000))
    except Exception as exc:
        st.error(f"MongoDB error: {exc}")
        return

    if not traces:
        st.info(
            "No traces yet — send a message in the Chat page first." if not is_ar
            else "لا توجد بيانات بعد — ابدأ محادثة من صفحة الدردشة."
        )
        return

    df = pd.DataFrame(traces)
    df["timestamp"]  = pd.to_datetime(df["timestamp"],  errors="coerce")
    df["date"]       = df["timestamp"].dt.date
    df["cost_usd"]   = pd.to_numeric(df["cost_usd"],    errors="coerce").fillna(0)
    df["latency_ms"] = pd.to_numeric(df["latency_ms"],  errors="coerce").fillna(0)
    df["input_tokens"]  = pd.to_numeric(df.get("input_tokens",  0), errors="coerce").fillna(0)
    df["output_tokens"] = pd.to_numeric(df.get("output_tokens", 0), errors="coerce").fillna(0)
    df["embed_tokens"]  = pd.to_numeric(df.get("embed_tokens",  0), errors="coerce").fillna(0)

    # ── KPI strip ──────────────────────────────────────────
    total_cost  = df["cost_usd"].sum()
    total_msgs  = len(df)
    avg_cost    = df["cost_usd"].mean()
    avg_latency = df["latency_ms"].mean()
    n_convs     = df["conversation_id"].nunique()
    n_users     = df["user_id"].nunique()

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Total Cost"    if not is_ar else "التكلفة الكلية",  f"${total_cost:.4f}")
    k2.metric("Messages"      if not is_ar else "الرسائل",         total_msgs)
    k3.metric("Avg / Message" if not is_ar else "متوسط / رسالة",  f"${avg_cost:.5f}")
    k4.metric("Conversations" if not is_ar else "المحادثات",       n_convs)
    k5.metric("Users"         if not is_ar else "المستخدمون",      n_users)
    k6.metric("Avg Latency"   if not is_ar else "متوسط الاستجابة",f"{avg_latency:.0f} ms")

    st.divider()

    tab1, tab2, tab3 = st.tabs([
        "📨 Per Message"      if not is_ar else "📨 لكل رسالة",
        "💬 Per Conversation" if not is_ar else "💬 لكل محادثة",
        "👤 Per User"         if not is_ar else "👤 لكل مستخدم",
    ])

    # ── Tab 1 ─────────────────────────────────────────────
    with tab1:
        st.subheader("Message-level cost & latency")

        daily_cost = df.groupby("date")["cost_usd"].sum().reset_index()
        daily_cost["date"] = pd.to_datetime(daily_cost["date"])
        fig_daily = px.bar(
            daily_cost, x="date", y="cost_usd",
            title="Daily Spend (USD)" if not is_ar else "الإنفاق اليومي (USD)",
            color_discrete_sequence=["#6366f1"],
            labels={"cost_usd": "Cost (USD)", "date": "Date"},
        )
        fig_daily.update_layout(height=280, margin=dict(t=40, b=20, l=40, r=20))
        st.plotly_chart(fig_daily, use_container_width=True)

        token_df = df[["timestamp", "input_tokens", "output_tokens", "embed_tokens"]].copy()
        token_df = token_df.sort_values("timestamp")
        fig_tok  = go.Figure()
        fig_tok.add_trace(go.Bar(name="Input tokens",  x=token_df["timestamp"], y=token_df["input_tokens"],  marker_color="#6366f1"))
        fig_tok.add_trace(go.Bar(name="Output tokens", x=token_df["timestamp"], y=token_df["output_tokens"], marker_color="#f97316"))
        fig_tok.add_trace(go.Bar(name="Embed tokens",  x=token_df["timestamp"], y=token_df["embed_tokens"],  marker_color="#22c55e"))
        fig_tok.update_layout(
            barmode="stack",
            title="Token Breakdown per Message" if not is_ar else "توزيع التوكن لكل رسالة",
            height=280, margin=dict(t=40, b=20, l=40, r=20),
        )
        st.plotly_chart(fig_tok, use_container_width=True)

        st.subheader("Most expensive messages" if not is_ar else "أغلى الرسائل تكلفة")
        top_msgs = df.nlargest(20, "cost_usd")[[
            "timestamp", "user_message", "intent",
            "input_tokens", "output_tokens", "embed_tokens", "cost_usd", "latency_ms",
        ]].copy()
        top_msgs["user_message"] = top_msgs["user_message"].astype(str).str[:80]
        top_msgs["cost_usd"]     = top_msgs["cost_usd"].map("${:.6f}".format)
        top_msgs["latency_ms"]   = top_msgs["latency_ms"].map("{:.0f} ms".format)
        st.dataframe(top_msgs, use_container_width=True, hide_index=True)

    # ── Tab 2 ─────────────────────────────────────────────
    with tab2:
        conv_df = df.groupby("conversation_id").agg(
            Messages      =("run_id",        "count"),
            Total_Cost    =("cost_usd",      "sum"),
            Avg_Cost      =("cost_usd",      "mean"),
            Total_In_Tok  =("input_tokens",  "sum"),
            Total_Out_Tok =("output_tokens", "sum"),
            Avg_Latency   =("latency_ms",    "mean"),
            First_Msg     =("timestamp",     "min"),
            Last_Msg      =("timestamp",     "max"),
        ).reset_index().sort_values("Total_Cost", ascending=False)

        fig_conv = px.bar(
            conv_df.head(20), x="conversation_id", y="Total_Cost",
            title="Top 20 Conversations by Cost" if not is_ar else "أغلى 20 محادثة",
            labels={"Total_Cost": "Cost (USD)", "conversation_id": "Conversation"},
            color="Total_Cost",
            color_continuous_scale="Purples",
        )
        fig_conv.update_layout(height=300, margin=dict(t=40, b=60, l=40, r=20))
        fig_conv.update_xaxes(tickangle=45, tickfont=dict(size=9))
        st.plotly_chart(fig_conv, use_container_width=True)

        conv_df["Total_Cost"]      = conv_df["Total_Cost"].map("${:.6f}".format)
        conv_df["Avg_Cost"]        = conv_df["Avg_Cost"].map("${:.6f}".format)
        conv_df["Avg_Latency"]     = conv_df["Avg_Latency"].map("{:.0f} ms".format)
        conv_df["conversation_id"] = conv_df["conversation_id"].astype(str).str[:16] + "…"
        st.dataframe(conv_df, use_container_width=True, hide_index=True)

    # ── Tab 3 ─────────────────────────────────────────────
    with tab3:
        user_df = df.groupby("user_id").agg(
            Messages      =("run_id",          "count"),
            Total_Cost    =("cost_usd",        "sum"),
            Avg_Cost      =("cost_usd",        "mean"),
            Conversations =("conversation_id", "nunique"),
            Avg_Latency   =("latency_ms",      "mean"),
        ).reset_index().sort_values("Total_Cost", ascending=False)

        fig_user = px.bar(
            user_df, x="user_id", y="Total_Cost",
            title="Cost by User" if not is_ar else "التكلفة لكل مستخدم",
            labels={"Total_Cost": "Cost (USD)", "user_id": "User"},
            color="Total_Cost",
            color_continuous_scale="Blues",
        )
        fig_user.update_layout(height=280, margin=dict(t=40, b=40, l=40, r=20))
        st.plotly_chart(fig_user, use_container_width=True)

        user_df["Total_Cost"]  = user_df["Total_Cost"].map("${:.6f}".format)
        user_df["Avg_Cost"]    = user_df["Avg_Cost"].map("${:.6f}".format)
        user_df["Avg_Latency"] = user_df["Avg_Latency"].map("{:.0f} ms".format)
        st.dataframe(user_df, use_container_width=True, hide_index=True)

    # ── Optimisation hints ─────────────────────────────────
    st.divider()
    with st.expander(
        "⚡ Optimisation Hints" if not is_ar else "⚡ اقتراحات التحسين",
        expanded=False,
    ):
        expensive_intent = (
            df.groupby("intent")["cost_usd"].mean()
            .sort_values(ascending=False)
            .reset_index()
        )
        if not expensive_intent.empty:
            top_intent = expensive_intent.iloc[0]
            st.warning(
                f"**Most expensive intent:** `{top_intent['intent']}` costs "
                f"${top_intent['cost_usd']:.5f}/msg on average. "
                "Consider caching results for frequent queries of this type."
            )

        high_tok = df[df["input_tokens"] > df["input_tokens"].quantile(0.9)]
        if not high_tok.empty:
            st.info(
                f"**{len(high_tok)} messages** are in the top-10% for input tokens "
                f"(>{int(df['input_tokens'].quantile(0.9))} tokens). "
                "Trim the system prompt or use selective RAG to cut context size."
            )

        slow = df[df["latency_ms"] > 4000]
        if not slow.empty:
            st.info(
                f"**{len(slow)} messages** exceeded 4 s latency. "
                "Batch independent retrieval calls or route simple questions to a faster model."
            )


# ──────────────────────────────────────────────────────────
# PAGE 5 — TRACE VIEWER
# ──────────────────────────────────────────────────────────
def page_trace_viewer():
    lang  = st.session_state["lang"]
    is_ar = lang == "ar"
    inject_direction(lang)

    st.title("🔍 متتبع الاستجابة" if is_ar else "🔍 Trace Viewer")
    st.caption(
        "أعد تشغيل كل خطوة اتخذها الوكيل للرد على رسالة واحدة"
        if is_ar else
        "Replay every step the agent took to answer one prompt"
    )

    try:
        recent = list(get_traces_col().find(
            {},
            {
                "run_id": 1, "timestamp": 1, "user_message": 1,
                "intent": 1, "cost_usd": 1, "latency_ms": 1, "lead_score": 1,
            },
        ).sort("timestamp", -1).limit(100))
    except Exception as exc:
        st.error(f"MongoDB error: {exc}")
        return

    if not recent:
        st.info(
            "No traces yet — send a message in the Chat page first." if not is_ar
            else "لا توجد بيانات بعد — ابدأ محادثة من صفحة الدردشة."
        )
        return

    options = {
        f"{safe(str(t.get('timestamp',''))[:19])}  |  {safe(str(t.get('user_message',''))[:60])}": str(t["run_id"])
        for t in recent
    }
    selected_label  = st.selectbox(
        "Select a message to inspect:" if not is_ar else "اختر رسالة لمراجعتها:",
        list(options.keys()),
    )
    selected_run_id = options[selected_label]

    try:
        trace = get_traces_col().find_one({"run_id": selected_run_id})
    except Exception as exc:
        st.error(str(exc))
        return

    if not trace:
        st.warning("Trace not found.")
        return

    h1, h2, h3, h4, h5 = st.columns(5)
    h1.metric("Intent",     safe(trace.get("intent"), "—"))
    h2.metric("Lead Score", trace.get("lead_score", 0))
    h3.metric("Total Cost", f"${trace.get('cost_usd', 0):.6f}")
    h4.metric("Latency",    f"{trace.get('latency_ms', 0):.0f} ms")
    h5.metric("Lang",       f"{safe(trace.get('lang'), '—')} / {safe(trace.get('accent'), '—')}")

    st.divider()

    with st.chat_message("user"):
        st.markdown(f"**{safe(trace.get('user_message'))}**")

    steps = trace.get("steps") or []
    if not steps:
        st.info("No steps recorded for this trace.")
    else:
        st.markdown(f"**{len(steps)} steps recorded**")

    step_colors = {
        "llm_call":  ("🧠", "#6366f1"),
        "tool_call": ("🔧", "#f97316"),
        "decision":  ("🎯", "#22c55e"),
    }

    for i, step in enumerate(steps, 1):
        stype = step.get("type", "unknown")
        icon, _ = step_colors.get(stype, ("⚙️", "#888"))
        node    = safe(step.get("node"), "?")

        with st.expander(
            f"{icon} Step {i} · `{node}` · {stype.replace('_', ' ').title()}",
            expanded=(i <= 3),
        ):
            if stype == "llm_call":
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Purpose",       safe(step.get("purpose"), "—"))
                c2.metric("Input tokens",  step.get("input_tokens",  0))
                c3.metric("Output tokens", step.get("output_tokens", 0))
                c4.metric("Cost",          f"${step.get('cost_usd', 0):.7f}")
                st.metric("Latency", f"{step.get('latency_ms', 0):.0f} ms")
                if step.get("prompt_snippet"):
                    st.markdown("**Prompt snippet:**")
                    st.code(step["prompt_snippet"], language="text")
                if step.get("result_snippet"):
                    st.markdown("**Result snippet:**")
                    st.code(step["result_snippet"], language="text")

            elif stype == "tool_call":
                c1, c2, c3 = st.columns(3)
                c1.metric("Tool",         safe(step.get("tool"), "—"))
                c2.metric("Embed tokens", step.get("embed_tokens", 0))
                c3.metric("Latency",      f"{step.get('latency_ms', 0):.0f} ms")
                if step.get("args"):
                    st.markdown("**Arguments:**")
                    st.json(step["args"])
                if step.get("result_snippet"):
                    st.markdown("**Result (snippet):**")
                    st.code(step["result_snippet"], language="text")

            elif stype == "decision":
                st.success(f"**Decision:** {safe(step.get('decision'), '—')}")
                if step.get("detail"):
                    st.markdown(f"_{step['detail']}_")

    st.divider()
    with st.chat_message("assistant"):
        st.markdown(safe(trace.get("final_response")))

    st.divider()
    st.subheader("Cost breakdown" if not is_ar else "تفصيل التكاليف")
    llm_steps = [s for s in steps if s.get("type") == "llm_call"]
    if llm_steps:
        node_costs: dict = {}
        for s in llm_steps:
            node_costs[s["node"]] = node_costs.get(s["node"], 0) + s.get("cost_usd", 0)
        fig_pie = px.pie(
            names=list(node_costs.keys()),
            values=list(node_costs.values()),
            title="Cost share by node" if not is_ar else "حصة التكلفة لكل عقدة",
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_pie.update_layout(height=300, margin=dict(t=40, b=20, l=20, r=20))
        st.plotly_chart(fig_pie, use_container_width=True)

    last = st.session_state.get("last_trace")
    if last and last.get("run_id") != selected_run_id:
        with st.expander("⚡ Latest trace from this session", expanded=False):
            st.json({k: v for k, v in last.items() if k != "steps"})


# ──────────────────────────────────────────────────────────
# PAGE 0 — LOGIN
# ──────────────────────────────────────────────────────────
def page_login():
    is_ar = st.session_state.get("lang") == "ar"
    inject_direction("ar" if is_ar else "en")

    _, col, _ = st.columns([1, 1.4, 1])
    with col:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.image("logo.png", use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f"<h2 style='text-align:center;'>{'تسجيل دخول المشرف' if is_ar else 'Admin Login'}</h2>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<p style='text-align:center;color:#94a3b8;'>"
            f"{'أدخل بيانات الدخول للوصول إلى لوحة الإدارة' if is_ar else 'Enter your credentials to access the admin dashboard'}"
            f"</p>",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        username = st.text_input(
            "اسم المستخدم" if is_ar else "Username",
            placeholder="admin",
            key="login_username",
        )
        password = st.text_input(
            "كلمة المرور" if is_ar else "Password",
            type="password",
            placeholder="••••••••",
            key="login_password",
        )
        st.markdown("<br>", unsafe_allow_html=True)
        login_btn = st.button(
            "🔑 تسجيل الدخول" if is_ar else "🔑 Login",
            use_container_width=True,
            type="primary",
        )

        if login_btn:
            if not username or not password:
                st.error(
                    "يرجى إدخال اسم المستخدم وكلمة المرور" if is_ar
                    else "Please enter both username and password."
                )
            elif _verify_admin(username, password):
                st.session_state["admin_logged_in"] = True
                st.session_state["nav_target"]      = "📋 CRM Dashboard"
                st.success("✅ تم تسجيل الدخول بنجاح!" if is_ar else "✅ Logged in successfully!")
                st.rerun()
            else:
                st.error(
                    "❌ بيانات الدخول غير صحيحة" if is_ar
                    else "❌ Invalid username or password."
                )

        st.markdown("<br>", unsafe_allow_html=True)
        st.caption(
            "💬 Chat متاح للجميع بدون تسجيل دخول" if is_ar
            else "💬 Chat is available to everyone without login."
        )


# ──────────────────────────────────────────────────────────
# ROUTER
# ──────────────────────────────────────────────────────────
page = render_sidebar()

if page == "💬 Chat":
    page_chat()
elif page == "🔑 Login":
    page_login()
elif page == "📋 CRM Dashboard":
    if _require_admin():
        page_crm()
elif page == "📊 Analytics":
    if _require_admin():
        page_analytics()
elif page == "💰 Cost Monitor":
    if _require_admin():
        page_cost_monitor()
elif page == "🔍 Trace Viewer":
    if _require_admin():
        page_trace_viewer()