#=============================================================
# MEDICAL APPOINTMENT & DOCTOR FINDER ASSISTANT
# Streamlit + LangChain agent (Gemini) + Tavily web search
#
# Features:
#   1. Find nearby doctors
#   2. Compare hospitals
#   3. Booking guidance
#   4. Medicine reminder chat
#=============================================================

import hashlib
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import SystemMessage, HumanMessage
from tavily import TavilyClient

st.set_page_config(page_title="Medical Appointment & Doctor Finder Assistant", layout="wide")

st.title("🩺 Medical Appointment & Doctor Finder Assistant")
st.caption("Find doctors, compare hospitals, get booking guidance, and manage medicine reminders.")

st.warning(
    "⚠️ This tool provides general information only and is **not** a substitute for "
    "professional medical advice, diagnosis, or treatment. Always verify details directly "
    "with the clinic/hospital, and consult a licensed doctor or pharmacist for medical decisions, "
    "dosages, or emergencies."
)

# ---------------- API KEYS ----------------
with st.sidebar:
    st.header("🔑 API Keys")
    GOOGLE_API_KEY = st.text_input("Gemini API Key", type="password")
    TAVILY_API_KEY = st.text_input("Tavily API Key", type="password")
    st.divider()

if not GOOGLE_API_KEY or not TAVILY_API_KEY:
    st.sidebar.warning("Please enter both API keys to continue.")
    st.stop()

st.sidebar.success("API keys loaded ✅")

# ---------------- MODEL ----------------
@st.cache_resource
def get_model(api_key):
    return ChatGoogleGenerativeAI(
        google_api_key=api_key,
        model = 'gemini-3.5-flash-lite',
        temperature=0.4,
    )

model = get_model(GOOGLE_API_KEY)

# A friendly, reusable message for quota/rate-limit errors so users understand what happened
QUOTA_HINT = (
    "⚠️ The API key hit a rate limit or quota error. This usually means too many requests "
    "were sent in a short window, or the free-tier daily limit was reached. Wait a bit and "
    "try again, or use a key with a higher quota."
)


def _is_quota_error(e: Exception) -> bool:
    msg = str(e).lower()
    return any(term in msg for term in ["quota", "rate limit", "429", "resource_exhausted"])


# ---------------- SEARCH (1 call, cached) ----------------
# Cached by (hash of tavily key, query, max_results) so re-running the script (Streamlit reruns
# on every widget interaction) or re-clicking with the same inputs does NOT fire a new API call.
@st.cache_data(show_spinner=False, ttl=3600)
def cached_search(tavily_key_hash: str, query: str, max_results: int = 5):
    client = TavilyClient(api_key=st.session_state["_tavily_key"])
    return client.search(query, max_results=max_results)


def search_once(query: str, max_results: int = 5):
    """Exactly ONE Tavily call per unique query (cached across reruns)."""
    st.session_state["_tavily_key"] = TAVILY_API_KEY  # stash so the cached fn can read it
    key_hash = hashlib.sha256(TAVILY_API_KEY.encode()).hexdigest()[:12]
    try:
        return cached_search(key_hash, query, max_results)
    except Exception as e:
        if _is_quota_error(e):
            return {"error": QUOTA_HINT}
        return {"error": f"⚠️ Search failed: {e}"}


# ---------------- FORMAT (1 LLM call, cached, no agent loop) ----------------
@st.cache_data(show_spinner=False, ttl=3600)
def cached_format(gemini_key_hash: str, system_text: str, user_text: str) -> str:
    messages = [SystemMessage(content=system_text), HumanMessage(content=user_text)]
    response = model.invoke(messages)
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [c.get("text", "") for c in content if isinstance(c, dict) and c.get("text")]
        if texts:
            return "\n".join(texts)
    return str(content)


def format_once(system_text: str, user_text: str) -> str:
    """Exactly ONE Gemini call — a plain completion, never an autonomous multi-step agent."""
    key_hash = hashlib.sha256(GOOGLE_API_KEY.encode()).hexdigest()[:12]
    try:
        return cached_format(key_hash, system_text, user_text)
    except Exception as e:
        if _is_quota_error(e):
            return QUOTA_HINT
        return f"⚠️ Something went wrong while contacting the assistant: {e}"


def search_and_format(query: str, format_instructions: str, max_results: int = 5) -> str:
    """The whole pipeline: 1 Tavily search + 1 Gemini formatting call. No hidden retries or loops."""
    results = search_once(query, max_results=max_results)
    if isinstance(results, dict) and results.get("error"):
        return results["error"]
    system_text = (
        "You turn raw web search results into clean, honest HTML output. "
        "Never invent facts not present in the results; say 'Not available' instead of guessing."
    )
    user_text = f"{format_instructions}\n\nSearch results (JSON):\n{results}"
    return format_once(system_text, user_text)


# ---------------- STATIC OPTIONS ----------------
CITIES = [
    "Delhi", "Noida", "Gurgaon/Gurugram", "Kanpur", "Lucknow",
    "Bangalore", "Pune", "Mumbai", "Chennai", "Hyderabad",
]

SPECIALTIES = [
    "General Physician", "Cardiologist", "Dermatologist", "Orthopedic",
    "Pediatrician", "Gynecologist", "Dentist", "ENT Specialist",
    "Neurologist", "Psychiatrist", "Ophthalmologist",
]

tab1, tab2, tab3, tab4 = st.tabs(
    ["🔍 Find Doctors", "🏥 Compare Hospitals", "📅 Booking Guidance", "💊 Medicine Reminder Chat"]
)

# ---------------- TAB 1: FIND DOCTORS ----------------
with tab1:
    st.subheader("Find Nearby Doctors")
    col1, col2 = st.columns(2)
    with col1:
        city = st.selectbox("City / Area", CITIES, key="doc_city")
    with col2:
        specialty = st.selectbox("Specialty", SPECIALTIES, key="doc_specialty")

    if st.button("Search Doctors"):
        with st.spinner("Searching for doctors..."):
            query = f"{specialty} doctors clinics in {city} India"
            instructions = f"""
            These are search results for {specialty} doctors/clinics in {city}, India.
            Return the result as clean HTML (no markdown fences, no explanations outside the HTML)
            styled as a responsive card grid. Each card should show: doctor/clinic name, specialty,
            area/address, approximate consultation fee if found, and a link if available.
            Never fabricate phone numbers.
            """
            st.session_state["doctor_results"] = search_and_format(query, instructions)

    if "doctor_results" in st.session_state:
        st.components.v1.html(st.session_state["doctor_results"], height=600, scrolling=True)

# ---------------- TAB 2: COMPARE HOSPITALS ----------------
with tab2:
    st.subheader("Compare Hospitals")
    city2 = st.selectbox("City / Area", CITIES, key="hosp_city")
    hospital_names = st.text_input(
        "Enter 2–4 hospital names to compare (comma separated), or leave blank for top hospitals in the city"
    )

    if st.button("Compare Hospitals"):
        with st.spinner("Gathering hospital details..."):
            if hospital_names.strip():
                query = f"{hospital_names} hospital India ratings specialties"
                target_desc = f"the following hospitals: {hospital_names}"
            else:
                query = f"top rated hospitals in {city2} India"
                target_desc = f"the top rated hospitals in {city2}, India"
            instructions = f"""
            These are search results about {target_desc}.
            Present a comparison as a single clean HTML table (no markdown fences) with columns:
            Hospital Name, Location, Key Specialties, Approx. Rating, Emergency Services (Y/N),
            Notable Facilities, Website/Link.
            If information is unavailable for a field, write "Not available" rather than guessing.
            """
            st.session_state["hospital_results"] = search_and_format(query, instructions)

    if "hospital_results" in st.session_state:
        st.components.v1.html(st.session_state["hospital_results"], height=500, scrolling=True)

# ---------------- TAB 3: BOOKING GUIDANCE ----------------
with tab3:
    st.subheader("Appointment Booking Guidance")
    doc_or_hospital = st.text_input("Doctor or Hospital name (optional)")
    reason = st.text_area("Briefly describe the reason for your visit (optional)")
    insurance = st.radio("Do you have health insurance?", ["Yes", "No", "Not sure"], horizontal=True)

    if st.button("Get Booking Guidance"):
        with st.spinner("Preparing guidance..."):
            # General logistics advice doesn't need a web search - this is a single LLM call,
            # which keeps quota usage minimal (0 Tavily calls, 1 Gemini call).
            system_text = (
                "You give general, practical, administrative guidance about booking medical "
                "appointments in India. You never give medical diagnoses, treatment advice, or "
                "medication advice - only logistics."
            )
            user_text = f"""
            Give step-by-step guidance (as clean HTML, no markdown fences) for booking
            a medical appointment {"at/with " + doc_or_hospital if doc_or_hospital else "with a suitable doctor"}
            in India.
            Visit reason (if any): {reason if reason else "not specified"}.
            Insurance status: {insurance}.
            Include: how to book (phone/app/walk-in), documents to carry (ID, prior reports, insurance card),
            what to expect at check-in, typical wait times, and questions to ask the receptionist.
            End with a short note to confirm final details directly with the clinic/hospital.
            """
            st.session_state["booking_guidance"] = format_once(system_text, user_text)

    if "booking_guidance" in st.session_state:
        st.components.v1.html(st.session_state["booking_guidance"], height=550, scrolling=True)

# ---------------- TAB 4: MEDICINE REMINDER CHAT ----------------
with tab4:
    st.subheader("Medicine Reminder Chat")
    st.caption(
        "Tell me your medicines and when you take them, and I'll help you organize a reminder schedule. "
        "I can't advise on dosages, interactions, or whether a medicine is right for you — please check "
        "with your doctor or pharmacist for that."
    )

    if "med_chat" not in st.session_state:
        st.session_state["med_chat"] = []

    for role, text in st.session_state["med_chat"]:
        with st.chat_message(role):
            st.markdown(text)

    user_msg = st.chat_input("e.g. I take Metformin at 8am and 8pm, and Vitamin D once a week on Sunday")

    if user_msg:
        st.session_state["med_chat"].append(("user", user_msg))
        with st.chat_message("user"):
            st.markdown(user_msg)

        history_text = "\n".join(f"{r}: {t}" for r, t in st.session_state["med_chat"][-10:])
        system_text = (
            "You are a medicine reminder organizer assistant, not a doctor or pharmacist. "
            "You help organize reminder schedules only. You never suggest dosages, dosage "
            "changes, or drug combinations, and never judge whether a medicine is appropriate - "
            "redirect those questions to a doctor or pharmacist instead of answering them."
        )
        user_text = f"""
        Conversation so far:
        {history_text}

        Based on what the user has told you, help them organize a clear reminder schedule
        (e.g. a simple table of medicine name, time, frequency). Ask clarifying questions if
        timings are unclear or missing. Keep the reply conversational (plain markdown, not HTML)
        and concise.
        """
        # A single, direct completion call - no autonomous agent loop, no hidden retries.
        reply = format_once(system_text, user_text)
        st.session_state["med_chat"].append(("assistant", reply))
        with st.chat_message("assistant"):
            st.markdown(reply)

    if st.session_state["med_chat"] and st.button("Clear Chat"):
        st.session_state["med_chat"] = []
        st.rerun()
