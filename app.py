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

import streamlit as st
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
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

# ---------------- MODEL & TOOLS ----------------
@st.cache_resource
def get_model(api_key):
    return ChatGoogleGenerativeAI(
        google_api_key=api_key,
        model="gemini-2.0-flash",
        temperature=0.4,
    )

model = get_model(GOOGLE_API_KEY)


def search_web(query: str):
    """Search the web for doctors, hospitals, clinics, or medical facility information near a location."""
    client = TavilyClient(api_key=TAVILY_API_KEY)
    return client.search(query, max_results=8)


agent = create_agent(model=model, tools=[search_web])


def run_agent(prompt: str) -> str:
    """Send a prompt to the agent and safely extract the text reply, regardless of content shape."""
    try:
        response = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
        msg = response["messages"][-1]
        content = msg.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = [c.get("text", "") for c in content if isinstance(c, dict) and c.get("text")]
            if texts:
                return "\n".join(texts)
        return str(content)
    except Exception as e:
        return f"⚠️ Something went wrong while contacting the assistant: {e}"


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
            search_prompt = f"""
            Use the search tool to find real, currently practicing {specialty} doctors or clinics
            in {city}, India (e.g. via hospital sites, Practo, Justdial-type listings).
            Return the result as clean HTML (no markdown fences, no explanations outside the HTML)
            styled as a responsive card grid. Each card should show: doctor/clinic name, specialty,
            area/address, approximate consultation fee if found, and a link if available.
            If exact data isn't found, clearly state that fewer results were available instead of
            inventing details, and never fabricate phone numbers.
            """
            st.session_state["doctor_results"] = run_agent(search_prompt)

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
            target = (
                f"the following hospitals: {hospital_names}"
                if hospital_names.strip()
                else f"the top rated hospitals in {city2}, India"
            )
            compare_prompt = f"""
            Use the search tool to find information about {target}.
            Present a comparison as a single clean HTML table (no markdown fences) with columns:
            Hospital Name, Location, Key Specialties, Approx. Rating, Emergency Services (Y/N),
            Notable Facilities, Website/Link.
            If information is unavailable for a field, write "Not available" rather than guessing.
            """
            st.session_state["hospital_results"] = run_agent(compare_prompt)

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
            guidance_prompt = f"""
            Give general, practical step-by-step guidance (as clean HTML, no markdown fences) for booking
            a medical appointment {"at/with " + doc_or_hospital if doc_or_hospital else "with a suitable doctor"}
            in India.
            Visit reason (if any): {reason if reason else "not specified"}.
            Insurance status: {insurance}.
            Include: how to book (phone/app/walk-in), documents to carry (ID, prior reports, insurance card),
            what to expect at check-in, typical wait times, and questions to ask the receptionist.
            This must stay administrative/logistics guidance only — do not give medical diagnoses,
            treatment advice, or medication advice.
            End with a short note to confirm final details directly with the clinic/hospital.
            """
            st.session_state["booking_guidance"] = run_agent(guidance_prompt)

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
        reminder_prompt = f"""
        You are a medicine reminder organizer assistant, not a doctor or pharmacist.
        Conversation so far:
        {history_text}

        Based on what the user has told you, help them organize a clear reminder schedule
        (e.g. a simple table of medicine name, time, frequency). Ask clarifying questions if
        timings are unclear or missing. Do NOT suggest dosages, dosage changes, drug combinations,
        or whether a medicine is appropriate for them — if asked about that, gently redirect the
        user to a doctor or pharmacist instead of answering. Keep the reply conversational
        (plain markdown, not HTML) and concise.
        """
        reply = run_agent(reminder_prompt)
        st.session_state["med_chat"].append(("assistant", reply))
        with st.chat_message("assistant"):
            st.markdown(reply)

    if st.session_state["med_chat"] and st.button("Clear Chat"):
        st.session_state["med_chat"] = []
        st.rerun()
