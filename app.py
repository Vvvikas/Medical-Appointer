#=============================================================
# Medical Appointment & Doctor Finder Assistant
# Features : Find nearby doctors | Compare hospitals |
#            Booking guidance | Medicine reminder chat
# Tools    : Tavily web search ONLY
#=============================================================

import json
from datetime import datetime

import streamlit as st
import pytesseract
from PIL import Image
import streamlit.components.v1 as components

from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient

st.set_page_config(page_title="Medical Appointment & Doctor Finder Assistant", layout="wide")

st.title("🩺 Medical Appointment & Doctor Finder Assistant")
st.caption("Find nearby doctors, compare hospitals, get booking guidance, and track medicine reminders.")

st.error(
    "⚠️ **Not medical advice.** This assistant helps with logistics only (finding providers, "
    "comparing facilities, booking steps). It does not diagnose, prescribe, or replace a licensed "
    "doctor. In an emergency, call your local emergency number immediately (India: 112 / 108, US: 911, UK: 999)."
)

st.info(
    "ℹ️ All search results below come from live web search (Tavily), not a verified medical "
    "directory. Details like phone numbers, addresses, and ratings are only as accurate as what "
    "the source pages say — always confirm directly with the clinic/hospital before relying on them."
)

# ---------------------------- API KEYS ----------------------------
st.sidebar.header("🔑 API Keys")
GOOGLE = st.sidebar.text_input("Gemini API Key", type="password")
GROQ = st.sidebar.text_input("Groq API Key", type="password")
TAVILY = st.sidebar.text_input("Tavily API Key", type="password")

if not TAVILY or not (GOOGLE or GROQ):
    st.sidebar.warning("Enter a Tavily key and at least one LLM key (Gemini or Groq) to continue.")
    st.stop()

st.sidebar.success("API keys loaded ✅")

# ---------------------------- MODEL ----------------------------
@st.cache_resource(show_spinner=False)
def load_model(google_key, groq_key):
    # NOTE: model names change over time — verify the current one in your
    # provider's docs (Google AI Studio / Groq console) before deploying.
    if google_key:
        return ChatGoogleGenerativeAI(google_api_key=google_key, model="gemini-2.5-flash", temperature=0.3)
    return ChatGroq(groq_api_key=groq_key, model="llama-3.3-70b-versatile", temperature=0.3)

model = load_model(GOOGLE, GROQ)

# ---------------------------- TOOLS ----------------------------
def search_medical_web(query: str):
    """Search the web for doctors, hospitals, reviews, health news, or booking pages using Tavily."""
    client = TavilyClient(api_key=TAVILY)
    return client.search(query, max_results=8, include_answer=False)

def tavily_raw_search(query: str, max_results: int = 8):
    """Direct Tavily call used to pre-fetch data before asking the LLM to format it."""
    client = TavilyClient(api_key=TAVILY)
    return client.search(query, max_results=max_results, include_answer=False)

agent = create_agent(
    model=model,
    tools=[search_medical_web],
)

SYSTEM_INSTRUCTIONS = """You are a careful, empathetic medical-logistics assistant.
Rules you must always follow:
- Never diagnose a condition, never prescribe medicine, never give specific dosage advice.
- Only give general, publicly-known information about medicines when asked, and always add a
  reminder to confirm with a pharmacist or doctor.
- When comparing hospitals or doctors, present facts neutrally — never claim one is medically
  "better" at treating a condition.
- NEVER invent a phone number, address, or rating that is not present in the provided search data.
  If a detail is missing, say "Not listed in search results" instead of guessing.
- When asked for a report, output clean HTML only (no markdown fences, no text outside the HTML).
- Always end any generated report with a short disclaimer line about consulting a licensed professional.
"""

def run_agent(user_prompt: str) -> str:
    full_prompt = SYSTEM_INSTRUCTIONS + "\n\nTASK:\n" + user_prompt
    response = agent.invoke({"messages": [{"role": "user", "content": full_prompt}]})
    return response["messages"][-1].content[-1]["text"]

def render_html(html_code: str, height: int = 650):
    components.html(html_code, height=height, scrolling=True)

# ---------------------------- TABS ----------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["🔎 Find Doctors", "🏥 Compare Hospitals", "📋 Booking Guidance", "💊 Medicine Reminders"]
)

# ======================= TAB 1: FIND DOCTORS =======================
with tab1:
    st.subheader("Find nearby doctors, clinics, or hospitals")
    col1, col2 = st.columns(2)
    with col1:
        specialty = st.selectbox(
            "Specialty",
            ["General Physician", "Cardiologist", "Dermatologist", "Dentist", "Pediatrician",
             "Orthopedic", "Gynecologist", "ENT Specialist", "Psychiatrist", "Neurologist", "Ophthalmologist"],
        )
    with col2:
        city = st.text_input("City / Area", placeholder="e.g. Gurugram, Haryana")

    if st.button("Search Doctors", key="search_doctors"):
        if not city:
            st.warning("Please enter a city or area.")
        else:
            with st.spinner("Searching the web via Tavily..."):
                raw = tavily_raw_search(f"top {specialty} doctors clinics in {city} contact address reviews")
                prompt = f"""
                Here is raw web search data (JSON) about {specialty} providers in {city}:
                {json.dumps(raw.get("results", []))}

                Generate clean HTML (no markdown) showing up to 8 provider options as cards.
                For each card include: name, area/address if mentioned in the text, phone if
                mentioned, a one-line note on what the source says, and a link to the source URL.
                If a field isn't in the text, write "Not listed in search results" — do not invent it.
                Use a light, professional color scheme with clear spacing between cards.
                """
                html_out = run_agent(prompt)
            render_html(html_out)

# ======================= TAB 2: COMPARE HOSPITALS =======================
with tab2:
    st.subheader("Compare hospitals side by side")
    city2 = st.text_input("City / Area", key="compare_city", placeholder="e.g. Noida")
    hospitals_raw = st.text_input("Hospital names (comma separated, 2-4)", placeholder="Fortis, Max, Apollo")

    if st.button("Compare", key="compare_btn"):
        names = [n.strip() for n in hospitals_raw.split(",") if n.strip()]
        if not city2 or len(names) < 2:
            st.warning("Enter a city and at least 2 hospital names.")
        else:
            bundle = {}
            with st.spinner("Searching the web via Tavily for each hospital..."):
                for name in names[:4]:
                    res = tavily_raw_search(f"{name} hospital {city2} reviews rating address contact services")
                    bundle[name] = res.get("results", [])

                prompt = f"""
                Here is raw web search data (JSON), keyed by hospital name, gathered for a
                hospital comparison in {city2}:
                {json.dumps(bundle)}

                Generate a single clean HTML comparison table (no markdown) with one row per
                hospital and columns: Hospital, Address, Contact, Notable Info (services/reviews
                mentioned in the text), Source (link to the most relevant URL found). If a field
                is not present in the text for that hospital, write "Not found in search results" —
                do not invent details. Use a light, readable color scheme.
                """
                html_out = run_agent(prompt)
            render_html(html_out)
            st.caption("Details are pulled from live web search snippets, not a verified hospital database.")

# ======================= TAB 3: BOOKING GUIDANCE =======================
with tab3:
    st.subheader("Get step-by-step booking guidance")
    provider_name = st.text_input("Doctor / Hospital name")
    city3 = st.text_input("City", key="booking_city")
    reason = st.text_area("Reason for visit (general category only, e.g. 'routine checkup', 'skin rash')")
    insurance = st.selectbox("Do you have health insurance / a TPA card?", ["Not sure", "Yes", "No"])

    if st.button("Get Booking Guidance"):
        if not provider_name:
            st.warning("Please enter a doctor or hospital name.")
        else:
            with st.spinner("Preparing guidance..."):
                prompt = f"""
                Use web search (the search_medical_web tool) if useful to check how patients
                typically book with this provider, then generate an HTML guide (no markdown) for
                booking an appointment with:
                Provider: {provider_name}
                City: {city3}
                Reason for visit (general category): {reason}
                Insurance status: {insurance}

                Include: how to find their contact number/booking page, what to ask the receptionist,
                documents to bring (ID, insurance card, prior reports), whether telemedicine may be an
                option, and a simple checklist. Use headings, a checklist styled as a list, and a light
                color scheme. End with a one-line disclaimer to consult a licensed doctor.
                """
                html_out = run_agent(prompt)
            render_html(html_out)

# ======================= TAB 4: MEDICINE REMINDERS =======================
with tab4:
    st.subheader("Medicine Reminder Chat")
    st.caption(
        "⚠️ Streamlit apps don't run in the background, so this can't send push notifications like a "
        "phone app. Use it as a running log/checklist — set your phone's own alarm for the actual time."
    )

    if "reminders" not in st.session_state:
        st.session_state.reminders = []

    st.markdown("**Add a reminder by describing it in plain language:**")
    chat_input = st.text_input("e.g. 'Paracetamol 500mg, twice daily, 9am and 9pm'", key="reminder_chat")
    add_clicked = st.button("Add Reminder")

    if add_clicked and chat_input:
        with st.spinner("Parsing reminder..."):
            parse_prompt = f"""
            Extract structured medicine reminder data from this text: "{chat_input}"
            Respond with ONLY a JSON object, no markdown fences, no extra text, with keys:
            medicine (string), dosage (string), times (array of strings like "09:00"), frequency (string).
            If something is unclear, make a reasonable simple assumption.
            """
            raw = run_agent(parse_prompt)
        try:
            clean = raw.strip().strip("`")
            if clean.lower().startswith("json"):
                clean = clean[4:].strip()
            data = json.loads(clean)
            data["added_on"] = datetime.now().strftime("%Y-%m-%d %H:%M")
            st.session_state.reminders.append(data)
            st.success(f"Added reminder for {data.get('medicine', 'medicine')}")
        except Exception:
            st.error("Couldn't parse that reminder — try rephrasing, e.g. 'Metformin 500mg every morning at 8am'.")

    st.divider()
    st.markdown("**Or upload a prescription photo to extract medicine names (OCR):**")
    presc_file = st.file_uploader("Prescription image", type=["jpg", "jpeg", "png"], key="presc_upload")
    if presc_file is not None:
        try:
            img = Image.open(presc_file)
            st.image(img, caption="Uploaded prescription", width=300)
            extracted_text = pytesseract.image_to_string(img)
            st.text_area(
                "Extracted text (OCR) — review before trusting it, OCR can misread handwriting",
                value=extracted_text, height=150,
            )
        except Exception as e:
            st.error(f"Could not process image (is tesseract-ocr installed on the server? see packages.txt): {e}")

    st.divider()
    st.markdown("**Your reminders:**")
    if not st.session_state.reminders:
        st.info("No reminders added yet.")
    else:
        for i, r in enumerate(st.session_state.reminders):
            with st.container(border=True):
                c1, c2 = st.columns([4, 1])
                with c1:
                    st.write(f"💊 **{r.get('medicine', '?')}** — {r.get('dosage', '?')}")
                    st.caption(f"Times: {', '.join(r.get('times', []))} | Frequency: {r.get('frequency', '?')}")
                with c2:
                    if st.button("Remove", key=f"remove_{i}"):
                        st.session_state.reminders.pop(i)
                        st.rerun()
