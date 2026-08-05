#=============================================================
# Medical Appointment & Doctor Finder Assistant
# Features : Find nearby doctors | Compare hospitals |
#            Booking guidance | Medicine reminder chat
# Tools    : Tavily (web search) + Google Places API (location data)
#=============================================================

import json
from datetime import datetime

import requests
import pandas as pd
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

# ---------------------------- API KEYS ----------------------------
st.sidebar.header("🔑 API Keys")
GOOGLE = st.sidebar.text_input("Gemini API Key", type="password")
GROQ = st.sidebar.text_input("Groq API Key", type="password")
TAVILY = st.sidebar.text_input("Tavily API Key", type="password")
GPLACES = st.sidebar.text_input("Google Places API Key", type="password")

if not TAVILY or not (GOOGLE or GROQ):
    st.sidebar.warning("Enter a Tavily key and at least one LLM key (Gemini or Groq) to continue.")
    st.stop()

if not GPLACES:
    st.sidebar.info("Add a Google Places API key to enable doctor/hospital search & comparison.")

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
    """Search the web for hospital reviews, doctor credentials, health news, or booking pages using Tavily."""
    client = TavilyClient(api_key=TAVILY)
    return client.search(query, max_results=5)

def find_nearby_places(query: str):
    """Find doctors, clinics, or hospitals via Google Places Text Search.
    query should include the specialty/type and the location, e.g. 'cardiologist in Gurugram'."""
    if not GPLACES:
        return {"error": "Google Places API key not provided."}
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    r = requests.get(url, params={"query": query, "key": GPLACES}, timeout=15)
    return r.json()

def get_place_details(place_id: str):
    """Get rating, reviews, phone number, hours and website for a specific Google Place ID."""
    if not GPLACES:
        return {"error": "Google Places API key not provided."}
    url = "https://maps.googleapis.com/maps/api/place/details/json"
    fields = "name,rating,user_ratings_total,formatted_address,formatted_phone_number,opening_hours,website"
    r = requests.get(url, params={"place_id": place_id, "fields": fields, "key": GPLACES}, timeout=15)
    return r.json()

agent = create_agent(
    model=model,
    tools=[search_medical_web, find_nearby_places, get_place_details],
)

SYSTEM_INSTRUCTIONS = """You are a careful, empathetic medical-logistics assistant.
Rules you must always follow:
- Never diagnose a condition, never prescribe medicine, never give specific dosage advice.
- Only give general, publicly-known information about medicines when asked, and always add a
  reminder to confirm with a pharmacist or doctor.
- When comparing hospitals or doctors, present facts (rating, distance, reviews, services)
  neutrally — never claim one is medically "better" at treating a condition.
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
            with st.spinner("Searching nearby providers..."):
                results = find_nearby_places(f"{specialty} in {city}")
            places = results.get("results", []) if isinstance(results, dict) else []
            if not places:
                st.info("No results found. Check your Google Places API key or try a different search.")
            else:
                for p in places[:10]:
                    with st.container(border=True):
                        c1, c2 = st.columns([3, 1])
                        with c1:
                            st.markdown(f"**{p.get('name', 'Unknown')}**")
                            st.caption(p.get("formatted_address", ""))
                            rating = p.get("rating")
                            total = p.get("user_ratings_total")
                            if rating:
                                st.write(f"⭐ {rating} ({total} reviews)")
                            open_now = p.get("opening_hours", {}).get("open_now")
                            if open_now is not None:
                                st.write("🟢 Open now" if open_now else "🔴 Closed now")
                        with c2:
                            pid = p.get("place_id")
                            if pid:
                                st.link_button("View on Maps", f"https://www.google.com/maps/place/?q=place_id:{pid}")

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
            rows = []
            with st.spinner("Fetching hospital data..."):
                for name in names[:4]:
                    search = find_nearby_places(f"{name} hospital in {city2}")
                    results = search.get("results", []) if isinstance(search, dict) else []
                    if not results:
                        rows.append({"Hospital": name, "Rating": "N/A", "Reviews": "N/A",
                                      "Address": "Not found", "Phone": "N/A", "Website": "N/A"})
                        continue
                    top = results[0]
                    details = get_place_details(top.get("place_id", ""))
                    d = details.get("result", {}) if isinstance(details, dict) else {}
                    rows.append({
                        "Hospital": d.get("name", name),
                        "Rating": d.get("rating", "N/A"),
                        "Reviews": d.get("user_ratings_total", "N/A"),
                        "Address": d.get("formatted_address", top.get("formatted_address", "N/A")),
                        "Phone": d.get("formatted_phone_number", "N/A"),
                        "Website": d.get("website", "N/A"),
                    })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)
            st.caption("Ratings/reviews are pulled live from Google Places and reflect general patient "
                       "sentiment, not clinical quality.")

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
                Generate an HTML guide (no markdown) for booking an appointment with:
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
