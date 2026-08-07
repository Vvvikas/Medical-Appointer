#=============================================================
# MEDICAL APPOINTMENT & DOCTOR FINDER ASSISTANT
# Streamlit + LangChain (Gemini) + Tavily web search
#
# Features:
#   1. Find nearby doctors (by locality/pincode, not just city)
#   2. Compare hospitals (by locality/pincode, not just city)
#   3. In-app "Request Appointment" flow (WhatsApp / Email — no external booking site needed)
#   4. Booking guidance
#   5. Medicine reminder chat
#   6. My Appointment Requests (session history of what you've sent)
#=============================================================

import hashlib
import json
import urllib.parse
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import SystemMessage, HumanMessage
from tavily import TavilyClient

st.set_page_config(page_title="Medical Appointment & Doctor Finder Assistant", layout="wide")

st.title("🩺 Medical Appointment & Doctor Finder Assistant")
st.caption("Find local doctors, compare hospitals, request appointments in-app, and manage medicine reminders.")

st.warning(
    "⚠️ This tool provides general information only and is **not** a substitute for "
    "professional medical advice, diagnosis, or treatment. Appointment requests generated here are "
    "**not confirmed bookings** — they're pre-filled messages you send yourself; always verify final "
    "details directly with the clinic/hospital, and consult a licensed doctor or pharmacist for medical "
    "decisions, dosages, or emergencies."
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
        model='gemini-3.5-flash-lite',
        temperature=0.4,
    )

model = get_model(GOOGLE_API_KEY)

QUOTA_HINT = (
    "⚠️ The API key hit a rate limit or quota error. This usually means too many requests "
    "were sent in a short window, or the free-tier daily limit was reached. Wait a bit and "
    "try again, or use a key with a higher quota."
)


def _is_quota_error(e: Exception) -> bool:
    msg = str(e).lower()
    return any(term in msg for term in ["quota", "rate limit", "429", "resource_exhausted"])


# ---------------- SEARCH (1 call, cached) ----------------
@st.cache_data(show_spinner=False, ttl=3600)
def cached_search(tavily_key_hash: str, query: str, max_results: int = 6):
    client = TavilyClient(api_key=st.session_state["_tavily_key"])
    return client.search(query, max_results=max_results)


def search_once(query: str, max_results: int = 6):
    """Exactly ONE Tavily call per unique query (cached across reruns)."""
    st.session_state["_tavily_key"] = TAVILY_API_KEY
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


def search_and_format(query: str, format_instructions: str, max_results: int = 6) -> str:
    """1 Tavily search + 1 Gemini formatting call -> HTML string. Used for tab3 only now."""
    results = search_once(query, max_results=max_results)
    if isinstance(results, dict) and results.get("error"):
        return results["error"]
    system_text = (
        "You turn raw web search results into clean, honest HTML output. "
        "Never invent facts not present in the results; say 'Not available' instead of guessing."
    )
    user_text = f"{format_instructions}\n\nSearch results (JSON):\n{results}"
    return format_once(system_text, user_text)


def _safe_json_parse(raw: str):
    """Best-effort JSON array parsing: strip code fences, then try to find the [...] span."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            return None
    return None


def search_and_format_json(query: str, format_instructions: str, max_results: int = 6) -> dict:
    """1 Tavily search + 1 Gemini call -> structured JSON array (needed so each result can get
    its own 'Request Appointment' button, rather than one opaque HTML blob)."""
    results = search_once(query, max_results=max_results)
    if isinstance(results, dict) and results.get("error"):
        return {"error": results["error"]}
    system_text = (
        "You convert raw web search results into STRICT JSON only. "
        "Output must be a single valid JSON array and nothing else — no markdown code fences, "
        "no prose before or after. Never invent facts not present in the results; use null for "
        "any field you cannot find. Never fabricate phone numbers, emails, or links."
    )
    user_text = f"{format_instructions}\n\nSearch results (JSON):\n{results}"
    raw = format_once(system_text, user_text)
    if raw == QUOTA_HINT or raw.strip().startswith("⚠️"):
        return {"error": raw}
    parsed = _safe_json_parse(raw)
    if parsed is None or not isinstance(parsed, list):
        return {"error": None, "raw_text": raw}
    return {"data": parsed}


# ---------------- APPOINTMENT REQUEST HELPERS ----------------
def _phone_to_wa_number(phone):
    if not phone:
        return ""
    digits = "".join(ch for ch in str(phone) if ch.isdigit())
    if not digits:
        return ""
    if len(digits) == 10:  # bare Indian mobile number, add country code
        digits = "91" + digits
    return digits


def _build_message(patient_name, patient_phone, pref_date, pref_time, reason, target_name, target_address):
    lines = [
        "Hello, I would like to request an appointment.",
        "",
        f"Patient Name: {patient_name or 'Not provided'}",
        f"Patient Phone: {patient_phone or 'Not provided'}",
        f"Preferred Date: {pref_date}",
        f"Preferred Time: {pref_time}",
        f"Reason for Visit: {reason.strip() if reason else 'Not specified'}",
        "",
        f"Appointment requested at: {target_name}",
    ]
    if target_address:
        lines.append(f"Address: {target_address}")
    lines += ["", "(Sent via Medical Appointment & Doctor Finder Assistant)"]
    return "\n".join(lines)


def render_booking_form(target: dict, target_type: str):
    """Shared in-app appointment request form. Generates a pre-filled WhatsApp / email message
    so the whole flow stays inside the app — there is no real-time booking API this app can call
    for arbitrary clinics, so this is the closest genuine 'book here' experience possible."""
    target_name = target.get("name") or "the clinic/hospital"
    target_address = target.get("address") or target.get("location")
    target_phone = target.get("phone")

    st.divider()
    st.markdown(f"### 📝 Request an appointment — **{target_name}**")
    if not target_phone:
        st.caption("No verified phone number was found for this listing — you'll pick the recipient "
                    "yourself when WhatsApp/Email opens.")

    with st.form("booking_form", clear_on_submit=False):
        c1, c2 = st.columns(2)
        with c1:
            patient_name = st.text_input("Your Name*")
            patient_phone = st.text_input("Your Phone Number*")
        with c2:
            pref_date = st.date_input("Preferred Date")
            pref_time = st.time_input("Preferred Time")
        reason = st.text_area("Reason for visit (optional)")
        submitted = st.form_submit_button("Generate Appointment Request")

    if submitted:
        if not patient_name.strip() or not patient_phone.strip():
            st.warning("Please enter your name and phone number.")
        else:
            message = _build_message(
                patient_name, patient_phone, pref_date, pref_time, reason, target_name, target_address
            )
            wa_number = _phone_to_wa_number(target_phone)
            wa_url = (
                f"https://wa.me/{wa_number}?text={urllib.parse.quote(message)}"
                if wa_number else f"https://wa.me/?text={urllib.parse.quote(message)}"
            )
            mail_url = (
                f"mailto:?subject={urllib.parse.quote('Appointment Request - ' + target_name)}"
                f"&body={urllib.parse.quote(message)}"
            )

            b1, b2 = st.columns(2)
            with b1:
                st.link_button("📲 Send via WhatsApp", wa_url, use_container_width=True)
            with b2:
                st.link_button("✉️ Send via Email", mail_url, use_container_width=True)
            st.text_area("Message preview (copy manually if you prefer)", message, height=170)

            st.session_state.setdefault("appointment_requests", []).append({
                "target": target_name,
                "type": target_type,
                "address": target_address,
                "patient_name": patient_name,
                "patient_phone": patient_phone,
                "date": str(pref_date),
                "time": str(pref_time),
                "reason": reason,
            })
            st.success("Request generated below — tap WhatsApp or Email to send it now.")

    if st.button("✖ Cancel / choose a different one", key=f"cancel_{target_type}"):
        st.session_state["booking_target"] = None
        st.session_state["booking_target_type"] = None
        st.rerun()


def render_doctor_card(d: dict, idx: int):
    with st.container(border=True):
        st.markdown(f"**{d.get('name') or 'Unnamed listing'}**")
        st.caption(d.get("specialty") or "—")
        if d.get("address"):
            st.write(f"📍 {d['address']}")
        if d.get("fee"):
            st.write(f"💰 Approx. fee: {d['fee']}")
        if d.get("phone"):
            st.write(f"📞 {d['phone']}")
        if d.get("link"):
            st.markdown(f"[🔗 More info]({d['link']})")
        if st.button("📅 Request Appointment", key=f"book_doc_{idx}", use_container_width=True):
            st.session_state["booking_target"] = d
            st.session_state["booking_target_type"] = "doctor"


def render_hospital_card(h: dict, idx: int):
    with st.container(border=True):
        st.markdown(f"**{h.get('name') or 'Unnamed listing'}**")
        if h.get("location"):
            st.write(f"📍 {h['location']}")
        if h.get("specialties"):
            st.write(f"🩺 {h['specialties']}")
        if h.get("rating"):
            st.write(f"⭐ {h['rating']}")
        if h.get("emergency"):
            st.write(f"🚑 Emergency services: {h['emergency']}")
        if h.get("facilities"):
            st.write(f"🏥 {h['facilities']}")
        if h.get("website"):
            st.markdown(f"[🔗 Website]({h['website']})")
        if st.button("📅 Request Appointment", key=f"book_hosp_{idx}", use_container_width=True):
            st.session_state["booking_target"] = {
                "name": h.get("name"), "address": h.get("location"), "phone": h.get("phone")
            }
            st.session_state["booking_target_type"] = "hospital"


# ---------------- STATIC OPTIONS ----------------
CITIES = [
    "Delhi", "Noida", "Gurgaon/Gurugram", "Kanpur", "Lucknow",
    "Bangalore", "Pune", "Mumbai", "Chennai", "Hyderabad", "Other (type below)",
]

SPECIALTIES = [
    "General Physician", "Cardiologist", "Dermatologist", "Orthopedic",
    "Pediatrician", "Gynecologist", "Dentist", "ENT Specialist",
    "Neurologist", "Psychiatrist", "Ophthalmologist",
]

if "booking_target" not in st.session_state:
    st.session_state["booking_target"] = None
    st.session_state["booking_target_type"] = None

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["🔍 Find Doctors", "🏥 Compare Hospitals", "📅 Booking Guidance",
     "💊 Medicine Reminder Chat", "📋 My Requests"]
)

# ---------------- TAB 1: FIND DOCTORS ----------------
with tab1:
    st.subheader("Find Nearby Doctors")
    col1, col2 = st.columns(2)
    with col1:
        city = st.selectbox("City", CITIES, key="doc_city")
        if city == "Other (type below)":
            city = st.text_input("Enter your city", key="doc_city_other")
    with col2:
        specialty = st.selectbox("Specialty", SPECIALTIES, key="doc_specialty")
    locality = st.text_input(
        "Area / Locality / Pincode (for more local results)",
        placeholder="e.g. Sector 62, HSR Layout, Andheri West, 122001",
        key="doc_locality",
    )

    if st.button("Search Doctors"):
        with st.spinner("Searching for doctors near you..."):
            location_str = f"{locality}, {city}" if locality.strip() else city
            query = f"{specialty} doctors clinics near {location_str} India"
            instructions = f"""
            These are search results for {specialty} doctors/clinics near {location_str}, India.
            Return a JSON array (max 8 items) of objects with EXACTLY these keys:
            "name", "specialty", "address", "fee", "phone", "link".
            - Prefer listings that are geographically closest to "{location_str}" over ones merely
              in the same broad city.
            - "specialty" should default to "{specialty}" if not explicitly stated.
            - Use null for any field not found in the results. Never fabricate phone numbers.
            Return ONLY the JSON array, nothing else.
            """
            st.session_state["doctor_results"] = search_and_format_json(query, instructions)
            st.session_state["booking_target"] = None

    result = st.session_state.get("doctor_results")
    if result:
        if result.get("error"):
            st.error(result["error"])
        elif "raw_text" in result:
            st.warning("Couldn't fully structure these results — showing raw output instead "
                        "(appointment-request buttons aren't available for this list).")
            st.markdown(result["raw_text"])
        else:
            doctors = result.get("data") or []
            if not doctors:
                st.info("No results found for that area — try a nearby locality or a broader city search.")
            else:
                cols = st.columns(2)
                for i, d in enumerate(doctors):
                    with cols[i % 2]:
                        render_doctor_card(d, i)

    if st.session_state.get("booking_target_type") == "doctor" and st.session_state.get("booking_target"):
        render_booking_form(st.session_state["booking_target"], "doctor")

# ---------------- TAB 2: COMPARE HOSPITALS ----------------
with tab2:
    st.subheader("Compare Hospitals")
    col1, col2 = st.columns(2)
    with col1:
        city2 = st.selectbox("City", CITIES, key="hosp_city")
        if city2 == "Other (type below)":
            city2 = st.text_input("Enter your city", key="hosp_city_other")
    with col2:
        locality2 = st.text_input(
            "Area / Locality / Pincode (for more local results)",
            placeholder="e.g. Whitefield, Dwarka, 400053",
            key="hosp_locality",
        )
    hospital_names = st.text_input(
        "Enter 2–4 hospital names to compare (comma separated), or leave blank for top hospitals nearby"
    )

    if st.button("Compare Hospitals"):
        with st.spinner("Gathering hospital details..."):
            location_str2 = f"{locality2}, {city2}" if locality2.strip() else city2
            if hospital_names.strip():
                query = f"{hospital_names} hospital India ratings specialties"
                target_desc = f"the following hospitals: {hospital_names}"
            else:
                query = f"top rated hospitals near {location_str2} India"
                target_desc = f"the top rated hospitals near {location_str2}, India"
            instructions = f"""
            These are search results about {target_desc}.
            Return a JSON array (max 8 items) of objects with EXACTLY these keys:
            "name", "location", "specialties", "rating", "emergency", "facilities", "website", "phone".
            - "emergency" should be "Yes", "No", or null.
            - Use null for any field not found. Never fabricate phone numbers, emails, or links.
            Return ONLY the JSON array, nothing else.
            """
            st.session_state["hospital_results"] = search_and_format_json(query, instructions)
            st.session_state["booking_target"] = None

    result2 = st.session_state.get("hospital_results")
    if result2:
        if result2.get("error"):
            st.error(result2["error"])
        elif "raw_text" in result2:
            st.warning("Couldn't fully structure these results — showing raw output instead "
                        "(appointment-request buttons aren't available for this list).")
            st.markdown(result2["raw_text"])
        else:
            hospitals = result2.get("data") or []
            if not hospitals:
                st.info("No results found for that area — try a nearby locality or a broader city search.")
            else:
                cols = st.columns(2)
                for i, h in enumerate(hospitals):
                    with cols[i % 2]:
                        render_hospital_card(h, i)

    if st.session_state.get("booking_target_type") == "hospital" and st.session_state.get("booking_target"):
        render_booking_form(st.session_state["booking_target"], "hospital")

# ---------------- TAB 3: BOOKING GUIDANCE ----------------
with tab3:
    st.subheader("Appointment Booking Guidance")
    doc_or_hospital = st.text_input("Doctor or Hospital name (optional)")
    reason3 = st.text_area("Briefly describe the reason for your visit (optional)")
    insurance = st.radio("Do you have health insurance?", ["Yes", "No", "Not sure"], horizontal=True)

    if st.button("Get Booking Guidance"):
        with st.spinner("Preparing guidance..."):
            system_text = (
                "You give general, practical, administrative guidance about booking medical "
                "appointments in India. You never give medical diagnoses, treatment advice, or "
                "medication advice - only logistics."
            )
            user_text = f"""
            Give step-by-step guidance (as clean HTML, no markdown fences) for booking
            a medical appointment {"at/with " + doc_or_hospital if doc_or_hospital else "with a suitable doctor"}
            in India.
            Visit reason (if any): {reason3 if reason3 else "not specified"}.
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
        reply = format_once(system_text, user_text)
        st.session_state["med_chat"].append(("assistant", reply))
        with st.chat_message("assistant"):
            st.markdown(reply)

    if st.session_state["med_chat"] and st.button("Clear Chat"):
        st.session_state["med_chat"] = []
        st.rerun()

# ---------------- TAB 5: MY APPOINTMENT REQUESTS ----------------
with tab5:
    st.subheader("My Appointment Requests (this session)")
    requests = st.session_state.get("appointment_requests", [])
    if not requests:
        st.info("No appointment requests generated yet. Go to 'Find Doctors' or 'Compare Hospitals', "
                 "pick a result, and tap 'Request Appointment'.")
    else:
        for i, r in enumerate(reversed(requests), start=1):
            with st.container(border=True):
                st.markdown(f"**{i}. {r['target']}** ({r['type']})")
                if r.get("address"):
                    st.caption(r["address"])
                st.write(f"👤 {r['patient_name']} · 📞 {r['patient_phone']}")
                st.write(f"🗓️ {r['date']} at {r['time']}")
                if r.get("reason"):
                    st.write(f"Reason: {r['reason']}")
        if st.button("Clear request history"):
            st.session_state["appointment_requests"] = []
            st.rerun()
