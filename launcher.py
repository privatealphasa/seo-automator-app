# launcher.py
import os
import streamlit as st
from shared import load_env_keys

# -------------------------------
# 🔐 APP PASSWORD LOGIN
# -------------------------------

APP_PASSWORD = os.getenv("APP_PASSWORD")

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.title("🔐 SEO Automator Login")

    password = st.text_input("Enter App Password", type="password")

    if password:
        if password == APP_PASSWORD:
            st.session_state["authenticated"] = True
            st.success("Access granted. Loading app...")
            st.rerun()
        else:
            st.error("Incorrect password")

    st.stop()  # Prevent app from loading until authenticated


# -------------------------------
# 🚀 MAIN APP CONFIG
# -------------------------------

st.set_page_config(
    page_title="SEO Automator (SerpAPI + OpenAI)",
    page_icon="🔎",
    layout="wide"
)

# ---- ENV + Sidebar ----
OPENAI_API_KEY, SERPAPI_KEY, AHREFS_API_TOKEN = load_env_keys()

st.sidebar.header("⚙️ Settings")

OPENAI_MODEL = st.sidebar.selectbox(
    "OpenAI Model",
    ["gpt-4o-mini", "gpt-4o", "gpt-4.1", "gpt-4.1-mini"],
    index=0
)

gl = st.sidebar.selectbox(
    "Google geolocation (gl)",
    ["us", "uk", "ca", "au", "za", "de", "fr", "se", "nl"],
    index=0
)

hl = st.sidebar.selectbox(
    "Google language (hl)",
    ["en", "fr", "de", "sv", "nl", "es", "it", "pt"],
    index=0
)

# Expose model globally
os.environ["OPENAI_MODEL"] = OPENAI_MODEL

# -------------------------------
# 🧠 APP UI
# -------------------------------

st.title("🔎 SEO Automator — SerpAPI + OpenAI")
st.caption("Configure model, gl, and hl in the left sidebar.")

# ---- Tabs ----
tab_internal, tab_rank, tab_audit, tab_brief, tab_update, tab_sc, tab_sf = st.tabs(
    [
        "Internal Links",
        "Rank Track",
        "Content Audit",
        "Content Brief",
        "Content Update",
        "Search Console",
        "Screaming Frog",
    ]
)

with tab_internal:
    import internallinks
    internallinks.render(gl, hl, OPENAI_API_KEY, SERPAPI_KEY)

with tab_rank:
    import ranking
    ranking.render(gl, hl, SERPAPI_KEY)

with tab_audit:
    import contentaudit
    contentaudit.render(OPENAI_API_KEY)

with tab_brief:
    import contentbrief
    contentbrief.render(OPENAI_API_KEY, SERPAPI_KEY, gl, hl, OPENAI_MODEL)

with tab_update:
    import contentupdate
    contentupdate.render(OPENAI_API_KEY, SERPAPI_KEY, gl, hl, OPENAI_MODEL)

with tab_sc:
    import searchconsole
    searchconsole.render()

with tab_sf:
    import screamingfrog
    screamingfrog.render()
