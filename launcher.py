# launcher.py
import os
import streamlit as st
from shared import load_env_keys

st.set_page_config(page_title="SEO Automator (SerpAPI + OpenAI)", page_icon="🔎", layout="wide")

# ---- ENV + Sidebar ----
OPENAI_API_KEY, SERPAPI_KEY, AHREFS_API_TOKEN = load_env_keys()

st.sidebar.header("⚙️ Settings")
OPENAI_MODEL = st.sidebar.selectbox(
    "OpenAI Model",
    ["gpt-4o-mini", "gpt-4o", "gpt-4.1", "gpt-4.1-mini"],  
    index=0
)
gl = st.sidebar.selectbox("Google geolocation (gl)", ["us","uk","ca","au","za","de","fr","se","nl"], index=0)
hl = st.sidebar.selectbox("Google language (hl)", ["en","fr","de","sv","nl","es","it","pt"], index=0)

# Expose the model globally for tabs that use OpenAI
os.environ["OPENAI_MODEL"] = OPENAI_MODEL

st.title("🔎 SEO Automator — SerpAPI + OpenAI")
st.caption("Configure model, gl, and hl in the left sidebar.")

# ---- Tabs ----
tab_internal, tab_rank, tab_audit, tab_brief, tab_sc, tab_sf = st.tabs(
    ["Internal Links", "Rank Track", "Content Audit", "Content Brief",
     "Search Console", "Screaming Frog"]
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

with tab_sc:
    import searchconsole
    searchconsole.render()

with tab_sf:
    import screamingfrog
    screamingfrog.render()
