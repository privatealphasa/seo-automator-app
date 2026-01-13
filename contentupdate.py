# contentupdate.py
import os
import streamlit as st
import requests
from openai import OpenAI
from collections import Counter
import re
from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_GL = os.getenv("GL", "us")
DEFAULT_HL = os.getenv("HL", "en")

# --- Helper functions ---
def fetch_serp_snippets(keyword, hl="en", gl="us"):
    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google",
        "q": keyword,
        "hl": hl,
        "gl": gl,
        "num": 10,
        "api_key": os.getenv("SERPAPI_KEY")
    }

    response = requests.get(url, params=params, timeout=30)
    data = response.json()

    snippets = [
        r.get("snippet", "")
        for r in data.get("organic_results", [])
        if r.get("snippet")
    ]

    return snippets

def extract_keywords(text):
    stopwords = set([
        "the","a","an","and","or","in","on","for","to","of","with","as","by","at",
        "from","is","it","this","that","these","those","be","are","was","were"
    ])
    words = re.findall(r'\b\w+\b', text.lower())
    keywords = [w for w in words if w not in stopwords and len(w) > 3]
    return Counter(words).most_common(20)

def highlight_missing_keywords(original, serp_snippets):
    serp_text = " ".join(serp_snippets)
    serp_keywords = set([w for w,_ in extract_keywords(serp_text)])
    original_keywords = set([w for w,_ in extract_keywords(original)])
    missing_keywords = serp_keywords - original_keywords
    return missing_keywords

# --- Render function for launcher.py ---
def render(OPENAI_API_KEY, SERPAPI_KEY, default_gl, default_hl, default_model):
    openai = OpenAI(api_key=OPENAI_API_KEY)

    # --- Minimal Content Update settings at top ---
    
    col1, col2 = st.columns([3,3])
    
    with col1:
        target_keyword = st.text_input("Target Keyword")
    with col2:
        mode = st.radio(
            "Mode",
            ["Full Rewrite","Update Only","SERP Overview Only"],
            horizontal=True
        )
    
    st.markdown("---")
    st.markdown("**Paste your page content below:**")
    
    # --- Main content area: Original + Updated ---
    col_orig, col_upd = st.columns(2)
    
    with col_orig:
        original_content = st.text_area("Original Content", height=400)
    with col_upd:
        updated_content_placeholder = st.empty()
    
    # --- Run AI analysis ---
    if st.button("Audit & Update Content") and original_content.strip() and target_keyword.strip():
        st.info("Fetching top SERP results...")
        serp_snippets = fetch_serp_snippets(target_keyword, DEFAULT_HL, DEFAULT_GL)
        
        missing_keywords = highlight_missing_keywords(original_content, serp_snippets)
        
        # --- Audit ---
        audit_prompt = f"""
        You are an SEO content expert.
        Do NOT include any introductory phrases like "Certainly! Here’s a fully updated article..."
        Do NOT add any extra commentary.
        Original content:
        {original_content}
        
        Target Keyword: {target_keyword}
        
        Top SERP snippets:
        {' '.join(serp_snippets)}
        
        1. Identify missing keywords and NLP terms compared to the top SERP pages.
        2. Suggest improvements to increase SEO relevance.
        3. Provide a list of recommended keywords and NLP terms to add.
        """
        st.info("Generating content audit...")
        audit_response = openai.chat.completions.create(
            model=default_model,
            messages=[{"role":"user","content":audit_prompt}]
        )
        audit_text = audit_response.choices[0].message.content
        
        # --- Generate updated content ---
        if mode != "SERP Overview Only":
            if mode == "Full Rewrite":
                update_prompt = f"""
                You are an SEO content expert.
                Original content:
                {original_content}
                
                Target Keyword: {target_keyword}
                
                Top SERP snippets:
                {' '.join(serp_snippets)}
                
                Based on the audit, generate a fully updated version optimized for the target keyword and include NLP terms naturally.
                """
            elif mode == "Update Only":
                update_prompt = f"""
                You are an SEO content expert.
                Original content:
                {original_content}
                
                Target Keyword: {target_keyword}
                
                Top SERP snippets:
                {' '.join(serp_snippets)}
                
                Update content by inserting missing keywords and NLP terms naturally without rewriting the whole content. Preserve structure and style.
                """
            st.info(f"Generating updated content ({mode})...")
            update_response = openai.chat.completions.create(
                model=default_model,
                messages=[{"role":"user","content":update_prompt}]
            )
            updated_content_text = update_response.choices[0].message.content
            
            with col_upd:
                updated_content_placeholder.text_area("AI Suggested Updated Content", updated_content_text, height=400)
                st.download_button(
                    label="Download Updated Content",
                    data=updated_content_text,
                    file_name=f"{target_keyword.replace(' ','_')}_updated_content.txt",
                    mime="text/plain"
                )
        else:
            with col_upd:
                updated_content_placeholder.text_area("AI Suggested Updated Content", "Mode: SERP Overview Only (no content generated)", height=400)
        
        # --- SERP snippets + audit below ---
        st.subheader("Top SERP Snippets")
        for i, snippet in enumerate(serp_snippets,1):
            st.markdown(f"{i}. {snippet}")
        
        st.subheader("Content Audit & Recommended Keywords")
        st.text_area("Audit Results", audit_text, height=300)
        
        st.markdown("**Missing Keywords from Your Content (compared to top SERPs):**")
        st.write(", ".join(missing_keywords))
