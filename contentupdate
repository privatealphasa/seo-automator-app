# contentupdate_tab.py
import streamlit as st
from collections import Counter
import re

from shared import serpapi_google_search, openai_chat

def render(openai_api_key: str):
    st.title("AI Content Auditor & Updater")
    
    # --- Sidebar ---
    with st.sidebar:
        st.header("Content Audit Settings")
        keyword = st.text_input("Target Keyword")
        language = st.selectbox("Language", ["en", "fr", "de"], index=0)
        country = st.selectbox("Country", ["us", "gb", "ca", "za"], index=0)

        st.subheader("AI Model (cheaper options available)")
        model = st.selectbox(
            "Select GPT Model", 
            ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4o", "gpt-4.1"], 
            index=0
        )

        st.subheader("Mode")
        mode = st.radio(
            "Select how AI should handle content",
            ["Full Rewrite", "Update Only", "SERP Overview Only"]
        )
        st.markdown("**Paste your page content in the main area.**")
    
    # --- Main content area ---
    col1, col2 = st.columns(2)
    with col1:
        original_content = st.text_area("Original Content", height=400)
    with col2:
        updated_content_placeholder = st.empty()
    
    # --- Helper functions ---
    def extract_keywords(text):
        stopwords = set([
            "the","a","an","and","or","in","on","for","to","of","with","as","by","at",
            "from","is","it","this","that","these","those","be","are","was","were"
        ])
        words = re.findall(r'\b\w+\b', text.lower())
        keywords = [w for w in words if w not in stopwords and len(w) > 3]
        return Counter(keywords).most_common(20)
    
    def highlight_missing_keywords(original, serp_snippets):
        serp_text = " ".join(serp_snippets)
        serp_keywords = set([w for w,_ in extract_keywords(serp_text)])
        original_keywords = set([w for w,_ in extract_keywords(original)])
        missing_keywords = serp_keywords - original_keywords
        return missing_keywords
    
    def fetch_serp_snippets(keyword, language, country):
        serp_json = serpapi_google_search(keyword, num=10, gl=country, hl=language, serpapi_key=os.getenv("SERPAPI_KEY"))
        snippets = []
        for r in serp_json.get("organic_results", []):
            snippet = r.get("snippet", "")
            if snippet:
                snippets.append(snippet)
        return snippets
    
    # --- Run analysis ---
    if st.button("Audit & Update Content") and original_content.strip() and keyword.strip():
        st.info("Fetching top SERP results...")
        serp_snippets = fetch_serp_snippets(keyword, language, country)
        missing_keywords = highlight_missing_keywords(original_content, serp_snippets)
        
        # --- Audit ---
        audit_prompt = f"""
You are an SEO content expert.
Original content:
{original_content}

Target Keyword: {keyword}

Top SERP snippets:
{' '.join(serp_snippets)}

1. Identify missing keywords and NLP terms compared to the top SERP pages.
2. Suggest improvements to increase SEO relevance.
3. Provide a list of recommended keywords and NLP terms to add.
"""
        st.info("Generating content audit...")
        audit_text = openai_chat(
            api_key=openai_api_key,
            messages=[{"role": "user", "content": audit_prompt}],
            model=model
        )
        
        # --- Updated content ---
        if mode != "SERP Overview Only":
            update_prompt = ""
            if mode == "Full Rewrite":
                update_prompt = f"""
You are an SEO content expert.
Original content:
{original_content}

Target Keyword: {keyword}

Top SERP snippets:
{' '.join(serp_snippets)}

Based on the audit, generate a fully updated version of the content optimized for the target keyword and include recommended NLP terms naturally.
"""
            elif mode == "Update Only":
                update_prompt = f"""
You are an SEO content expert.
Original content:
{original_content}

Target Keyword: {keyword}

Top SERP snippets:
{' '.join(serp_snippets)}

Based on the audit, update the content by inserting missing keywords and NLP terms naturally, without rewriting the entire content.
Keep the original structure and style intact.
"""
            st.info(f"Generating updated content ({mode})...")
            updated_content_text = openai_chat(
                api_key=openai_api_key,
                messages=[{"role": "user", "content": update_prompt}],
                model=model
            )
            with col2:
                updated_content_placeholder.text_area("AI Suggested Updated Content", updated_content_text, height=400)
                st.download_button(
                    label="Download Updated Content",
                    data=updated_content_text,
                    file_name=f"{keyword.replace(' ','_')}_updated_content.txt",
                    mime="text/plain"
                )
        else:
            with col2:
                updated_content_placeholder.text_area("AI Suggested Updated Content", "Mode: SERP Overview Only (no content generated)", height=400)
        
        # --- SERP + audit display ---
        st.subheader("Top SERP Snippets")
        for i, snippet in enumerate(serp_snippets, 1):
            st.markdown(f"{i}. {snippet}")
        
        st.subheader("Content Audit & Recommended Keywords")
        st.text_area("Audit Results", audit_text, height=300)
        
        st.markdown("**Missing Keywords from Your Content (compared to top SERPs):**")
        st.write(", ".join(missing_keywords))
