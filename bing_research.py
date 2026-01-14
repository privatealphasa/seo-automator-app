# bing-research.py
import re
import os
import pandas as pd
import streamlit as st
from collections import Counter
from serpapi import GoogleSearch

EXCLUDED_FILE = "excluded_domains.json"

# -----------------------------
# Utility Functions
# -----------------------------
def load_excluded_domains():
    if os.path.exists(EXCLUDED_FILE):
        return pd.read_json(EXCLUDED_FILE)
    # Default excluded domains
    return pd.DataFrame([
        {"Domain": "reddit.com", "Category": "Forum"},
        {"Domain": "youtube.com", "Category": "Social"},
        {"Domain": "microsoft.com", "Category": "Big Tech"},
        {"Domain": "bing.com", "Category": "Search Engine"},
        {"Domain": "expressvpn.com", "Category": "VPN Brand"},
        {"Domain": "nordvpn.com", "Category": "VPN Brand"},
        {"Domain": "surfshark.com", "Category": "VPN Brand"},
        {"Domain": "protonvpn.com", "Category": "VPN Brand"},
    ])

def save_excluded_domains(df):
    df.to_json(EXCLUDED_FILE, orient="records")

def extract_phrases(text, min_len=2, max_len=5):
    text = re.sub(r"[^a-zA-Z0-9 ]", "", text.lower())
    words = text.split()
    stop_words = {
        "the","and","for","with","that","this","you","your","from","are","was","were",
        "have","has","how","why","what","when","where","can","not","but","all","any",
        "get","use","using","into","out","about","more","also","will","just","than","then",
        "them","they","their","its","our","who","which","while","because","over","under",
        "guide","tips","fix","error","issue"
    }
    words = [w for w in words if w not in stop_words and len(w) > 2]
    phrases = []
    for i in range(len(words)):
        for n in range(min_len, max_len + 1):
            phrase = words[i:i+n]
            if len(phrase) == n:
                phrases.append(" ".join(phrase))
    return phrases

def detect_intent(phrase):
    buyer_keywords = ["buy","best","coupon","discount","deal","review","compare","top"]
    return "Buyer" if any(k in phrase for k in buyer_keywords) else "Informational"

def affiliate_score(domain):
    score = 50
    if any(b in domain for b in ["expressvpn","nordvpn","surfshark"]):
        score = 0
    elif len(domain.split(".")) > 2:
        score += 20
    return score

def ai_summary(text, OPENAI_API_KEY):
    if not OPENAI_API_KEY:
        return "OpenAI API key not set."
    import openai
    openai.api_key = OPENAI_API_KEY
    try:
        resp = openai.ChatCompletion.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role":"system","content":"You are an SEO assistant."},
                {"role":"user","content":f"Summarize this page content in 2 sentences for SEO purposes: {text}"}
            ],
            max_tokens=150
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"AI Error: {e}"

# -----------------------------
# Main render function
# -----------------------------
def render(gl, hl, OPENAI_API_KEY, SERPAPI_KEY):

    # -----------------------------
    # Excluded Domains Manager
    # -----------------------------
    if "excluded_domains" not in st.session_state:
        st.session_state.excluded_domains = load_excluded_domains()

    st.markdown("---")
    st.subheader("⚙️ Excluded Domains Manager")
    col_add, col_remove = st.columns(2)

    # Add domain
    with col_add:
        st.text("➕ Add Domain")
        new_domain = st.text_input("Domain (e.g., example.com)", key="add_domain")
        new_category = st.text_input("Category (e.g., Forum, Big Tech, VPN Brand)", key="add_category")
        if st.button("Add Domain", key="add_domain_btn"):
            if new_domain.strip() == "":
                st.warning("Please enter a valid domain.")
            else:
                st.session_state.excluded_domains = pd.concat([
                    st.session_state.excluded_domains,
                    pd.DataFrame([{"Domain": new_domain.strip(), "Category": new_category.strip()}])
                ], ignore_index=True)
                save_excluded_domains(st.session_state.excluded_domains)
                st.success(f"✅ Domain '{new_domain}' added!")

    # Remove domain
    with col_remove:
        st.text("🗑 Remove Domain(s)")
        remove_domains = st.multiselect(
            "Select domains to remove",
            options=st.session_state.excluded_domains["Domain"].tolist(),
            key="remove_domains_select"
        )
        if st.button("Remove Selected", key="remove_domain_btn"):
            if remove_domains:
                st.session_state.excluded_domains = st.session_state.excluded_domains[
                    ~st.session_state.excluded_domains["Domain"].isin(remove_domains)
                ]
                save_excluded_domains(st.session_state.excluded_domains)
                st.success(f"✅ Removed {len(remove_domains)} domain(s)!")
            else:
                st.warning("No domains selected.")

    # View & edit existing domains
    st.subheader("📝 Excluded Domains")
    edited_df = st.data_editor(
        st.session_state.excluded_domains,
        num_rows=0,
        use_container_width=True
    )
    st.session_state.excluded_domains = edited_df
    save_excluded_domains(st.session_state.excluded_domains)

    # -----------------------------
    # Competitor Analysis Inputs
    # -----------------------------
    st.markdown("---")
    st.subheader("📌 Find Keywords & Top Pages by Domain")
    col1, col2, col3 = st.columns(3)
    with col1:
        domain = st.text_input("Competitor Domain", "appuals.com", key="analysis_domain")
    with col2:
        keyword = st.text_input("Keyword / Topic", "vpn", key="analysis_keyword")
    with col3:
        result_count = st.selectbox("Results", [20,50,100], index=1, key="analysis_results")

    run_analysis = st.button("Analyze Competitor", key="run_analysis")

    if run_analysis:
        query = f"site:{domain} {keyword}"
        params = {
            "engine":"bing",
            "q":query,
            "api_key":SERPAPI_KEY,
            "num":result_count,
            "gl": gl,  # pass sidebar geolocation
            "hl": hl   # pass sidebar language
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        organic = results.get("organic_results", [])

        if not organic:
            st.error("No results found. Try another keyword.")
            return

        data = []
        all_phrases = []

        for r in organic:
            title = r.get("title","")
            link = r.get("link","")
            snippet = r.get("snippet","")
            text_blob = f"{title} {snippet} {link}"

            phrases = extract_phrases(text_blob)
            all_phrases.extend(phrases)

            data.append({
                "Title": title,
                "URL": link,
                "Snippet": snippet,
                "AI Summary": ai_summary(snippet, OPENAI_API_KEY),
                "Affiliate Score": affiliate_score(link)
            })

        df = pd.DataFrame(data)
        st.subheader("📄 Top Ranking Pages on Bing")
        st.dataframe(df, use_container_width=True)

        # Keyword Phrases
        phrase_counts = Counter(all_phrases).most_common(40)
        phrases_df = pd.DataFrame(phrase_counts, columns=["Keyword Phrase","Frequency"])
        phrases_df["Intent"] = phrases_df["Keyword Phrase"].apply(detect_intent)
        st.subheader("🔑 Top Extracted Keyword Phrases with Intent")
        st.dataframe(phrases_df, use_container_width=True)

        # Content Gap
        top_phrases = set([k for k,_ in phrase_counts])
        user_content = keyword.lower().split()
        missing_phrases = [p for p in top_phrases if not any(word in p for word in user_content)]
        st.subheader("🕵️ Content Gap (Opportunities)")
        st.write(", ".join(missing_phrases[:20]))

        # CSV Export
        st.subheader("⬇️ Export Data")
        colA, colB = st.columns(2)
        with colA:
            st.download_button("Download Pages CSV", df.to_csv(index=False), "bing_competitor_pages.csv", "text/csv")
        with colB:
            st.download_button("Download Keyword Phrases CSV", phrases_df.to_csv(index=False), "bing_competitor_keywords.csv", "text/csv")

        # Bing Rank Tracking
        st.subheader("📊 Bing Rank Tracking")
        df['Rank'] = range(1,len(df)+1)
        st.dataframe(df[['Rank','Title','URL']], use_container_width=True)

        st.success("✅ Analysis complete! Ready for SEO domination 😎")

    # -----------------------------
    # Affiliate Competitor Finder
    # -----------------------------
    st.markdown("---")
    st.subheader("🏆 Bing Affiliate Competitor Finder")
    comp_keyword = st.text_input("Search Keyword for Competitor Finder", "best vpn for streaming", key="affiliate_keyword")
    category_filter = st.multiselect(
        "Exclude by Category",
        options=st.session_state.excluded_domains["Category"].unique(),
        default=st.session_state.excluded_domains["Category"].unique(),
        key="affiliate_category_filter"
    )
    find_comps = st.button("Find Affiliate Competitors", key="find_comps")

    if find_comps:
        params = {"engine":"bing","q":comp_keyword,"api_key":SERPAPI_KEY,"num":50, "gl": gl, "hl": hl}
        search = GoogleSearch(params)
        results = search.get_dict()
        organic = results.get("organic_results", [])

        excluded_domains_list = st.session_state.excluded_domains[
            st.session_state.excluded_domains["Category"].isin(category_filter)
        ]["Domain"].tolist()

        competitors = []
        for r in organic:
            link = r.get("link","")
            comp_domain = link.split("/")[2].replace("www.","")
            if not any(excl in comp_domain for excl in excluded_domains_list):
                competitors.append({
                    "Domain": comp_domain,
                    "Title": r.get("title",""),
                    "URL": link,
                    "Affiliate Score": affiliate_score(comp_domain)
                })

        comp_df = pd.DataFrame(competitors).drop_duplicates(subset="Domain")
        st.subheader("🎯 Affiliate-Style Competitors on Bing")
        st.dataframe(comp_df, use_container_width=True)
        st.download_button("Download Competitor List", comp_df.to_csv(index=False),"bing_affiliate_competitors.csv","text/csv")
