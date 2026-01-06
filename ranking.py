# ranking.py
# Minimal Rank Checker: fresh results only, with SERP pagination (no cache/snapshots/KPIs/history/baseline)

from typing import List, Dict, Any
from urllib.parse import urlparse
import requests
import pandas as pd
import streamlit as st

# If you prefer using your shared helpers for Excel only, keep this import:
from shared import to_excel_bytes  # uses xlsxwriter under the hood


def normalize_host(domain_or_url: str) -> str:
    parsed = urlparse(domain_or_url if "://" in domain_or_url else f"http://{domain_or_url}")
    return parsed.netloc.lower()


def host_matches(link: str, target_host: str) -> bool:
    try:
        link_host = urlparse(link).netloc.lower().lstrip("www.")
        target = target_host.lower().lstrip("www.")
        return (link_host == target) or link_host.endswith("." + target) or (target in link_host)
    except Exception:
        return False


def serpapi_fetch_pages(keyword: str, gl: str, hl: str, api_key: str, total: int) -> List[Dict[str, Any]]:
    """
    Fetch up to `total` organic results by paging through Google SERPs via SerpAPI.
    We explicitly request 10 per page and advance with `start=0,10,20,...` to avoid
    providers that cap a single response at 10 results.
    """
    endpoint = "https://serpapi.com/search.json"
    collected: List[Dict[str, Any]] = []
    per_page = 10
    start = 0

    while len(collected) < total:
        params = {
            "engine": "google",
            "q": keyword,
            "hl": hl,
            "gl": gl,
            "num": per_page,  # ask for 10; some engines ignore >10
            "start": start,   # page offset
            "api_key": api_key,
        }
        r = requests.get(endpoint, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        page_results = data.get("organic_results", []) or []

        if not page_results:
            break  # no more results

        collected.extend(page_results)
        start += per_page

    # Trim in case we got more than needed
    return collected[:total]


def render(gl: str, hl: str, SERPAPI_KEY: str):
    st.header("Simple Rank Checker")

    col_domain, col_num = st.columns([2, 1])
    domain_input = col_domain.text_input(
        "Your domain (host or URL)",
        value="example.com",
        help="Examples: example.com or https://www.example.com/"
    )
    num_results = int(col_num.number_input("Top results (num)", min_value=10, max_value=100, value=50, step=10))

    st.markdown("**Keywords**")
    kw_text = st.text_area(
        "Paste one keyword per line",
        height=160,
        placeholder="e.g.\nweb hosting canada\nbest vpn canada\nwebsite development"
    )
    uploaded_kw = st.file_uploader("…or upload a .txt or .csv with one keyword per line/row",
                                   type=["txt", "csv"], key="kw_upload")

    # Gather keywords
    keywords: List[str] = []
    if kw_text.strip():
        keywords.extend([line.strip() for line in kw_text.splitlines() if line.strip()])
    if uploaded_kw is not None:
        try:
            if uploaded_kw.name.lower().endswith(".txt"):
                txt = uploaded_kw.read().decode("utf-8", errors="ignore")
                keywords.extend([line.strip() for line in txt.splitlines() if line.strip()])
            else:
                df_u = pd.read_csv(uploaded_kw, header=None)
                keywords.extend([str(x).strip() for x in df_u.iloc[:, 0].tolist() if str(x).strip()])
        except Exception as e:
            st.error(f"Could not read uploaded keywords: {e}")

    # De-dupe
    seen = set()
    keywords = [k for k in keywords if not (k in seen or seen.add(k))]
    st.write(f"**Keywords loaded:** {len(keywords)}")

    run_btn = st.button("▶️ Check Rankings", type="primary")
    if not run_btn:
        return

    # Validations
    if not SERPAPI_KEY:
        st.error("Please set SERPAPI_KEY in environment or Streamlit secrets.")
        return
    if not domain_input.strip():
        st.error("Please provide your domain.")
        return
    if not keywords:
        st.error("Please add at least one keyword.")
        return

    target_host = normalize_host(domain_or_url=domain_input)
    progress = st.progress(0.0)
    status = st.empty()
    rows = []

    for i, kw in enumerate(keywords, start=1):
        status.write(f"Checking **{kw}** ({i}/{len(keywords)}) …")

        try:
            organic = serpapi_fetch_pages(
                keyword=kw,
                gl=gl,
                hl=hl,
                api_key=SERPAPI_KEY,
                total=num_results,
            )

            pos = ""
            url_found = ""
            for idx, res in enumerate(organic, start=1):
                link = res.get("link", "")
                if link and host_matches(link, target_host):
                    pos = idx
                    url_found = link
                    break

            rows.append({"Keyword": kw, "URL": url_found, "Position": pos})
        except requests.HTTPError as e:
            rows.append({"Keyword": kw, "URL": f"ERROR HTTP {e.response.status_code}", "Position": ""})
        except Exception as e:
            rows.append({"Keyword": kw, "URL": f"ERROR {type(e).__name__}", "Position": ""})

        progress.progress(i / len(keywords))

    df = pd.DataFrame(rows, columns=["Keyword", "URL", "Position"])
    st.success("Done! Preview below.")
    st.dataframe(df, use_container_width=True)

    # Downloads only
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "⬇️ Download CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="rankings.csv",
            mime="text/csv"
        )
    with c2:
        st.download_button(
            "⬇️ Download Excel",
            data=to_excel_bytes(df),
            file_name="rankings.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
