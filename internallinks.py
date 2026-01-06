# internallinks.py
import json
import re
import requests
import pandas as pd
import streamlit as st
from typing import List, Dict, Any
from shared import ts, to_excel_bytes
from shared import openai_chat


def serpapi_google_search(query: str, num: int, gl: str, hl: str, api_key: str) -> Dict[str, Any]:
    if not api_key:
        raise RuntimeError("SERPAPI_KEY is missing. Add it to your .env")
    params = {"engine": "google", "q": query, "num": int(num), "gl": gl, "hl": hl, "api_key": api_key}
    r = requests.get("https://serpapi.com/search.json", params=params, timeout=60)
    r.raise_for_status()
    return r.json()

def render(gl: str, hl: str, OPENAI_API_KEY: str, SERPAPI_KEY: str, show_header: bool = False):
    if show_header:
        st.subheader("Internal Link Opportunities")
    st.markdown(
        "Find natural internal link opportunities on your site using SerpAPI + OpenAI "
        "(site: operator searches, anchor variants, and placement hints)."
    )

    domain = st.text_input(
        "Your domain (e.g., example.com or https://example.com)",
        value="bitcoingg.com", key="il_domain"
    )
    targets = st.text_area(
        "Target URLs (one per line)",
        value="https://www.bitcoingg.com/bitcoin-games/poker/",
        key="il_targets"
    )
    anchors = st.text_area(
        "Anchor phrases (one per line)",
        value="bitcoin poker\ncrypto poker", key="il_anchors"
    )
    il_num = st.number_input(
        "Max results per anchor search", min_value=10, max_value=100,
        value=20, step=10, key="il_num"
    )
    run_il = st.button("Find Internal Link Opportunities", key="il_run")

    if not run_il:
        return

    if not (SERPAPI_KEY and OPENAI_API_KEY):
        st.error("Please set SERPAPI_KEY and OPENAI_API_KEY in your environment/.env.")
        return

    d = domain.strip().rstrip("/")
    if not d.startswith(("http://", "https://")):
        d = "https://" + d
    host = d.split("://", 1)[-1].rstrip("/")

    def find_sources(anchor: str) -> List[str]:
        q = f'site:{host} "{anchor}"'
        data = serpapi_google_search(q, int(il_num), gl, hl, SERPAPI_KEY)
        return [r.get("link", "") for r in data.get("organic_results", []) if r.get("link")]

    rows = []
    t_urls = [u.strip() for u in targets.splitlines() if u.strip()]
    a_list = [a.strip() for a in anchors.splitlines() if a.strip()]

    for turl in t_urls:
        for a in a_list:
            sources = find_sources(a)
            prompt = f"""
We have a target URL: {turl}
Candidate source URLs (same domain) to link from: {sources[:20]}
Anchor phrase: "{a}"

Return JSON:
- top_sources: top 10 source URLs in priority order
- anchor_variants: 5 natural variants of the anchor
- placement_hint: 1-2 sentence advice where the link best fits
"""
            try:
                plan_raw = openai_chat(OPENAI_API_KEY, [{"role": "user", "content": prompt}], temperature=0.2)
                try:
                    plan = json.loads(plan_raw)
                except Exception:
                    m = re.search(r"\{.*\}", plan_raw, re.S)
                    plan = json.loads(m.group(0)) if m else {"top_sources": sources[:10], "anchor_variants": [a], "placement_hint": ""}
            except Exception as e:
                st.error(f"OpenAI error: {e}")
                plan = {"top_sources": sources[:10], "anchor_variants": [a], "placement_hint": ""}

            for i, src in enumerate(plan.get("top_sources", []), start=1):
                rows.append({
                    "target_url": turl,
                    "anchor_seed": a,
                    "proposed_anchor": (plan.get("anchor_variants") or [a])[0],
                    "source_url": src,
                    "priority": i,
                    "placement_hint": plan.get("placement_hint", "")
                })

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        st.download_button("⬇️ CSV", df.to_csv(index=False).encode("utf-8"),
                           file_name=f"internal_links_{ts()}.csv", mime="text/csv", key="il_dl_csv")
        st.download_button("⬇️ Excel", to_excel_bytes(df),
                           file_name=f"internal_links_{ts()}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           key="il_dl_xlsx")
    else:
        st.info("No internal link opportunities found for the given inputs.")
