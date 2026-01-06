# contentaudit.py
import os
import re, json, requests
import pandas as pd
import streamlit as st
from collections import Counter

from shared import ts, to_excel_bytes
from shared import openai_chat


def render(OPENAI_API_KEY: str, show_header: bool = False):
    """
    Content Audit tab. Uses SerpAPI for fetching pages and OpenAI for
    recommendations. Now respects the model selected in the sidebar via
    os.environ['OPENAI_MODEL'] (see launcher.py).
    """
    if show_header:
        st.subheader("Content Audit")
    st.markdown(
        "Analyze your page vs competitors: headings, on-page snapshot, n-gram density, "
        "coverage overlap, rich-result hints, and quick recommendations."
    )

    # ---- session persistence ----
    AUDIT_STATE_KEY = "content_audit_state"
    if AUDIT_STATE_KEY not in st.session_state:
        st.session_state[AUDIT_STATE_KEY] = {}
    audit_state = st.session_state[AUDIT_STATE_KEY]

    # --- HTML parsing / text utilities
    try:
        from bs4 import BeautifulSoup  # noqa
    except Exception:
        BeautifulSoup = None

    PRIMARY_STOPWORDS = {
        "a","an","the","and","or","but","if","then","else","when","at","by","for","in","of","on","to","with",
        "is","are","was","were","be","been","being","as","from","that","this","it","its","you","your","yours",
        "me","my","we","our","ours","they","their","them","he","she","his","her","hers","not","no","do","does",
        "did","doing","can","could","should","would","will","just","so","than","too","very","into","over","under",
        "again","further","here","there","both","each","few","more","most","other","some","such","only","own","same",
        "until","while","about","against","between","through","during","before","after","above","below","up","down",
        "out","off"
    }
    UA_HEADERS = {"User-Agent": "SEO-Reporting/1.0"}

    def tokenise(text):
        import re as _re
        return _re.findall(r"[A-Za-z0-9]+", (text or "").lower())

    def get_ngrams(words, n=2, stop_words=PRIMARY_STOPWORDS):
        filtered = [w for w in words if w not in stop_words]
        return [" ".join(filtered[i:i+n]) for i in range(len(filtered)-n+1)]

    def density(count, total_tokens):
        return 0 if total_tokens == 0 else round((count/total_tokens)*100, 2)

    # ---------- Analyzer with on-page metrics & schema ----------
    def analyze_url(url, stop_words=PRIMARY_STOPWORDS):
        """
        Fetch a URL, extract title/meta/H1 + all H2/H3, compute bigrams/trigrams on full text,
        and collect lightweight on-page metrics for quick wins.
        """
        from urllib.parse import urlparse
        import re as _re, json as _json

        result = {
            "url": url, "title": "", "description": "", "h1": "",
            "h1_list": [], "h2_list": [], "h3_list": [], "schema_types": [],
            "bigrams": Counter(), "trigrams": Counter(), "total_tokens": 0,
            "metrics": {}, "error": ""
        }
        if not url:
            result["error"] = "No URL provided"
            return result
        if BeautifulSoup is None:
            result["error"] = "BeautifulSoup (bs4) not available"
            return result

        def _readability_stats(text: str):
            sents = _re.split(r"[.!?]+", text)
            sents = [s.strip() for s in sents if s.strip()]
            words = _re.findall(r"[A-Za-z0-9']+", text)
            n_s, n_w = max(1, len(sents)), max(1, len(words))
            avg_slen = n_w / n_s
            vowels = "aeiouy"
            syl = 0
            for w in words:
                w2 = w.lower()
                prev_v = False
                for ch in w2:
                    is_v = ch in vowels
                    if is_v and not prev_v:
                        syl += 1
                    prev_v = is_v
                if w2.endswith("e") and syl > 0:
                    syl -= 1
            syl = max(1, syl)
            fk_grade = 0.39 * (n_w / n_s) + 11.8 * (syl / n_w) - 15.59
            return {
                "word_count": n_w,
                "avg_sentence_len": round(avg_slen, 2),
                "fk_grade": round(fk_grade, 2),
                "est_read_time_min": round(n_w / 200, 1)
            }

        errs = []
        title, description, h1_text, text = "", "", "", ""
        h1s, h2s, h3s, schema_types = [], [], [], []
        try:
            resp = requests.get(url, headers=UA_HEADERS, timeout=30)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            t = soup.find("title")
            title = t.get_text(strip=True) if t else ""
            meta = soup.find("meta", attrs={"name":"description"}) or soup.find("meta", property="og:description")
            if meta and meta.get("content"):
                description = meta["content"]

            canonical = soup.find("link", rel=lambda v: v and "canonical" in v.lower())
            canonical_href = canonical.get("href") if canonical else ""
            meta_robots = soup.find("meta", attrs={"name":"robots"})
            robots_content = (meta_robots.get("content") or "").lower() if meta_robots else ""
            noindex = "noindex" in robots_content

            h1_elems = soup.find_all("h1")
            h2_elems = soup.find_all("h2")
            h3_elems = soup.find_all("h3")
            h1s = [h.get_text(" ", strip=True) for h in h1_elems]
            h2s = [h.get_text(" ", strip=True) for h in h2_elems]
            h3s = [h.get_text(" ", strip=True) for h in h3_elems]
            h1_text = h1s[0] if h1s else ""

            imgs = soup.find_all("img")
            img_count = len(imgs)
            img_alt_with = sum(1 for im in imgs if (im.get("alt") or "").strip())
            lists_count = len(soup.find_all(["ul", "ol"]))
            tables_count = len(soup.find_all("table"))
            code_blocks = len(soup.find_all(["pre", "code"]))

            # Links
            parsed_host = urlparse(url).netloc.lower()
            a_tags = soup.find_all("a", href=True)
            internal, external = 0, 0
            for a in a_tags:
                try:
                    host = urlparse(a["href"]).netloc.lower()
                    if not host or host.endswith(parsed_host) or parsed_host.endswith(host):
                        internal += 1
                    else:
                        external += 1
                except Exception:
                    internal += 1

            # JSON-LD schema types
            schema_types = []
            for sc in soup.find_all("script", type="application/ld+json"):
                try:
                    data = _json.loads(sc.string or "{}")
                    def collect_types(obj):
                        if isinstance(obj, dict):
                            t = obj.get("@type")
                            if isinstance(t, list):
                                schema_types.extend([str(x) for x in t])
                            elif t:
                                schema_types.append(str(t))
                            for v in obj.values():
                                collect_types(v)
                        elif isinstance(obj, list):
                            for it in obj:
                                collect_types(it)
                    collect_types(data)
                except Exception:
                    pass
            schema_types = sorted(set(schema_types))

            text = soup.get_text(" ", strip=True)

            rb = _readability_stats(text)
            metrics = {
                **rb,
                "title_len": len(title or ""),
                "meta_desc_len": len(description or ""),
                "h1_count": len(h1s),
                "h2_count": len(h2s),
                "h3_count": len(h3s),
                "h23_ratio": round((len(h3s) / max(1, len(h2s))), 2) if (len(h2s) or len(h3s)) else 0.0,
                "images": img_count,
                "images_with_alt_%": (round(img_alt_with / img_count * 100, 1) if img_count else 0.0),
                "lists": lists_count,
                "tables": tables_count,
                "code_blocks": code_blocks,
                "links_internal": internal,
                "links_external": external,
                "canonical_present": bool(canonical_href),
                "noindex": bool(noindex),
            }

        except Exception as e:
            errs.append(f"Fetch/parse failed: {e}")
            metrics = {}

        words = tokenise(text)
        filtered = [w for w in words if w not in PRIMARY_STOPWORDS]
        total_tokens = len(filtered)
        bigrams = Counter(get_ngrams(words, 2, stop_words=PRIMARY_STOPWORDS))
        trigrams = Counter(get_ngrams(words, 3, stop_words=PRIMARY_STOPWORDS))

        result.update({
            "title": title,
            "description": description,
            "h1": h1_text,
            "h1_list": h1s,
            "h2_list": h2s,
            "h3_list": h3s,
            "schema_types": schema_types,
            "bigrams": bigrams,
            "trigrams": trigrams,
            "total_tokens": total_tokens,
            "metrics": metrics,
            "error": "; ".join(errs) if errs else ""
        })
        return result

    def rows_for_table(slot_label, record, top_b=4, top_t=2):
        bigs = record["bigrams"].most_common(top_b)
        tris = record["trigrams"].most_common(top_t)
        while len(bigs) < top_b: bigs.append(("", 0))
        while len(tris) < top_t: tris.append(("", 0))

        row = {
            "Slot": slot_label,
            "URL": record.get("url",""),
            "Title": record.get("title",""),
            "H1": record.get("h1",""),
            "Description": record.get("description","")
        }
        total = max(record.get("total_tokens",0),1)
        for i,(phrase,cnt) in enumerate(bigs, start=1):
            row[f"Bigram {i}"]=phrase; row[f"B{i} Mentions"]=cnt; row[f"B{i} Density %"]=density(cnt,total)
        for i,(phrase,cnt) in enumerate(tris, start=1):
            row[f"Trigram {i}"]=phrase; row[f"T{i} Mentions"]=cnt; row[f"T{i} Density %"]=density(cnt,total)
        return row

    # ---------- Heading gap + coverage + rich-result hints ----------
    def _norm_heading(s: str) -> str:
        import re as _re
        s = (s or "").lower().strip()
        s = _re.sub(r"\s+", " ", s)
        s = _re.sub(r"[^\w\s]", "", s)
        return s

    def collect_headings_gap(records):
        our_all = set(); our_h1=[]; our_h2=[]; our_h3=[]
        comps = []
        for rec in records:
            slot = rec.get("slot","")
            all_norm = []
            for h in (rec.get("h1_list") or []): all_norm.append(_norm_heading(h))
            for h in (rec.get("h2_list") or []): all_norm.append(_norm_heading(h))
            for h in (rec.get("h3_list") or []): all_norm.append(_norm_heading(h))
            all_norm = [h for h in all_norm if h]
            if slot.lower().startswith("our"):
                our_all = set(all_norm)
                our_h1 = [ _norm_heading(x) for x in (rec.get("h1_list") or []) if _norm_heading(x) ]
                our_h2 = [ _norm_heading(x) for x in (rec.get("h2_list") or []) if _norm_heading(x) ]
                our_h3 = [ _norm_heading(x) for x in (rec.get("h3_list") or []) if _norm_heading(x) ]
            else:
                comps.append({
                    "slot": slot, "url": rec.get("url",""),
                    "h1": [_norm_heading(x) for x in (rec.get("h1_list") or []) if _norm_heading(x)],
                    "h2": [_norm_heading(x) for x in (rec.get("h2_list") or []) if _norm_heading(x)],
                    "h3": [_norm_heading(x) for x in (rec.get("h3_list") or []) if _norm_heading(x)],
                    "all_norm": set(all_norm)
                })

        from collections import Counter as C
        cnt = C()
        for c in comps:
            cnt.update([h for h in c["all_norm"] if h not in our_all])

        missing_candidates = [{"heading": k, "competitor_count": v} for k, v in cnt.most_common()]
        return {
            "our": {"all_norm": our_all, "h1": our_h1, "h2": our_h2, "h3": our_h3},
            "competitors": comps,
            "missing_candidates": missing_candidates
        }

    def coverage_and_hints(records):
        our_rec = next((r for r in records if str(r.get("slot","")).lower().startswith("our")), None)
        our_set = set([_norm_heading(x) for x in (our_rec.get("h2_list") or [])] +
                      [_norm_heading(x) for x in (our_rec.get("h3_list") or [])]) if our_rec else set()

        rows = []
        faq_comp, howto_comp = 0, 0
        our_has_faq = our_rec and "FAQPage" in (our_rec.get("schema_types") or [])
        our_has_howto = our_rec and "HowTo"   in (our_rec.get("schema_types") or [])

        for rec in records:
            if str(rec.get("slot","")).lower().startswith("our"):
                continue
            comp_set = set([_norm_heading(x) for x in (rec.get("h2_list") or [])] +
                           [_norm_heading(x) for x in (rec.get("h3_list") or [])])
            inter = len(our_set & comp_set)
            union = len(our_set | comp_set)
            jacc = (inter / union) if union else 0.0
            rows.append({
                "Competitor": rec.get("slot",""),
                "Jaccard (H2+H3)": round(jacc, 3),
                "Overlap": inter,
                "Our H2+H3": len(our_set),
                "Their H2+H3": len(comp_set)
            })
            stypes = set(rec.get("schema_types") or [])
            if "FAQPage" in stypes: faq_comp += 1
            if "HowTo" in stypes:   howto_comp += 1

        cov_df = pd.DataFrame(rows).sort_values("Jaccard (H2+H3)", ascending=False) if rows else pd.DataFrame(
            columns=["Competitor","Jaccard (H2+H3)","Overlap","Our H2+H3","Their H2+H3"])

        hints = {
            "our_has_faq": bool(our_has_faq),
            "our_has_howto": bool(our_has_howto),
            "competitors_with_faq": faq_comp,
            "competitors_with_howto": howto_comp
        }
        return cov_df, hints

    # ---------- Recommendations (OpenAI) ----------
    def generate_recommendations(keyword, ngram_df, heading_gap, coverage_df, rr_hints):
        if not OPENAI_API_KEY:
            return "OpenAI key not set. Add OPENAI_API_KEY to your environment to get recommendations."

        ngram_rows = ngram_df.to_dict(orient="records")
        our_all = sorted(list(heading_gap["our"]["all_norm"]))
        comp_sets = [{
            "slot": cp["slot"], "url": cp["url"],
            "h1": cp["h1"], "h2": cp["h2"], "h3": cp["h3"]
        } for cp in heading_gap["competitors"]]
        quick_candidates = heading_gap["missing_candidates"]
        coverage_rows = coverage_df.to_dict(orient="records")

        system = (
            "You are an elite SEO content strategist. Produce concise, implementable recommendations. "
            "NO long outline. Focus on quick wins that can be added fast. "
            "When suggesting content blocks, choose from: H2, H3, paragraph, bullet list, numbered steps, table, FAQ item (Q&A), "
            "callout/note, code/example. Keep tone practical. Use the coverage and schema hints when relevant."
        )

        user = {
            "keyword": keyword,
            "ngrams_table": ngram_rows,
            "our_headings_norm": our_all,
            "competitor_headings_norm": comp_sets,
            "missing_topic_candidates": quick_candidates,
            "coverage_rows": coverage_rows,
            "rich_result_hints": rr_hints,
            "instructions": [
                "1) Recommend 3–8 phrases to INCREASE or DECREASE. For each, specify: "
                "target density range (%) AND how many mentions to add/remove to hit that range.",
                "2) Pick 5–12 Quick Wins (missing topics). For each: short label, block type, 1-line guidance.",
                "3) Suggest one improved Title, Meta Description, H1.",
                "4) End with 2–3 bullet summary."
            ]
        }

        # Read the selected model from env (set in launcher.py)
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        content = openai_chat(
            OPENAI_API_KEY,
            [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user)}
            ],
            temperature=0.35,
            model=model,  # <- respect sidebar model
        )

        return (content or "").strip()

    # ------------ UI inputs ------------
    keyword = st.text_input("Primary keyword", value=audit_state.get("keyword", "best website builder canada"), key="audit_kw")
    our_url = st.text_input("Your landing page URL", value=audit_state.get("our_url", ""), key="audit_our")
    c1, c2, c3 = st.columns(3)
    with c1: comp1 = st.text_input("Competitor 1 URL", value=audit_state.get("comp1",""), key="audit_c1")
    with c2: comp2 = st.text_input("Competitor 2 URL", value=audit_state.get("comp2",""), key="audit_c2")
    with c3: comp3 = st.text_input("Competitor 3 URL", value=audit_state.get("comp3",""), key="audit_c3")

    run_btn = st.button("Run Content Audit", type="primary", use_container_width=True, key="audit_run")

    if run_btn:
        urls, slots = [], []
        if our_url.strip():
            urls.append(our_url.strip()); slots.append("Our page")
        for i, u in enumerate([comp1, comp2, comp3], start=1):
            if u.strip():
                urls.append(u.strip()); slots.append(f"Competitor {i}")
        if not urls:
            st.warning("Please provide at least one URL (your page and/or competitors).")
            st.stop()

        # Analyze pages
        try:
            from bs4 import BeautifulSoup  # ensure available
        except Exception:
            st.error("BeautifulSoup (bs4) is not installed. Add 'beautifulsoup4' to requirements.")
            st.stop()

        records = []
        prog = st.progress(0)
        for i,(slot,url) in enumerate(zip(slots, urls)):
            rec = analyze_url(url)
            rec["slot"] = slot
            records.append(rec)
            prog.progress((i+1)/len(urls))
        prog.empty()

        # ---------- 1) Results (Top Bigrams & Trigrams) ----------
        rows = [rows_for_table(rec["slot"], rec) for rec in records]
        df = pd.DataFrame(rows)
        df["Error"] = [rec.get("error","") for rec in records]

        # ---------- 3) Headings (side-by-side) ----------
        def _flatten_headings(rec):
            out = []
            for h in rec.get("h1_list", []) or []:
                if h: out.append(f"H1 • {h}")
            for h in rec.get("h2_list", []) or []:
                if h: out.append(f"H2 • {h}")
            for h in rec.get("h3_list", []) or []:
                if h: out.append(f"H3 • {h}")
            if not out and rec.get("h1"):
                out.append(f"H1 • {rec['h1']}")
            return out

        columns_wide, max_rows = {}, 0
        for rec in records:
            header = f"{rec['slot']}\n{rec.get('url','')}"
            items = _flatten_headings(rec)
            columns_wide[header] = items
            max_rows = max(max_rows, len(items))
        for k, items in columns_wide.items():
            if len(items) < max_rows:
                columns_wide[k] = items + [""] * (max_rows - len(items))
        matrix_df = pd.DataFrame(columns_wide)
        matrix_df.index = [f"{i+1}" for i in range(max_rows)]

        # ---------- Coverage vs competitors + rich results hints ----------
        gap_info = collect_headings_gap(records)
        cov_df, rr_hints = coverage_and_hints(records)

        # ---------- Recommendations (OpenAI) ----------
        recs_text = generate_recommendations(keyword, df, gap_info, cov_df, rr_hints)

        # ---------- Build download bytes (persist) ----------
        ngram_xlsx = to_excel_bytes(df)
        head_csv = matrix_df.to_csv(index=False).encode("utf-8")
        head_xlsx = to_excel_bytes(matrix_df)

        # ---------- Save everything to session state ----------
        audit_state.update({
            "keyword": keyword,
            "our_url": our_url,
            "comp1": comp1, "comp2": comp2, "comp3": comp3,
            "records": records,
            "ngrams_df": df,
            "ngram_xlsx": ngram_xlsx,
            "matrix_df": matrix_df,
            "head_csv": head_csv,
            "head_xlsx": head_xlsx,
            "gap_info": gap_info,
            "cov_df": cov_df,
            "rr_hints": rr_hints,
            "recs_text": recs_text,
            "ready": True,
        })
        st.success("Audit complete. Downloads and results will persist until you run again or change inputs.")

    # ---------- Render from session (survives reruns & downloads) ----------
    if audit_state.get("ready"):
        df = audit_state["ngrams_df"]
        records = audit_state["records"]
        matrix_df = audit_state["matrix_df"]
        cov_df = audit_state["cov_df"]
        rr_hints = audit_state["rr_hints"]

        st.subheader("Results (Top Bigrams & Trigrams)")
        st.caption("Top 4 bigrams and top 2 trigrams per page, with mentions and density (% of non-stopword tokens).")
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.download_button(
            "⬇️ Download Excel (N-gram table)",
            audit_state["ngram_xlsx"],
            file_name=f"content_audit_ngrams_{ts()}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="audit_dl_xlsx"
        )

        # ---------- 2) Per-page Cards ----------
        st.markdown("---")
        st.subheader("Per-page Cards")
        for rec in records:
            colA, colB = st.columns([1.4, 2.6])
            with colA:
                st.markdown(f"**{rec.get('slot','')}**")
                st.write(rec.get("url",""))
                st.write(f"**Title:** {rec.get('title','')}")
                st.write(f"**Meta Description:** {rec.get('description','')}")
                st.write(f"**Primary H1:** {rec.get('h1','')}")
                if rec.get("error"):
                    st.caption(f"⚠️ {rec['error']}")
            with colB:
                total = max(rec.get("total_tokens", 0), 1)
                bigs = rec["bigrams"].most_common(4)
                tris = rec["trigrams"].most_common(2)
                bb = [{"Bigram": p, "Mentions": c, "Density %": density(c, total)} for (p, c) in bigs]
                tb = [{"Trigram": p, "Mentions": c, "Density %": density(c, total)} for (p, c) in tris]
                st.write("**Top Bigrams**")
                st.dataframe(pd.DataFrame(bb), use_container_width=True, hide_index=True)
                st.write("**Top Trigrams**")
                st.dataframe(pd.DataFrame(tb), use_container_width=True, hide_index=True)

                m = rec.get("metrics", {}) or {}
                if m:
                    st.write("**On-page Snapshot**")
                    snap = pd.DataFrame([{
                        "Word Count": m.get("word_count"),
                        "Read Time (min)": m.get("est_read_time_min"),
                        "FK Grade": m.get("fk_grade"),
                        "Title Len": m.get("title_len"),
                        "Meta Len": m.get("meta_desc_len"),
                        "H1/H2/H3": f"{m.get('h1_count')}/{m.get('h2_count')}/{m.get('h3_count')}",
                        "H3:H2 Ratio": m.get("h23_ratio"),
                        "Images (alt %)": f"{m.get('images')} ({m.get('images_with_alt_%')}%)",
                        "Lists/Tables/Code": f"{m.get('lists')}/{m.get('tables')}/{m.get('code_blocks')}",
                        "Links Int/Ext": f"{m.get('links_internal')}/{m.get('links_external')}",
                        "Canonical?": "Yes" if m.get("canonical_present") else "No",
                        "Noindex?": "Yes" if m.get("noindex") else "No",
                    }])
                    st.dataframe(snap, use_container_width=True, hide_index=True)

                st.caption(f"Schema types: {', '.join(rec.get('schema_types', [])) or '—'}")

        # ---------- 3) Headings (side-by-side) ----------
        st.markdown("---")
        st.subheader("Headings (H1 → H2 → H3): Side-by-Side")
        st.dataframe(matrix_df, use_container_width=True, hide_index=True)

        st.download_button(
            "⬇️ Download Headings Matrix (CSV)",
            audit_state["head_csv"],
            file_name=f"headings_matrix_{ts()}.csv",
            mime="text/csv",
            key="audit_head_csv",
        )
        st.download_button(
            "⬇️ Download Headings Matrix (Excel)",
            audit_state["head_xlsx"],
            file_name=f"headings_matrix_{ts()}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="audit_head_xlsx",
        )

        # ---------- Coverage vs competitors + rich results hints ----------
        st.markdown("---")
        st.subheader("Coverage vs Competitors (H2+H3 overlap)")
        if cov_df.empty:
            st.info("Provide at least one competitor to compute coverage.")
        else:
            st.dataframe(cov_df, use_container_width=True, hide_index=True)

        hints_msgs = []
        if rr_hints["competitors_with_faq"] >= 2 and not rr_hints["our_has_faq"]:
            hints_msgs.append("≥2 competitors use **FAQPage** schema; consider adding a small FAQ block (and schema).")
        if rr_hints["competitors_with_howto"] >= 2 and not rr_hints["our_has_howto"]:
            hints_msgs.append("≥2 competitors use **HowTo** schema; add a concise steps section (and HowTo schema).")
        if hints_msgs:
            st.info("Rich-results hints:\n\n- " + "\n- ".join(hints_msgs))

        # ---------- 4) Recommendations (OpenAI) ----------
        st.markdown("---")
        st.subheader("Recommendations (OpenAI)")
        st.write(audit_state.get("recs_text", ""))
    else:
        st.info("Run the Content Audit to see results and downloads.")
