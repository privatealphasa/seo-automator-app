#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Screaming Frog — Quick Fixes (tab-only)
"""

from __future__ import annotations
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# =============================================================================
# IO & column helpers
# =============================================================================

def _read_csv_guess(buf) -> pd.DataFrame:
    """Read CSV with sensible fallbacks (keeps UI from crashing on odd encodings)."""
    try:
        return pd.read_csv(buf, on_bad_lines="skip")
    except Exception:
        try:
            buf.seek(0)
        except Exception:
            pass
        try:
            return pd.read_csv(buf, encoding="latin-1", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame()

def _open_any(upload) -> Dict[str, pd.DataFrame]:
    """Open a CSV or XLSX and return dict of {sheet_or_file_name: DataFrame}."""
    if upload is None:
        return {}
    nm = (upload.name or "").lower()
    if nm.endswith(".csv"):
        return {upload.name: _read_csv_guess(upload)}
    if nm.endswith(".xlsx"):
        xls = pd.ExcelFile(upload)
        return {name: pd.read_excel(xls, sheet_name=name) for name in xls.sheet_names}
    return {}

def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Trim + collapse whitespace in headers."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out.columns = [re.sub(r"\s+", " ", str(c or "")).strip() for c in out.columns]
    return out

def _pick_table_by_name_or_largest(tables: Dict[str, pd.DataFrame],
                                   name_hints: List[str]) -> pd.DataFrame:
    """Prefer a sheet by name; otherwise pick the largest table."""
    if not tables:
        return pd.DataFrame()
    for name, df in tables.items():
        if any(h in str(name).lower() for h in name_hints):
            return _norm_cols(df)
    best, best_len = None, -1
    for _, df in tables.items():
        if isinstance(df, pd.DataFrame) and len(df) > best_len:
            best, best_len = df, len(df)
    return _norm_cols(best) if best is not None else pd.DataFrame()

def _pick_inlinks_table(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    if not tables:
        return pd.DataFrame()
    for name, df in tables.items():
        if any(h in str(name).lower() for h in ["inlinks", "all inlinks", "all_inlinks", "links"]):
            return _norm_cols(df)
    return _pick_table_by_name_or_largest(tables, [])

def _pick_col(cols: List[str], *names) -> Optional[str]:
    """Find a column by exact name (case/space-insensitive), then by fuzzy contains."""
    lc = {re.sub(r"\s+", " ", str(c or "")).strip().lower(): c for c in cols}
    for n in names:
        key = re.sub(r"\s+", " ", str(n)).strip().lower()
        if key in lc:
            return lc[key]
    # fuzzy fallbacks
    for c in cols:
        cl = re.sub(r"\s+", " ", str(c or "")).strip().lower()
        for n in names:
            if re.sub(r"\s+", " ", str(n)).strip().lower() in cl:
                return c
    return None

def _pick_col_exact(cols: List[str], name: str) -> Optional[str]:
    """Exact header picker (case/space-insensitive)."""
    norm = {re.sub(r"\s+", " ", str(c or "")).strip().lower(): c for c in cols}
    return norm.get(re.sub(r"\s+", " ", name).strip().lower())

def _as_tz_naive(series: pd.Series) -> pd.Series:
    """
    Parse datetimes robustly and return tz-naive (no timezone) pandas datetimes.
    Works whether the input is mixed/aware/naive.
    """
    s = pd.to_datetime(series, errors="coerce", utc=True)
    try:
        s = s.dt.tz_convert(None)
    except Exception:
        try:
            s = s.dt.tz_localize(None)
        except Exception:
            s = pd.to_datetime(series, errors="coerce")
    return s

# =============================================================================
# N-gram helpers
# =============================================================================

def _parse_phrase_and_mentions(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Parse 'keyword: 12' / 'keyword (12)' / 'keyword ... 12' → (phrase, mentions)."""
    phrases, counts = [], []
    for raw in series.fillna("").astype(str):
        s = raw.strip()
        m = (re.search(r"^(.*?)[\s]*[:\-]\s*(\d+)\s*$", s) or
             re.search(r"^(.*?)\s*\((\d+)\)\s*$", s) or
             re.search(r"^(.*?)[^\d]*(\d+)\s*$", s))
        if m:
            phrases.append(m.group(1).strip(' "\'')); counts.append(pd.to_numeric(m.group(2), errors="coerce"))
        else:
            phrases.append(s); counts.append(np.nan)
    return pd.Series(phrases), pd.to_numeric(pd.Series(counts), errors="coerce")

def _list_ngram_phrase_cols(df: pd.DataFrame, kind: str) -> List[Tuple[str, int]]:
    """Return [(column_name, lang_number)] for bigram/trigram phrase columns."""
    cols = list(map(str, df.columns))
    out, base = [], (r"bigrams?" if kind == "bigram" else r"trigrams?")
    for c in cols:
        cl = c.lower()
        if not re.search(base, cl, re.I):
            continue
        if re.search(r"(mentions?|occurrences?|count|density)", cl):
            continue
        m = re.search(r"(?:en\s*)?(\d+)\s*$", cl, re.I)
        out.append((c, int(m.group(1)) if m else 0))
    out.sort(key=lambda t: (0 if t[1] == 0 else 1, t[1]))
    return out

def _find_density_col_near(df: pd.DataFrame, phrase_col: Optional[str]) -> Optional[str]:
    """Heuristic: density column tends to sit to the right of phrase column."""
    if not phrase_col:
        return None
    cols = list(df.columns)
    try:
        idx = cols.index(phrase_col); right = cols[idx + 1: idx + 6]
    except ValueError:
        right = []
    for c in right:
        if "density" in str(c).lower():
            return c
    return None

# ------------------------------------------------------------------
# Hard filters for Screaming Frog "page" analyses (no UI)
# ------------------------------------------------------------------
BAN_RE = re.compile(
    r'/wp(?:/|$)|'
    r'/wp-login(?:\.php)?(?:/|$)|'
    r'/wp-admin(?:/|$)|'
    r'/xmlrpc\.php(?:/|$)|'
    r'/go/|/app/|/uploads/|'
    r'/amp/|/author/|/feed/|'
    r'/page/|/tag/|'
    r'[?&]utm_[^=&]+='
    r'|[?&]s=',
    re.I
)

def _filter_sf_pages(df: pd.DataFrame,
                     url_col: Optional[str],
                     stat_col: Optional[str],
                     ct_col: Optional[str],
                     index_col: Optional[str]) -> pd.DataFrame:
    """Return only 2xx + HTML + indexable pages and remove system/utility URLs."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()

    if stat_col in out:
        code = pd.to_numeric(out[stat_col], errors="coerce")
        out = out[code.between(200, 299)]

    if ct_col in out:
        out = out[out[ct_col].astype(str).str.contains("text/html", case=False, na=False)]

    if index_col in out:
        out = out[out[index_col].astype(str).str.contains("indexable", case=False, na=False)]

    if url_col in out:
        out = out[~out[url_col].astype(str).str.contains(BAN_RE)]

    return _norm_cols(out)

# =============================================================================
# UI / MAIN
# =============================================================================

def render(show_header: bool = False) -> None:
    # ---- Header (tab-safe) ----
    if show_header:
        st.subheader("Screaming Frog")
    st.markdown(
        "Upload **Internal** (.xlsx/.csv) and optionally **All Inlinks** to analyze n-grams, "
        "on-page basics, freshness, and internal links."
    )

    # ---- Uploads ----
    c1, c2 = st.columns(2)
    with c1:
        up_internal = st.file_uploader(
            "Internal (required for N-grams, On-page & Freshness) — .xlsx / .csv",
            type=["xlsx", "csv"], key="SF_internal"
        )
    with c2:
        up_inlinks = st.file_uploader(
            "All Inlinks (optional) — .xlsx / .csv",
            type=["xlsx", "csv"], key="SF_inlinks"
        )

    # ---- Read uploads ----
    internal_tables = _open_any(up_internal)
    inlinks_tables  = _open_any(up_inlinks)

    df_int = _pick_table_by_name_or_largest(
        internal_tables, ["internal", "content", "crawl", "full crawl", "all pages"]
    ) if internal_tables else pd.DataFrame()
    df_inl = _pick_inlinks_table(inlinks_tables) if inlinks_tables else pd.DataFrame()

    df_int = _norm_cols(df_int)
    cols_int = list(df_int.columns)

    # Page columns (look up once)
    URL_COL   = _pick_col(cols_int, "Address", "URL", "Uri", "URI")
    WC_COL    = _pick_col(cols_int, "Word Count", "WordCount", "Words")
    STAT_COL  = _pick_col(cols_int, "Status Code", "Status", "Status Code 1")
    CT_COL    = _pick_col(cols_int, "Content Type", "Content-Type")
    INDEX_COL = _pick_col(cols_int, "Indexability", "Indexable")

    # Apply the hard filter once; reuse in N-grams + On-page + Freshness
    df_cur = _filter_sf_pages(df_int, URL_COL, STAT_COL, CT_COL, INDEX_COL)

    # =========================================================================
    # N-grams
    # =========================================================================
    st.subheader("N-grams")

    df_ng = df_cur.copy()
    TOP_BIGRAM_COLS = 4
    TOP_TRIGRAM_COLS = 2

    def _find_sf_ngram_cols(df, kind="bigram"):
        """
        Detect Screaming Frog custom JS extraction columns:
        accepts 'Bigram-EN 1', 'Bigrams EN 2', 'Trigrams EN 1', etc.
        Returns a list of column names sorted by their trailing number.
        """
        pat = re.compile(rf"^{kind}s?\s*[- ]?en\s*(\d+)$", re.I)
        hits = []
        for c in df.columns:
            m = pat.match(str(c).strip())
            if m:
                hits.append((int(m.group(1)), c))
        hits.sort(key=lambda x: x[0])
        return [c for _, c in hits]

    def _split_phrase_count(val):
        s = "" if pd.isna(val) else str(val)
        if not s:
            return "", np.nan
        parts = s.rsplit(":", 1)
        if len(parts) == 2:
            phrase = parts[0].strip()
            cnt = pd.to_numeric(parts[1], errors="coerce")
            return phrase, cnt
        return s.strip(), np.nan

    def _phrase_and_mentions(series):
        phrases, mentions = [], []
        for v in series.astype(str):
            p, c = _split_phrase_count(v)
            phrases.append(p)
            mentions.append(c)
        return pd.Series(phrases, index=series.index), pd.to_numeric(mentions, errors="coerce")

    def _safe_density(mentions, wc_series):
        m = pd.to_numeric(mentions, errors="coerce")
        w = pd.to_numeric(wc_series, errors="coerce")
        with np.errstate(divide="ignore", invalid="ignore"):
            return (m / w) * 100.0

    big_cols = _find_sf_ngram_cols(df_ng, "bigram")
    tri_cols = _find_sf_ngram_cols(df_ng, "trigram")

    if df_ng.empty or not URL_COL:
        st.info("No eligible pages for N-grams.")
    else:
        view_mode = st.radio("View", ["Site-wide", "Per page"], horizontal=True, key="ng_view")

        base = pd.DataFrame({
            "URL": df_ng[URL_COL].astype(str),
            "Word Count": pd.to_numeric(df_ng.get(WC_COL), errors="coerce") if (WC_COL in df_ng.columns) else np.nan
        })

        # -------------------- Site-wide --------------------
        if view_mode == "Site-wide":
            site_df = base.copy()

            # Bigram-EN 1..4
            for i in range(1, TOP_BIGRAM_COLS + 1):
                if i <= len(big_cols):
                    col = big_cols[i-1]
                    phr, men = _phrase_and_mentions(df_ng[col])
                    site_df[f"Bigram-EN {i}"] = phr
                    site_df[f"Bigram-EN {i} Mentions"] = men
                    site_df[f"Bigram-EN {i} Density %"] = _safe_density(men, site_df["Word Count"])
                else:
                    site_df[f"Bigram-EN {i}"] = ""
                    site_df[f"Bigram-EN {i} Mentions"] = np.nan
                    site_df[f"Bigram-EN {i} Density %"] = np.nan

            # Trigrams-EN 1..2
            for i in range(1, TOP_TRIGRAM_COLS + 1):
                if i <= len(tri_cols):
                    col = tri_cols[i-1]
                    phr, men = _phrase_and_mentions(df_ng[col])
                    site_df[f"Trigrams-EN {i}"] = phr
                    site_df[f"Trigrams-EN {i} Mentions"] = men
                    site_df[f"Trigrams-EN {i} Density %"] = _safe_density(men, site_df["Word Count"])
                else:
                    site_df[f"Trigrams-EN {i}"] = ""
                    site_df[f"Trigrams-EN {i} Mentions"] = np.nan
                    site_df[f"Trigrams-EN {i} Density %"] = np.nan

            order = ["URL", "Word Count"]
            for i in range(1, TOP_BIGRAM_COLS + 1):
                order += [f"Bigram-EN {i}", f"Bigram-EN {i} Mentions", f"Bigram-EN {i} Density %"]
            for i in range(1, TOP_TRIGRAM_COLS + 1):
                order += [f"Trigrams-EN {i}", f"Trigrams-EN {i} Mentions", f"Trigrams-EN {i} Density %"]

            colcfg = {"Word Count": st.column_config.NumberColumn("Word Count", format="%.0f")}
            for i in range(1, TOP_BIGRAM_COLS + 1):
                colcfg[f"Bigram-EN {i} Mentions"] = st.column_config.NumberColumn("Mentions")
                colcfg[f"Bigram-EN {i} Density %"] = st.column_config.NumberColumn("Density %", format="%.2f")
            for i in range(1, TOP_TRIGRAM_COLS + 1):
                colcfg[f"Trigrams-EN {i} Mentions"] = st.column_config.NumberColumn("Mentions")
                colcfg[f"Trigrams-EN {i} Density %"] = st.column_config.NumberColumn("Density %", format="%.2f")

            st.dataframe(site_df[order], use_container_width=True, hide_index=True, column_config=colcfg)
            st.download_button(
                "⬇️ CSV – N-grams (site-wide: top 4 bigrams + top 2 trigrams)",
                site_df[order].to_csv(index=False).encode("utf-8"),
                file_name="sf_ngrams_sitewide_top4_2.csv",
                mime="text/csv",
            )

        # -------------------- Per page (one table like your 2nd image) --------------------
        else:
            urls = df_ng[URL_COL].astype(str).dropna().unique().tolist()
            sel = st.selectbox("Select a URL", urls, index=0 if urls else None, key="ng_url")

            # how many rows to show side-by-side
            top_n = st.number_input("Top N", min_value=1, max_value=50, value=20, step=1, key="ng_topn")

            if sel:
                row = df_ng[df_ng[URL_COL].astype(str) == sel].iloc[0]
                wc_val = float(pd.to_numeric(row.get(WC_COL), errors="coerce")) if WC_COL in df_ng.columns else np.nan

                def _page_top_list(cols, k, label):
                    """Return a DataFrame of top-k phrases for this row: label, Mentions, Density %."""
                    recs = []
                    cap = min(k, len(cols))
                    for i in range(cap):
                        cell = row.get(cols[i], "")
                        p, c = _split_phrase_count(cell)
                        m = pd.to_numeric(c, errors="coerce")
                        d = (float(m) / wc_val * 100.0) if (pd.notna(m) and pd.notna(wc_val) and wc_val > 0) else np.nan
                        recs.append({label: p or "", "Mentions": m, "Density %": d})
                    return pd.DataFrame(recs)

                # build top lists
                b_df = _page_top_list(big_cols, top_n, "Bigrams")
                t_df = _page_top_list(tri_cols, top_n, "Trigrams")

                # combine them row-wise (ith bigram next to ith trigram)
                from itertools import zip_longest

                rows = []
                for b_row, t_row in zip_longest(b_df.to_dict("records"), t_df.to_dict("records"), fillvalue=None):
                    b = b_row or {"Bigrams": "", "Mentions": np.nan, "Density %": np.nan}
                    t = t_row or {"Trigrams": "", "Mentions": np.nan, "Density %": np.nan}
                    # row with duplicate headings Mentions/Density % (matches your layout)
                    rows.append([
                        wc_val, b["Bigrams"], b["Mentions"], b["Density %"],
                        t["Trigrams"], t["Mentions"], t["Density %"]
                    ])

                # use unique internal column names
                columns = [
                    "Word Count",
                    "Bigrams", "Bigrams Mentions", "Bigrams Density %",
                    "Trigrams", "Trigrams Mentions", "Trigrams Density %",
                ]
                page_df = pd.DataFrame(rows, columns=columns)

                st.dataframe(
                    page_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Word Count": st.column_config.NumberColumn("Word Count", format="%.0f"),
                        "Bigrams Mentions": st.column_config.NumberColumn("Mentions"),
                        "Bigrams Density %": st.column_config.NumberColumn("Density %", format="%.2f"),
                        "Trigrams Mentions": st.column_config.NumberColumn("Mentions"),
                        "Trigrams Density %": st.column_config.NumberColumn("Density %", format="%.2f"),
                    },
                )


                st.download_button(
                    "⬇️ CSV – Page N-grams (Top N, side-by-side)",
                    page_df.to_csv(index=False).encode("utf-8"),
                    file_name="sf_page_ngrams_side_by_side.csv",
                    mime="text/csv",
                )

    # =========================================================================
    # On-page Basics (runs on filtered pages)
    # =========================================================================
    if df_cur.empty:
        st.subheader("On-page Basics")
        st.info("No eligible (2xx, HTML, indexable) pages after filtering.")
    else:
        st.subheader("On-page Basics")
        df_pg = df_cur  # filtered set

        C_URL   = URL_COL
        C_TITLE = _pick_col(df_pg.columns, "Title 1", "Title")
        C_TLEN  = _pick_col(df_pg.columns, "Title 1 Length", "Title Length")
        C_MD    = _pick_col(df_pg.columns, "Meta Description 1", "Meta Description")
        C_MDL   = _pick_col(df_pg.columns, "Meta Description 1 Length", "Meta Description Length")

        st.markdown("**Recommended ranges**")
        col1, col2 = st.columns(2, gap="large")
        with col1:
            t_min, t_max = st.slider("Title length", min_value=0, max_value=120, value=(30, 65), step=1, key="SF_title_range")
        with col2:
            d_min, d_max = st.slider("Meta description length", min_value=0, max_value=320, value=(70, 160), step=1, key="SF_desc_range")

        t_len = pd.to_numeric(df_pg.get(C_TLEN), errors="coerce") if C_TLEN else df_pg.get(C_TITLE, "").astype(str).str.len()
        d_len = pd.to_numeric(df_pg.get(C_MDL), errors="coerce") if C_MDL else df_pg.get(C_MD, "").astype(str).str.len()

        def _simple(df, cols):
            return df[cols].dropna(how="all").head(300)

        # Titles
        if C_TITLE and C_URL:
            titles = df_pg[[C_URL, C_TITLE]].copy()
            titles["Length"] = t_len
            too_short_t = titles[titles["Length"] < t_min]
            too_long_t  = titles[titles["Length"] > t_max]

            st.markdown("**Meta Titles**")
            cta, ctb = st.columns(2)
            with cta:
                st.write("Too short")
                st.dataframe(_simple(too_short_t, [C_URL, C_TITLE, "Length"]).rename(columns={C_URL: "URL", C_TITLE: "Title"}),
                             use_container_width=True, hide_index=True)
            with ctb:
                st.write("Too long")
                st.dataframe(_simple(too_long_t, [C_URL, C_TITLE, "Length"]).rename(columns={C_URL: "URL", C_TITLE: "Title"}),
                             use_container_width=True, hide_index=True)
        else:
            st.info("Title columns not found (need Title and optionally Title Length).")

        # Meta descriptions
        if C_MD and C_URL:
            metas = df_pg[[C_URL, C_MD]].copy()
            metas["Length"] = d_len
            too_short_d = metas[metas["Length"] < d_min]
            too_long_d  = metas[metas["Length"] > d_max]

            st.markdown("**Meta Descriptions**")
            cdc, cdd = st.columns(2)
            with cdc:
                st.write("Too short")
                st.dataframe(_simple(too_short_d, [C_URL, C_MD, "Length"]).rename(columns={C_URL: "URL", C_MD: "Meta Description"}),
                             use_container_width=True, hide_index=True)
            with cdd:
                st.write("Too long")
                st.dataframe(_simple(too_long_d, [C_URL, C_MD, "Length"]).rename(columns={C_URL: "URL", C_MD: "Meta Description"}),
                             use_container_width=True, hide_index=True)
        else:
            st.info("Meta description columns not found.")

    # =========================================================================
    # Freshness & Quick Fix Candidates (runs on filtered pages)
    # =========================================================================
    st.subheader("Freshness & Quick Fix Candidates")

    if df_cur.empty:
        st.info("Upload your **Internal** export to use freshness, thin content, inlinks, and readability checks.")
    else:
        df_src = df_cur.copy()

        # Column lookups from filtered set
        URL_COL   = _pick_col(df_src.columns, "Address", "URL", "Uri", "URI")
        WC_COL    = _pick_col(df_src.columns, "Word Count", "WordCount", "Words")
        LM_COL    = _pick_col_exact(df_src.columns, "Last modified 1")  # your custom XPath column (EXACT)
        INL_UNIQ_COL = _pick_col(df_src.columns, "Unique Inlinks", "Unique Inlinks (Follow)", "Inlinks", "URL Inlinks")

        base = pd.DataFrame({
            "URL": df_src.get(URL_COL, "").astype(str) if URL_COL else "",
            "Word Count": pd.to_numeric(df_src.get(WC_COL), errors="coerce") if WC_COL else np.nan,
            "Last Modified": _as_tz_naive(df_src.get(LM_COL)) if LM_COL else pd.NaT,
        })

        # Unique inlinks (prefer Internal; else compute from All Inlinks)
        if INL_UNIQ_COL in df_src:
            base["Unique Inlinks"] = pd.to_numeric(df_src[INL_UNIQ_COL], errors="coerce")
        else:
            if not df_inl.empty:
                df_inl_norm = _norm_cols(df_inl.copy())
                c_from = _pick_col(df_inl_norm.columns, "From", "Source")
                c_to   = _pick_col(df_inl_norm.columns, "To", "Destination", "Destination Address", "Address")
                if c_from and c_to:
                    tmp = (df_inl_norm[[c_from, c_to]].dropna()
                           .drop_duplicates()
                           .groupby(c_to)[c_from].nunique()
                           .reset_index()
                           .rename(columns={c_to: "__dest", c_from: "__uin"}))
                    base = base.merge(tmp, left_on="URL", right_on="__dest", how="left")
                    base["Unique Inlinks"] = pd.to_numeric(base["__uin"], errors="coerce")
                    base.drop(columns=[c for c in ["__dest", "__uin"] if c in base], inplace=True)
                else:
                    base["Unique Inlinks"] = np.nan
            else:
                base["Unique Inlinks"] = np.nan

        # Readability (auto-detect)
        READ_EASE  = _pick_col(df_src.columns, "Flesch Reading Ease", "Reading Ease")
        READ_GRADE = _pick_col(df_src.columns, "Flesch-Kincaid Grade", "Flesch Kincaid Grade",
                               "Flesch-Kincaid Grade Level", "Grade Level", "Gunning Fog Score", "Readability")
        if READ_EASE in df_src:
            base["Readability Score"] = pd.to_numeric(df_src[READ_EASE], errors="coerce")  # higher = easier
            readability_rule = "ease"   # hard if <= threshold
            read_label = "Hard if Flesch Reading Ease ≤"
            read_default = 50.0
        elif READ_GRADE in df_src:
            base["Readability Score"] = pd.to_numeric(df_src[READ_GRADE], errors="coerce")  # higher = harder
            readability_rule = "grade"  # hard if >= threshold
            read_label = "Hard if Grade/Fog ≥"
            read_default = 10.0
        else:
            base["Readability Score"] = np.nan
            readability_rule, read_label, read_default = None, "Readability threshold", 50.0

        # Thresholds
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            stale_months = st.number_input("Stale if not updated ≥ (months)", 1, 60, 9, 1, key="SF_stale_months")
        with c2:
            wc_thresh = st.number_input("Thin if Word Count <", 0, 10000, 400, 50, key="SF_wc_thresh")
        with c3:
            inl_thresh = st.number_input("Low internal links if <", 0, 1000, 2, 1, key="SF_inl_thresh")
        with c4:
            read_thresh = st.number_input(read_label, 0.0, 100.0, float(read_default), 1.0, key="SF_read_thresh")

        # Stale pages
        now = pd.Timestamp.utcnow().tz_localize(None).normalize()
        delta_days = (now - base["Last Modified"]).dt.days
        base["Months Since Update"] = (delta_days / 30.44).round(1)

        stale = base[base["Last Modified"].notna() & (base["Months Since Update"] >= stale_months)]
        stale_view = (stale[["URL","Last Modified","Months Since Update","Word Count","Unique Inlinks","Readability Score"]]
                      .sort_values("Months Since Update", ascending=False).head(500))
        st.markdown("**Stale pages**")
        st.dataframe(stale_view, use_container_width=True, hide_index=True)
        st.download_button("⬇️ CSV – Stale pages",
                           stale_view.to_csv(index=False).encode("utf-8"),
                           file_name="sf_stale_pages.csv", mime="text/csv")

        # Thin content
        thin = base[(base["Word Count"].fillna(0) < wc_thresh)]
        thin_view = (thin[["URL","Word Count","Unique Inlinks","Readability Score","Last Modified"]]
                     .sort_values("Word Count", ascending=True).head(500))
        st.markdown("**Thin content**")
        st.dataframe(thin_view, use_container_width=True, hide_index=True)
        st.download_button("⬇️ CSV – Thin content",
                           thin_view.to_csv(index=False).encode("utf-8"),
                           file_name="sf_thin_content.csv", mime="text/csv")

        # Low unique inlinks
        low_inl = base[(base["Unique Inlinks"].fillna(0) < inl_thresh)]
        low_inl_view = (low_inl[["URL","Unique Inlinks","Word Count","Readability Score","Last Modified"]]
                        .sort_values("Unique Inlinks", ascending=True).head(500))
        st.markdown("**Low internal authority** (few unique inlinks)")
        st.dataframe(low_inl_view, use_container_width=True, hide_index=True)
        st.download_button("⬇️ CSV – Low unique inlinks",
                           low_inl_view.to_csv(index=False).encode("utf-8"),
                           file_name="sf_low_unique_inlinks.csv", mime="text/csv")

        # Hard to read
        if readability_rule is None:
            st.info("No readability metric detected (look for Flesch Reading Ease or Grade/Fog columns in Internal).")
        else:
            if readability_rule == "ease":
                hard = base[base["Readability Score"].fillna(1000) <= read_thresh]
                sort_asc = True    # lower ease = harder
            else:
                hard = base[base["Readability Score"].fillna(-1) >= read_thresh]
                sort_asc = False   # higher grade = harder

            hard_view = (hard[["URL","Readability Score","Word Count","Unique Inlinks","Last Modified"]]
                         .sort_values("Readability Score", ascending=sort_asc).head(500))
            st.markdown("**Hard to read**")
            st.dataframe(hard_view, use_container_width=True, hide_index=True)
            st.download_button("⬇️ CSV – Hard to read",
                               hard_view.to_csv(index=False).encode("utf-8"),
                               file_name="sf_hard_to_read.csv", mime="text/csv")

    # =========================================================================
    # Quick Wins – Keywords & Landing Pages vs Internal links
    # =========================================================================
    st.subheader("Keywords & Landing Pages vs Internal Links")

    if df_cur.empty:
        st.info("Upload your **Internal** export first to use this view.")
    else:
        try:
            cols = ["Address", "Word Count", "Inlinks", "Unique Inlinks", "Impressions", "CTR", "Position"]
            sg_view = df_cur[cols].copy()

            # Coerce numerics
            for c in ["Word Count", "Inlinks", "Unique Inlinks", "Impressions", "CTR", "Position"]:
                sg_view[c] = pd.to_numeric(sg_view[c], errors="coerce")

            # ✅ Only show Position 11–20
            qw = sg_view[(sg_view["Position"] >= 11) & (sg_view["Position"] <= 20)].copy()

            # Sort by impressions (high → low)
            qw = qw.sort_values(["Impressions"], ascending=[False])

            st.dataframe(
                qw,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Word Count": st.column_config.NumberColumn(format="%.0f"),
                    "Inlinks": st.column_config.NumberColumn(format="%.0f"),
                    "Unique Inlinks": st.column_config.NumberColumn(format="%.0f"),
                    "Impressions": st.column_config.NumberColumn(format="%.0f"),
                    "CTR": st.column_config.NumberColumn(format="%.3f"),
                    "Position": st.column_config.NumberColumn(format="%.2f"),
                },
            )

            st.download_button(
                "⬇️ CSV — Quick Wins (Position 11–20)",
                qw.to_csv(index=False).encode("utf-8"),
                file_name="quick_wins_pos_11_20.csv",
                mime="text/csv",
            )
        except KeyError as e:
            st.error(f"One or more required columns not found in Internal: {e}")




    # =========================================================================
    # Internal Links (simple) — destination auto-filtered (no system URLs)
    # =========================================================================
    st.subheader("Internal Links")
    if df_inl.empty:
        st.info("Upload **All Inlinks** (.xlsx / .csv) to see internal links.")
    else:
        df_inl = _norm_cols(df_inl)
        c_from   = _pick_col(df_inl.columns, "From", "Source")
        c_to     = _pick_col(df_inl.columns, "To", "Destination", "Destination Address", "Address")
        c_anchor = _pick_col(df_inl.columns, "Anchor", "Link Text", "Alt Text")
        c_stat   = _pick_col(df_inl.columns, "Status Code", "Status")
        c_pos    = _pick_col(df_inl.columns, "Link Position", "Position")

        if not c_from or not c_to:
            st.info("Couldn’t detect required columns in All Inlinks (need Source/From and Destination/To).")
            return

        # keep only "content" positions (when available)
        if c_pos:
            df_inl = df_inl[df_inl[c_pos].astype(str).str.contains(r"\bcontent\b", case=False, na=False)]

        # Auto-drop system/utility destinations (same BAN_RE + /app/themes/)
        dest = df_inl[c_to].astype(str)
        drop_dest_re = re.compile(rf'(?:{BAN_RE.pattern})|/app/themes/', re.I)
        mask_keep = ~dest.str.contains(drop_dest_re)
        df_inl = df_inl[mask_keep]

        base_links = pd.DataFrame({
            "Source": df_inl[c_from].astype(str),
            "Status Code": pd.to_numeric(df_inl[c_stat], errors="coerce") if c_stat else np.nan,
            "Destination": df_inl[c_to].astype(str),
            "Anchor": df_inl[c_anchor].astype(str) if c_anchor else "",
        })

        left, right = st.columns(2)
        with left:
            status_values = sorted([int(x) for x in pd.to_numeric(base_links["Status Code"], errors="coerce").dropna().unique()])
            pick_status = st.multiselect("Status code (optional)", status_values, default=[], key="SF_link_status")
        with right:
            dest_list = sorted(base_links["Destination"].dropna().unique().tolist())
            sel_dest = st.selectbox("Destination page (optional)", ["— All —"] + dest_list, index=0, key="SF_link_dest")

        filt = base_links.copy()
        if pick_status:
            filt = filt[filt["Status Code"].isin(pick_status)]
        if sel_dest != "— All —":
            filt = filt[filt["Destination"] == sel_dest]

        st.dataframe(filt[["Source", "Status Code", "Destination", "Anchor"]],
                     use_container_width=True, hide_index=True)

        # --- CSV download for Internal Links (respects current filters) ---
        # Build a friendly file name from the current filters
        fname = "sf_internal_links.csv"
        if sel_dest != "— All —":
            safe_dest = re.sub(r"[^a-z0-9]+", "-", sel_dest.lower()).strip("-")[:60]
            fname = f"sf_internal_links_to_{safe_dest}.csv"
        if pick_status:
            fname = fname.replace(".csv", f"_status_{'-'.join(map(str, pick_status))}.csv")

        # Encode filtered table and render button
        csv_bytes = filt[["Source", "Status Code", "Destination", "Anchor"]].to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ CSV – Internal links (current filters)",
            csv_bytes,
            file_name=fname,
            mime="text/csv",
        )