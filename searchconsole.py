#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Search Console Analyzer (CSV/XLSX/ZIP uploads only)
---------------------------------------------------
`render()` can be used inside launcher.py as a tab, or run standalone:
    streamlit run searchconsole.py

Data sources:
- File upload (.xlsx / .zip of CSVs / .csv)

Panels:
- Low-performing pages
- Mid-pack headroom
- Losses (via optional baseline upload)
- FAQ fuel
- Cannibalization (exact if page+query present; heuristic otherwise)
"""

from __future__ import annotations

import io
import zipfile
import re
from pathlib import Path
from urllib.parse import urlparse
from datetime import datetime, date, timedelta

import numpy as np
import pandas as pd
import streamlit as st

# ======================================================================
#                          FILE / CSV HELPERS
# ======================================================================

def _clean_header(h: str) -> str:
    return re.sub(r"\s+", " ", str(h or "")).strip()

def _read_csv_guess(buf) -> pd.DataFrame:
    """Read CSV, try utf-8 then utf-16 TSV fallback (for Screaming Frog style exports)."""
    try:
        return pd.read_csv(buf)
    except Exception:
        try:
            buf.seek(0)
            return pd.read_csv(buf, encoding="utf-16", sep="\t")
        except Exception:
            return pd.DataFrame()

def _open_xlsx_all(file) -> dict:
    xls = pd.ExcelFile(file)
    return {name: pd.read_excel(xls, sheet_name=name) for name in xls.sheet_names}

def _open_zip_all(file) -> dict:
    zf = zipfile.ZipFile(file)
    out = {}
    for n in zf.namelist():
        if n.lower().endswith(".csv"):
            with zf.open(n) as f:
                out[n] = _read_csv_guess(io.BytesIO(f.read()))
    return out

def _open_any_all(upload) -> dict:
    if upload is None:
        return {}
    nm = (upload.name or "").lower()
    if nm.endswith(".xlsx"):
        return _open_xlsx_all(upload)
    if nm.endswith(".zip"):
        return _open_zip_all(upload)
    if nm.endswith(".csv"):
        return {upload.name: _read_csv_guess(upload)}
    try:
        return {upload.name: _read_csv_guess(upload)}
    except Exception:
        return {}

from datetime import timedelta

def _prev_period(start: date, end: date):
    days = (end - start).days + 1
    base_end = start - timedelta(days=1)
    base_start = base_end - timedelta(days=days-1)
    return base_start, base_end

# ======================================================================
#                       NORMALIZATION & AGGREGATION
# ======================================================================

METRIC_TAIL = re.compile(r"(?i)(clicks|impressions|ctr|position)\s*$")

def _extract_latest_metric_cols(df: pd.DataFrame) -> dict:
    """Pick the right-most occurrence of metric names (supports Compare headers)."""
    cols = list(map(str, df.columns))
    low = [c.lower().strip() for c in cols]

    def pick_latest(name: str):
        idxs = [i for i, c in enumerate(low) if re.search(rf"(^|\b){name}\s*$", c)]
        if idxs:
            return cols[max(idxs)]
        tail_idxs = [i for i, c in enumerate(low) if METRIC_TAIL.search(c) and name in c]
        return cols[max(tail_idxs)] if tail_idxs else None

    clicks = impressions = ctr = position = None
    for i in reversed(range(len(cols))):
        c = low[i]
        if clicks is None and re.search(r"(?i)\bclicks\b", c): clicks = cols[i]
        if impressions is None and re.search(r"(?i)\bimpressions\b", c): impressions = cols[i]
        if ctr is None and (re.search(r"(?i)\bctr\b", c) or re.search(r"(?i)ctr \(.*\)", c)): ctr = cols[i]
        if position is None and re.search(r"(?i)\bposition\b", c): position = cols[i]

    return {
        "clicks": clicks or pick_latest("clicks"),
        "impressions": impressions or pick_latest("impressions"),
        "ctr": ctr or pick_latest("ctr"),
        "position": position or pick_latest("position"),
    }

def _normalize_one(df_in: pd.DataFrame, label_hint: str) -> pd.DataFrame:
    """
    Normalize one raw sheet/CSV to common schema:
    [page, query, country, device, date, clicks, impressions, ctr_pct, position]
    """
    if df_in is None or df_in.empty:
        return pd.DataFrame()

    df = df_in.copy()
    df.columns = [_clean_header(c) for c in df.columns]
    df = df.dropna(how="all")

    # Fix first column headers frequently used in GSC exports
    if len(df.columns):
        first = df.columns[0]
        fl = str(first).lower()
        if fl.startswith("top queries"): df.rename(columns={first: "Query"}, inplace=True)
        elif fl.startswith("top pages"): df.rename(columns={first: "Page"}, inplace=True)
        elif fl.startswith("unnamed"):
            if "quer" in label_hint.lower(): df.rename(columns={first: "Query"}, inplace=True)
            if "page" in label_hint.lower(): df.rename(columns={first: "Page"}, inplace=True)

    low = {str(c).lower(): c for c in df.columns}
    c_page    = low.get("page") or low.get("url") or low.get("landing page") or low.get("top pages")
    c_query   = low.get("query") or low.get("search query") or low.get("top queries")
    c_country = low.get("country")
    c_device  = low.get("device")
    c_date    = low.get("date")

    met = _extract_latest_metric_cols(df)

    out = pd.DataFrame(index=df.index)
    if c_page is not None and c_page in df:       out["page"]    = df[c_page].astype(str)
    if c_query is not None and c_query in df:     out["query"]   = df[c_query].astype(str)
    if c_country is not None and c_country in df: out["country"] = df[c_country].astype(str)
    if c_device is not None and c_device in df:   out["device"]  = df[c_device].astype(str)
    if c_date is not None and c_date in df:       out["date"]    = pd.to_datetime(df[c_date], errors="coerce")

    if "date" in out.columns:
        # If values look like Excel serials (e.g., 45875), convert from 1899-12-30
        ser = out["date"]
        if pd.api.types.is_numeric_dtype(ser) and ser.dropna().between(35000, 60000).all():
            out["date"] = pd.to_datetime(ser, unit="D", origin="1899-12-30", errors="coerce")


    if met.get("clicks") and met["clicks"] in df: out["clicks"] = pd.to_numeric(df[met["clicks"]], errors="coerce")
    if met.get("impressions") and met["impressions"] in df: out["impressions"] = pd.to_numeric(df[met["impressions"]], errors="coerce")

    if met.get("ctr") and met["ctr"] in df:
        ctr_raw = (df[met["ctr"]]
                   .astype(str)
                   .str.replace("%","",regex=False)
                   .str.replace(",",".",regex=False))
        out["ctr_pct"] = pd.to_numeric(ctr_raw, errors="coerce")
    else:
        out["ctr_pct"] = np.nan

    if met.get("position") and met["position"] in df:
        pos_raw = df[met["position"]].astype(str).str.replace(",",".",regex=False)
        out["position"] = pd.to_numeric(pos_raw, errors="coerce")
    else:
        out["position"] = np.nan

    # >>> ADD THIS BLOCK (normalize 0–1 fractions to 0–100 %) <<<
    if "ctr_pct" in out.columns:
        vals = out["ctr_pct"].dropna()
        # If CTR looks like fractions (e.g., 0.69 = 69%), scale to percent
        if not vals.empty and vals.quantile(0.90) <= 1.0 and vals.max() <= 5.0:
            out["ctr_pct"] = out["ctr_pct"] * 100
    # <<< END ADD >>>

    # Fix CTR if missing/invalid
    if "clicks" in out and "impressions" in out:
        denom = out["impressions"].where(out["impressions"] > 0)
        recomputed = (out["clicks"] / denom) * 100
        bad = out["ctr_pct"].isna() | (out["ctr_pct"] < 0) | (out["ctr_pct"] > 100)
        out.loc[bad, "ctr_pct"] = recomputed[bad]

    # Drop fully empty metric rows
    if {"clicks","impressions","ctr_pct","position"}.issubset(out.columns):
        mask_all_nan = out[["clicks","impressions","ctr_pct","position"]].isna().all(axis=1)
        out = out[~mask_all_nan]

    return out.reset_index(drop=True)

def _normalize_all(raw_map: dict) -> dict:
    """Normalize uploads and classify by CONTENT (has 'page'/'query') not just sheet name."""
    if not raw_map:
        return {}

    # include pickers
    norm = {
        "pages": pd.DataFrame(),
        "queries": pd.DataFrame(),
        "countries": pd.DataFrame(),
        "devices": pd.DataFrame(),
    }
    # REQUIRED: used to keep any extra sheets
    others = {}

    for name, df in raw_map.items():
        cleaned = _normalize_one(df, name)
        if cleaned.empty:
            continue

        # collect picker values from ANY sheet that has these columns
        if "country" in cleaned.columns:
            norm["countries"] = pd.concat([norm["countries"], cleaned[["country"]]], ignore_index=True)
        if "device" in cleaned.columns:
            norm["devices"] = pd.concat([norm["devices"], cleaned[["device"]]], ignore_index=True)

        has_page  = "page"  in cleaned.columns
        has_query = "query" in cleaned.columns

        lname = str(name).lower()
        # keep original behavior when sheet/file name already hints type
        if "pages" in lname and has_page:
            norm["pages"] = pd.concat([norm["pages"], cleaned], ignore_index=True)
        elif "queries" in lname and has_query:
            norm["queries"] = pd.concat([norm["queries"], cleaned], ignore_index=True)
        else:
            # classify by columns
            if has_page and not has_query:
                norm["pages"] = pd.concat([norm["pages"], cleaned], ignore_index=True)
            elif has_query and not has_page:
                norm["queries"] = pd.concat([norm["queries"], cleaned], ignore_index=True)
            elif has_page and has_query:
                # single sheet (like SEO Tools) has both → feed both views
                norm["pages"]   = pd.concat([norm["pages"],   cleaned], ignore_index=True)
                norm["queries"] = pd.concat([norm["queries"], cleaned], ignore_index=True)
            else:
                others[f"other:{name}"] = cleaned

    # attach any extras (countries/devices) if they exist
    norm.update(others)

    # de-dupe picker values
    if not norm["countries"].empty:
        norm["countries"] = norm["countries"].dropna().drop_duplicates()
    if not norm["devices"].empty:
        norm["devices"] = norm["devices"].dropna().drop_duplicates()

    return norm

def find_query_page_table(parts: dict) -> pd.DataFrame:
    """Return the first table that has BOTH 'query' and 'page'."""
    if "queries" in parts:
        df = parts["queries"]
        if isinstance(df, pd.DataFrame) and {"query","page"}.issubset(df.columns):
            return df
    for _, df in parts.items():
        if isinstance(df, pd.DataFrame) and {"query","page"}.issubset(df.columns):
            return df
    return pd.DataFrame()

def _apply_filters(df: pd.DataFrame, country_sel, device_sel) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    if country_sel and "country" in out.columns:
        out = out[out["country"].isin(country_sel)]
    if device_sel and "device" in out.columns:
        out = out[out["device"].isin(device_sel)]
    return out

def _agg_pages(df):
    if df is None or df.empty or "page" not in df.columns:
        return pd.DataFrame(columns=["page","clicks","impressions","ctr_pct","position"])
    return df.groupby("page", dropna=False, as_index=False).agg(
        clicks=("clicks","sum"),
        impressions=("impressions","sum"),
        ctr_pct=("ctr_pct","mean"),
        position=("position","mean"),
    )

def _agg_queries(df):
    if df is None or df.empty or "query" not in df.columns:
        return pd.DataFrame(columns=["query","page","clicks","impressions","ctr_pct","position"])
    keys = ["query"] + (["page"] if "page" in df.columns else [])
    return df.groupby(keys, dropna=False, as_index=False).agg(
        clicks=("clicks","sum"),
        impressions=("impressions","sum"),
        ctr_pct=("ctr_pct","mean"),
        position=("position","mean"),
    )

# ======================================================================
#                 CANNIBALIZATION (heuristic fallback)
# ======================================================================

_STOP = set("""
a an and are as at be but by for from how in into is it its of on or that the this to what when where which who why with your you
""".split())
_token_re = re.compile(r"[a-z0-9]+")

def _tokens_from_query(q: str) -> set:
    toks = _token_re.findall(str(q).lower())
    return {t for t in toks if len(t) > 2 and t not in _STOP}

def _tokens_from_url(url: str) -> set:
    try:
        path = urlparse(str(url)).path.lower()
    except Exception:
        path = str(url).lower()
    toks = _token_re.findall(path)
    return {t for t in toks if len(t) > 2 and t not in _STOP}

def _approximate_cannibalization(pages_cur: pd.DataFrame,
                                 queries_cur: pd.DataFrame,
                                 min_impr: int,
                                 threshold: float = 0.5,
                                 max_pages: int = 3) -> pd.DataFrame:
    """Build (query, page) pairs by token overlap between query tokens and page URL tokens."""
    if pages_cur.empty or queries_cur.empty:
        return pd.DataFrame()

    pc = pages_cur.loc[pages_cur.get("impressions", 0) >= min_impr, ["page","clicks","impressions","ctr_pct","position"]].copy()
    qc = queries_cur.loc[queries_cur.get("impressions", 0) >= min_impr, ["query","clicks","impressions","ctr_pct","position"]].copy()
    if pc.empty or qc.empty:
        return pd.DataFrame()

    pc["__tok"] = pc["page"].map(_tokens_from_url)
    token_to_pages = {}
    for i, row in pc.iterrows():
        for t in row["__tok"]:
            token_to_pages.setdefault(t, set()).add(i)

    rows = []
    for _, qrow in qc.iterrows():
        q = qrow["query"]
        q_toks = _tokens_from_query(q)
        if not q_toks:
            continue
        candidate_idx = set()
        for t in q_toks:
            if t in token_to_pages:
                candidate_idx |= token_to_pages[t]
        if not candidate_idx:
            continue
        scored = []
        for idx in candidate_idx:
            p_toks = pc.at[idx, "__tok"]
            shared = len(q_toks & p_toks)
            ratio = shared / max(1, len(q_toks))
            if ratio >= threshold:
                scored.append((ratio, idx))
        if not scored:
            continue
        scored.sort(reverse=True)
        for ratio, idx in scored[:max_pages]:
            prow = pc.loc[idx]
            rows.append({
                "query": q,
                "query_impressions": qrow.get("impressions", np.nan),
                "query_clicks": qrow.get("clicks", np.nan),
                "page": prow["page"],
                "page_impressions": prow.get("impressions", np.nan),
                "page_clicks": prow.get("clicks", np.nan),
                "overlap_ratio": round(float(ratio), 3),
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    multi = df.groupby("query")["page"].nunique().reset_index(name="page_variants")
    df = df.merge(multi, on="query", how="inner")
    df = df[df["page_variants"] >= 2].copy()

    return df.sort_values(
        by=["page_variants", "query_impressions", "overlap_ratio"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

# ======================================================================
#                               UI / MAIN
# ======================================================================

def render(show_header: bool = True):
    # (Header text removed to keep UI exactly like the file-upload mode.)

    # ---------------------- File Upload Source ----------------------
    c1, c2 = st.columns([2, 2])
    with c1:
        primary = st.file_uploader("Primary export (.xlsx / .zip / .csv)", type=["xlsx", "zip", "csv"], key="SC_ALL_primary")
    with c2:
        baseline = st.file_uploader("Optional baseline (.xlsx / .zip / .csv)", type=["xlsx", "zip", "csv"], key="SC_ALL_baseline")

    raw_cur  = _open_any_all(primary)
    raw_base = _open_any_all(baseline)

    cur_parts  = _normalize_all(raw_cur)
    base_parts = _normalize_all(raw_base)

    # File-upload-only “Filters & thresholds” (includes country/device multiselect)
    st.markdown("### 2) Filters & thresholds")
    country_opts = sorted(cur_parts.get("countries", pd.DataFrame()).get("country", pd.Series(dtype=str)).dropna().unique().tolist())
    device_opts  = sorted(cur_parts.get("devices",   pd.DataFrame()).get("device",  pd.Series(dtype=str)).dropna().unique().tolist())

    cA, cB, cC = st.columns(3)
    with cA:
        min_impr = st.number_input("Min impressions (filter)", 0, 10_000_000, 50, 10, key="SC_min_impr_file")
    with cB:
        low_ctr_max = st.number_input("Low-performing: CTR ≤ (%)", 0.0, 100.0, 1.0, 0.1, key="SC_low_ctr_file")
    with cC:
        pos_mid_min, pos_mid_max = st.slider("Mid pack window (11–20)", 1, 100, (11, 20), key="SC_mid_window_file")

    cD, cE = st.columns(2)
    with cD:
        country_sel = st.multiselect("Country (from Countries sheet)", country_opts, key="SC_country_sel_file")
    with cE:
        device_sel  = st.multiselect("Device (from Devices sheet)", device_opts, key="SC_device_sel_file")

    # Apply only for file uploads
    def _apply_for_upload(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty: return df
        out = df.copy()
        if country_sel and "country" in out.columns: out = out[out["country"].isin(country_sel)]
        if device_sel and "device"  in out.columns: out = out[out["device"].isin(device_sel)]
        return out

    if cur_parts:
        cur_parts["pages"]   = _apply_for_upload(cur_parts.get("pages", pd.DataFrame()))
        cur_parts["queries"] = _apply_for_upload(cur_parts.get("queries", pd.DataFrame()))

    if base_parts:
        base_parts["pages"] = _apply_for_upload(base_parts.get("pages", pd.DataFrame()))

    # ✅ If nothing loaded (yet), keep the tab alive and exit cleanly
    if not cur_parts:
        st.info("Upload a GSC Excel/ZIP/CSV (or Screaming Frog GSC CSV). I’ll read all sheets/files at once.")
        return

    loaded_badges = " · ".join(sorted(cur_parts.keys()))
    st.markdown(f"**Loaded parts:** {loaded_badges if loaded_badges else '—'}")

    # ---------------------- Aggregations ----------------------
    pages_df   = cur_parts.get("pages",   pd.DataFrame())
    queries_df = cur_parts.get("queries", pd.DataFrame())

    pages_cur   = _agg_pages(pages_df)
    queries_cur = _agg_queries(queries_df)

    # ---------------------- Low-performing pages ----------------------
    st.markdown("### Low-performing pages (high impressions, low CTR)")
    if pages_cur.empty:
        st.info("No page-level data found.")
    low = pages_cur[(pages_cur["impressions"] >= min_impr) & (pages_cur["ctr_pct"].fillna(0) <= low_ctr_max)].copy()
    low = low.sort_values(["impressions","ctr_pct"], ascending=[False, True])
    st.dataframe(low.head(50), use_container_width=True, hide_index=True)
    st.download_button("⬇️ CSV – Low-performing pages", data=low.to_csv(index=False).encode("utf-8"),
                       file_name="gsc_low_performing_pages.csv", mime="text/csv")

    # ---------------------- Mid-pack headroom ----------------------
    st.markdown("### Mid-pack (pos 11–20) with headroom")
    mid = pages_cur.copy()
    if not mid.empty:
        # Strip only the URL fragment (keep query params as-is)
        mid["page"] = mid["page"].astype(str).str.split("#", n=1).str[0].str.split("?", n=1).str[0]
        # Re-aggregate after normalization
        mid = (mid.groupby("page", as_index=False, dropna=False)
                  .agg(clicks=("clicks", "sum"),
                       impressions=("impressions", "sum"),
                       ctr_pct=("ctr_pct", "mean"),
                       position=("position", "mean")))
    if not mid.empty and "position" in mid.columns:
        mid = mid[mid["position"].between(pos_mid_min, pos_mid_max, inclusive="both")]
        mid = mid[mid["impressions"] >= min_impr]
        if not mid.empty:
            q25 = max(1.0, mid["clicks"].quantile(0.25))
            mid = mid[mid["clicks"] <= q25]
            def exp_ctr(p):
                if pd.isna(p): return np.nan
                if p<=3: return 20.0
                if p<=5: return 12.0
                if p<=10: return 6.0
                if p<=20: return 2.5
                if p<=30: return 1.5
                return 1.0
            mid["expected_ctr_pct"] = mid["position"].apply(exp_ctr)
            mid["ctr_uplift_pts"] = (mid["expected_ctr_pct"] - mid["ctr_pct"]).round(2)
            mid["extra_clicks_if_hit_expected"] = ((mid["expected_ctr_pct"]/100.0)*mid["impressions"] - mid["clicks"]).round(0)
            mid = mid.sort_values(["extra_clicks_if_hit_expected","impressions"], ascending=[False, False])
    st.dataframe(mid.head(50), use_container_width=True, hide_index=True)
    st.download_button("⬇️ CSV – Mid-pack headroom", data=mid.to_csv(index=False).encode("utf-8"),
                       file_name="gsc_midpack_headroom.csv", mime="text/csv")

    # ---------------------- Quick-win extras ----------------------
    st.markdown("### Quick-win extras")

    # ---- Brand / Non-brand segmentation (affects only the panels below) ----
    b1, b2 = st.columns([2, 1])
    with b1:
        brand_terms_raw = st.text_input(
            "Brand terms (comma-separated, regex-aware; leave blank for no brand split)",
            value=st.session_state.get("SC_brand_terms", ""),
            key="SC_brand_terms"
        ).strip()

    with b2:
        seg_choice = st.radio("Segment", ["All", "Brand only", "Non-brand"], horizontal=True, key="SC_brand_segment")

    def _build_brand_regex(raw: str) -> str | None:
        if not raw:
            return None
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if not parts:
            return None
        # Regex: word-ish boundary when possible, but allow spaces/phrases
        escaped = [re.escape(p) for p in parts]
        return r"(?i)\b(?:%s)\b" % "|".join(escaped)

    brand_regex = _build_brand_regex(brand_terms_raw)

    def _apply_brand_segment(df: pd.DataFrame, seg: str, regex: str | None) -> pd.DataFrame:
        if df is None or df.empty or "query" not in df.columns or not regex or seg == "All":
            return df
        mask = df["query"].astype(str).str.contains(regex, na=False)
        return df[mask] if seg == "Brand only" else df[~mask]

    # We'll use the raw frames for new panels (no need to re-aggregate for CTR/pos unless specified)
    qs_raw = _apply_brand_segment(queries_df.copy(), seg_choice, brand_regex)
    pg_raw = _apply_brand_segment(pages_df.copy(), seg_choice, brand_regex)

    # Helper: collapse to one row per query, keeping a representative page
    def _collapse_query_level(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty or "query" not in df.columns:
            return pd.DataFrame(columns=["query","page","clicks","impressions","ctr_pct","position"])
        g = df.copy()
        g["__w"] = g.get("impressions", 0).fillna(0)
        # Sums per query
        agg = (g.groupby("query", as_index=False)
                 .agg(clicks=("clicks", "sum"),
                      impressions=("impressions", "sum")))
        # Weighted avg position by impressions (if present)
        if "position" in g.columns:
            pos = (g.groupby("query")
                     .apply(lambda d: float(np.average(d["position"].fillna(0), weights=(d["__w"] + 1e-9))))
                     .reset_index(name="position"))
            agg = agg.merge(pos, on="query", how="left")
        else:
            agg["position"] = np.nan
        # CTR from sums
        agg["ctr_pct"] = (agg["clicks"] / agg["impressions"].replace(0, np.nan)) * 100
        # Representative page (highest impressions within that query)
        if "page" in g.columns:
            rep = (g.sort_values(["query","impressions"], ascending=[True, False])
                     .groupby("query")
                     .head(1))[["query","page"]]
            agg = agg.merge(rep, on="query", how="left")
        else:
            agg["page"] = np.nan
        return agg

    # ---- A) Featured snippet / PAA opportunities (query-level) ----
    st.markdown("#### Featured snippet / PAA opportunities")
    qs_q = _collapse_query_level(qs_raw)  # per-query metrics + representative page
    if qs_q.empty:
        st.info("No query data available for featured snippet opportunities.")
    else:
        # Filter to question-like queries with decent impressions and rank just shy of #1
        def _exp_ctr(p):
            if pd.isna(p):  return np.nan
            if p <= 1:      return 28.0
            if p <= 2:      return 20.0
            if p <= 3:      return 12.0
            if p <= 5:      return 9.0
            if p <= 10:     return 5.0
            return 2.0

        cand = qs_q.copy()
        cand["is_question"] = cand["query"].str.match(r"(?i)^(who|what|why|how|when|where)\b", na=False)
        cand = cand[
            (cand["is_question"]) &
            (cand["impressions"] >= min_impr) &
            (cand["position"].between(2, 5, inclusive="both"))
        ]
        if cand.empty:
            st.info("No near-snippet questions found given current thresholds and segment.")
        else:
            cand["expected_ctr_pct"] = cand["position"].apply(_exp_ctr)
            cand["ctr_uplift_pts"] = (cand["expected_ctr_pct"] - cand["ctr_pct"]).round(2)
            cand["extra_clicks_if_hit_expected"] = (
                (cand["expected_ctr_pct"]/100.0)*cand["impressions"] - cand["clicks"]
            ).round(0)
            out = (cand.sort_values(["extra_clicks_if_hit_expected","impressions"], ascending=[False, False])
                        .head(50))
            cols = ["query","page","impressions","position","ctr_pct","expected_ctr_pct","ctr_uplift_pts","extra_clicks_if_hit_expected"]
            st.dataframe(out[cols], use_container_width=True, hide_index=True)
            st.download_button("⬇️ CSV – Featured snippet opportunities",
                               data=out[cols].to_csv(index=False).encode("utf-8"),
                               file_name="gsc_snippet_opportunities.csv", mime="text/csv")

    # ---- B) Volatility finder (ranking swings per page+query, requires daily rows) ----
    st.markdown("#### Volatility (ranking swings)")
    if "date" not in pg_raw.columns:
        st.caption("Volatility requires daily rows (with a 'date' column).")
    else:
        g = pg_raw.dropna(subset=["date"]).copy()
        if g.empty:
            st.info("No dated rows found for volatility.")
        else:
            vol = (g.groupby(["page","query"])
                     .agg(impressions=("impressions","sum"),
                          clicks=("clicks","sum"),
                          avg_pos=("position","mean"),
                          pos_std=("position","std"))
                     .reset_index())
            vol["pos_std"] = vol["pos_std"].fillna(0.0)
            vol = vol[(vol["impressions"] >= min_impr) & (vol["pos_std"] >= 2.0)]
            if vol.empty:
                st.info("No volatile page+query pairs found with current thresholds and segment.")
            else:
                vol = vol.sort_values(["pos_std","impressions"], ascending=[False, False]).head(50)
                st.dataframe(vol, use_container_width=True, hide_index=True)
                st.download_button("⬇️ CSV – Volatile pairs",
                                   data=vol.to_csv(index=False).encode("utf-8"),
                                   file_name="gsc_volatility.csv", mime="text/csv")

    # ---------------------- Losses (Clicks & Impressions) ----------------------
    st.markdown("### Losses (Clicks & Impressions)")
    base_pages_df = base_parts.get("pages", pd.DataFrame())
    if isinstance(base_pages_df, pd.DataFrame) and not base_pages_df.empty and not pages_df.empty:
        cur_p  = _agg_pages(pages_df)
        base_p = _agg_pages(base_pages_df)
        merged = cur_p.merge(base_p, on="page", how="outer", suffixes=("_curr","_prev")).fillna(0)

        merged["Δ_clicks"] = (merged["clicks_curr"] - merged["clicks_prev"]).round(0)
        merged["Δ_impr"]   = (merged["impressions_curr"] - merged["impressions_prev"]).round(0)

        lost_clicks = merged.sort_values("Δ_clicks").head(50).copy()
        lost_clicks_view = lost_clicks.rename(columns={
            "clicks_prev":       "Clicks (prev)",
            "clicks_curr":       "Clicks (curr)",
            "Δ_clicks":          "Clicks lost",
            "impressions_prev":  "Impr (prev)",
            "impressions_curr":  "Impr (curr)",
        })[["page","Clicks (prev)","Clicks (curr)","Clicks lost","Impr (prev)","Impr (curr)"]]
        st.write("**Top pages that lost clicks**")
        st.dataframe(lost_clicks_view, use_container_width=True, hide_index=True)
        st.download_button("⬇️ CSV – Lost Clicks",
            data=lost_clicks_view.to_csv(index=False).encode("utf-8"),
            file_name="gsc_lost_clicks.csv", mime="text/csv")

        lost_impr = merged.sort_values("Δ_impr").head(50).copy()
        lost_impr_view = lost_impr.rename(columns={
            "impressions_prev":  "Impr (prev)",
            "impressions_curr":  "Impr (curr)",
            "Δ_impr":            "Impr lost",
            "clicks_prev":       "Clicks (prev)",
            "clicks_curr":       "Clicks (curr)",
        })[["page","Impr (prev)","Impr (curr)","Impr lost","Clicks (prev)","Clicks (curr)"]]
        st.write("**Top pages that lost impressions**")
        st.dataframe(lost_impr_view, use_container_width=True, hide_index=True)
        st.download_button("⬇️ CSV – Lost Impressions",
            data=lost_impr_view.to_csv(index=False).encode("utf-8"),
            file_name="gsc_lost_impressions.csv", mime="text/csv")
    else:
        st.info("Provide a baseline file to enable the Losses tables (upload an optional baseline).")

    # ---------------------- FAQ fuel ----------------------
    st.markdown("### FAQ fuel – Who/What/Why/How queries")
    if not queries_cur.empty and "query" in queries_cur.columns:
        faq = queries_cur[queries_cur["query"].str.match(
            r"(?i)(who|what|why|how|where|when|which).*",
            na=False
        )]
        faq = faq[faq["impressions"] >= min_impr].sort_values(["impressions","ctr_pct"], ascending=[False, True])
        st.dataframe(faq.head(100), use_container_width=True, hide_index=True)
        st.download_button("⬇️ CSV – FAQ queries", data=faq.to_csv(index=False).encode("utf-8"),
                           file_name="gsc_faq_queries.csv", mime="text/csv")
    else:
        st.info("No Queries sheet with a 'query' column was found.")

    # ---------------------- Cannibalization ----------------------
    st.markdown("### Cannibalization")
    qp_table = find_query_page_table(cur_parts)

    if not qp_table.empty:
        st.caption("Exact cannibalization (table includes both query + page).")

        qp_f = qp_table.copy()
        if "impressions" in qp_f.columns:
            qp_f = qp_f[qp_f["impressions"].fillna(0) >= min_impr]

        if qp_f.empty:
            st.info("No rows left after min-impressions filter to compute cannibalization.")
        else:
            g = qp_f.copy()
            g["__w"] = g.get("impressions", 0).fillna(0)

            agg = (g.groupby(["query", "page"], as_index=False)
                     .agg(clicks=("clicks", "sum"),
                          impressions=("impressions", "sum")))

            if "position" in g.columns:
                pos = (g.groupby(["query", "page"])
                         .apply(lambda df: float(np.average(
                             df["position"].fillna(0), weights=(df["__w"] + 1e-9)
                         )))
                         .reset_index(name="position"))
                agg = agg.merge(pos, on=["query", "page"], how="left")
            else:
                agg["position"] = np.nan

            agg["ctr_pct"] = (agg["clicks"] / agg["impressions"].replace(0, np.nan)) * 100

            variants = (agg.groupby("query")["page"].nunique()
                        .reset_index(name="page_variants"))
            cann_q = variants[variants["page_variants"] >= 2]

            if cann_q.empty:
                st.success("No clear cannibalization detected above the impression threshold.")
            else:
                top = (agg.merge(cann_q, on="query", how="inner")
                         .sort_values(["query", "clicks"], ascending=[True, False])
                         .groupby("query")
                         .head(3)
                         .reset_index(drop=True))

                cols = [c for c in ["query","page","clicks","impressions",
                                    "ctr_pct","position","page_variants"]
                        if c in top.columns]
                st.dataframe(top[cols], use_container_width=True, hide_index=True)
                st.download_button("⬇️ CSV – Cannibalization (top 3 pages/query)",
                                   data=top[cols].to_csv(index=False).encode("utf-8"),
                                   file_name="gsc_cannibalization.csv", mime="text/csv")
    else:
        st.caption("Heuristic cannibalization (matching queries to pages by URL-token overlap).")
        t1, t2 = st.columns(2)
        with t1:
            heur_thresh = st.slider("Match threshold (overlap ratio)", 0.1, 1.0, 0.5, 0.05, key="SC_cann_thresh")
        with t2:
            max_pages_per_q = st.number_input("Max pages per query", 2, 10, 3, 1, key="SC_cann_maxpages")

        approx = _approximate_cannibalization(
            pages_cur=pages_cur,
            queries_cur=queries_cur,
            min_impr=min_impr,
            threshold=heur_thresh,
            max_pages=max_pages_per_q
        )

        if approx.empty:
            st.info("No heuristic cannibalization found given the current thresholds.")
        else:
            view = approx[[
                "query", "page", "overlap_ratio",
                "query_impressions", "query_clicks",
                "page_impressions", "page_clicks",
                "page_variants"
            ]]
            st.dataframe(view, use_container_width=True, hide_index=True)
            st.download_button("⬇️ CSV – Heuristic cannibalization",
                               data=view.to_csv(index=False).encode("utf-8"),
                               file_name="gsc_cannibalization_heuristic.csv",
                               mime="text/csv")

    st.markdown("---")
    st.caption("Uploads mode with filters & thresholds. Losses enabled when a baseline file is provided.")

# Standalone run (optional)
if __name__ == "__main__":
    render()
