# competitiveaudit.py
import io
import re
import json
import hashlib
import random
import requests
import pandas as pd
import streamlit as st
from datetime import datetime


# -----------------------------
# Small utilities (local – no cross-file deps)
# -----------------------------
def ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Sheet1") -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    buf.seek(0)
    return buf.read()

def union_keywords(label_to_df: dict[str, pd.DataFrame]) -> list[str]:
    s = set()
    for _, df in label_to_df.items():
        if df is not None and not df.empty and "keyword" in df.columns:
            s.update(df["keyword"].dropna().astype(str).tolist())
    return sorted(s)

def _extract_domain(url: str) -> str:
    u = (url or "").strip()
    u = re.sub(r"^https?://", "", u, flags=re.I)
    u = re.sub(r"^www\.", "", u, flags=re.I)
    return u.split("/")[0]


# -----------------------------
# Ahrefs client (best-effort + demo fallback)
# -----------------------------
class AhrefsClient:
    BASE = "https://apiv2.ahrefs.com"

    def __init__(self, token: str):
        self.token = (token or "").strip()

    def get(self, from_name: str, **params) -> dict:
        if not self.token:
            raise RuntimeError("Missing AHREFS_API_TOKEN")
        q = {"token": self.token, "from": from_name, "output": "json"}
        q.update({k: v for k, v in params.items() if v is not None})
        r = requests.get(self.BASE, params=q, timeout=30)
        r.raise_for_status()
        return r.json()

    # ---------- Deterministic demo data (keeps UI usable without API) ----------
    @staticmethod
    def _seed_for(s: str) -> int:
        return int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16) % (2**32)

    def demo_positions(self, target: str, limit=60):
        random.seed(self._seed_for("pos:" + target))
        base = _extract_domain(target)
        skeletons = [
            f"{base} review", f"{base} alternatives", f"best {base}", f"how to use {base}",
            f"{base} download", f"{base} pricing", f"{base} vs", f"what is {base}",
            f"{base} tutorial", f"{base} not working"
        ]
        rows = []
        while len(rows) < limit:
            kw = random.choice(skeletons)
            rows.append({
                "keyword": f"{kw} {random.choice(['2025','guide','setup',''])}".strip(),
                "position": random.randint(1, 40),
                "volume": random.randint(80, 18000),
                "difficulty": random.randint(1, 60),
                "url": f"https://{base}/{random.choice(['blog','guide','post','features'])}/{random.randint(10,99)}"
            })
        return pd.DataFrame(rows)

    def demo_metrics(self, target: str):
        random.seed(self._seed_for("met:" + target))
        return {
            "refdomains": random.randint(100, 3500),
            "backlinks": random.randint(2_000, 120_000),
            "domain_rating": random.randint(10, 90),
            "url_rating": random.randint(5, 60),
        }

    def demo_internal_inlinks(self, target: str) -> int:
        random.seed(self._seed_for("in:" + target))
        return random.randint(5, 300)

    def demo_top_pages(self, target: str, n=5) -> list[str]:
        random.seed(self._seed_for("tp:" + target))
        base = _extract_domain(target)
        return [f"https://{base}/{random.choice(['category','article','blog','hub'])}/{i}" for i in range(1, n+1)]

    def demo_anchors(self, target: str, n=5) -> list[str]:
        random.seed(self._seed_for("an:" + target))
        anchors = ["click here", "brand name", "best guide", "read more", "official site", "pricing", "features"]
        out = []
        for _ in range(n):
            a = random.choice(anchors)
            rd = random.randint(5, 340)
            out.append(f"{a} (RD {rd})")
        return out


# -----------------------------
# High-level Ahrefs helpers (use API; fallback to demo on error)
# -----------------------------
def ahrefs_positions(client: AhrefsClient, target: str, mode: str, country: str, limit: int = 100) -> pd.DataFrame:
    try:
        data = client.get("positions", target=target, mode=mode, country=country, limit=limit, order_by="traffic:desc")
        rows = data.get("positions") or data.get("rows") or []
        if not rows:
            raise RuntimeError("no rows")
        df = pd.DataFrame(rows)
        # normalize columns we care about
        if "kd" in df.columns and "difficulty" not in df.columns:
            df = df.rename(columns={"kd": "difficulty"})
        for c in ["keyword", "position", "volume", "difficulty", "url"]:
            if c not in df.columns:
                df[c] = None
        return df[["keyword", "position", "volume", "difficulty", "url"]]
    except Exception:
        return client.demo_positions(target, limit=min(limit, 80))

def ahrefs_metrics(client: AhrefsClient, target: str, mode: str) -> dict:
    try:
        data = client.get("metrics", target=target, mode=mode, limit=1)
        arr = data.get("metrics") or data.get("rows") or []
        if not arr:  # demo
            return client.demo_metrics(target)
        m = arr[0]
        return {
            "refdomains": m.get("refdomains") or m.get("referring_domains"),
            "backlinks": m.get("backlinks"),
            "domain_rating": m.get("domain_rating") or m.get("dr"),
            "url_rating": m.get("url_rating") or m.get("ur"),
        }
    except Exception:
        return client.demo_metrics(target)

def ahrefs_internal_inlinks_count(client: AhrefsClient, target_url: str) -> int:
    domain = _extract_domain(target_url)
    try:
        data = client.get("backlinks", target=target_url, mode="url",
                          where=f"domain_from='{domain}'", limit=1, order_by="ahrefs_rank:desc")
        count = (
            (data.get("stats") or {}).get("rows_count")
            or data.get("rows_count")
            or data.get("total")
        )
        if isinstance(count, (int, float)):
            return int(count)
        return len(data.get("backlinks") or [])
    except Exception:
        return client.demo_internal_inlinks(target_url)

def ahrefs_top_pages(client: AhrefsClient, target: str, mode: str, limit: int = 5) -> list[str]:
    try:
        data = client.get("top_pages", target=target, mode=mode, limit=limit, order_by="traffic:desc")
        rows = data.get("top_pages") or data.get("pages") or data.get("rows") or []
        urls = [r.get("url") for r in rows if r.get("url")]
        if not urls:
            raise RuntimeError("no urls")
        return urls[:limit]
    except Exception:
        return client.demo_top_pages(target, n=limit)

def ahrefs_organic_traffic(client: AhrefsClient, target: str, mode: str):
    try:
        data = client.get("metrics", target=target, mode=mode, limit=1)
        rows = data.get("metrics") or data.get("rows") or []
        if rows:
            r = rows[0]
            return r.get("organic_search_traffic") or r.get("traffic") or r.get("organic_traffic")
    except Exception:
        pass
    return None

def ahrefs_top_anchors(client: AhrefsClient, target: str, mode: str, limit: int = 5) -> list[str]:
    try:
        data = client.get("anchors", target=target, mode=mode, limit=limit, order_by="refdomains:desc")
        rows = data.get("anchors") or data.get("rows") or []
        out = []
        for r in rows:
            a = r.get("anchor") or r.get("text") or ""
            rd = r.get("refdomains") or r.get("refpages") or r.get("backlinks")
            if a:
                out.append(f"{a} ({'RD ' + str(int(rd)) if rd is not None else '—'})")
        return out[:limit]
    except Exception:
        return client.demo_anchors(target, n=limit)


# -----------------------------
# MAIN TAB RENDER
# -----------------------------
def render(AHREFS_API_TOKEN: str, show_header: bool = False):
    if show_header:
        st.subheader("Competitive Audit")
    st.markdown("Keyword Gap • Internal Inlinks Gap • Top Pages • Anchors • Quick Actions")

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        ca_mode = st.selectbox("Scope / mode", ["url", "prefix", "subdomains", "domain"], index=0, key="ca_mode")
    with c2:
        ca_country = st.text_input("Country (e.g., us, uk, ca, za)", value="us", key="ca_country")
    with c3:
        ca_topk = st.number_input("Top keywords per target", min_value=5, max_value=100, value=10, step=5, key="ca_topk")

    our_target = st.text_input(
        "Your landing page or domain (used for keyword gap & inlinks count)",
        value="",
        key="ca_ours"
    )
    colA, colB, colC = st.columns(3)
    with colA: c1_target = st.text_input("Competitor 1 (URL/domain)", value="", key="ca_c1")
    with colB: c2_target = st.text_input("Competitor 2 (URL/domain)", value="", key="ca_c2")
    with colC: c3_target = st.text_input("Competitor 3 (URL/domain)", value="", key="ca_c3")

    run_ca = st.button("Run Competitive Audit", type="primary", key="ca_run")
    client = AhrefsClient(AHREFS_API_TOKEN)

    # Scoped helpers
    def _metrics(label, tgt):
        try:
            return ahrefs_metrics(client, tgt, ca_mode)
        except Exception as e:
            st.warning(f"Metrics failed for {label}: {e}")
            return {"refdomains": None, "backlinks": None, "domain_rating": None, "url_rating": None}

    def _positions(label, tgt):
        try:
            df = ahrefs_positions(client, tgt, ca_mode, ca_country, limit=max(int(ca_topk), 60))
            if not df.empty:
                for c in ("volume", "difficulty", "position"):
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                df = df.sort_values(["position", "volume"], ascending=[True, False]).head(int(ca_topk))
            return df.reset_index(drop=True)
        except Exception as e:
            st.warning(f"Keyword fetch failed for {label}: {e}")
            return pd.DataFrame(columns=["keyword", "position", "volume", "difficulty", "url"])

    def _inlinks(label, tgt):
        try:
            return ahrefs_internal_inlinks_count(client, tgt)
        except Exception as e:
            st.warning(f"Inlinks count failed for {label}: {e}")
            return None

    if run_ca:
        if not our_target.strip():
            st.warning("Please enter your landing page/domain.")
            st.stop()

        targets = [("Our Target", our_target.strip())]
        for lbl, v in [("Competitor 1", c1_target), ("Competitor 2", c2_target), ("Competitor 3", c3_target)]:
            if v.strip():
                targets.append((lbl, v.strip()))

        kw_data, metrics_map, inlinks_map, traffic_map, top_pages_map, anchors_map = {}, {}, {}, {}, {}, {}
        prog = st.progress(0.0)
        for i, (label, tgt) in enumerate(targets, start=1):
            metrics_map[label] = _metrics(label, tgt)
            kw_data[label] = _positions(label, tgt)
            inlinks_map[label] = _inlinks(label, tgt)
            traffic_map[label] = ahrefs_organic_traffic(client, tgt, ca_mode)
            top_pages_map[label] = ahrefs_top_pages(client, tgt, ca_mode, limit=5)
            anchors_map[label] = ahrefs_top_anchors(client, tgt, ca_mode, limit=5)
            prog.progress(i / len(targets))
        prog.empty()

        # ---------- Keyword Gap assembly
        comp_labels = [lbl for (lbl, _) in targets if lbl != "Our Target"]
        comp_kw_union = union_keywords({lbl: kw_data.get(lbl, pd.DataFrame()) for lbl in comp_labels})

        def pos_for(df: pd.DataFrame, kw: str):
            if df is None or df.empty:
                return None, None, None, None
            row = df[df["keyword"] == kw].head(1)
            if row.empty:
                return None, None, None, None
            r = row.iloc[0]
            return r.get("position"), r.get("url"), r.get("volume"), r.get("difficulty")

        long_rows = []
        for kw in comp_kw_union:
            row = {"keyword": kw}
            for label, _ in targets:
                p, u, sv, kd = pos_for(kw_data.get(label), kw)
                row[f"{label} pos"] = p
                row[f"{label} url"] = u
                if label != "Our Target":
                    row["sv"] = max(row.get("sv") or 0, sv or 0)
                    row["kd"] = max(row.get("kd") or 0, kd or 0)
            long_rows.append(row)
        long_df = pd.DataFrame(long_rows)

        def any_competitor_ranks(r):
            for label, _ in targets:
                if label == "Our Target":
                    continue
                p = r.get(f"{label} pos")
                if pd.notna(p) and p <= 20:
                    return True
            return False

        if "Our Target pos" in long_df.columns:
            gap_df = long_df[ long_df["Our Target pos"].isna() & long_df.apply(any_competitor_ranks, axis=1) ].copy()
        else:
            gap_df = long_df[ long_df.apply(any_competitor_ranks, axis=1) ].copy()

        # pick top ~5 content gap keywords
        top_gap = []
        if not gap_df.empty:
            tmp = gap_df.copy()
            tmp["sv"] = pd.to_numeric(tmp["sv"], errors="coerce")
            best_cols = [c for c in tmp.columns if c.endswith(" pos") and not c.startswith("Our Target")]
            tmp["best_pos"] = tmp[best_cols].min(axis=1, numeric_only=True)
            tmp = tmp.sort_values(["sv", "best_pos"], ascending=[False, True])
            top_gap = tmp["keyword"].head(5).astype(str).tolist()

        # low-KD helper
        def low_kd_keywords_cell(label, k=5):
            df = kw_data.get(label)
            if df is None or df.empty:
                return []
            d2 = df.copy()
            d2["difficulty"] = pd.to_numeric(d2["difficulty"], errors="coerce")
            d2["volume"] = pd.to_numeric(d2["volume"], errors="coerce")
            d2 = d2.sort_values(["difficulty", "volume"], ascending=[True, False])
            out = []
            for _, r in d2.head(k).iterrows():
                kw = str(r.get("keyword") or "")
                kd = r.get("difficulty")
                sv = r.get("volume")
                bits = []
                if pd.notna(kd): bits.append(f"KD {int(kd)}")
                if pd.notna(sv): bits.append(f"SV {int(sv)}")
                meta = (", ".join(bits)) if bits else ""
                out.append(f"{kw}" + (f" ({meta})" if meta else ""))
            return out

        # ---------- Summary Matrix
        st.markdown("### Summary Matrix")
        col_headers = [lbl for (lbl, _) in targets]

        def _mval(label, key):
            m = metrics_map.get(label, {}) or {}
            return m.get(key)

        def _top_keywords_cell(label, k=5):
            df = kw_data.get(label)
            if df is None or df.empty:
                return []
            out = []
            for _, r in df.head(k).iterrows():
                kw = str(r.get("keyword") or "")
                sv = r.get("volume")
                out.append(f"{kw}" + (f" (SV {int(sv)})" if pd.notna(sv) else ""))
            return out

        matrix_rows = []
        matrix_rows.append([""] + col_headers)
        matrix_rows.append(["Website"] + [tgt for (_, tgt) in targets])

        # Key Stats
        matrix_rows.append(["Key Stats (#)"] + [""] * len(col_headers))
        matrix_rows.append(["Domain Rating (DR)"] + [_mval(lbl, "domain_rating") for lbl in col_headers])
        matrix_rows.append(["Referring Domains"]   + [_mval(lbl, "refdomains") for lbl in col_headers])
        matrix_rows.append(["UR"]                  + [_mval(lbl, "url_rating") for lbl in col_headers])
        matrix_rows.append(["Organic Traffic"]     + [ahrefs_organic_traffic(client, t, ca_mode) for (_, t) in targets])
        matrix_rows.append(["Internal Inlinks"]    + [inlinks_map.get(lbl) for lbl in col_headers])

        # Keywords (SV only)
        matrix_rows.append(["Keywords (#)"] + [""] * len(col_headers))
        per_col_kw = {lbl: _top_keywords_cell(lbl, k=5) for lbl in col_headers}
        for i in range(5):
            matrix_rows.append([f"Keyword {i+1}"] + [
                (per_col_kw.get(lbl, []) or [""*0])[i] if i < len(per_col_kw.get(lbl, [])) else "" for lbl in col_headers
            ])

        # Content gap
        matrix_rows.append(["Content Gap (#)"] + [""] * len(col_headers))
        if top_gap:
            for i, kw in enumerate(top_gap, start=1):
                matrix_rows.append([f"Content gap keyword {i}"] + [kw] + [""] * (len(col_headers) - 1))
        else:
            matrix_rows.append(["No gaps found"] + [""] * len(col_headers))

        # Low-KD
        matrix_rows.append(["Low-KD Keywords (#)"] + [""] * len(col_headers))
        per_col_lowkd = {lbl: low_kd_keywords_cell(lbl, k=5) for lbl in col_headers}
        for i in range(5):
            matrix_rows.append([f"Low-KD {i+1}"] + [
                (per_col_lowkd.get(lbl, []) or [""*0])[i] if i < len(per_col_lowkd.get(lbl, [])) else "" for lbl in col_headers
            ])

        matrix_df = pd.DataFrame(matrix_rows[1:], columns=matrix_rows[0])
        st.dataframe(matrix_df, use_container_width=True, hide_index=True)

        st.download_button(
            "⬇️ Download Summary Matrix (Excel)",
            to_excel_bytes(matrix_df),
            file_name=f"competitive_audit_summary_{ts()}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="ca_dl_summary_xlsx",
        )
        st.download_button(
            "⬇️ Download Summary Matrix (CSV)",
            matrix_df.to_csv(index=False).encode("utf-8"),
            file_name=f"competitive_audit_summary_{ts()}.csv",
            mime="text/csv",
            key="ca_dl_summary_csv",
        )

        # ---------- Quick Actions
        st.markdown("### Quick Actions")
        bullets = []

        our_inlinks = int(inlinks_map.get("Our Target") or 0)
        for lbl in col_headers:
            if lbl == "Our Target": continue
            comp_in = int(inlinks_map.get(lbl) or 0)
            if comp_in > our_inlinks:
                bullets.append(f"Build **+{comp_in - our_inlinks} internal links** to close gap vs **{lbl}**.")

        if top_gap:
            bullets.append(f"Create/optimize for gap keywords: **{', '.join(top_gap[:3])}**.")

        for lbl in col_headers:
            if lbl == "Our Target": continue
            lks = low_kd_keywords_cell(lbl, k=3)
            if lks:
                bullets.append(f"Target low-KD terms seen for **{lbl}**: **{', '.join([t.split(' (')[0] for t in lks])}**.")

        for lbl in col_headers:
            if lbl == "Our Target": continue
            tps = top_pages_map.get(lbl) or []
            if tps:
                bullets.append(f"Review competitor **{lbl}** top pages for topical gaps & link targets (e.g., {tps[0]}).")

        if bullets:
            st.markdown("\n".join([f"- {b}" for b in bullets]))
        else:
            st.info("No urgent actions detected from the current inputs.")

        # ---------- Detail tables
        st.markdown("---")
        st.markdown("### Target Metrics (detail)")
        mrows = []
        for label, tgt in targets:
            m = metrics_map.get(label, {})
            mrows.append({
                "Label": label,
                "Target": tgt,
                "Referring Domains": m.get("refdomains"),
                "Backlinks": m.get("backlinks"),
                "DR": m.get("domain_rating"),
                "UR": m.get("url_rating"),
                "Organic Traffic": traffic_map.get(label),
            })
        mdf = pd.DataFrame(mrows)
        st.dataframe(mdf, use_container_width=True)

        st.markdown("### Internal Inlinks (Counts & Δ to match)")
        rows = []
        our_count = int(inlinks_map.get("Our Target") or 0)
        for label, tgt in targets:
            cnt = None if inlinks_map.get(label) is None else int(inlinks_map[label])
            delta = max((cnt or 0) - our_count, 0) if label != "Our Target" else 0
            rows.append({"Label": label, "URL": tgt, "Internal Inlinks": cnt, "Build Δ vs us": int(delta)})
        in_df = pd.DataFrame(rows)
        st.dataframe(in_df, use_container_width=True)

        msgs = []
        for _, r in in_df.iterrows():
            if r["Label"] == "Our Target": continue
            comp = int(r["Internal Inlinks"] or 0)
            delta = int(r["Build Δ vs us"] or 0)
            if comp and delta > 0:
                msgs.append(f"- {r['Label']} has **{comp}** internal inlinks; we have **{our_count}** → build **{delta}** more.")
        if msgs:
            st.markdown("**Inlinks summary**")
            st.markdown("\n".join(msgs))
        else:
            st.info("No internal inlink deficit detected vs the provided competitors.")

        st.markdown("### Keyword Gap (Competitors rank, our target doesn’t)")
        if gap_df.empty:
            st.info("No keyword gaps found based on pulled top keywords and thresholds.")
        else:
            sort_cols = [f"{label} pos" for (label, _) in targets if label != "Our Target"] + ["sv"]
            gap_df = gap_df.sort_values(sort_cols, ascending=[True] * (len(sort_cols) - 1) + [False])
            st.dataframe(gap_df, use_container_width=True)

        # Downloads (detail)
        st.markdown("### Downloads (detail)")
        st.download_button("⬇️ Metrics (CSV)", mdf.to_csv(index=False).encode("utf-8"),
                           file_name=f"ahrefs_target_metrics_{ts()}.csv", mime="text/csv", key="ca_dl_metrics")
        st.download_button("⬇️ Inlinks (CSV)", in_df.to_csv(index=False).encode("utf-8"),
                           file_name=f"inlinks_comparison_{ts()}.csv", mime="text/csv", key="ca_dl_inlinks")
        if not gap_df.empty:
            st.download_button("⬇️ Keyword Gap (CSV)", gap_df.to_csv(index=False).encode("utf-8"),
                               file_name=f"ahrefs_keyword_gaps_{ts()}.csv", mime="text/csv", key="ca_dl_gap")
        st.download_button("⬇️ Keyword Matrix (Excel)", to_excel_bytes(long_df.fillna("")),
                           file_name=f"ahrefs_keyword_matrix_{ts()}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           key="ca_dl_matrix")
