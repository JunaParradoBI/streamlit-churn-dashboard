import re
import streamlit as st
import pandas as pd
import altair as alt

# =============================== #
#  PAGE CONFIGURATION
# =============================== #
st.set_page_config(
    page_title="Telco Retention Desk · Juan Parrado",
    page_icon="📶",
    layout="wide",
)

# "Telco · Retention desk" identity: deep plum, signal magenta, warm paper (IBM Plex nods to the IBM Telco dataset)
ACCENT = "#E4007C"
ACCENT_SOFT = "#FCE3EF"
INK = "#23062E"
MUTED = "#6E5A69"
NEUTRAL = "#E6D6DF"
RISK_THRESHOLD = 0.70

LOGO = (
    '<svg width="26" height="22" viewBox="0 0 26 22" aria-hidden="true">'
    '<rect x="0" y="14" width="4" height="8" rx="1.5" fill="#FCE3EF"/><rect x="7" y="9" width="4" height="13" rx="1.5" fill="#FCE3EF"/>'
    '<rect x="14" y="4" width="4" height="18" rx="1.5" fill="#E4007C"/><rect x="21" y="0" width="4" height="22" rx="1.5" fill="#E4007C" opacity=".35"/></svg>'
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Familjen+Grotesk:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500&display=swap');
    html, body, [class*="css"], .stMarkdown, p, li, label, .stTextInput, .stSelectbox, button { font-family: 'IBM Plex Sans', sans-serif; }
    h1, h2, h3, .hero-title { font-family: 'Familjen Grotesk', sans-serif !important; letter-spacing: -0.02em; color: #23062E; }
    .stApp { background: #FBF6F2; }
    .block-container { padding-top: 1.2rem; max-width: 1240px; }
    .hero { background: radial-gradient(70% 120% at 100% 0%, rgba(228,0,124,.35), transparent 60%), #23062E; color: #fff; border-radius: 26px; padding: 26px 34px 30px; margin-bottom: 18px; }
    .brandbar { display: flex; align-items: center; gap: 10px; font-family: 'IBM Plex Mono', monospace; font-size: 13px; letter-spacing: .04em; color: #FCE3EF; margin-bottom: 18px; }
    .brandbar b { color: #fff; font-weight: 500; }
    .brandbar .src { margin-left: auto; color: #C9AFC0; font-size: 12px; }
    .hero-title { font-size: clamp(32px, 4.2vw, 54px) !important; font-weight: 700; line-height: 1.02 !important; color: #fff !important; margin: 0 0 12px; }
    .hero-title span { color: #FF4FA7; }
    .lede { font-size: 17px; color: #E9D9E3; max-width: 64ch; margin: 0; }
    .kpi { background: #fff; border: 1px solid #EFDDE7; border-radius: 18px; padding: 18px 20px; height: 100%; box-shadow: 0 1px 0 rgba(35,6,46,.04); }
    .kpi .v { font-family: 'Familjen Grotesk', sans-serif; font-size: 40px; font-weight: 700; color: #23062E; line-height: 1; letter-spacing: -0.03em; }
    .kpi.hot { background: #E4007C; border-color: #E4007C; }
    .kpi.hot .v, .kpi.hot .l { color: #fff; }
    .kpi .l { font-size: 14px; color: #6E5A69; margin-top: 8px; line-height: 1.4; }
    .plain { background: #FCE3EF; border-radius: 16px; padding: 14px 18px; margin: 20px 0 8px; font-size: 16.5px; color: #23062E; max-width: 90ch; }
    .plain, .plain * { font-family: 'IBM Plex Sans', sans-serif !important; }
    .plain .k { font-family: 'IBM Plex Mono', monospace !important; font-size: 12px; font-weight: 500; letter-spacing: .06em; text-transform: uppercase; color: #B8005F; display: block; margin-bottom: 4px; }
    .card-top { display: flex; flex-wrap: wrap; align-items: baseline; gap: 6px 14px; }
    .card-top .p { font-family: 'Familjen Grotesk', sans-serif; font-size: 30px; font-weight: 700; color: #E4007C; }
    .card-top .id { font-family: 'IBM Plex Mono', monospace; font-weight: 500; color: #23062E; }
    .card-top .meta { color: #6E5A69; font-size: 14px; }
    .tag { display: inline-block; font-size: 12px; font-weight: 600; background: #23062E; color: #FCE3EF; border-radius: 6px; padding: 3px 9px; }
    .why, .act { font-size: 15px; margin-top: 8px; }
    .why b, .act b { font-family: 'IBM Plex Mono', monospace; font-size: 11.5px; font-weight: 500; letter-spacing: .05em; text-transform: uppercase; color: #6E5A69; display: block; }
    .act { background: #FBF6F2; border-left: 3px solid #E4007C; border-radius: 0 10px 10px 0; padding: 9px 12px; }
    [data-testid="stVerticalBlockBorderWrapper"] { background: #fff; border-color: #EFDDE7 !important; border-radius: 18px !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================== #
#  DATA SOURCES
# =============================== #
DEFAULT_GSHEETS_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "10vq7PsjVoonwVnjqM0o161n5ybslQAuV8xe0dwDoKbw/edit?gid=11827755#gid=11827755"
)
# Public IBM Telco dataset the model was trained on (adds monthly bill, contract and tenure)
TELCO_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"


def extract_spreadsheet_id(url: str) -> str:
    """Extracts the Google Sheets spreadsheet ID from a full URL."""
    match = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url)
    if not match:
        raise ValueError("Could not extract spreadsheet ID from URL. Check the format.")
    return match.group(1)


@st.cache_data(show_spinner=False, ttl=3600)
def load_data_from_gsheets(url: str, sheet: str) -> pd.DataFrame:
    """Loads one tab of a public Google Sheet through the CSV export endpoint."""
    spreadsheet_id = extract_spreadsheet_id(url)
    csv_url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/gviz/tq?tqx=out:csv&sheet={sheet}"
    # The sheet uses comma decimals (e.g. 0,85)
    return pd.read_csv(csv_url, decimal=",")


@st.cache_data(show_spinner=False, ttl=86400)
def load_telco() -> pd.DataFrame | None:
    try:
        return pd.read_csv(TELCO_URL)
    except Exception:
        return None


# Main determinants are free text written by the AI; group spelling variants into a few reasons
def reason_group(text: str) -> str:
    t = str(text).lower()
    if "contract" in t:
        return "No long-term contract"
    if "charge" in t or "price" in t or "cost" in t:
        return "High monthly bill"
    if "tenure" in t:
        return "New customer (short tenure)"
    if "payment" in t:
        return "Payment method"
    if "support" in t:
        return "Tech support issues"
    return "Other"


with st.spinner("Loading the latest model results…"):
    try:
        predicted = load_data_from_gsheets(DEFAULT_GSHEETS_URL, "predicted")
        risk = load_data_from_gsheets(DEFAULT_GSHEETS_URL, "risk_customers")
    except Exception as e:
        st.error(f"Couldn't load the model results from Google Sheets: {e}")
        st.stop()
    telco = load_telco()

risk = risk.copy()
risk["Reason"] = risk["MainDeterminant"].map(reason_group)
if telco is not None:
    risk = risk.merge(
        telco[["tenure", "Contract", "MonthlyCharges"]], left_on="CustomerIndex", right_index=True, how="left"
    )

# Model quality on the test customers
def auc(y: pd.Series, score: pd.Series) -> float:
    """ROC-AUC: chance the model ranks a real leaver above a customer who stayed."""
    r = score.rank()
    pos = int((y == 1).sum()); neg = len(y) - pos
    return (r[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg) if pos and neg else 0


def scores(pred_col: str) -> dict:
    y, yh = predicted["churn_actual"], predicted[pred_col]
    proba = pd.to_numeric(predicted[pred_col.replace("_pred", "_proba")].astype(str).str.replace(",", "."), errors="coerce")
    tp = int(((y == 1) & (yh == 1)).sum())
    fp = int(((y == 0) & (yh == 1)).sum())
    fn = int(((y == 1) & (yh == 0)).sum())
    return {
        "Recall (leavers the model caught)": tp / (tp + fn) if tp + fn else 0,
        "Precision (flagged who really left)": tp / (tp + fp) if tp + fp else 0,
        "ROC-AUC": auc(y, proba),
        "Accuracy": (y == yh).mean(),
    }

models = {"Logistic Regression": "logistic_pred", "Random Forest": "random_forest_pred", "XGBoost": "xgboost_pred"}
model_scores = {name: scores(col) for name, col in models.items()}
best = max(model_scores, key=lambda m: model_scores[m]["ROC-AUC"])
baseline_acc = 1 - predicted["churn_actual"].mean()  # accuracy of always guessing "nobody leaves"
flagged_really_left = predicted.set_index("customerID").reindex(risk["CustomerIndex"])["churn_actual"].mean()
n_risk = len(risk)
revenue_at_risk = risk["MonthlyCharges"].sum() if "MonthlyCharges" in risk else None

# =============================== #
#  HERO
# =============================== #
st.markdown(
    f'<div class="hero"><div class="brandbar">{LOGO}<b>Telco</b>· Retention desk'
    '<span class="src">Portfolio project · public IBM Telco dataset · built by Juan Parrado</span></div>'
    f'<div class="hero-title"><span>{n_risk} customers</span> are about to cancel.<br>Here is who to call first.</div>'
    f'<p class="lede">A machine-learning model scored {len(predicted):,} subscribers on how likely they are to leave. '
    f'Everyone above {RISK_THRESHOLD:.0%} risk gets an AI-written retention plan below.</p></div>',
    unsafe_allow_html=True,
)

kpis = [
    (f"{n_risk}", f"customers above {RISK_THRESHOLD:.0%} risk of cancelling"),
    (f"${revenue_at_risk:,.0f}" if revenue_at_risk is not None else "n/a", "monthly revenue at risk from them"),
    (f"{flagged_really_left:.0%}", "of the customers the model flagged really did cancel"),
    (f"{model_scores[best]['ROC-AUC']:.2f}", "ROC-AUC score (1.0 = perfect, 0.5 = a coin flip)"),
]
for i, (col, (v, l)) in enumerate(zip(st.columns(4), kpis)):
    col.markdown(f'<div class="kpi{" hot" if i == 1 else ""}"><div class="v">{v}</div><div class="l">{l}</div></div>', unsafe_allow_html=True)

reason_counts = risk["Reason"].value_counts()
top_reason = reason_counts.index[0]
plain = (
    f"The riskiest customers look alike: "
    + (f"all {n_risk} are on month-to-month contracts, " if "Contract" in risk and (risk["Contract"] == "Month-to-month").all() else "")
    + (f"most joined only {risk['tenure'].median():.0f} months ago, and they pay ${risk['MonthlyCharges'].mean():.0f} a month on average. " if "tenure" in risk else "")
    + "Offering a discounted 12-month contract in the first months is where retention money works hardest."
)
st.markdown(f'<div class="plain"><span class="k">In plain words</span>{plain}</div>', unsafe_allow_html=True)

# =============================== #
#  CHARTS
# =============================== #
alt.data_transformers.disable_max_rows()
axis = dict(labelColor=MUTED, titleColor=MUTED, gridColor="#F1E6EC", domainColor="#E6D6DF", labelFont="IBM Plex Sans", titleFont="IBM Plex Sans")

c1, c2 = st.columns([1, 1.15], gap="large")
with c1:
    st.subheader("Why they are at risk")
    st.caption("Main reason the AI gave for each high-risk customer")
    rc = reason_counts.reset_index()
    rc.columns = ["Reason", "Customers"]
    rc["Top"] = rc["Reason"] == top_reason
    bars = (
        alt.Chart(rc)
        .mark_bar(cornerRadiusEnd=4, height=22)
        .encode(
            y=alt.Y("Reason:N", sort="-x", title=None, axis=alt.Axis(labelLimit=320, labelPadding=8)),
            x=alt.X("Customers:Q", title="Customers", scale=alt.Scale(domain=[0, int(rc["Customers"].max() * 1.18) + 1])),
            color=alt.condition("datum.Top", alt.value(ACCENT), alt.value(NEUTRAL)),
            tooltip=["Reason", "Customers"],
        )
    )
    labels = bars.mark_text(align="left", dx=6, color=INK, font="IBM Plex Sans", fontWeight=600).encode(text="Customers:Q", color=alt.value(INK))
    st.altair_chart((bars + labels).properties(height=260).configure_axis(**axis).configure_view(stroke=None), width="stretch")

with c2:
    st.subheader("How risk is spread across all customers")
    st.caption(f"Each bar is a group of customers by churn probability (XGBoost). Magenta bars are above the {RISK_THRESHOLD:.0%} line.")
    hist = predicted[["xgboost_proba"]].copy()
    hist["bin"] = (hist["xgboost_proba"] * 20).clip(upper=19.999).astype(int) / 20
    hist = hist.groupby("bin").size().reset_index(name="Customers")
    hist["High risk"] = hist["bin"] >= RISK_THRESHOLD
    hist["Range"] = hist["bin"].map(lambda b: f"{b:.0%}–{b + 0.05:.0%}")
    order = hist["Range"].tolist()
    h = (
        alt.Chart(hist)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("Range:N", sort=order, title="Probability of cancelling", axis=alt.Axis(labelAngle=-45, labelExpr="split(datum.label, '–')[0]")),
            y=alt.Y("Customers:Q", title="Customers"),
            color=alt.condition("datum['High risk']", alt.value(ACCENT), alt.value(NEUTRAL)),
            tooltip=[alt.Tooltip("Range:N", title="Probability"), "Customers:Q"],
        )
    )
    rule = alt.Chart(pd.DataFrame({"Range": [f"{RISK_THRESHOLD:.0%}–{RISK_THRESHOLD + 0.05:.0%}"], "t": [f"{RISK_THRESHOLD:.0%} line"]})).mark_text(
        align="left", dx=-6, dy=-8, color=INK, font="IBM Plex Sans", fontWeight=600).encode(x=alt.X("Range:N", sort=order), y=alt.value(12), text="t:N")
    st.altair_chart((h + rule).properties(height=260).configure_axis(**axis).configure_view(stroke=None), width="stretch")

# =============================== #
#  CUSTOMERS TO CALL FIRST
# =============================== #
st.subheader("Customers to call first")
f1, f2 = st.columns([2, 1])
with f1:
    reasons = ["All reasons"] + reason_counts.index.tolist()
    chosen = st.segmented_control("Filter by main reason", reasons, default="All reasons", label_visibility="collapsed") or "All reasons"
with f2:
    show_n = st.selectbox("Show", [6, 12, 24, n_risk], format_func=lambda n: "All" if n == n_risk else f"Top {n}", label_visibility="collapsed")

view = risk if chosen == "All reasons" else risk[risk["Reason"] == chosen]
view = view.sort_values("Churn_Probability", ascending=False).head(show_n)

cols = st.columns(2, gap="medium")
for i, (_, r) in enumerate(view.iterrows()):
    meta = []
    if "MonthlyCharges" in r and pd.notna(r["MonthlyCharges"]):
        meta.append(f"${r['MonthlyCharges']:.0f}/month")
    if "tenure" in r and pd.notna(r["tenure"]):
        m = int(r["tenure"])
        meta.append(f"{m} month{'s' if m != 1 else ''} as a customer")
    with cols[i % 2].container(border=True):
        st.markdown(
            f'<div class="card-top"><span class="p">{r["Churn_Probability"]:.0%}</span>'
            f'<span class="id">Customer #{int(r["CustomerIndex"])}</span><span class="meta">{" · ".join(meta)}</span></div>'
            f'<span class="tag">{r["Reason"]}</span>'
            f'<div class="why"><b>Why</b>{r["Causes"]}</div>'
            f'<div class="act"><b>Suggested action (AI)</b>{r["Recommendation"]}</div>',
            unsafe_allow_html=True,
        )

st.download_button(
    "⬇️ Download the call list (CSV)",
    data=risk.sort_values("Churn_Probability", ascending=False).to_csv(index=False).encode("utf-8"),
    file_name="high_risk_customers.csv",
    mime="text/csv",
)

# =============================== #
#  MODEL COMPARISON
# =============================== #
st.subheader("How the three models compare")
st.caption(
    f"Tested on {len(predicted):,} customers the models never saw during training. "
    f"Accuracy alone is misleading here: guessing \"nobody leaves\" is already {baseline_acc:.0%} accurate, "
    "so recall, precision and ROC-AUC are the numbers that matter."
)
ms = pd.DataFrame(model_scores).T
st.dataframe(ms.style.format({c: "{:.0%}" for c in ms.columns if c != "ROC-AUC"} | {"ROC-AUC": "{:.2f}"}), width="stretch")

# =============================== #
#  ADVANCED: RAW DATA EXPLORER (original tool, kept for analysts)
# =============================== #
with st.expander("Explore the raw data (for analysts)"):
    gsheets_url = st.text_input("Google Sheets URL", value=DEFAULT_GSHEETS_URL,
                                help="The sheet must be shared as 'Anyone with the link – Viewer'.")
    sheet_name = st.selectbox("Sheet/tab", ["predicted", "risk_customers"], index=0)
    try:
        df = load_data_from_gsheets(gsheets_url, sheet_name)
    except Exception as e:
        st.error(f"Error loading data from Google Sheets: {e}")
        st.stop()

    cols_to_filter = st.multiselect("Columns to filter", df.columns.tolist())
    filtered = df.copy()
    for col in cols_to_filter:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            lo, hi = st.slider(f"{col} (range)", float(series.min()), float(series.max()), (float(series.min()), float(series.max())))
            filtered = filtered[(filtered[col] >= lo) & (filtered[col] <= hi)]
        else:
            uniq = sorted(series.dropna().astype(str).unique().tolist())[:500]
            sel = st.multiselect(col, uniq)
            if sel:
                filtered = filtered[filtered[col].astype(str).isin(sel)]

    st.caption(f"Rows after filters: {len(filtered):,} / {len(df):,}")
    st.dataframe(filtered, width="stretch")
    st.download_button("⬇️ Download filtered CSV", data=filtered.to_csv(index=False).encode("utf-8"),
                       file_name=f"{sheet_name}_filtered.csv", mime="text/csv")
