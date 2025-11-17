import re
import streamlit as st
import pandas as pd
import altair as alt

# =============================== #
#  BASIC PAGE CONFIGURATION
# =============================== #
st.set_page_config(
    page_title="Dataset Loader & Filters from Google Sheets",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Load Data from Google Sheets and Apply Filters")

st.markdown("""
**Steps:**  
1) Enter your Google Sheets URL.  
2) Select which sheet/tab to load (`predicted` or `risk_customers`).  
3) Apply filters, explore the data, and download the filtered result.
""")

# =============================== #
#  CONSTANT – DEFAULT GOOGLE SHEETS URL
# =============================== #
# You can change this URL to any other Google Sheets document you want
DEFAULT_GSHEETS_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "10vq7PsjVoonwVnjqM0o161n5ybslQAuV8xe0dwDoKbw/edit?gid=11827755#gid=11827755"
)

# =============================== #
#  HELPER – EXTRACT SHEET ID FROM URL
# =============================== #
def extract_spreadsheet_id(url: str) -> str:
    """
    Extracts the Google Sheets spreadsheet ID from a full URL.

    Example:
      https://docs.google.com/spreadsheets/d/SPREADSHEET_ID/edit#gid=0
      -> returns "SPREADSHEET_ID"
    """
    match = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url)
    if not match:
        raise ValueError("Could not extract spreadsheet ID from URL. Check the format.")
    return match.group(1)

# =============================== #
#  SIDEBAR – GOOGLE SHEETS CONFIG
# =============================== #
with st.sidebar:
    st.header("1) Google Sheets Settings")

    # Text input so you can change the sheet URL if needed
    gsheets_url = st.text_input(
        "Google Sheets URL",
        value=DEFAULT_GSHEETS_URL,
        help="https://docs.google.com/spreadsheets/d/10vq7PsjVoonwVnjqM0o161n5ybslQAuV8xe0dwDoKbw/edit?usp=sharing"
    )

    st.caption(
        "Make sure the Google Sheet is shared as "
        "**'Anyone with the link – Viewer'** so the app can read it."
    )

    # Choose which tab (sheet) to load
    sheet_name = st.selectbox(
        "Select sheet/tab",
        ["predicted", "risk_customers"],
        index=0
    )

# If no URL provided, stop the app
if not gsheets_url.strip():
    st.info("⬅️ Please paste a valid Google Sheets URL in the sidebar to begin.")
    st.stop()

# =============================== #
#  LOAD DATA FROM GOOGLE SHEETS
# =============================== #
@st.cache_data(show_spinner=True)
def load_data_from_gsheets(url: str, sheet: str) -> pd.DataFrame:
    """
    Loads data from a specific sheet (tab) of a Google Sheets document
    using the public CSV export endpoint.

    - url: full Google Sheets URL
    - sheet: sheet/tab name (e.g. 'predicted', 'risk_customers')
    """
    spreadsheet_id = extract_spreadsheet_id(url)

    # Google Sheets CSV export endpoint:
    # https://docs.google.com/spreadsheets/d/<ID>/gviz/tq?tqx=out:csv&sheet=<SHEET_NAME>
    csv_url = (
        f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/gviz/tq?"
        f"tqx=out:csv&sheet={sheet}"
    )

    df = pd.read_csv(csv_url)
    return df

# Try to load the selected sheet
try:
    df = load_data_from_gsheets(gsheets_url, sheet_name)
except Exception as e:
    st.error(f"Error loading data from Google Sheets: {e}")
    st.stop()

# Stop if dataframe is empty
if df.empty:
    st.warning("The sheet was loaded, but it is empty or has no rows.")
    st.stop()

# =============================== #
#  SIDEBAR – DATE PARSING OPTION
# =============================== #
with st.sidebar:
    st.header("2) Options")

    # Option to automatically detect and convert date-like columns
    try_dates = st.checkbox("Try to detect date columns", value=True)

    if try_dates:
        for col in df.columns:
            if df[col].dtype == "object":  # Only convert text columns
                try:
                    parsed = pd.to_datetime(df[col], errors="raise", infer_datetime_format=True)
                    # Accept conversion only if at least 80% parse successfully
                    if parsed.notna().mean() > 0.8:
                        df[col] = parsed
                except Exception:
                    # Ignore columns that cannot be safely converted
                    pass

# =============================== #
#  SIDEBAR – FILTER SELECTION
# =============================== #
with st.sidebar:
    st.header("3) Filters")

    # Choose which columns will have filters
    cols_to_filter = st.multiselect(
        "Choose columns to filter",
        df.columns.tolist()
    )

# =============================== #
#  BUILD FILTER WIDGETS DYNAMICALLY
# =============================== #
filters = {}

for col in cols_to_filter:
    series = df[col]
    container = st.sidebar.container()

    with container:
        # Numeric columns → range slider
        if pd.api.types.is_numeric_dtype(series):
            min_v, max_v = float(series.min()), float(series.max())
            sel = st.slider(f"{col} (range)", min_v, max_v, (min_v, max_v))
            filters[col] = ("numeric_range", sel)

        # Datetime columns → date range selector
        elif pd.api.types.is_datetime64_any_dtype(series):
            min_d = pd.to_datetime(series.min()).date()
            max_d = pd.to_datetime(series.max()).date()
            sel = st.date_input(f"{col} (date range)", (min_d, max_d))

            if isinstance(sel, tuple) and len(sel) == 2:
                filters[col] = ("date_range", sel)

        # Other types (categorical/text) → multiselect of unique values
        else:
            uniq = sorted(series.dropna().astype(str).unique().tolist())

            # Avoid extremely large dropdowns
            if len(uniq) > 500:
                st.caption(f"{col}: too many unique values ({len(uniq)}). Showing first 500.")
                uniq = uniq[:500]

            sel = st.multiselect(col, uniq)
            filters[col] = ("isin", sel)

# =============================== #
#  APPLY FILTERS
# =============================== #
filtered = df.copy()

for col, (kind, val) in filters.items():

    if kind == "numeric_range":
        lo, hi = val
        filtered = filtered[(filtered[col] >= lo) & (filtered[col] <= hi)]

    elif kind == "date_range":
        start, end = val
        start, end = pd.to_datetime(start), pd.to_datetime(end)
        filtered = filtered[
            (pd.to_datetime(filtered[col]) >= start)
            & (pd.to_datetime(filtered[col]) <= end)
        ]

    elif kind == "isin":
        if val:
            str_vals = [str(x) for x in val]
            filtered = filtered[filtered[col].astype(str).isin(str_vals)]

# =============================== #
#  SUMMARY MESSAGE
# =============================== #
st.success(
    f"Active sheet: **{sheet_name}** · "
    f"Rows after filters: {len(filtered):,} / {len(df):,} · "
    f"Columns: {len(df.columns)}"
)

# =============================== #
#  SIDEBAR – CHOOSE COLUMNS TO DISPLAY
# =============================== #
with st.sidebar:
    show_cols = st.multiselect(
        "Columns to display (optional)",
        df.columns.tolist(),
        default=df.columns.tolist()
    )

# =============================== #
#  DISPLAY DATAFRAME
# =============================== #
st.subheader("Data Preview")
st.dataframe(filtered[show_cols], use_container_width=True)

# =============================== #
#  DOWNLOAD FILTERED DATA
# =============================== #
csv_out = filtered[show_cols].to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download filtered CSV",
    data=csv_out,
    file_name=f"{sheet_name}_filtered.csv",
    mime="text/csv"
)

# =============================== #
#  INTERACTIVE CHART
# =============================== #
st.subheader("Interactive Chart")

if not filtered.empty:

    # Identify numeric and non-numeric columns
    num_cols = [c for c in filtered.columns if pd.api.types.is_numeric_dtype(filtered[c])]
    cat_cols = [c for c in filtered.columns if not pd.api.types.is_numeric_dtype(filtered[c])]

    # Chart configuration controls
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        chart_type = st.selectbox("Chart Type", ["Bar", "Line", "Scatter"])
    with col2:
        x_col = st.selectbox("X Axis (category/date)", cat_cols or filtered.columns.tolist())
    with col3:
        y_col = st.selectbox("Y Axis (numeric)", num_cols or filtered.columns.tolist())
    with col4:
        agg = st.selectbox("Aggregation", ["sum", "mean", "count", "min", "max"], index=0)

    # Prepare dataframe for plotting
    plot_df = filtered.copy()

    # For Bar and Line charts, aggregate by X
    if chart_type in ["Bar", "Line"] and len(plot_df):

        if agg == "count":
            plot_df = (
                plot_df.groupby(x_col, dropna=False)[y_col]
                .count()
                .reset_index(name=y_col)
            )
        else:
            plot_df = getattr(
                plot_df.groupby(x_col, dropna=False)[y_col],
                agg
            )().reset_index()

    # Altair configuration
    alt.data_transformers.disable_max_rows()

    # Bar chart
    if chart_type == "Bar":
        chart = (
            alt.Chart(plot_df)
            .mark_bar()
            .encode(
                x=alt.X(x_col, sort="-y"),
                y=alt.Y(y_col),
                tooltip=[x_col, y_col]
            )
            .interactive()
        )

    # Line chart
    elif chart_type == "Line":
        chart = (
            alt.Chart(plot_df)
            .mark_line(point=True)
            .encode(
                x=alt.X(x_col),
                y=alt.Y(y_col),
                tooltip=[x_col, y_col]
            )
            .interactive()
        )

    # Scatter chart – uses non-aggregated filtered data
    else:
        chart = (
            alt.Chart(filtered)
            .mark_circle(size=60)
            .encode(
                x=alt.X(x_col),
                y=alt.Y(y_col),
                tooltip=list(filtered.columns)
            )
            .interactive()
        )

    st.altair_chart(chart, use_container_width=True)

else:
    st.info("No data available to plot.")
