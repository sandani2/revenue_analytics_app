"""
Revenue Analytics App - Streamlit Dashboard
Industry Project - All Modules

Run with:
    streamlit run app.py

Expects merged_revenue_data.csv in the same folder (or update DATA_PATH below).
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from scipy import stats
from scipy.stats import chi2_contingency
from statsmodels.tsa.holtwinters import Holt

# =============================================================================
# PAGE CONFIG + BLUE THEME
# =============================================================================
st.set_page_config(
    page_title="Revenue Analytics App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Blue palette used consistently across every chart and UI element ----
NAVY = "#02075D"
DARK_BLUE = "#173A5E"
PRIMARY_BLUE = "#2E75B6"
MID_BLUE = "#2F5597"
LIGHT_BLUE = "#8CBCE6"
PALE_BLUE = "#B8D5F0"
BLUE_SEQUENCE = [PRIMARY_BLUE, DARK_BLUE, MID_BLUE, LIGHT_BLUE, NAVY, PALE_BLUE]

sns.set_theme(style="whitegrid")
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["font.family"] = "DejaVu Sans"

st.markdown(f"""
<style>
    .main {{ background-color: #F4F8FC; }}
    section[data-testid="stSidebar"] {{
        background-color: {NAVY};
    }}
    section[data-testid="stSidebar"] * {{ color: white !important; }}
    /* Fix: selected value in every dropdown (sidebar filters + selectboxes)
       was invisible - white text on white background. Different Streamlit
       versions render selectboxes differently (older versions use BaseWeb
       classes, newer versions use a React Aria ComboBox with the value in
       a literal <input>), so both structures are targeted here at once to
       be safe regardless of which version is running. -webkit-text-fill-color
       is set alongside color since some browsers apply it separately on
       input elements and ignore color alone. forced-color-adjust: none
       opts these elements out of Edge's/Windows' forced-contrast repainting,
       which can override page CSS (including !important) at the browser
       level for accessibility - without this, no amount of page CSS can
       fix the colors if that setting is active. */
    div[data-testid="stSelectbox"] input,
    div[data-testid="stSelectbox"] div,
    div[data-testid="stSelectbox"] span,
    div[data-baseweb="select"] input,
    div[data-baseweb="select"] div,
    div[data-baseweb="select"] span {{
        color: #000000 !important;
        -webkit-text-fill-color: #b5bec8 !important;
        forced-color-adjust: none !important;
    }}
    div[data-testid="stSelectbox"] input,
    div[data-baseweb="select"] {{
        background-color: #ffffff !important;
        forced-color-adjust: none !important;
    }}
    /* The open dropdown options list - covers both the React Aria listbox
       portal and the older BaseWeb virtualized dropdown */
    div[role="listbox"] [role="option"],
    ul[data-testid="stSelectboxVirtualDropdown"] * {{
        color: #000000 !important;
        forced-color-adjust: none !important;
        -webkit-text-fill-color: #000000 !important;
    }}
    div[data-testid="stFileUploader"] {{
    background: white !important;
    padding:12px;
    border:2px solid #8CBCE6;
    border-radius:10px;
}}

div[data-testid="stFileUploader"] * {{
    color:#173A5E !important;
}}
    div[data-testid="stFileUploader"] * {{
        color:#173A5E !important;
    }}
        label[data-testid="stWidgetLabel"] {{
        color:#173A5E !important;
        font-weight:600;
    }}
    div[data-testid="stFileUploaderDropzone"] * {{
        color:#173A5E !important;
    }}
    div[data-testid="stMetric"] {{
        background-color: white;
        border: 1px solid #D6E4F0;
        border-left: 5px solid {PRIMARY_BLUE};
        border-radius: 8px;
        padding: 12px 16px;
    }}
    /* Metric card background is deliberately always white (light card on
       either theme), so label/value text must be forced dark too - otherwise
       dark theme's light default text becomes invisible on the white card,
       same root cause as the dropdown issue. Delta (the up/down indicator)
       is intentionally left untouched so its red/green semantic color still
       shows correctly. */
    div[data-testid="stMetric"] [data-testid="stMetricLabel"],
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
        color: {DARK_BLUE} !important;
        forced-color-adjust: none !important;
    }}
    h1, h2, h3 {{ color: {NAVY}; }}
    div[data-testid="stTabs"] button {{ color: {DARK_BLUE}; }}
    .stDataFrame {{ border: 1px solid #D6E4F0; }}
</style>
""", unsafe_allow_html=True)

DATA_PATH = "merged_revenue_data.csv"


# =============================================================================
# DATA LOADING (cached so it only runs once)
# =============================================================================
@st.cache_data
def load_data(uploaded_file=None):
    df = pd.read_csv(uploaded_file if uploaded_file is not None else DATA_PATH)
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    if "customer_since" in df.columns:
        df["customer_since"] = pd.to_datetime(df["customer_since"])
    if "start_date" in df.columns:
        df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    if "end_date" in df.columns:
        df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
    df["is_campaign"] = df["campaign_id"].notna()
    df["year_month"] = df["transaction_date"].dt.to_period("M")
    return df


uploaded_file = st.sidebar.markdown("### 📁 Upload New Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
try:
    df = load_data(uploaded_file)
except FileNotFoundError:
    st.error(f"Could not find `{DATA_PATH}`. Upload a CSV or place it in the same folder as this app.")
    st.stop()

REFERENCE_DATE = df["transaction_date"].max()


# =============================================================================
# SIDEBAR NAVIGATION
# =============================================================================
st.sidebar.title("📊 Revenue Analytics")
st.sidebar.caption(f"{len(df):,} transactions | {df['customer_id'].nunique():,} customers | "
                    f"{df['transaction_date'].min().date()} to {REFERENCE_DATE.date()}")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Executive Dashboard",
        "👥 Customer Analytics",
        "📦 Product & Channel",
        "💷 Pricing & Discount",
        "📈 Revenue Forecasting",
        "⚠️ Revenue Leakage",
        "✅ Recommendations",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption("Filters")

df_f = df.copy()
filter_columns=["branch_name","customer_type","channel","payment_status","country","category"]
for col in filter_columns:
    if col in df_f.columns:
        options=["All"]+sorted(df_f[col].dropna().astype(str).unique().tolist())
        selected=st.sidebar.selectbox(col.replace("_"," ").title(),options,key=f"filter_{col}")
        if selected!="All":
            df_f=df_f[df_f[col].astype(str)==selected]


# =============================================================================
# HELPER: consistent blue bar chart
# =============================================================================
def blue_bar(data, x, y, title, xlabel, ylabel, horizontal=False, figsize=(8, 4.5),
             color=PRIMARY_BLUE, fmt="{:,.0f}"):
    fig, ax = plt.subplots(figsize=figsize)
    if horizontal:
        bars = ax.barh(data[x], data[y], color=color, edgecolor="white")
        ax.invert_yaxis()
    else:
        bars = ax.bar(data[x], data[y], color=color, edgecolor="white")
        plt.xticks(rotation=30, ha="right")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    return fig


# =============================================================================
# EXECUTIVE REVENUE DASHBOARD
# =============================================================================
def render_module1(data):
    st.title("🏠 Executive Revenue Dashboard")
    

    total_net_revenue = data["net_revenue"].sum()
    total_gross_revenue = data["gross_revenue"].sum()
    total_transactions = len(data)
    avg_order_value = total_net_revenue / total_transactions
    overall_margin_pct = data["gross_margin"].sum() / total_net_revenue * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Net Revenue", f"£{total_net_revenue:,.0f}")
    c2.metric("Total Transactions", f"{total_transactions:,}")
    c3.metric("Average Order Value", f"£{avg_order_value:,.2f}")
    c4.metric("Overall Gross Margin", f"{overall_margin_pct:.1f}%")

    st.markdown("### Monthly Revenue Trend")
    monthly = data.groupby("year_month")["net_revenue"].sum().reset_index()
    monthly["net_revenue"] = monthly["net_revenue"].round(2)
    monthly["mom_growth_pct"] = monthly["net_revenue"].pct_change().mul(100).round(2)
    monthly["year_month"] = monthly["year_month"].astype(str)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(monthly["year_month"], monthly["net_revenue"], marker="o", color=NAVY, linewidth=2)
    ax.set_title("Monthly Net Revenue Trend")
    ax.set_xlabel("Month")
    ax.set_ylabel("Net Revenue (£)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)
    with st.expander("View monthly table (with MoM growth %)"):
        st.dataframe(monthly, use_container_width=True)

    st.markdown("### Revenue Breakdown")
    col1, col2 = st.columns(2)
    with col1:
        by_channel = data.groupby("channel")["net_revenue"].sum().sort_values(ascending=False).reset_index()
        st.pyplot(blue_bar(by_channel, "channel", "net_revenue", "Net Revenue by Channel", "Channel", "Net Revenue (£)"))
    with col2:
        by_branch = data.groupby("branch_name")["net_revenue"].sum().sort_values(ascending=False).reset_index()
        st.pyplot(blue_bar(by_branch, "branch_name", "net_revenue", "Net Revenue by Branch", "Branch", "Net Revenue (£)"))

    by_category = data.groupby("category")["net_revenue"].sum().sort_values(ascending=False).reset_index()
    st.pyplot(blue_bar(by_category.head(10), "category", "net_revenue",
                        "Top 10 Categories by Net Revenue", "Net Revenue (£)", "Category",
                        horizontal=True, figsize=(9, 5)))

    st.markdown("### Gross Margin % by Channel / Branch")
    col1, col2 = st.columns(2)
    with col1:
        margin_by_channel = data.groupby("channel").apply(
            lambda g: g["gross_margin"].sum() / g["net_revenue"].sum() * 100, include_groups=False
        ).reset_index(name="margin_pct").sort_values("margin_pct", ascending=False)
        st.pyplot(blue_bar(margin_by_channel, "channel", "margin_pct", "Gross Margin % by Channel", "Channel", "Margin %"))
    with col2:
        margin_by_branch = data.groupby("branch_name").apply(
            lambda g: g["gross_margin"].sum() / g["net_revenue"].sum() * 100, include_groups=False
        ).reset_index(name="margin_pct").sort_values("margin_pct", ascending=False)
        st.pyplot(blue_bar(margin_by_branch, "branch_name", "margin_pct", "Gross Margin % by Branch", "Branch", "Margin %"))

    with st.expander("net_revenue summary statistics"):
        summary_stats = pd.DataFrame([{
            "mean": data["net_revenue"].mean(), "median": data["net_revenue"].median(),
            "std_dev": data["net_revenue"].std(), "min": data["net_revenue"].min(),
            "max": data["net_revenue"].max()
        }]).round(2)
        st.dataframe(summary_stats, use_container_width=True)


# =============================================================================
# CUSTOMER ANALYTICS (RFM)
# =============================================================================
@st.cache_data
def compute_rfm(data):
    rfm = data.groupby("customer_id").agg(
        last_purchase_date=("transaction_date", "max"),
        frequency=("transaction_id", "count"),
        monetary=("net_revenue", "sum"),
        customer_type=("customer_type", "first"),
        acquisition_channel=("acquisition_channel", "first"),
        location_category=("location_category", "first"),
        customer_since=("customer_since", "first"),
    ).reset_index()
    rfm["recency"] = (REFERENCE_DATE - rfm["last_purchase_date"]).dt.days
    rfm["R_score"] = pd.qcut(rfm["recency"], q=3, labels=[3, 2, 1]).astype(int)
    rfm["F_score"] = pd.qcut(rfm["frequency"].rank(method="first"), q=3, labels=[1, 2, 3]).astype(int)
    rfm["M_score"] = pd.qcut(rfm["monetary"].rank(method="first"), q=3, labels=[1, 2, 3]).astype(int)
    rfm["RFM_score"] = rfm["R_score"] + rfm["F_score"] + rfm["M_score"]
    rfm["rfm_segment"] = rfm["RFM_score"].apply(
        lambda s: "High Value" if s >= 7 else ("Medium Value" if s >= 4 else "Low Value")
    )
    rfm["avg_order_value"] = rfm["monetary"] / rfm["frequency"]
    rfm["tenure_years"] = ((REFERENCE_DATE - rfm["customer_since"]).dt.days / 365).clip(lower=0.1)
    rfm["annual_frequency"] = rfm["frequency"] / rfm["tenure_years"]
    rfm["clv_projected_1yr"] = rfm["avg_order_value"] * rfm["annual_frequency"]
    rfm["is_inactive"] = rfm["recency"] > 90
    return rfm


def render_module2(data):
    st.title("👥 Customer Analytics")
    

    rfm = compute_rfm(data)
    seg_order = ["High Value", "Medium Value", "Low Value"]
    colors_seg = {"High Value": PRIMARY_BLUE, "Medium Value": DARK_BLUE, "Low Value": PALE_BLUE}
    seg_counts = rfm["rfm_segment"].value_counts()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{len(rfm):,}")
    c2.metric("High Value", f"{seg_counts.get('High Value', 0):,}")
    c3.metric("Inactive (90+ days)", f"{rfm['is_inactive'].sum():,}")
    c4.metric("Repeat Purchase Rate", f"{(rfm['frequency'] > 1).mean() * 100:.1f}%")

    st.markdown("### RFM Segment Distribution")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        vals = [seg_counts.get(s, 0) for s in seg_order]
        ax.bar(seg_order, vals, color=[colors_seg[s] for s in seg_order], edgecolor="white")
        ax.set_title("Customers per RFM Segment")
        ax.set_ylabel("Number of Customers")
        plt.tight_layout()
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie(vals, labels=[f"{s}\n{v:,}" for s, v in zip(seg_order, vals)],
               colors=[colors_seg[s] for s in seg_order], startangle=140,
               wedgeprops={"edgecolor": "white", "linewidth": 2})
        ax.set_title("Segment Share")
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("### Customer Lifetime Value (CLV)")
    clv_by_seg = rfm.groupby("rfm_segment")["clv_projected_1yr"].mean().reindex(seg_order)
    col1, col2 = st.columns([2, 1])
    with col1:
        clv_limit = rfm["clv_projected_1yr"].quantile(0.95)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(rfm["clv_projected_1yr"].clip(upper=clv_limit), bins=40, color=PRIMARY_BLUE, edgecolor="white")
        ax.axvline(rfm["clv_projected_1yr"].mean(), color="red", linestyle="--",
                   label=f"Mean: £{rfm['clv_projected_1yr'].mean():,.0f}")
        ax.set_title("Projected 1-Year CLV Distribution (top 5% clipped)")
        ax.set_xlabel("CLV (£)")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
    with col2:
        st.dataframe(clv_by_seg.round(2).rename("Avg Annual CLV (£)"), use_container_width=True)

    st.markdown("### Inactive Customers (90+ days)")
    inactive = rfm[rfm["is_inactive"]]
    col1, col2, col3 = st.columns(3)
    col1.metric("Inactive Customers", f"{len(inactive):,} ({len(inactive)/len(rfm)*100:.1f}%)")
    col2.metric("Historical Revenue at Risk", f"£{inactive['monetary'].sum():,.0f}")
    col3.metric("Median Spend (inactive)", f"£{inactive['monetary'].median():,.2f}")

    st.markdown("### Top 10 Customers by Total Revenue")
    top10 = rfm.nlargest(10, "monetary")[["customer_id", "monetary", "frequency", "recency", "rfm_segment"]]
    st.dataframe(top10.style.format({"monetary": "£{:,.2f}"}), use_container_width=True)

    st.markdown("### Acquisition Channel Quality")
    channel_quality = rfm.groupby("acquisition_channel").agg(
        num_customers=("customer_id", "count"),
        avg_spend=("monetary", "mean"),
        pct_high_value=("rfm_segment", lambda x: (x == "High Value").mean() * 100)
    ).round(2).reset_index().sort_values("avg_spend", ascending=False)
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(blue_bar(channel_quality, "acquisition_channel", "avg_spend",
                            "Avg Lifetime Spend by Acquisition Channel", "Channel", "Avg Spend (£)"))
    with col2:
        st.pyplot(blue_bar(channel_quality, "acquisition_channel", "pct_high_value",
                            "% High Value Customers by Acquisition Channel", "Channel", "% High Value"))


# =============================================================================
# PRODUCT & CHANNEL ANALYSIS
# =============================================================================
def render_module3(data):
    st.title("📦 Product & Channel Revenue Analysis")
    

    product_rev = (
        data.groupby(["product_id", "product_name"])["net_revenue"].sum()
        .reset_index().sort_values("net_revenue", ascending=False).reset_index(drop=True)
    )
    product_rev["cumulative_pct"] = product_rev["net_revenue"].cumsum() / product_rev["net_revenue"].sum() * 100
    n_products_for_80pct = (product_rev["cumulative_pct"] <= 80).sum() + 1
    pct_of_products = n_products_for_80pct / len(product_rev) * 100

    c1, c2 = st.columns(2)
    c1.metric("Products driving 80% of revenue", f"{n_products_for_80pct:,} of {len(product_rev):,}")
    c2.metric("% of product range", f"{pct_of_products:.1f}%")

    st.markdown("### Pareto Chart (Top 40 Products)")
    top_n = 40
    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.bar(range(top_n), product_rev["net_revenue"][:top_n], color=MID_BLUE)
    ax1.set_xlabel("Product rank")
    ax1.set_ylabel("Net revenue (£)", color=MID_BLUE)
    ax2 = ax1.twinx()
    ax2.plot(range(top_n), product_rev["cumulative_pct"][:top_n], color="#C00000", marker="o", markersize=3)
    ax2.axhline(80, color="gray", linestyle="--")
    ax2.set_ylabel("Cumulative % of revenue", color="#C00000")
    plt.title(f"Pareto Chart (top 40 of {len(product_rev):,} products; 80% reached at rank {n_products_for_80pct})")
    plt.tight_layout()
    st.pyplot(fig)

    product_summary = data.groupby(["product_id", "product_name", "category"]).agg(
        net_revenue=("net_revenue", "sum"), gross_margin=("gross_margin", "sum"),
        n_transactions=("transaction_id", "nunique")
    ).reset_index()

    st.markdown("### Top / Bottom Products")
    col1, col2 = st.columns(2)
    with col1:
        top10 = product_summary.sort_values("net_revenue", ascending=False).head(10)
        st.pyplot(blue_bar(top10, "product_name", "net_revenue", "Top 10 Products by Revenue",
                            "Net Revenue (£)", "", horizontal=True))
    with col2:
        bottom10 = product_summary[product_summary["n_transactions"] >= 5].sort_values("net_revenue").head(10)
        st.pyplot(blue_bar(bottom10, "product_name", "net_revenue", "Bottom 10 Products (>=5 sales)",
                            "Net Revenue (£)", "", horizontal=True, color=LIGHT_BLUE))

    st.markdown("### Category Contribution")
    category_revenue = data.groupby("category").agg(
        net_revenue=("net_revenue", "sum"), gross_margin=("gross_margin", "sum")
    ).reset_index().sort_values("net_revenue", ascending=False)
    category_revenue["revenue_share_pct"] = category_revenue["net_revenue"] / category_revenue["net_revenue"].sum() * 100
    category_revenue["margin_pct"] = category_revenue["gross_margin"] / category_revenue["net_revenue"] * 100
    st.pyplot(blue_bar(category_revenue, "category", "revenue_share_pct", "Revenue Share by Category (%)",
                        "% of Total Revenue", "", horizontal=True, figsize=(9, 5)))

    st.markdown("### Channel Comparison (Revenue vs Margin %)")
    channel_comparison = data.groupby("channel").agg(
        net_revenue=("net_revenue", "sum"), gross_margin=("gross_margin", "sum")
    ).reset_index().sort_values("net_revenue", ascending=False)
    channel_comparison["margin_pct"] = channel_comparison["gross_margin"] / channel_comparison["net_revenue"] * 100
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(blue_bar(channel_comparison, "channel", "net_revenue", "Net Revenue by Channel", "Channel", "Revenue (£)"))
    with col2:
        st.pyplot(blue_bar(channel_comparison, "channel", "margin_pct", "Margin % by Channel", "Channel", "Margin %", color=DARK_BLUE))

    st.markdown("### Branch Ranking")
    branch_ranking = data.groupby(["branch_id", "branch_name"]).agg(
        net_revenue=("net_revenue", "sum"), gross_margin=("gross_margin", "sum")
    ).reset_index().sort_values("net_revenue", ascending=False)
    branch_ranking["margin_pct"] = branch_ranking["gross_margin"] / branch_ranking["net_revenue"] * 100
    st.dataframe(branch_ranking[["branch_name", "net_revenue", "margin_pct"]]
                 .style.format({"net_revenue": "£{:,.0f}", "margin_pct": "{:.1f}%"}), use_container_width=True)


# =============================================================================
# PRICING & DISCOUNT ANALYSIS
# =============================================================================
def render_module4(data):
    st.title("💷 Pricing & Discount Analysis")
    

    bands = sorted(data["discount_percentage"].unique())

    st.markdown("### Discount Distribution")
    discount_counts = data["discount_percentage"].value_counts().sort_index().reset_index()
    discount_counts.columns = ["discount_percentage", "count"]
    st.pyplot(blue_bar(discount_counts, "discount_percentage", "count", "Discount Distribution",
                        "Discount %", "Number of Transactions"))

    st.markdown("### Revenue & Margin by Discount Band")
    band_stats = data.groupby("discount_percentage").agg(
        avg_net_revenue=("net_revenue", "mean"), avg_gross_margin=("gross_margin", "mean"),
        transaction_count=("transaction_id", "count")
    ).reset_index()
    band_stats["margin_pct"] = data.groupby("discount_percentage").apply(
        lambda g: g["gross_margin"].sum() / g["net_revenue"].sum() * 100, include_groups=False
    ).values
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(blue_bar(band_stats, "discount_percentage", "margin_pct",
                            "Average Margin % by Discount Band", "Discount %", "Margin %"))
    with col2:
        st.dataframe(band_stats.round(2), use_container_width=True)

    st.markdown("### Correlation: Discount % vs Quantity / Revenue")
    spear_qty, spear_qty_p = stats.spearmanr(data["discount_percentage"], data["quantity"])
    spear_rev, spear_rev_p = stats.spearmanr(data["discount_percentage"], data["net_revenue"])

    def interpret_corr(r):
        strength = "Weak" if abs(r) < 0.3 else "Moderate" if abs(r) < 0.7 else "Strong"
        direction = "negative" if r < 0 else "positive"
        return f"{strength} {direction} relationship"

    col1, col2 = st.columns(2)
    col1.metric("Spearman r (discount vs quantity)", f"{spear_qty:.4f}", interpret_corr(spear_qty))
    col2.metric("Spearman r (discount vs revenue)", f"{spear_rev:.4f}", interpret_corr(spear_rev))
    st.caption(f"p-values: quantity p={spear_qty_p:.4f}, revenue p={spear_rev_p:.6f} "
               f"({'significant' if spear_rev_p < 0.05 else 'not significant'} at 0.05 level)")

    st.markdown("### Shapiro-Wilk Normality Test & Kruskal-Wallis (Nonparametric)")
    groups = [data.loc[data["discount_percentage"] == b, "gross_margin"] for b in bands]

    # Shapiro-Wilk tests whether each discount band's margin distribution is normal.
    # Large samples are capped at 5000 (scipy's practical limit / sensitivity threshold)
    # so the test remains numerically stable and interpretable.
    shapiro_rows = []
    for b, g in zip(bands, groups):
        sample = g.sample(n=min(len(g), 5000), random_state=42) if len(g) > 5000 else g
        stat, p = stats.shapiro(sample)
        shapiro_rows.append({
            "discount_band": b, "n": len(g), "shapiro_stat": round(stat, 4),
            "p_value": round(p, 6), "normal_at_5%": "Yes" if p >= 0.05 else "No"
        })
    shapiro_df = pd.DataFrame(shapiro_rows)

    kw_stat, kw_p = stats.kruskal(*groups)
    levene_stat, levene_p = stats.levene(*groups)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(shapiro_df, use_container_width=True)
    with col2:
        st.metric("Kruskal-Wallis H", f"{kw_stat:.2f}", f"p={kw_p:.6f}")
        st.metric("Levene's test (equal variance)", f"{levene_stat:.2f}", f"p={levene_p:.4f}")

    any_non_normal = (shapiro_df["normal_at_5%"] == "No").any()
    if any_non_normal:
        st.info("Shapiro-Wilk finds at least one discount band's margin distribution is **not normal** "
                "(p < 0.05), and/or Levene's test shows unequal variances. Both violate the assumptions "
                "behind a standard ANOVA, so **Kruskal-Wallis is the appropriate test here** - it makes "
                "no assumption of normality or equal variance.")
    else:
        st.success("All discount bands pass the Shapiro-Wilk normality check. Kruskal-Wallis is still "
                    "reported since it is a fully non-parametric, assumption-light test.")

    st.markdown("### Campaign vs Non-Campaign")
    campaign_comparison = data.groupby("is_campaign").agg(
        avg_net_revenue=("net_revenue", "mean"), avg_gross_margin=("gross_margin", "mean"),
        transaction_count=("transaction_id", "count")
    ).reset_index()
    campaign_comparison["is_campaign"] = campaign_comparison["is_campaign"].map({True: "Campaign", False: "Non-Campaign"})
    margin_camp = data.loc[data["is_campaign"], "gross_margin"]
    margin_noncamp = data.loc[~data["is_campaign"], "gross_margin"]
    t_stat, t_p = stats.ttest_ind(margin_camp, margin_noncamp, equal_var=False)
    col1, col2 = st.columns([1, 1])
    with col1:
        st.dataframe(campaign_comparison.round(2), use_container_width=True)
    with col2:
        st.metric("Welch's t-test (margin, campaign vs non)", f"t={t_stat:.3f}", f"p={t_p:.4f}")

    st.markdown("### Over-Discounted, Loss-Making Transactions")
    over_discounted = data[(data["discount_percentage"] >= 20) & (data["gross_margin"] < 0)]
    c1, c2 = st.columns(2)
    c1.metric("Transactions flagged", f"{len(over_discounted):,}")
    c2.metric("Total margin lost", f"£{over_discounted['gross_margin'].sum():,.0f}")
    with st.expander("Top 10 products in the over-discounted list"):
        top_products = over_discounted.groupby("product_name")["gross_margin"].agg(["count", "sum"]).sort_values("count", ascending=False).head(10)
        st.dataframe(top_products, use_container_width=True)


# =============================================================================
# REVENUE FORECASTING
# =============================================================================
def render_module5(data):
    st.title("📈 Revenue Forecasting")
    

    monthly = data.groupby("year_month")["net_revenue"].sum().reset_index()
    monthly["year_month"] = monthly["year_month"].astype(str)
    series = monthly.set_index("year_month")["net_revenue"]
    series.index = pd.PeriodIndex(series.index, freq="M")

    n_forecast = 6
    ma3 = series.rolling(window=3).mean()
    last_ma = ma3.iloc[-1]
    ma_forecast = pd.Series([last_ma] * n_forecast)

    ses_model = Holt(series.values, initialization_method="estimated").fit()
    ses_forecast = ses_model.forecast(n_forecast)
    alpha = ses_model.params["smoothing_level"]
    beta = ses_model.params["smoothing_trend"]

    future_periods = pd.period_range(series.index[-1] + 1, periods=n_forecast, freq="M")
    hist_df = pd.DataFrame({"year_month": series.index.astype(str), "actual_revenue": series.values})
    fc_df = pd.DataFrame({
        "year_month": future_periods.astype(str),
        "moving_avg_forecast": ma_forecast.round(2).values,
        "holt_trend_forecast": ses_forecast.round(2),
    })
    fc_df["best_case_x1.10"] = (fc_df["holt_trend_forecast"] * 1.10).round(2)
    fc_df["base_case"] = fc_df["holt_trend_forecast"]
    fc_df["low_case_x0.90"] = (fc_df["holt_trend_forecast"] * 0.90).round(2)

    st.markdown("### Historical Revenue + 6-Month Forecast")
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(hist_df["year_month"], hist_df["actual_revenue"], marker="o", color=NAVY, label="Actual Revenue")
    ax.plot(fc_df["year_month"], fc_df["base_case"], marker="o", linestyle="--", color="#C9772F", label="Forecast (Base Case)")
    ax.fill_between(fc_df["year_month"], fc_df["low_case_x0.90"], fc_df["best_case_x1.10"],
                     color="#C9772F", alpha=0.15, label="Low-High Case Range")
    ax.set_title("Revenue Forecast - Next 6 Months (Holt's Trend Method)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Net Revenue (£)")
    plt.xticks(rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("### Forecast Accuracy (back-tested on last 3 known months)")
    test_size = 3
    train, test = series.iloc[:-test_size], series.iloc[-test_size:]
    ma_bt_value = train.rolling(window=3).mean().iloc[-1]
    ma_bt_preds = np.array([ma_bt_value] * test_size)
    ses_bt_model = Holt(train.values, initialization_method="estimated").fit()
    ses_bt_preds = ses_bt_model.forecast(test_size)

    def accuracy(actual, pred):
        actual, pred = np.array(actual), np.array(pred)
        mae = np.mean(np.abs(actual - pred))
        rmse = np.sqrt(np.mean((actual - pred) ** 2))
        mape = np.mean(np.abs((actual - pred) / actual)) * 100
        return mae, rmse, mape

    ma_mae, ma_rmse, ma_mape = accuracy(test.values, ma_bt_preds)
    ses_mae, ses_rmse, ses_mape = accuracy(test.values, ses_bt_preds)
    accuracy_df = pd.DataFrame([
        {"Method": "Moving Average (3mo)", "MAE": round(ma_mae, 2), "RMSE": round(ma_rmse, 2), "MAPE %": round(ma_mape, 2)},
        {"Method": "Holt's Trend Smoothing", "MAE": round(ses_mae, 2), "RMSE": round(ses_rmse, 2), "MAPE %": round(ses_mape, 2)},
    ])

    st.dataframe(accuracy_df, use_container_width=True)
    better = "Holt's Trend Smoothing" if ses_mape < ma_mape else "Moving Average"
    st.success(f"Lower MAPE (more accurate): **{better}**")
    st.caption(f"Holt's method parameters - alpha (level): {alpha:.3f}, beta (trend): {beta:.3f}")

    with st.expander("Full forecast table"):
        st.dataframe(fc_df, use_container_width=True)


# =============================================================================
# REVENUE LEAKAGE
# =============================================================================
def render_module6(data):
    st.title("⚠️ Revenue Leakage & Underperformance")
    

    overdue = data[data["payment_status"] == "Overdue"]
    partial = data[data["payment_status"] == "Partially Paid"]
    neg_margin = data[data["gross_margin"] < 0]
    over_discounted = data[(data["discount_percentage"] >= 20) & (data["gross_margin"] < 0)]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overdue Revenue", f"£{overdue['net_revenue'].sum():,.0f}")
    c2.metric("Partially Paid Revenue", f"£{partial['net_revenue'].sum():,.0f}")
    c3.metric("Negative-Margin Loss", f"£{neg_margin['gross_margin'].sum():,.0f}")
    c4.metric("Over-Discounted Loss", f"£{over_discounted['gross_margin'].sum():,.0f}")

    st.markdown("### Overdue Rate by Channel & Branch (the view that matters)")
    st.caption("Absolute £ overdue is skewed by transaction volume - rate (%) reveals the real problem areas.")
    overdue_rate_channel = data.groupby("channel")["payment_status"].apply(
        lambda x: (x == "Overdue").mean() * 100).sort_values(ascending=False).reset_index()
    overdue_rate_channel.columns = ["channel", "overdue_rate_pct"]
    overdue_rate_branch = data.groupby("branch_name")["payment_status"].apply(
        lambda x: (x == "Overdue").mean() * 100).sort_values(ascending=False).reset_index()
    overdue_rate_branch.columns = ["branch_name", "overdue_rate_pct"]

    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(blue_bar(overdue_rate_channel, "channel", "overdue_rate_pct",
                            "Overdue Rate (%) by Channel", "Channel", "Overdue Rate %"))
    with col2:
        st.pyplot(blue_bar(overdue_rate_branch, "branch_name", "overdue_rate_pct",
                            "Overdue Rate (%) by Branch", "Branch", "Overdue Rate %"))

    st.markdown("### Chi-Square Test: Is overdue rate related to channel?")
    chi_data = data[data["payment_status"].isin(["Paid", "Overdue"])]
    contingency = pd.crosstab(chi_data["channel"], chi_data["payment_status"])
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    col1, col2 = st.columns([1, 1])
    with col1:
        st.dataframe(contingency, use_container_width=True)
    with col2:
        st.metric("Chi-square statistic", f"{chi2:.2f}", f"p={p_value:.6f}, df={dof}")
        if p_value < 0.05:
            st.info("p < 0.05: overdue rate IS significantly different across channels - "
                    "this is a genuine, systemic pattern, not random chance.")
        else:
            st.info("p >= 0.05: no statistically significant difference in overdue rate across channels.")

    st.markdown("### Negative Margin Transactions")
    col1, col2 = st.columns(2)
    with col1:
        neg_by_category = neg_margin.groupby("category")["gross_margin"].sum().sort_values().head(5).reset_index()
        st.pyplot(blue_bar(neg_by_category, "category", "gross_margin", "Worst 5 Categories by Lost Margin",
                            "Category", "Lost Margin (£)", color=DARK_BLUE))
    with col2:
        neg_by_channel = neg_margin.groupby("channel")["gross_margin"].sum().sort_values().reset_index()
        st.pyplot(blue_bar(neg_by_channel, "channel", "gross_margin", "Lost Margin by Channel",
                            "Channel", "Lost Margin (£)", color=DARK_BLUE))

    st.markdown("### Inactive Customer Loss Estimate")
    MAX_DATE = data["transaction_date"].max()
    customer_last_purchase = data.groupby("customer_id")["transaction_date"].max()
    inactive_customers = customer_last_purchase[(MAX_DATE - customer_last_purchase).dt.days > 90]
    customer_median_spend = data.groupby("customer_id")["net_revenue"].median()
    inactive_loss_estimate = customer_median_spend.loc[inactive_customers.index].sum()
    col1, col2 = st.columns(2)
    col1.metric("Inactive Customers (90+ days)", f"{len(inactive_customers):,} "
                f"({len(inactive_customers)/data['customer_id'].nunique()*100:.1f}%)")
    col2.metric("Estimated At-Risk Revenue", f"£{inactive_loss_estimate:,.0f}",
                help="Median-based, not mean-based - net_revenue is heavily right-skewed")


# =============================================================================
# RECOMMENDATIONS
# =============================================================================
def render_module7(data):
    st.title("✅ Business Recommendations")
    

    # Pull a few live numbers so recommendations stay grounded in the current filtered data
    channel_comparison = data.groupby("channel").agg(
        net_revenue=("net_revenue", "sum"), gross_margin=("gross_margin", "sum")
    ).reset_index()
    channel_comparison["margin_pct"] = channel_comparison["gross_margin"] / channel_comparison["net_revenue"] * 100
    worst_margin_channel = channel_comparison.sort_values("margin_pct").iloc[0]

    overdue_rate_channel = data.groupby("channel")["payment_status"].apply(lambda x: (x == "Overdue").mean() * 100)
    worst_overdue_channel = overdue_rate_channel.idxmax()

    neg_margin = data[data["gross_margin"] < 0]
    top_neg_category = neg_margin.groupby("category")["gross_margin"].sum().idxmin() if len(neg_margin) else "N/A"

    recommendations = [
        {"Finding": f"{worst_overdue_channel} channel has the highest overdue-payment rate "
                    f"({overdue_rate_channel.max():.1f}%)",
         "Recommendation": "Introduce stricter credit terms specifically for this channel",
         "Priority": "Quick Win"},
        {"Finding": f"{worst_margin_channel['channel']} has the weakest margin "
                    f"({worst_margin_channel['margin_pct']:.1f}%) despite meaningful revenue",
         "Recommendation": "Review commission/operating costs on this channel",
         "Priority": "Medium-term"},
        {"Finding": f"{top_neg_category} is the category with the largest total negative-margin loss",
         "Recommendation": "Apply a hard discount cap for this category to prevent below-cost sales",
         "Priority": "Quick Win"},
        {"Finding": "A meaningful share of customers have not purchased in 90+ days",
         "Recommendation": "Launch a targeted reactivation campaign for inactive customers",
         "Priority": "Medium-term"},
        {"Finding": "A small subset of products drives the large majority of revenue (Pareto pattern)",
         "Recommendation": "Prioritise stock and supplier reliability for this core product group",
         "Priority": "Strategic"},
        {"Finding": "Discount % shows a weak/negative correlation with revenue and quantity",
         "Recommendation": "Reduce blanket discounting; target discounts to specific segments instead",
         "Priority": "Strategic"},
    ]
    rec_df = pd.DataFrame(recommendations)
    rec_df.insert(0, "#", range(1, len(rec_df) + 1))

    priority_colors = {"Quick Win": "🟢", "Medium-term": "🟡", "Strategic": "🔵"}
    for _, row in rec_df.iterrows():
        with st.container():
            st.markdown(f"**{priority_colors.get(row['Priority'], '')} {row['Priority']} — Finding #{row['#']}**")
            st.markdown(f"*Finding:* {row['Finding']}")
            st.markdown(f"*Recommendation:* {row['Recommendation']}")
            st.markdown("---")

    st.download_button("Download recommendations as CSV", rec_df.to_csv(index=False),
                        file_name="recommendations.csv", mime="text/csv")


# =============================================================================
# ROUTER
# =============================================================================
if page.startswith("🏠"):
    render_module1(df_f)
elif page.startswith("👥"):
    render_module2(df_f)
elif page.startswith("📦"):
    render_module3(df_f)
elif page.startswith("💷"):
    render_module4(df_f)
elif page.startswith("📈"):
    render_module5(df_f)
elif page.startswith("⚠️"):
    render_module6(df_f)
elif page.startswith("✅"):
    render_module7(df_f)
