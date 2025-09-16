import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

st.set_page_config(page_title="LLM Cost & Latency Tracker (Mock)", page_icon="🧮", layout="wide")

# -------------------------
# MOCK MODEL CATALOG (fictional numbers!)
# -------------------------
MODELS = {
    "GPT-4o-mini (mock)": {"in_per_1k": 0.15, "out_per_1k": 0.60, "base_latency_ms": 350},
    "GPT-4o (mock)": {"in_per_1k": 5.00, "out_per_1k": 15.00, "base_latency_ms": 800},
    "Claude 3.5 Sonnet (mock)": {"in_per_1k": 3.00, "out_per_1k": 15.00, "base_latency_ms": 900},
    "Llama-3.1 8B (mock)": {"in_per_1k": 0.05, "out_per_1k": 0.08, "base_latency_ms": 300},
    "Gemini Pro (mock)": {"in_per_1k": 0.50, "out_per_1k": 1.50, "base_latency_ms": 450},
}

# -------------------------
# SIDEBAR CONTROLS
# -------------------------
with st.sidebar:
    st.title("⚙️ Settings")
    model_name = st.selectbox("Model", list(MODELS.keys()))
    rpm = st.slider("Requests per minute", 1, 600, 60, help="How many requests you send per minute.")
    avg_in_tokens = st.slider("Avg input tokens / request", 1, 8000, 1000)
    avg_out_tokens = st.slider("Avg output tokens / request", 1, 8000, 500)
    hours_per_day = st.slider("Hours active per day", 1, 24, 8)
    jitter_pct = st.slider("Latency jitter ± (%)", 0, 200, 25, help="Random variation around base latency.")
    sample_points = st.slider("Simulation samples", 50, 1000, 200)

    st.markdown("---")
    st.caption("⚠️ All numbers are **MOCK/SIMULATED** for demo purposes only!")

cfg = MODELS[model_name]

# -------------------------
# COST CALCULATIONS (mock)
# -------------------------
cost_per_req = (avg_in_tokens / 1000.0) * cfg["in_per_1k"] + (avg_out_tokens / 1000.0) * cfg["out_per_1k"]
cost_per_min = rpm * cost_per_req
cost_per_hour = cost_per_min * 60
cost_per_day = cost_per_hour * hours_per_day
cost_per_30d = cost_per_day * 30

# -------------------------
# LATENCY SIMULATION (mock)
# -------------------------
np.random.seed(42)
base = cfg["base_latency_ms"]
jitter = jitter_pct / 100.0

# Generate latencies with normal distribution + jitter
latencies = np.random.normal(loc=base, scale=max(5, base * 0.15 * (1 + jitter)), size=sample_points)
latencies = np.clip(latencies, 10, None)  # minimum 10ms

# Add queue pressure effect (higher RPM = more tail latency)
tail_boost = np.log1p(rpm / 60.0)  # logarithmic scaling
latencies = latencies * (1 + 0.3 * tail_boost)

# Add some realistic spikes (10% of requests have 2-5x latency)
spike_indices = np.random.choice(sample_points, int(sample_points * 0.1), replace=False)
latencies[spike_indices] *= np.random.uniform(2, 5, len(spike_indices))

# Build simulation dataframe
df = pd.DataFrame({
    "request_id": np.arange(1, sample_points + 1),
    "latency_ms": latencies,
    "in_tokens": np.random.normal(avg_in_tokens, max(10, avg_in_tokens*0.15), sample_points).clip(1),
    "out_tokens": np.random.normal(avg_out_tokens, max(10, avg_out_tokens*0.15), sample_points).clip(1),
})
df["cost_usd"] = ((df["in_tokens"]/1000.0) * cfg["in_per_1k"] +
                  (df["out_tokens"]/1000.0) * cfg["out_per_1k"])

# Cumulative cost over simulated day
minutes = hours_per_day * 60
t = pd.date_range(datetime.now().replace(hour=9, minute=0, second=0), periods=minutes, freq="min")
per_min_cost = np.full(minutes, cost_per_min)
# Add some realistic variation throughout the day
daily_pattern = 0.5 + 0.5 * np.sin(np.linspace(0, 2*np.pi, minutes))  # sine wave pattern
per_min_cost = per_min_cost * daily_pattern
cum_cost = np.cumsum(per_min_cost)
cost_series = pd.DataFrame({"time": t, "cumulative_cost_usd": cum_cost})

# -------------------------
# MAIN APP LAYOUT
# -------------------------
st.title("🧮 LLM Cost & Latency Tracker")
st.markdown("### Interactive **MOCK** dashboard to explore cost and latency trade-offs")
st.markdown("---")

# Top metrics row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("💰 Cost/Request", f"${cost_per_req:.4f}")
with col2:
    st.metric("📅 Daily Cost", f"${cost_per_day:.2f}")
with col3:
    st.metric("📈 30-Day Cost", f"${cost_per_30d:,.0f}")
with col4:
    st.metric("⚡ Avg Latency", f"{df['latency_ms'].mean():.0f} ms")

st.caption(f"**{model_name}** • Input: ${cfg['in_per_1k']}/1k tokens • Output: ${cfg['out_per_1k']}/1k tokens • Base: {cfg['base_latency_ms']}ms")

# -------------------------
# CHARTS AND DATA TABS
# -------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📈 Cost Projection", "⏱️ Latency Analysis", "📊 Performance Stats", "🔢 Raw Data"])

with tab1:
    st.subheader("💸 Cumulative Cost Throughout Day")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(cost_series["time"], cost_series["cumulative_cost_usd"], color='#1f77b4', linewidth=2)
    ax1.fill_between(cost_series["time"], cost_series["cumulative_cost_usd"], alpha=0.3)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Cumulative Cost (USD)")
    ax1.set_title("Projected Daily Spending (Mock)")
    ax1.grid(True, alpha=0.3)
    fig1.autofmt_xdate()
    plt.tight_layout()
    st.pyplot(fig1, clear_figure=True)
    
    st.info(f"💡 **Insight**: At {rpm} RPM for {hours_per_day}h, you'd spend ~${cost_per_day:.2f}/day")

with tab2:
    st.subheader("🚀 Latency Distribution & Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.hist(df["latency_ms"], bins=25, alpha=0.7, color='orange', edgecolor='black')
        ax2.axvline(df["latency_ms"].mean(), color='red', linestyle='--', label=f'Mean: {df["latency_ms"].mean():.0f}ms')
        ax2.axvline(df["latency_ms"].quantile(0.95), color='green', linestyle='--', label=f'P95: {df["latency_ms"].quantile(0.95):.0f}ms')
        ax2.set_xlabel("Latency (ms)")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Latency Distribution (Mock)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2, clear_figure=True)
    
    with col2:
        st.markdown("**Latency Percentiles:**")
        percentiles = [50, 90, 95, 99]
        for p in percentiles:
            val = df["latency_ms"].quantile(p/100)
            st.metric(f"P{p}", f"{val:.0f} ms")

with tab3:
    st.subheader("📊 Performance & Cost Breakdown")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Token Usage:**")
        st.metric("Avg Input Tokens", f"{df['in_tokens'].mean():.0f}")
        st.metric("Avg Output Tokens", f"{df['out_tokens'].mean():.0f}")
        st.metric("Total Tokens/Day", f"{(avg_in_tokens + avg_out_tokens) * rpm * 60 * hours_per_day:,.0f}")
    
    with col2:
        st.markdown("**Cost Efficiency:**")
        tokens_per_dollar = (avg_in_tokens + avg_out_tokens) / cost_per_req
        st.metric("Tokens per $1", f"{tokens_per_dollar:,.0f}")
        st.metric("Cost per 1M tokens", f"${cost_per_req * 1000000 / (avg_in_tokens + avg_out_tokens):.2f}")
        
        if cost_per_30d > 1000:
            st.warning("💸 High monthly cost! Consider optimizing.")
        elif cost_per_30d < 100:
            st.success("💚 Very cost-effective setup!")
        else:
            st.info("📊 Moderate monthly spending.")

with tab4:
    st.subheader("🔍 Detailed Simulation Data")
    
    # Summary stats
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Requests", len(df))
    col2.metric("Failed Requests", len(df[df["latency_ms"] > 10000]))  # mock failure threshold
    col3.metric("Avg Cost/Request", f"${df['cost_usd'].mean():.4f}")
    
    # Data table with formatting
    display_df = df.copy()
    display_df = display_df.round({
        'latency_ms': 0,
        'in_tokens': 0, 
        'out_tokens': 0,
        'cost_usd': 4
    })
    
    st.dataframe(
        display_df.style.format({
            'latency_ms': '{:.0f} ms',
            'in_tokens': '{:.0f}',
            'out_tokens': '{:.0f}',
            'cost_usd': '${:.4f}'
        }),
        use_container_width=True,
        height=300
    )
    
    # Download options
    csv = df.to_csv(index=False)
    st.download_button(
        "📥 Download Full Dataset (CSV)",
        csv.encode('utf-8'),
        f"llm_cost_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        "text/csv",
        key='download-csv'
    )

# -------------------------
# FOOTER
# -------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>⚠️ DISCLAIMER:</strong> This dashboard uses <strong>FICTIONAL</strong> pricing and latency data for demonstration only.<br>
    Do NOT use these numbers for real budgeting or production planning.</p>
    <p>Built with Streamlit • Mock data simulation • Educational purposes</p>
</div>
""", unsafe_allow_html=True)
