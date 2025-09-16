import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

st.set_page_config(
    page_title="LLM Cost & Latency Tracker (Mock)",
    page_icon="🧮",
    layout="wide"
)

# ---------------------
# Model Setup - Note: just mock data, nothing real here
# ---------------------
MODELS = {
    "GPT-4o-mini (mock)": {"in_per_1k": 0.15, "out_per_1k": 0.60, "base_latency_ms": 350},
    "GPT-4o (mock)": {"in_per_1k": 5.00, "out_per_1k": 15.00, "base_latency_ms": 800},
    "Claude 3.5 Sonnet (mock)": {"in_per_1k": 3.00, "out_per_1k": 15.00, "base_latency_ms": 900},
    "Llama-3.1 8B (mock)": {"in_per_1k": 0.05, "out_per_1k": 0.08, "base_latency_ms": 300},
    "Gemini Pro (mock)": {"in_per_1k": 0.50, "out_per_1k": 1.50, "base_latency_ms": 450},
}

# ---------------------
# Sidebar Sliders - pick your poison
# ---------------------
with st.sidebar:
    st.title("⚙️ Settings")
    
    model_choice = st.selectbox("Model", list(MODELS.keys()))
    rpm = st.slider("Requests per minute", 1, 600, 60)
    in_tok = st.slider("Avg input tokens / request", 1, 8000, 1000)
    out_tok = st.slider("Avg output tokens / request", 1, 8000, 500)
    hours = st.slider("Hours active per day", 1, 24, 8)
    jitter = st.slider("Latency jitter ± (%)", 0, 200, 25)
    samples = st.slider("Simulation samples", 50, 1000, 200)
    
    st.markdown("---")
    st.caption("⚠️ Just playing with numbers — nothing here is real!")

model = MODELS[model_choice]

# ---------------------
# Cost math — a bit simplistic, but fine for mock
# ---------------------
req_cost = (in_tok / 1000) * model["in_per_1k"] + (out_tok / 1000) * model["out_per_1k"]
cost_min = rpm * req_cost
cost_hour = cost_min * 60
cost_day = cost_hour * hours
cost_month = cost_day * 30  # just assuming a month is 30 days

# ---------------------
# Latency simulation — using normal distribution plus some chaos
# ---------------------
np.random.seed(42)

base_latency = model["base_latency_ms"]
jitter_amount = jitter / 100

latencies = np.random.normal(
    loc=base_latency,
    scale=max(5, base_latency * 0.15 * (1 + jitter_amount)),
    size=samples
)
latencies = np.clip(latencies, 10, None)  # let’s not go below 10ms

# Add pressure multiplier for high load situations
rpm_effect = np.log1p(rpm / 60)
latencies *= (1 + 0.3 * rpm_effect)

# Inject random latency spikes for realism
spikes = np.random.choice(samples, int(samples * 0.1), replace=False)
latencies[spikes] *= np.random.uniform(2, 5, len(spikes))

# ---------------------
# DataFrame for simulation output
# ---------------------
df = pd.DataFrame({
    "request_id": np.arange(1, samples + 1),
    "latency_ms": latencies,
    "in_tokens": np.random.normal(in_tok, max(10, in_tok * 0.15), samples).clip(1),
    "out_tokens": np.random.normal(out_tok, max(10, out_tok * 0.15), samples).clip(1),
})

df["cost_usd"] = (df["in_tokens"] / 1000 * model["in_per_1k"] +
                  df["out_tokens"] / 1000 * model["out_per_1k"])

# ---------------------
# Time-based cost series — kinda synthetic but shows trends
# ---------------------
minutes = hours * 60
time_series = pd.date_range(
    datetime.now().replace(hour=9, minute=0, second=0), periods=minutes, freq="min"
)

min_costs = np.full(minutes, cost_min)
daily_wave = 0.5 + 0.5 * np.sin(np.linspace(0, 2 * np.pi, minutes))
min_costs *= daily_wave
cum_cost = np.cumsum(min_costs)

cost_df = pd.DataFrame({
    "time": time_series,
    "cumulative_cost_usd": cum_cost
})

# ---------------------
# UI Layout
# ---------------------
st.title("🧮 LLM Cost & Latency Tracker")
st.markdown("### Simulated dashboard — play around to see the cost/latency effects")
st.markdown("---")

# Main metrics
col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("💰 Cost/Request", f"${req_cost:.4f}")
col_b.metric("📅 Daily Cost", f"${cost_day:.2f}")
col_c.metric("📈 30-Day Cost", f"${cost_month:,.0f}")
col_d.metric("⚡ Avg Latency", f"{df['latency_ms'].mean():.0f} ms")

st.caption(f"**{model_choice}** • ${model['in_per_1k']}/1k in • ${model['out_per_1k']}/1k out • Base: {model['base_latency_ms']}ms")

# ---------------------
# Tabs — Cost, Latency, Stats, Raw Data
# ---------------------
tab1, tab2, tab3, tab4 = st.tabs(["📈 Cost Projection", "⏱️ Latency Analysis", "📊 Performance Stats", "🔢 Raw Data"])

with tab1:
    st.subheader("Cumulative Cost Over the Day (Simulated)")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(cost_df["time"], cost_df["cumulative_cost_usd"], color="steelblue")
    ax.fill_between(cost_df["time"], cost_df["cumulative_cost_usd"], alpha=0.2)
    ax.set_title("Estimated Daily Spend")
    ax.set_ylabel("USD")
    ax.set_xlabel("Time")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.autofmt_xdate()
    st.pyplot(fig)
    
    st.info(f"💡 If you ran this for {hours} hours/day at {rpm} RPM, you'd spend about ${cost_day:.2f}/day")

with tab2:
    st.subheader("Latency Simulation (MS)")
    
    c1, c2 = st.columns(2)
    with c1:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.hist(df["latency_ms"], bins=30, color='orange', edgecolor='black', alpha=0.8)
        ax2.axvline(df["latency_ms"].mean(), linestyle='--', color='red', label="Mean")
        ax2.axvline(df["latency_ms"].quantile(0.95), linestyle='--', color='green', label="P95")
        ax2.set_title("Latency Spread")
        ax2.set_xlabel("ms")
        ax2.set_ylabel("Frequency")
        ax2.legend()
        ax2.grid(True, alpha=0.2)
        st.pyplot(fig2)
    
    with c2:
        st.markdown("**Latency Percentiles:**")
        for p in [50, 90, 95, 99]:
            st.metric(f"P{p}", f"{df['latency_ms'].quantile(p/100):.0f} ms")

with tab3:
    st.subheader("Token Usage & Cost Efficiency")
    
    left, right = st.columns(2)
    with left:
        st.metric("Avg Input Tokens", f"{df['in_tokens'].mean():.0f}")
        st.metric("Avg Output Tokens", f"{df['out_tokens'].mean():.0f}")
        total_toks = (in_tok + out_tok) * rpm * 60 * hours
        st.metric("Est. Total Tokens/Day", f"{total_toks:,.0f}")
    
    with right:
        toks_per_dollar = (in_tok + out_tok) / req_cost
        st.metric("Tokens per $1", f"{toks_per_dollar:,.0f}")
        st.metric("Cost per 1M tokens", f"${req_cost * 1_000_000 / (in_tok + out_tok):.2f}")
        
        if cost_month > 1000:
            st.warning("💸 Might wanna dial this back — that’s $$$")
        elif cost_month < 100:
            st.success("🟢 Cost setup is very reasonable.")
        else:
            st.info("🟡 Not bad — probably worth tracking.")

with tab4:
    st.subheader("Simulation Data (Details)")
    
    a, b, c = st.columns(3)
    a.metric("Total Requests", f"{len(df)}")
    b.metric("Failed Requests", f"{len(df[df['latency_ms'] > 10000])}")  # arbitrarily say >10s is a fail
    c.metric("Avg Cost/Req", f"${df['cost_usd'].mean():.4f}")
    
    shown_df = df.copy()
    shown_df = shown_df.round({
        'latency_ms': 0,
        'in_tokens': 0,
        'out_tokens': 0,
        'cost_usd': 4
    })
    
    st.dataframe(
        shown_df.style.format({
            'latency_ms': '{:.0f} ms',
            'in_tokens': '{:.0f}',
            'out_tokens': '{:.0f}',
            'cost_usd': '${:.4f}'
        }),
        use_container_width=True,
        height=300
    )

    csv = df.to_csv(index=False)
    st.download_button(
        "📥 Download CSV",
        csv.encode('utf-8'),
        f"llm_mock_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        "text/csv"
    )

# ---------------------
# Footer
# ---------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    ⚠️ This is totally fictional. Nothing here should guide your infra spend.<br>
    Made for fun, demo, and dashboard experimentation.
</div>
""", unsafe_allow_html=True)
