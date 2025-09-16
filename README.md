LLM Cost & Latency Tracker (Mock)

A full-fledged Streamlit App simulating LLM cost analysis and latency performance with real-world mock data. Good to understand the cost optimization and performance trade-offs in LLM applications.

## **[LIVE DEMO] (https://llm-cost-tracker-qjru45xwrgp6tkz7vbmjpw.streamlit.app/)**

Features


Interactive Cost Analysis

- Cost projections based on total costs per request, daily and monthly cost
- Support for multiple mock LLM models (GPT-4, Claude, Llama, and others)
- Dynamic pricing based on the number of input/output tokens used
- Future cost projection based on daily usage patterns

Latency Performance Simulation


- A realistic distribution of potential latency with a configurable level of jitter
- Experience the effects of queue pressure depending upon request rate
- Breakdown of latency performance into percentiles (P50, P90-P95-P99)
- Timing outliers that simulate real-world spikes in latency

Configurable Parameters

- Requests per minute (1-600 RPM)
- Average number of input/output tokens per request
- Total hours of use on a daily basis
- Range for latency jitter and controlling total sample size


Rich Visualizations


- Cost projected over time with usage patterns
- Latency distribution histogram with percentile lines / markers
- Performance metrics and efficiency calculations
- Ability to export simulation data (CSV)


Technical Features


- Built with Streamlit to create responsive web-based user interface
- Used NumPy/Pandas for fast processing of data
- Created Matplotlib visualizations to a publication-level of formatting
- Used professional looking metrics cards and tabs.


Getting Started
Option 1: Streamlit Cloud (Recommended)


Fork Repository
Navigate to share.streamlit.io
Connect your GitHub and deploy!

Option 2: Local Development
# Clone the repo
git clone https://github.com/YOUR_USERNAME/llm-cost-tracker.git
cd llm-cost-tracker

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py

Model Catalog (Mock Data)

Model Input ($/1k tokens) | Output ($/1k tokens) | Base Latency
GPT-4o-mini: $0.15 | $0.60 | 350ms
GPT-4o: $5.00 | $15.00 | 800ms 
Claude 3.5 Sonnet: $3.00 | $15.00 | 900ms 
Llama-3.1 8B: $0.05 | $0.08 | 300ms 
Gemini Pro: $0.50 | $1.50 | 450ms


 Use Cases


Cost Planning: Estimate costs of LLM usage for different workloads
Performance Analysis: Understand latency distributions and bottlenecks
Model Comparison: Compare cost-performance of different models
Capacity Planning: Optimize request rates and instances
Educational: Learn about the economics and performance characteristics of LLMs

Important Disclaimer

The pricing and latency data is FICTIONAL and for demo purposes only. Do not rely on this tool or data for actual production budgeting or planning. Always refer to official documentation provided by the provider for pricing.

Tech Stack


Front End: Streamlit
Data Processing: Pandas, NumPy
Visualization: Matplotlib
Deployment: Streamlit Community Cloud

License 
MIT License - feel free to use this for educational and portfolio purposes as desired!

