# CrowdWisdomTrading – Crypto Predictions Agent

A backend Python agent system that predicts 5-minute crypto price moves for **Ethereum (ETH)** and **Bitcoin (BTC)** by combining:

- **Crowd wisdom** from Polymarket & Kalshi prediction markets
- **Statistical ML** prediction using a Kronos-inspired Markov chain + momentum model
- **Kelly Criterion** risk management for position sizing
- **Hermes Agent** framework for the multi-agent orchestration loop

---

## Architecture

```
main.py  (Orchestrator)
│
├── Agent 1: PredictionSearchAgent
│   ├── Queries Polymarket CLOB API
│   └── Queries Kalshi public API
│
├── Agent 2: DataFetcherAgent
│   ├── Primary:  Apify (Yahoo Finance scraper)
│   └── Fallback: CoinGecko public API
│
├── Agent 3: KronosPredictionAgent
│   ├── Markov chain transition probabilities
│   ├── Momentum signal (12-bar & 60-bar)
│   ├── Multi-timeframe arbitrage (1-min n+5 vs 5-min n+1)
│   └── LLM synthesis → final UP/DOWN + confidence
│
├── Agent 4: RiskManagementAgent
│   ├── Blends crowd (40%) + model (60%) confidence
│   ├── Kelly Criterion position sizing (capped at 25%)
│   └── LLM → trade recommendations with SL/TP
│
└── Agent 5: FeedbackAgent (Agnes Loop)
    ├── Records predictions to logs/feedback_history.json
    ├── Computes rolling accuracy per asset
    └── LLM → parameter adjustment suggestions
```

---

## Setup

### 1. Clone & install

```bash
git clone <your-repo-url>
cd crypto_agent
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and fill in your keys:
```

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENROUTER_API_KEY` | ✅ Yes | Get free key at [openrouter.ai](https://openrouter.ai) |
| `APIFY_API_TOKEN` | ✅ Yes | Get free token at [apify.com](https://apify.com) |
| `LLM_MODEL` | Optional | Default: `mistralai/mistral-7b-instruct` |

### 3. Run

```bash
# Single prediction cycle
python main.py

# Continuous loop (every 5 minutes)
python main.py --loop

# Custom bankroll + save output
python main.py --bankroll 5000 --output results/output.json
```

---

## Project Structure

```
crypto_agent/
├── main.py                        # Orchestrator / entry point
├── requirements.txt
├── .env.example
├── README.md
│
├── config/
│   ├── __init__.py
│   └── settings.py               # All config from .env
│
├── agents/
│   ├── __init__.py
│   ├── base_agent.py             # Abstract BaseAgent (Hermes-style)
│   ├── prediction_search_agent.py # Agent 1: Polymarket + Kalshi
│   ├── data_fetcher_agent.py     # Agent 2: Apify / CoinGecko OHLCV
│   ├── kronos_prediction_agent.py # Agent 3: Markov + Momentum ML
│   ├── risk_management_agent.py  # Agent 4: Kelly Criterion
│   └── feedback_agent.py         # Agent 5: Agnes loop
│
├── utils/
│   ├── __init__.py
│   ├── logger.py                 # Loguru-based structured logging
│   ├── llm_client.py             # OpenRouter API wrapper
│   └── kelly.py                  # Kelly Criterion math
│
└── logs/
    ├── crypto_agent.log          # Rotating structured log
    └── feedback_history.json     # Prediction accuracy tracking
```

---

## Key Design Decisions

### Multi-Timeframe Arbitrage
The `KronosPredictionAgent` computes both a 60-bar (1-min proxy) and 12-bar (5-min) momentum signal. When both agree, confidence is boosted. This implements the "predict 1min n+5, use it on 5min n+1" idea from the assessment.

### Kelly Criterion
Implemented in `utils/kelly.py`. The fraction is capped at **25%** (`MAX_KELLY_FRACTION`) to prevent overbetting. Crowd and model predictions are blended before sizing:
- Crowd wisdom: **40%** weight
- Model prediction: **60%** weight

### Agnes Feedback Loop
After each cycle, `FeedbackAgent` records predictions and (when actuals are available) scores them. The rolling accuracy is fed back as context to improve subsequent cycles – a lightweight reinforcement signal without retraining.

### Error Handling
Every agent wraps external calls in try/except with graceful fallbacks. The system continues even if individual data sources fail.

---

## Scaling Ideas (Extra Points)

- **Containerise** with Docker + run on a cron schedule in cloud
- **Add more assets** – update `ASSETS` in `config/settings.py`
- **Webhook alerts** – Slack/Telegram notifications on high-confidence signals
- **Database persistence** – swap JSON feedback log for PostgreSQL
- **Real trading integration** – connect to Kalshi or a CEX API for live execution

---

## Submission

- GitHub repository link
- APIFY tokens used: add `APIFY_API_TOKEN` to `.env`
- Short demo video showing the agent running in terminal

---

*Built for CrowdWisdomTrading Intern Position Assessment*
