# AI Anomaly Agent

Detect → Explain → Alert. Converts a passive Excel business report into an
active monitoring system, orchestrated as a **LangGraph** state machine with
an explanation node powered by an LLM through **OpenRouter** (OpenAI-compatible Chat Completions + Structured Outputs; plain OpenAI works too).

## Key design principle

The statistical engine decides **what changed** (percentage change, moving
average, Z-score). The LLM only explains **what a change might mean** — it
is never asked to detect anomalies itself. This keeps detection testable
and deterministic, and keeps the AI layer honest (it must separate
observations from hypotheses and cannot invent facts).

## Architecture (LangGraph state graph)

```
START → ingest → validate ─(invalid)→ report → END
                    │(valid)
                    ▼
                  clean → analyze* → metrics → detect_and_score ─(no anomalies)→ report
                                          │(anomalies found)
                                          ▼
                                       explain (LLM, structured output)
                                          ▼
                                       persist (SQLite)
                                          ▼
                                       alert (SMTP email)
                                          ▼
                                       report → END
```

\* `analyze` (optional) builds the anomaly-free dataset, business report and charts - see
[Historical analysis & reporting](#historical-analysis--reporting).

Each box in `src/graph/pipeline.py` is a plain Python function; LangGraph
just wires the control flow (including the two conditional short-circuits:
skip everything downstream of validation on bad data, and skip the LLM/DB/
email calls entirely when nothing is anomalous).

## Project structure

```
ai-anomaly-agent/
├── app.py                     # Streamlit dashboard + upload entry point
├── cli.py                     # Headless CLI entry point (same pipeline, terminal output)
├── config/config.yaml         # All thresholds - no magic numbers in code
├── data/sample/               # generate_sample_data.py -> sample_metrics.xlsx
├── src/
│   ├── ingestion/excel_reader.py
│   ├── validation/validator.py
│   ├── cleaning/cleaner.py
│   ├── metrics/metric_engine.py      # % change, moving avg, Z-score
│   ├── anomaly/detector.py           # statistical detection only
│   ├── anomaly/severity.py           # Normal/Low/Warning/Critical
│   ├── ai/openai_client.py           # the ONLY module that talks to the LLM provider (endpoint, auth, retries, errors)
│   ├── ai/explainer.py               # per-anomaly explanation via structured output
│   ├── analysis/anomaly_flagger.py   # whole-history anomaly flagging (Hampel / robust z-score)
│   ├── analysis/anomaly_free.py      # builds the separate anomaly-free dataset
│   ├── analysis/narrative.py         # executive narrative (LLM, with template fallback)
│   ├── analysis/visualiser.py        # PNG (300 dpi) + interactive HTML charts
│   ├── analysis/reporting.py         # generate_business_report(data_frame, ...)
│   ├── alerts/email_alert.py         # SMTP, credentials from env only
│   ├── database/db.py                # SQLAlchemy models + SQLite
│   ├── reports/report_generator.py
│   └── graph/pipeline.py             # LangGraph StateGraph wiring it all together
└── tests/                     # pytest: metrics, detection, severity, validation
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in OPENROUTER_API_KEY ([OPENROUTER_API_KEY]), SMTP_*, ALERT_RECIPIENT
python data/sample/generate_sample_data.py   # creates a sample file with a real anomaly
```

## Run

```bash
# Dashboard
streamlit run app.py

# CLI (no DB writes, no emails). [DATA_SOURCE_PATH] = your .xlsx file, read-only
python cli.py [DATA_SOURCE_PATH]

# CLI, saving results and sending alert emails
python cli.py [DATA_SOURCE_PATH] --persist

# Choose where reports go / skip the report and only monitor
python cli.py [DATA_SOURCE_PATH] --output-dir reports
python cli.py [DATA_SOURCE_PATH] --no-report

# CLI, raw JSON output (for piping into other tools)
python cli.py [DATA_SOURCE_PATH] --json
```

The CLI is the same LangGraph pipeline the dashboard uses, so it's a
natural fit for a scheduled job (cron, GitHub Actions, a cloud
scheduler - TRD Sec. 33). It exits non-zero when attention is needed:
`0` = clean run, `1` = a Critical anomaly was found, `2` = validation
failed on the input file, `3` = unexpected failure - handy for alerting on job failure in CI.

## Historical analysis & reporting

Besides monitoring the latest row, each run audits the **whole history**:

1. **Flag** anomalies in every metric (centred rolling median + MAD, a.k.a. a Hampel filter;
   thresholds in `config.yaml` → `anomaly_analysis`).
2. **Build a separate anomaly-free dataset** (`interpolate` keeps every date; `drop_rows` removes affected rows).
3. **Write a business report** (Markdown + JSON), the flagged list (CSV), and an annotated copy of the raw data.
4. **Chart raw vs anomaly-free**: a 300-dpi PNG, an anomaly-count PNG, and an interactive Plotly HTML page.

Everything lands in a fresh `outputs/run_<timestamp>/` folder. **The input file is never modified or
overwritten**, and the cleaned data is always a separate file. Use it directly from Python:

```python
from src.analysis.reporting import generate_business_report
artifacts = generate_business_report(df, output_dir="outputs")   # df = [DATA_FRAME]
print(artifacts.output_dir, artifacts.summary["total_anomalies"], artifacts.errors)
```

If the LLM call fails or no key is set, the report is still produced with a template narrative.

## Test

```bash
pytest
```

No API key is required to run the test suite. The LLM integration tests run the real `openai` SDK
against a fake HTTP transport (`httpx.MockTransport`), so request shape, structured-output
parsing and every error path are verified offline.

## Security

- The LLM key (variable name set by `ai.api_key_env_var`, default `OPENROUTER_API_KEY` -
  the `[OPENROUTER_API_KEY]` placeholder), SMTP credentials, and the alert recipient are read
  exclusively from environment variables (`.env`, gitignored).
- Requests carry OpenRouter's `provider.data_collection = "deny"` preference, and the model only
  ever receives aggregated anomaly data, never the raw spreadsheet.

## LLM provider configuration

All provider settings live under `ai:` in `config/config.yaml`. To use OpenAI directly instead of
OpenRouter, change four values (no code changes):

| Setting | OpenRouter (default) | OpenAI direct |
|---|---|---|
| `provider` | `openrouter` | `openai` |
| `base_url` | `https://openrouter.ai/api/v1` | `https://api.openai.com/v1` |
| `api_key_env_var` | `OPENROUTER_API_KEY` | `OPENAI_API_KEY` |
| `model` | `openai/gpt-5-mini` | `gpt-5-mini` |

Override the model per environment with `LLM_MODEL`.
- `.env` is never committed; see `.env.example` for the required keys.
- If the AI call fails for any reason, the pipeline falls back to a
  statistics-only explanation rather than losing the anomaly report.

## Roadmap

- **V2**: CSV/Postgres input, Slack alerts, multi-user support, per-user thresholds.
- **V3**: forecasting, seasonality detection, root-cause analysis, natural-language Q&A.
- **V4**: a more autonomous LangGraph agent that investigates and recommends, not just alerts.
