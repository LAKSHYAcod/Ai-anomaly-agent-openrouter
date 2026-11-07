"""Business reporting for anomaly findings.

Public entry point: :func:`generate_business_report`.

Given a dataset (``[DATA_FRAME]``) it
  1. flags anomalies across the whole history,
  2. builds a separate anomaly-free dataset,
  3. writes a business report plus supporting files,
  4. renders charts contrasting raw vs anomaly-free data.

Safety guarantees
-----------------
* The caller's DataFrame is never mutated.
* Nothing is ever written next to, or over, the source file: every run
  writes into its own new ``run_<timestamp>`` folder, and a guard refuses
  any output path that equals ``source_path``.
* Individual artefacts (a chart, one file) can fail without losing the
  rest: failures are collected in ``ReportArtifacts.errors``.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.analysis.anomaly_flagger import AnomalyFlags, FlaggingError, flag_anomalies
from src.analysis.anomaly_free import CleaningError, build_anomaly_free_dataset
from src.analysis.narrative import ExecutiveNarrative, build_narrative
from src.analysis.visualiser import (
    VisualisationError,
    plot_anomaly_summary,
    plot_interactive_html,
    plot_raw_vs_clean,
)
from src.config_loader import load_config

logger = logging.getLogger(__name__)

# Fixed output names: easy for downstream tooling to find, and none can equal an input file.
FILENAMES = {
    "report": "business_report.md",
    "summary": "report_summary.json",
    "flagged": "flagged_anomalies.csv",
    "annotated_raw": "raw_data_annotated.csv",
    "anomaly_free": "anomaly_free_data.csv",
    "chart_timeseries": "raw_vs_clean_timeseries.png",
    "chart_summary": "anomaly_summary.png",
    "chart_interactive": "raw_vs_clean_interactive.html",
}


class ReportingError(RuntimeError):
    """Raised when a report cannot be produced at all (bad input, unusable output location)."""


@dataclass
class ReportArtifacts:
    """Everything :func:`generate_business_report` produces."""

    output_dir: Path
    summary: dict[str, Any]
    anomaly_free_df: pd.DataFrame
    flagged_df: pd.DataFrame
    narrative: ExecutiveNarrative
    narrative_source: str  # "llm" | "template"
    files: dict[str, Path] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


# --------------------------------------------------------------------------- input handling


def _normalise(data_frame: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
    """Return a sorted copy with parsed dates and numeric metrics (values otherwise unchanged)."""
    if not isinstance(data_frame, pd.DataFrame):
        raise ReportingError(f"Expected a pandas DataFrame, got {type(data_frame).__name__}.")
    if data_frame.empty:
        raise ReportingError("The DataFrame is empty; there is nothing to analyse.")
    missing = [c for c in ["Date", *metric_cols] if c not in data_frame.columns]
    if missing:
        raise ReportingError(f"Missing required column(s): {', '.join(missing)}")

    work = data_frame.copy()
    work["Date"] = pd.to_datetime(work["Date"], errors="coerce")
    dropped = int(work["Date"].isna().sum())
    if dropped:
        logger.warning("Ignoring %d row(s) with unparseable dates", dropped)
        work = work.dropna(subset=["Date"])
    for col in metric_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.sort_values("Date").reset_index(drop=True)
    if work.empty:
        raise ReportingError("No rows with a valid date remain after parsing.")
    return work


def _make_run_dir(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for attempt in range(100):
        run_dir = base / (f"run_{stamp}" if attempt == 0 else f"run_{stamp}_{attempt}")
        try:
            run_dir.mkdir(exist_ok=False)  # never reuse (and so never overwrite) an earlier run
            return run_dir
        except FileExistsError:
            continue
    raise ReportingError(f"Could not create a unique run folder under {base}.")


def _guard_source(source_path: str | Path | None, targets: list[Path]) -> None:
    if source_path is None:
        return
    src = Path(source_path).resolve()
    for t in targets:
        if t.resolve() == src:
            raise ReportingError(f"Refusing to write to '{t}': it is the original data source.")


# --------------------------------------------------------------------------- summary


def _f(x: Any) -> float | None:
    return None if x is None or pd.isna(x) else float(x)


def _metric_impacts(raw: pd.DataFrame, clean: pd.DataFrame, flags: AnomalyFlags, strategy: str) -> list[dict]:
    impacts = []
    for col in flags.metrics:
        n = int(flags.mask[col].sum())
        raw_mean, clean_mean = _f(raw[col].mean()), _f(clean[col].mean())
        raw_std, clean_std = _f(raw[col].std()), _f(clean[col].std())
        # Totals are only meaningful for additive metrics and when every date is retained.
        additive = strategy == "interpolate" and "rate" not in col.lower()
        raw_total = _f(raw[col].sum()) if additive else None
        clean_total = _f(clean[col].sum()) if additive else None
        diff = None if raw_total is None else clean_total - raw_total
        impacts.append({
            "metric": col, "anomalies": n,
            "raw_mean": raw_mean, "clean_mean": clean_mean,
            "mean_shift_pct": (clean_mean - raw_mean) / abs(raw_mean) * 100 if raw_mean else None,
            "raw_std": raw_std, "clean_std": clean_std,
            "volatility_reduction_pct": (raw_std - clean_std) / raw_std * 100 if raw_std else None,
            "raw_total": raw_total, "clean_total": clean_total, "total_difference": diff,
            "total_difference_pct": (diff / abs(raw_total) * 100) if (diff is not None and raw_total) else None,
        })
    return sorted(impacts, key=lambda d: (-d["anomalies"], d["metric"]))


def _build_summary(raw, clean, flags, replaced, strategy, residual) -> dict[str, Any]:
    d = flags.details
    top = (
        d.assign(_sev=d["Severity"].map({"Critical": 0, "Warning": 1}), _z=d["Robust_Z"].abs())
        .sort_values(["_sev", "_z"], ascending=[True, False]).head(10)
        .drop(columns=["_sev", "_z"]) if not d.empty else d
    )
    top_records = [
        {**r, "Date": pd.Timestamp(r["Date"]).strftime("%Y-%m-%d")} for r in top.to_dict("records")
    ]
    return {
        "period": {"start": raw["Date"].min().strftime("%Y-%m-%d"), "end": raw["Date"].max().strftime("%Y-%m-%d")},
        "rows_analysed": int(len(raw)),
        "metrics": list(flags.metrics),
        "total_anomalies": flags.total,
        "critical_count": int((d["Severity"] == "Critical").sum()) if not d.empty else 0,
        "warning_count": int((d["Severity"] == "Warning").sum()) if not d.empty else 0,
        "rows_with_anomalies": int(flags.row_mask.sum()),
        "anomalies_by_metric": {m: int(flags.mask[m].sum()) for m in flags.metrics},
        "cleaning_strategy": strategy,
        "rows_after_cleaning": int(len(clean)),
        "cells_replaced": int(replaced.to_numpy().sum()),
        "residual_anomalies_after_cleaning": residual,
        "top_anomalies": top_records,
        "metric_impacts": _metric_impacts(raw, clean, flags, strategy),
    }


# --------------------------------------------------------------------------- markdown


def _num(x: Any, nd: int = 2) -> str:
    return "n/a" if x is None else f"{x:,.{nd}f}"


def _render_markdown(s: dict, nar: ExecutiveNarrative, source: str, charts: dict[str, str]) -> str:
    L: list[str] = []
    add = L.append
    add("# Business Anomaly Report\n")
    add(f"*Period analysed:* **{s['period']['start']} → {s['period']['end']}**  ")
    add(f"*Generated:* {datetime.now():%Y-%m-%d %H:%M}  ")
    add(f"*Narrative written by:* {'AI model' if source == 'llm' else 'built-in template (AI unavailable or not requested)'}\n")

    add("## 1. Executive summary\n")
    add(f"**{nar.headline}**\n")
    for item in nar.key_findings:
        add(f"- {item}")
    add("")

    add("## 2. Headline numbers\n")
    add("| Measure | Value |\n|---|---|")
    add(f"| Rows analysed | {s['rows_analysed']} |")
    add(f"| Anomalous values flagged | {s['total_anomalies']} ({s['critical_count']} critical, {s['warning_count']} warning) |")
    add(f"| Rows affected | {s['rows_with_anomalies']} |")
    add(f"| Cleaning strategy | `{s['cleaning_strategy']}` |")
    add(f"| Values replaced with estimates | {s['cells_replaced']} |")
    add(f"| Rows in anomaly-free dataset | {s['rows_after_cleaning']} |")
    resid = s["residual_anomalies_after_cleaning"]
    add(f"| Anomalies remaining after cleaning | {'not assessed' if resid is None else resid} |\n")

    add("## 3. Most significant anomalies\n")
    if s["top_anomalies"]:
        add("| Date | Metric | Observed | Expected | Deviation | Severity |\n|---|---|---:|---:|---:|---|")
        for a in s["top_anomalies"]:
            add(f"| {a['Date']} | {a['Metric']} | {_num(a['Value'])} | {_num(a['Expected'])} | "
                f"{_num(a['Deviation_Pct'], 1)}% | {a['Severity']} |")
        add(f"\nThe complete list is in `{FILENAMES['flagged']}`.\n")
    else:
        add("No anomalies were detected.\n")

    add("## 4. Effect of removing the anomalies\n")
    add("| Metric | Flagged | Mean (raw) | Mean (anomaly-free) | Mean shift | Volatility reduction |\n|---|---:|---:|---:|---:|---:|")
    for m in s["metric_impacts"]:
        add(f"| {m['metric']} | {m['anomalies']} | {_num(m['raw_mean'])} | {_num(m['clean_mean'])} | "
            f"{_num(m['mean_shift_pct'], 1)}% | {_num(m['volatility_reduction_pct'], 1)}% |")
    totals = [m for m in s["metric_impacts"] if m["total_difference"] not in (None, 0) and m["anomalies"]]
    if totals:
        add("\n**Estimated effect on period totals** (anomaly-free minus raw):\n")
        for m in totals:
            add(f"- {m['metric']}: {_num(m['total_difference'])} ({_num(m['total_difference_pct'], 1)}%)")
    add("")

    add("## 5. Risks and recommended actions\n")
    add("**Risks**\n")
    for item in nar.business_risks:
        add(f"- {item}")
    add("\n**Recommended actions**\n")
    for item in nar.recommended_actions:
        add(f"- {item}")
    add("")

    add("## 6. Visualisations\n")
    if "chart_timeseries" in charts:
        add(f"![Raw vs anomaly-free time series]({charts['chart_timeseries']})\n")
    if "chart_summary" in charts:
        add(f"![Anomalies per metric]({charts['chart_summary']})\n")
    if "chart_interactive" in charts:
        add(f"Interactive version: [{charts['chart_interactive']}]({charts['chart_interactive']})\n")

    add("## 7. Method, assumptions and limitations\n")
    add("- **Detection:** each value is compared with the centred rolling median of its neighbours and scaled by "
        "the rolling median absolute deviation (a Hampel filter). Thresholds are set in `config/config.yaml`.")
    add("- **Anomalies are not necessarily errors.** A flagged value may be a real event (promotion, outage, "
        "stock-out). Verify against source systems before treating it as bad data.")
    add("- **Replaced values are estimates**, not observations. Each metric is estimated independently, so "
        "related metrics (e.g. orders, conversion rate and revenue) may not reconcile exactly in the anomaly-free data.")
    add("- **Level shifts:** a sustained step change is treated as the new normal once it outlasts about half the "
        "rolling window, so only short-lived deviations are flagged.")
    add("- The original data was not modified; this report and its datasets are separate outputs.\n")
    return "\n".join(L)


# --------------------------------------------------------------------------- public API


def generate_business_report(
    data_frame: pd.DataFrame,
    output_dir: str | Path | None = None,
    *,
    source_path: str | Path | None = None,
    strategy: str | None = None,
    use_ai_narrative: bool = True,
    chart_dpi: int | None = None,
    plotly_js: str | bool = "cdn",
) -> ReportArtifacts:
    """Analyse ``data_frame`` and write a business report plus visualisations.

    Args:
        data_frame: ``[DATA_FRAME]`` - a table with a ``Date`` column and the
            metric columns listed in ``config.yaml`` (``metrics``). It is
            never modified.
        output_dir: parent folder for outputs (default: ``anomaly_analysis.output_dir``
            from config). A fresh ``run_<timestamp>`` sub-folder is created in it.
        source_path: path of the file ``data_frame`` was loaded from, used only
            as a safety guard so that it can never be overwritten.
        strategy: ``"interpolate"`` or ``"drop_rows"`` (default from config).
        use_ai_narrative: let the configured LLM write the executive narrative.
            Falls back to a template automatically if the API is unavailable.
        chart_dpi: resolution of PNG charts (default from config, 300).
        plotly_js: ``"cdn"`` (small HTML, needs internet to view) or ``True``
            (embeds Plotly, ~4 MB, works offline).

    Returns:
        :class:`ReportArtifacts` with the summary dict, the anomaly-free
        DataFrame, the flagged-anomaly table, and the path of every file written.

    Raises:
        ReportingError: bad input, or the output location is unusable.
    """
    cfg = load_config()
    acfg = cfg["anomaly_analysis"]
    metric_cols = list(cfg["metrics"].values())
    strategy = strategy or acfg["cleaning_strategy"]
    dpi = chart_dpi or acfg["chart_dpi"]

    raw = _normalise(data_frame, metric_cols)

    try:
        flags = flag_anomalies(raw, metric_cols)
        anomaly_free, replaced = build_anomaly_free_dataset(raw, flags, strategy)
    except (FlaggingError, CleaningError) as exc:
        raise ReportingError(str(exc)) from exc

    try:
        residual: int | None = flag_anomalies(anomaly_free, metric_cols).total
    except FlaggingError:
        residual = None  # too few rows left (e.g. after drop_rows) to re-check

    summary = _build_summary(raw, anomaly_free, flags, replaced, strategy, residual)
    narrative, narrative_source = build_narrative(summary, use_ai=use_ai_narrative)
    summary["narrative_source"] = narrative_source

    # ---- output location (new folder every run; source file is guarded) -----
    try:
        run_dir = _make_run_dir(Path(output_dir or acfg["output_dir"]))
    except OSError as exc:
        raise ReportingError(f"Cannot create output folder: {exc}") from exc
    targets = {k: run_dir / v for k, v in FILENAMES.items()}
    _guard_source(source_path, list(targets.values()))

    result = ReportArtifacts(
        output_dir=run_dir, summary=summary, anomaly_free_df=anomaly_free,
        flagged_df=flags.details, narrative=narrative, narrative_source=narrative_source,
    )

    def attempt(key: str, writer) -> None:
        """Run one writer; record the file on success, the error on failure."""
        try:
            writer(targets[key])
            result.files[key] = targets[key]
        except (OSError, VisualisationError, ValueError, RuntimeError) as exc:
            msg = f"Could not write {FILENAMES[key]}: {exc}"
            logger.error(msg)
            result.errors.append(msg)

    annotated = raw.copy()
    for col in flags.metrics:
        annotated[f"{col}__is_anomaly"] = flags.mask[col].to_numpy()

    attempt("flagged", lambda p: flags.details.to_csv(p, index=False))
    attempt("annotated_raw", lambda p: annotated.to_csv(p, index=False))
    attempt("anomaly_free", lambda p: anomaly_free.to_csv(p, index=False))
    plot_clean = anomaly_free if strategy == "interpolate" else _align_for_plot(raw, anomaly_free)
    attempt("chart_timeseries", lambda p: plot_raw_vs_clean(raw, plot_clean, flags, p, dpi))
    attempt("chart_summary", lambda p: plot_anomaly_summary(flags, p, dpi))
    attempt("chart_interactive", lambda p: plot_interactive_html(raw, plot_clean, flags, p, plotly_js))
    attempt("summary", lambda p: p.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8"))

    charts = {k: targets[k].name for k in ("chart_timeseries", "chart_summary", "chart_interactive") if k in result.files}
    attempt("report", lambda p: p.write_text(_render_markdown(summary, narrative, narrative_source, charts), encoding="utf-8"))

    logger.info("Report written to %s (%d file(s), %d error(s))", run_dir, len(result.files), len(result.errors))
    return result


def _align_for_plot(raw: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    """For ``drop_rows`` the cleaned frame is shorter; reindex on Date so lines show gaps, not false links."""
    return raw[["Date"]].merge(clean, on="Date", how="left")
