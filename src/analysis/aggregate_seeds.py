"""
Aggregate per-seed results into across-seed statistics.

Reads outputs/seed_<seed>/results/*.json for each seed and, for every
experimental condition (baseline + each corruption_type x ratio, including
*_recovered), collects the four headline scalar metrics across seeds:

    - alignment_score   (alignment.alignment_score)
    - reasoning_acc     (reasoning.aggregate_accuracy)
    - drift_kl          (drift.mean_drift)
    - cka               (geometry.summary.mean_cka, fallback: mean of layer_cka)

For each (condition, metric) it computes mean, sample std, and a 95% CI using
the Student-t critical value (correct for small n, e.g. 5 seeds) and reports
n (number of seeds contributing).

Outputs (under <base_output_dir>/aggregated/):
    - seed_metrics.json   full nested structure
    - seed_metrics.csv    flat table, one row per (condition, metric)
    - main_results.tex    LaTeX table fragment (mean +/- CI half-width)

Importable: aggregate_seeds(...) returns a small summary dict.
"""

import csv
import json
import math
from pathlib import Path


# Student-t two-tailed critical values at 95% confidence, indexed by
# degrees of freedom (n-1). Beyond df=30 the normal approx (1.96) is fine.
_T_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    20: 2.086, 25: 2.060, 30: 2.042,
}


def _t_critical(df: int) -> float:
    if df <= 0:
        return float("nan")
    if df in _T_95:
        return _T_95[df]
    if df > 30:
        return 1.96
    # Nearest available smaller-or-equal df key (conservative)
    keys = sorted(k for k in _T_95 if k <= df)
    return _T_95[keys[-1]] if keys else 1.96


def _mean_std_ci(values: list[float]) -> dict:
    """Mean, sample std, and 95% t-CI half-width for a list of seed values."""
    vals = [v for v in values if v is not None and not _is_nan(v)]
    n = len(vals)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "ci95": None,
                "ci_lower": None, "ci_upper": None}
    mean = sum(vals) / n
    if n == 1:
        return {"n": 1, "mean": mean, "std": 0.0, "ci95": None,
                "ci_lower": mean, "ci_upper": mean}
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)  # sample variance
    std = math.sqrt(var)
    sem = std / math.sqrt(n)
    half = _t_critical(n - 1) * sem
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "ci95": half,
        "ci_lower": mean - half,
        "ci_upper": mean + half,
    }


def _is_nan(x) -> bool:
    try:
        return math.isnan(x)
    except (TypeError, ValueError):
        return False


def _extract_metrics(result: dict) -> dict:
    """Pull the four headline scalars from one result JSON (robust to gaps)."""
    align = (result.get("alignment") or {}).get("alignment_score")
    reason = (result.get("reasoning") or {}).get("aggregate_accuracy")
    drift = (result.get("drift") or {}).get("mean_drift")

    geom = result.get("geometry") or {}
    cka = (geom.get("summary") or {}).get("mean_cka")
    if cka is None:
        layer_cka = (geom.get("cka") or {}).get("layer_cka") or {}
        layer_vals = [v for v in layer_cka.values() if v is not None]
        cka = sum(layer_vals) / len(layer_vals) if layer_vals else None

    return {
        "alignment_score": align,
        "reasoning_acc": reason,
        "drift_kl": drift,
        "cka": cka,
    }


def _discover_seed_dirs(base_output_dir: str, seeds) -> dict[int, Path]:
    """Map seed -> results dir, for seeds that actually have results."""
    base = Path(base_output_dir)
    found = {}
    for seed in seeds:
        rdir = base / f"seed_{seed}" / "results"
        if rdir.is_dir() and any(rdir.glob("*.json")):
            found[seed] = rdir
    # Legacy fallback: a flat outputs/results with no seed dirs == one run.
    if not found:
        legacy = base / "results"
        if legacy.is_dir() and any(legacy.glob("*.json")):
            found[42] = legacy
    return found


METRIC_KEYS = ["alignment_score", "reasoning_acc", "drift_kl", "cka"]


def aggregate_seeds(base_output_dir: str, seeds, corruption_types) -> dict:
    """Aggregate metrics across seeds; write JSON, CSV, and LaTeX. Return summary."""
    seed_dirs = _discover_seed_dirs(base_output_dir, seeds)

    # condition -> seed -> {metric: value}
    per_condition: dict[str, dict[int, dict]] = {}
    for seed, rdir in seed_dirs.items():
        for jf in rdir.glob("*.json"):
            condition = jf.stem  # e.g. "toxic_0.5", "baseline_clean", "toxic_1.0_recovered"
            try:
                with open(jf) as fp:
                    result = json.load(fp)
            except (json.JSONDecodeError, OSError):
                continue
            per_condition.setdefault(condition, {})[seed] = _extract_metrics(result)

    # Aggregate per condition per metric
    aggregated: dict[str, dict] = {}
    for condition, seed_map in per_condition.items():
        agg_metrics = {}
        for mkey in METRIC_KEYS:
            vals = [seed_map[s].get(mkey) for s in sorted(seed_map)]
            agg_metrics[mkey] = _mean_std_ci(vals)
        aggregated[condition] = {
            "seeds_present": sorted(seed_map),
            "metrics": agg_metrics,
        }

    out_dir = Path(base_output_dir) / "aggregated"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- JSON ---
    with open(out_dir / "seed_metrics.json", "w") as f:
        json.dump(
            {
                "n_seeds_found": len(seed_dirs),
                "seeds_found": sorted(seed_dirs),
                "conditions": aggregated,
            },
            f, indent=2,
        )

    # --- CSV ---
    with open(out_dir / "seed_metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "metric", "n", "mean", "std", "ci95_halfwidth",
                    "ci_lower", "ci_upper"])
        for condition in sorted(aggregated):
            for mkey in METRIC_KEYS:
                m = aggregated[condition]["metrics"][mkey]
                w.writerow([
                    condition, mkey, m["n"],
                    _fmt(m["mean"]), _fmt(m["std"]), _fmt(m["ci95"]),
                    _fmt(m["ci_lower"]), _fmt(m["ci_upper"]),
                ])

    # --- LaTeX fragment (alignment + drift + cka, ordered baseline->ratios) ---
    _write_latex(aggregated, corruption_types, out_dir / "main_results.tex")

    return {
        "n_seeds_found": len(seed_dirs),
        "seeds_found": sorted(seed_dirs),
        "n_conditions": len(aggregated),
        "output_dir": str(out_dir),
    }


def _fmt(x) -> str:
    if x is None:
        return ""
    return f"{x:.4f}"


def _cell(m: dict) -> str:
    """Format a metric as 'mean $\\pm$ ci' or 'mean' if n<2."""
    if m["mean"] is None:
        return "--"
    if m.get("ci95"):
        return f"{m['mean']:.3f} $\\pm$ {m['ci95']:.3f}"
    return f"{m['mean']:.3f}"


def _write_latex(aggregated: dict, corruption_types, path: Path):
    lines = [
        "% Auto-generated by aggregate_seeds.py: across-seed mean $\\pm$ 95% CI",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Headline metrics across random seeds (mean $\\pm$ 95\\% CI). "
        "$n$ is the number of seeds.}",
        "\\label{tab:multiseed}",
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "Type & Ratio & Alignment & Reasoning & Drift (KL) & CKA \\\\",
        "\\midrule",
    ]

    def row(label, ratio_label, cond):
        if cond not in aggregated:
            return None
        m = aggregated[cond]["metrics"]
        n = max(m[k]["n"] for k in METRIC_KEYS)
        return (f"{label} & {ratio_label} & {_cell(m['alignment_score'])} & "
                f"{_cell(m['reasoning_acc'])} & {_cell(m['drift_kl'])} & "
                f"{_cell(m['cka'])} \\\\  % n={n}")

    base_row = row("Baseline", "0.0", "baseline_clean")
    if base_row:
        lines.append(base_row)
        lines.append("\\midrule")

    for ctype in corruption_types:
        ratios = sorted({
            float(c.rsplit("_", 1)[-1])
            for c in aggregated
            if c.startswith(f"{ctype}_") and not c.endswith("_recovered")
            and _is_floatable(c.rsplit("_", 1)[-1])
        })
        pretty = ctype.replace("_", " ").title()
        for ratio in ratios:
            r = row(pretty, f"{ratio:g}", f"{ctype}_{ratio}")
            if r:
                lines.append(r)
        lines.append("\\midrule")

    if lines[-1] == "\\midrule":
        lines.pop()
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]

    with open(path, "w") as f:
        f.write("\n".join(lines))


def _is_floatable(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Aggregate multi-seed results")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--seeds", type=int, nargs="+",
                   default=[42, 123, 456, 789, 1024])
    p.add_argument("--corruption-types", nargs="+",
                   default=["toxic", "misinformation", "semantic_noise",
                            "slang_compression"])
    args = p.parse_args()

    summary = aggregate_seeds(args.output_dir, args.seeds, args.corruption_types)
    print(json.dumps(summary, indent=2))
