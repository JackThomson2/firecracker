#!/usr/bin/env python3
# Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Script for creating visualizations for A/B runs.

Usage:
ab_plot.py path_to_run_a path_to_run_b path_to_run_c ... --output_type pdf/table
"""

import argparse
import glob
import json
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

pd.set_option("display.float_format", "{:.2f}".format)


def check_regression(
    a_samples: List[float],
    b_samples: List[float],
    statistic: Callable = np.mean,
    *,
    n_resamples=9999,
):
    """
    Check if 2 sample groups have a statistically big enough difference
    """
    result = scipy.stats.permutation_test(
        (a_samples, b_samples),
        lambda x, y: statistic(y) - statistic(x),
        vectorized=False,
        n_resamples=n_resamples,
    )
    statistic_a = statistic(a_samples)

    return result.pvalue, result.statistic / statistic_a, result.statistic


def load_data(data_path: Path):
    """
    Recursively collects `metrics.json` files in provided path
    """
    data = []
    for name in glob.glob(f"{data_path}/**/metrics.json", recursive=True):
        with open(name, encoding="utf-8") as f:
            j = json.load(f)

        if "performance_test" not in j["dimensions"]:
            print(f"skipping: {name}")
            continue

        metrics = j["metrics"]
        # Move test name from dimensions into a separate column
        perf_test = j["dimensions"]["performance_test"]
        del j["dimensions"]["performance_test"]

        # These are host specific and will prevent comparison of
        # different hosts
        for key in ["instance", "cpu_model", "host_kernel"]:
            j["dimensions"].pop(key, None)

        dimensions = frozenset(j["dimensions"].items())

        for m in metrics:
            if "cpu_utilization" in m:
                continue
            mm = metrics[m]
            unit = mm["unit"]
            values = mm["values"]
            for i, v in enumerate(values):
                data.append(
                    {
                        "index": i,
                        "test": perf_test,
                        "metric": m,
                        "value": v,
                        "unit": unit,
                        "dimensions": dimensions,
                    }
                )

    return data


def p50(a):
    """Returns 50th percentile of 1d-array a"""
    return np.percentile(a, 50)


def p90(a):
    """Returns 90th percentile of 1d-array a"""
    return np.percentile(a, 90)


def create_markdown_summary(df: pd.DataFrame, group_order: list):
    """Create a markdown summary with key statistics"""
    baseline = group_order[0]
    lines = [
        "# A/B Test Summary",
        f"\nComparing: {' → '.join(group_order)}",
        f"\nBaseline: **{baseline}**\n",
    ]

    # Metrics where higher values are better (positive change = improvement)
    # Matches if any of these substrings appear in the metric name
    higher_is_better = {"throughput", "iops", "bandwidth", "bw", "requests_per_sec", "rps"}

    def is_regression(metric: str, pct: float) -> bool:
        """Returns True if the change represents a regression"""
        metric_lower = metric.lower()
        if any(h in metric_lower for h in higher_is_better):
            return pct < -5  # decrease in throughput = regression
        return pct > 5  # increase in latency = regression

    def is_improvement(metric: str, pct: float) -> bool:
        """Returns True if the change represents an improvement"""
        metric_lower = metric.lower()
        if any(h in metric_lower for h in higher_is_better):
            return pct > 5  # increase in throughput = improvement
        return pct < -5  # decrease in latency = improvement

    all_diffs = []  # Collect all differences for overall stats

    for test_value in sorted(df["test"].unique()):
        df_test = df[df["test"] == test_value]
        lines.append(f"\n## {test_value}\n")

        test_diffs = []
        for metric in sorted(df_test["metric"].unique()):
            df_metric = df_test[df_test["metric"] == metric]
            baseline_vals = df_metric[df_metric["group"] == baseline]["value"].values
            if len(baseline_vals) == 0:
                continue
            baseline_p50 = np.percentile(baseline_vals, 50)

            for group in group_order[1:]:
                group_vals = df_metric[df_metric["group"] == group]["value"].values
                if len(group_vals) == 0:
                    continue
                group_p50 = np.percentile(group_vals, 50)
                if baseline_p50 != 0:
                    pct_diff = ((group_p50 - baseline_p50) / baseline_p50) * 100
                    abs_diff = group_p50 - baseline_p50
                    unit = df_metric["unit"].iloc[0]

                    # Statistical significance (only for 2-group comparison)
                    pvalue = None
                    if len(group_order) == 2:
                        pvalue, _, _ = check_regression(baseline_vals, group_vals)

                    test_diffs.append({
                        "metric": metric,
                        "group": group,
                        "pct": pct_diff,
                        "abs": abs_diff,
                        "unit": unit,
                        "pvalue": pvalue,
                        "baseline_p50": baseline_p50,
                        "group_p50": group_p50,
                    })
                    all_diffs.append(test_diffs[-1] | {"test": test_value})

        if not test_diffs:
            lines.append("No comparable metrics.\n")
            continue

        # Sort by absolute percentage change to find outliers
        sorted_diffs = sorted(test_diffs, key=lambda x: abs(x["pct"]), reverse=True)

        # Summary table
        lines.append("| Metric | Comparison | Change | p50 (base → new) |")
        lines.append("|--------|------------|--------|------------------|")
        for d in sorted_diffs[:10]:  # Top 10 by magnitude
            if is_regression(d["metric"], d["pct"]):
                sign = "🔴"
            elif is_improvement(d["metric"], d["pct"]):
                sign = "🟢"
            else:
                sign = "⚪"
            sig = " *" if d["pvalue"] and d["pvalue"] <= 0.05 else ""
            lines.append(
                f"| {d['metric']} | {baseline}→{d['group']} | "
                f"{sign} {d['pct']:+.2f}%{sig} | "
                f"{d['baseline_p50']:.2f} → {d['group_p50']:.2f} {d['unit']} |"
            )

        # Test-level stats
        if test_diffs:
            pcts = [d["pct"] for d in test_diffs]
            lines.append(f"\n**Stats:** avg {np.mean(pcts):+.2f}%, "
                        f"median {np.median(pcts):+.2f}%, "
                        f"range [{min(pcts):+.2f}%, {max(pcts):+.2f}%]")
            reg_count = sum(1 for d in test_diffs if is_regression(d["metric"], d["pct"]))
            imp_count = sum(1 for d in test_diffs if is_improvement(d["metric"], d["pct"]))
            if reg_count or imp_count:
                lines.append(f"  - {reg_count} potential regressions, "
                            f"{imp_count} improvements")

    # Overall summary
    if all_diffs:
        lines.insert(3, "\n## Overall Summary\n")
        all_pcts = [d["pct"] for d in all_diffs]
        lines.insert(4, f"- **Total metrics compared:** {len(all_diffs)}")
        lines.insert(5, f"- **Average change:** {np.mean(all_pcts):+.2f}%")
        lines.insert(6, f"- **Median change:** {np.median(all_pcts):+.2f}%")

        regressions = sorted([d for d in all_diffs if is_regression(d["metric"], d["pct"])],
                             key=lambda x: abs(x["pct"]), reverse=True)
        improvements = sorted([d for d in all_diffs if is_improvement(d["metric"], d["pct"])],
                              key=lambda x: abs(x["pct"]), reverse=True)
        lines.insert(7, f"- **Regressions (>5%):** {len(regressions)}")
        lines.insert(8, f"- **Improvements (<-5%):** {len(improvements)}")

        insert_idx = 9
        if regressions:
            lines.insert(insert_idx, "\n### 🔴 Regressions\n")
            insert_idx += 1
            lines.insert(insert_idx, "| Test | Metric | Change | p50 (base → new) |")
            insert_idx += 1
            lines.insert(insert_idx, "|------|--------|--------|------------------|")
            insert_idx += 1
            for d in regressions:
                sig = " *" if d["pvalue"] and d["pvalue"] <= 0.05 else ""
                lines.insert(insert_idx, f"| {d['test']} | {d['metric']} | {d['pct']:+.2f}%{sig} | "
                                        f"{d['baseline_p50']:.2f} → {d['group_p50']:.2f} {d['unit']} |")
                insert_idx += 1

        if improvements:
            lines.insert(insert_idx, "\n### 🟢 Improvements\n")
            insert_idx += 1
            lines.insert(insert_idx, "| Test | Metric | Change | p50 (base → new) |")
            insert_idx += 1
            lines.insert(insert_idx, "|------|--------|--------|------------------|")
            insert_idx += 1
            for d in improvements:
                sig = " *" if d["pvalue"] and d["pvalue"] <= 0.05 else ""
                lines.insert(insert_idx, f"| {d['test']} | {d['metric']} | {d['pct']:+.2f}%{sig} | "
                                        f"{d['baseline_p50']:.2f} → {d['group_p50']:.2f} {d['unit']} |")
                insert_idx += 1

    lines.append("\n---\n*Legend: 🔴 >5% regression, 🟢 >5% improvement, ⚪ within 5%, "
                 "\\* statistically significant (p≤0.05)*\n")

    with open("summary.md", "w", encoding="UTF-8") as f:
        f.write("\n".join(lines))
    print("Ready: summary.md")


def create_index_html(test_names: list, group_order: list):
    """Create an index.html linking to all test result pages"""
    with open("index.html", "w", encoding="UTF-8") as f:
        f.write(f"""<!DOCTYPE html>
<html>
<head>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px auto; max-width: 800px; background: #f5f5f5; }}
h1 {{ color: #333; }}
.info {{ color: #666; margin-bottom: 20px; }}
ul {{ list-style: none; padding: 0; }}
li {{ margin: 8px 0; }}
a {{ display: block; padding: 12px 16px; background: white; border-radius: 6px; text-decoration: none; color: #4a90d9; box-shadow: 0 1px 3px rgba(0,0,0,0.1); transition: transform 0.1s, box-shadow 0.1s; }}
a:hover {{ transform: translateX(4px); box-shadow: 0 2px 6px rgba(0,0,0,0.15); }}
</style>
</head>
<body>
<h1>A/B Test Results</h1>
<p class="info">Comparing: {' → '.join(group_order)}</p>
<ul>
""")
        for test in sorted(test_names):
            f.write(f'<li><a href="{test}.html">{test}</a></li>\n')
        f.write("</ul></body></html>")
    print("Ready: index.html")


def create_table(df: pd.DataFrame, group_order: list):
    """Create an html table per test in the data frame"""

    baseline = group_order[0]

    for test_value in df["test"].unique():
        df_test = df[df["test"] == test_value]

        # Split dimensions into separate columns
        df_expanded = df_test.copy()
        dim_df = pd.DataFrame(df_expanded["dimensions"].apply(dict).tolist())
        dim_df = dim_df.reset_index(drop=True)
        df_data = df_expanded.drop("dimensions", axis=1).reset_index(drop=True)
        df_expanded = pd.concat([df_data, dim_df], axis=1)

        # Use dimension columns as index
        dim_cols = sorted(dim_df.columns)
        df_pivoted = df_expanded.pivot_table(
            values=["value"],
            index=["metric", "unit"] + dim_cols,
            columns="group",
            aggfunc=[p50, p90],
        )

        # Add comparison columns for each group vs first group (baseline)
        for group in group_order[1:]:
            for stat in ["p50", "p90"]:
                diff_col = (stat, "value", f"{baseline}->{group} %")
                df_pivoted[diff_col] = (
                    (
                        df_pivoted[(stat, "value", group)]
                        - df_pivoted[(stat, "value", baseline)]
                    )
                    / df_pivoted[(stat, "value", baseline)]
                    * 100.0
                )
                diff_col = (stat, "value", f"{baseline}->{group} abs")
                df_pivoted[diff_col] = (
                    df_pivoted[(stat, "value", group)]
                    - df_pivoted[(stat, "value", baseline)]
                )

        # Sort columns to have a persistent table representation
        df_pivoted = df_pivoted[sorted(df_pivoted.columns)]

        # Prepare chart data - one chart per unit
        chart_data = df_expanded.groupby(["metric", "unit", "group"])["value"].agg(["median"]).reset_index()
        units = chart_data["unit"].unique().tolist()
        colors = ["#4a90d9", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]
        
        charts_config = []
        for unit in units:
            unit_data = chart_data[chart_data["unit"] == unit]
            metrics_for_unit = unit_data["metric"].unique().tolist()
            datasets = []
            for i, group in enumerate(group_order):
                group_data = unit_data[unit_data["group"] == group]
                values = [group_data[group_data["metric"] == m]["median"].values[0] 
                          if len(group_data[group_data["metric"] == m]) > 0 else 0 
                          for m in metrics_for_unit]
                datasets.append({"label": group, "data": values, "backgroundColor": colors[i % len(colors)]})
            charts_config.append({"unit": unit, "labels": metrics_for_unit, "datasets": datasets})

        test_output_path = f"{test_value}.html"
        with open(test_output_path, "w", encoding="UTF-8") as writer:
            # Build filter checkboxes and chart divs for each unit
            chart_sections = []
            for i, cfg in enumerate(charts_config):
                checkboxes = " ".join(f'<label><input type="checkbox" checked onchange="updateChart{i}()" value="{m}"> {m}</label>' for m in cfg["labels"])
                chart_sections.append(f'''<div class="filters" id="filters-{i}"><strong>Filter ({cfg["unit"]}):</strong><br>{checkboxes}</div>
<div class="chart-container"><canvas id="chart-{i}"></canvas></div>''')
            
            writer.write(f"""<!DOCTYPE html>
<html>
<head>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; background: #f5f5f5; }}
h1, h2 {{ color: #333; }}
.chart-container {{ background: white; padding: 20px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.12); }}
.filters {{ background: white; padding: 15px; margin-bottom: 10px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.12); }}
.filters label {{ margin-right: 15px; cursor: pointer; }}
table {{ border-collapse: collapse; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.12); width: 100%; table-layout: auto; }}
th, td {{ padding: 8px 12px; text-align: right; border: 1px solid #ddd; white-space: nowrap; }}
th {{ background: #4a90d9; color: white; position: sticky; top: 0; z-index: 1; }}
thead th {{ vertical-align: bottom; }}
tr:nth-child(even) {{ background: #f9f9f9; }}
tr:hover {{ background: #e8f4fc; }}
.overflow-wrapper {{ overflow-x: auto; }}
</style>
</head>
<body>
<h1>{test_value}</h1>
{chr(10).join(chart_sections)}
<h2>Data Table</h2>
""")
            styled = df_pivoted.style.format(precision=2)
            styled = styled.set_table_styles(
                [{"selector": 'th:contains("->")', "props": [("min-width", "80px")]}]
            )

            # Apply color gradient to all comparison columns
            for group in group_order[1:]:
                for stat in ["p50", "p90"]:
                    diff_col = (stat, "value", f"{baseline}->{group} %")
                    styled = styled.background_gradient(
                        subset=[diff_col], cmap="RdYlGn"
                    )

            writer.write('<div class="overflow-wrapper">')
            writer.write(styled.to_html())
            writer.write('</div>')
            
            # Generate JS for each chart with its own filter function
            chart_scripts = []
            for i, cfg in enumerate(charts_config):
                chart_scripts.append(f"""
const allLabels{i} = {json.dumps(cfg["labels"])};
const allDatasets{i} = {json.dumps(cfg["datasets"])};
const chart{i} = new Chart(document.getElementById('chart-{i}'), {{
    type: 'bar',
    data: {{ labels: allLabels{i}, datasets: JSON.parse(JSON.stringify(allDatasets{i})) }},
    options: {{
        responsive: true,
        plugins: {{ title: {{ display: true, text: 'Metrics ({cfg["unit"]})' }} }},
        scales: {{ y: {{ beginAtZero: true }} }}
    }}
}});
function updateChart{i}() {{
    const checked = [...document.querySelectorAll('#filters-{i} input:checked')].map(c => c.value);
    const indices = checked.map(m => allLabels{i}.indexOf(m));
    chart{i}.data.labels = checked;
    chart{i}.data.datasets = allDatasets{i}.map(ds => ({{ ...ds, data: indices.map(j => ds.data[j]) }}));
    chart{i}.update();
}}""")
            
            writer.write(f"<script>{''.join(chart_scripts)}</script></body></html>")
        print(f"Ready: {test_output_path}")


def _create_single_pdf(args_tuple):
    """Create a single PDF for one test (for parallel execution)"""
    test_value, df_test, metrics, group_order, errorbar = args_tuple
    
    sns.set_style("whitegrid")
    test_output_path = f"{test_value}.pdf"
    with PdfPages(test_output_path) as pdf:
        for dim_value in df_test["dimensions"].unique():
            for metric in metrics:
                metric_data = df_test[
                    (df_test["metric"] == metric)
                    & (df_test["dimensions"] == dim_value)
                ]

                if len(metric_data) == 0:
                    continue

                additional_title = ""
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                if len(group_order) == 2:
                    # Check if difference is significant
                    a_values = metric_data[metric_data["group"] == group_order[0]][
                        "value"
                    ].values
                    b_values = metric_data[metric_data["group"] == group_order[1]][
                        "value"
                    ].values
                    pvalue, diff_rel, diff_abs = check_regression(
                        a_values, b_values
                    )

                    if (
                        pvalue <= 0.1
                        and abs(diff_rel) >= 0.05
                        and abs(diff_abs) >= 0.0
                    ):
                        fig.patch.set_facecolor("lightcoral")
                        additional_title = (
                            f"{diff_rel * 100:+.2f}% ({diff_abs:+.2f}) difference"
                        )

                # Make a multi-line title since single line will be too long
                dim_items = sorted(str(item) for item in dim_value)
                dim_chunks = [
                    ", ".join(dim_items[i : i + 4])
                    for i in range(0, len(dim_items), 4)
                ]
                dim_str = "\n".join(dim_chunks)
                title = f"{metric}\n{dim_str}\n{additional_title}"
                weight = "bold" if additional_title else "normal"
                fig.suptitle(title, fontsize=10, weight=weight)

                sns.boxenplot(data=metric_data, x="group", y="value", ax=ax1)
                ax1.set_ylabel(f"{metric} ({metric_data['unit'].iloc[0]})")

                metric_data_indexed = metric_data.reset_index()
                sns.lineplot(
                    data=metric_data_indexed,
                    x="index",
                    y="value",
                    hue="group",
                    ax=ax2,
                    errorbar=errorbar,
                )
                ax2.set_ylabel(f"{metric} ({metric_data['unit'].iloc[0]})")

                plt.tight_layout()
                pdf.savefig()
                plt.close()
    print(f"Ready: {test_output_path}")
    return test_output_path


def create_pdf(args, df: pd.DataFrame, group_order: list):
    """Create a pdf per test in the data frame (parallelized)"""
    metrics = df["metric"].unique()
    errorbar = (args.errorbar[0], int(args.errorbar[1]))
    
    tasks = [
        (test_value, df[df["test"] == test_value], metrics, group_order, errorbar)
        for test_value in df["test"].unique()
    ]
    
    with ProcessPoolExecutor() as executor:
        list(executor.map(_create_single_pdf, tasks))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Executes Firecracker's A/B testsuite across the specified commits"
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Paths to directories with test runs",
        type=Path,
    )
    parser.add_argument(
        "--errorbar",
        nargs=2,
        default=["pi", "95"],
        help="Errorbar configuration for lineplot (type, value)",
    )
    parser.add_argument(
        "--output_type",
        nargs="+",
        default=["pdf"],
        choices=["pdf", "table", "markdown", "all"],
        help="Type(s) of output to generate",
    )
    parser.add_argument(
        "-n", "--use-folder-names",
        action="store_true",
        help="Use folder names as group labels instead of A, B, C, ...",
    )
    parser.add_argument(
        "-z", "--zip",
        metavar="FILENAME",
        help="Package all output files into specified zip file",
    )
    args = parser.parse_args()

    # Data retrieval
    start_time = time.time()
    all_data = []
    group_order = []
    for i, path in enumerate(args.paths):
        data = load_data(path)
        print(f"getting data {i} from {path}: {len(data)}")
        df = pd.DataFrame(data)
        group_name = path.name if args.use_folder_names else chr(65 + i)
        df["group"] = group_name
        group_order.append(group_name)
        all_data.append(df)
    print(f"Data retrieval: {time.time() - start_time:.2f}s")

    # Data processing
    start_time = time.time()
    df_combined = pd.concat(all_data, ignore_index=True)
    print(f"Data processing: {time.time() - start_time:.2f}s")

    # Plotting
    start_time = time.time()
    test_names = df_combined["test"].unique().tolist()
    output_files = []
    output_types = args.output_type if "all" not in args.output_type else ["pdf", "table", "markdown"]

    if "pdf" in output_types:
        create_pdf(args, df_combined, group_order)
        output_files.extend(f"{t}.pdf" for t in test_names)
    if "table" in output_types:
        create_table(df_combined, group_order)
        create_index_html(test_names, group_order)
        output_files.extend(f"{t}.html" for t in test_names)
        output_files.append("index.html")
    if "markdown" in output_types:
        create_markdown_summary(df_combined, group_order)
        output_files.append("summary.md")

    print(f"Plotting: {time.time() - start_time:.2f}s")

    if args.zip:
        folder_name = Path(args.zip).stem
        with zipfile.ZipFile(args.zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in output_files:
                zf.write(f, f"{f}")
                Path(f).unlink()
        print(f"Ready: {args.zip}")
