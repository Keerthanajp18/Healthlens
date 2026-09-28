"""
HealthLens - Stage 7: Medical Report Trend Analysis and Visualization
Module: src/comparison/visualizer.py

Generates simple, clean line charts using Matplotlib for individual medical tests.
Saves figures to data/processed/charts/.

STRICT CONSTRAINTS:
1. Visualization only.
2. Strictly NO medical diagnosis or clinical interpretation.
3. Do NOT claim that a trend is clinically good or bad.
4. Do NOT use an LLM.
5. Keep graphs simple and suitable for a student capstone.
6. One single-panel graph per test (no complex multi-panel subplots).
7. Handle missing dates, missing values, and unshared tests safely.
8. Do NOT connect values with incompatible units.
9. Always run Matplotlib in headless mode ('Agg') and close figures to prevent memory leaks.
"""

from pathlib import Path
import re
from typing import Any
import pandas as pd

# Configure Matplotlib in headless mode before importing pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import CHARTS_DIR


def sanitize_filename(name: str) -> str:
    """Convert test name to safe, clean filesystem filename."""
    clean = re.sub(r"[^\w\s-]", "", name).strip()
    return re.sub(r"[-\s]+", "_", clean).lower()


# Palette for clinical status markers
STATUS_COLORS = {
    "NORMAL": "#16a34a",   # Green
    "LOW": "#2563eb",      # Blue
    "HIGH": "#dc2626",     # Red
    "UNKNOWN": "#6b7280",  # Gray
}


class MedicalTrendVisualizer:
    """
    Renders simple, elegant single-panel line charts for longitudinal medical test trends.
    """

    def __init__(self, output_dir: Path | str | None = None):
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = CHARTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_test_trend(
        self,
        test_name: str,
        records: list[dict],
        output_path: Path | str | None = None
    ) -> Path | None:
        """
        Generates a line chart for a single medical test across chronological dates.

        Parameters:
            test_name: Display name of the medical test.
            records: List of chronological observation dicts containing:
                     'date', 'value', 'unit', 'status', and optionally 'lower_bound', 'upper_bound'.
            output_path: Destination path for the saved PNG chart.

        Returns:
            Path to the saved PNG chart, or None if no valid numeric points exist.
        """
        # Filter for valid numeric records
        valid_records = [
            r for r in records
            if r.get("value") is not None and isinstance(r.get("value"), (int, float))
        ]

        if not valid_records:
            return None

        # Verify unit consistency across valid records
        units = list({r.get("unit", "").strip() for r in valid_records if r.get("unit")})
        units_consistent = len(units) <= 1
        primary_unit = units[0] if units else ""

        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)

        # Plot observations
        dates = [str(r.get("date", "Unknown")) for r in valid_records]
        values = [float(r["value"]) for r in valid_records]
        x_indices = list(range(len(dates)))

        if units_consistent:
            # Units are compatible: Connect observations with a continuous line
            ax.plot(
                x_indices,
                values,
                color="#2563eb",
                linestyle="-",
                linewidth=2.0,
                alpha=0.85,
                zorder=2,
                label=f"Observed ({primary_unit})" if primary_unit else "Observed Value",
            )
        else:
            # INCOMPATIBLE UNITS: Do NOT connect points across incompatible units!
            # Group points by unit and connect only points sharing the exact same unit.
            unit_groups: dict[str, list[tuple[int, float]]] = {}
            for idx, r in zip(x_indices, valid_records):
                u = r.get("unit", "").strip() or "No Unit"
                unit_groups.setdefault(u, []).append((idx, float(r["value"])))

            colors = ["#2563eb", "#d97706", "#7c3aed", "#059669"]
            for color_idx, (u, pts) in enumerate(unit_groups.items()):
                pts_x = [p[0] for p in pts]
                pts_y = [p[1] for p in pts]
                c = colors[color_idx % len(colors)]
                if len(pts) >= 2:
                    ax.plot(
                        pts_x,
                        pts_y,
                        color=c,
                        linestyle="--",
                        linewidth=1.8,
                        alpha=0.8,
                        zorder=2,
                        label=f"Series ({u})",
                    )
                else:
                    # Isolated point - no connecting line
                    ax.plot(
                        pts_x,
                        pts_y,
                        linestyle="none",
                    )

        # Draw markers colored by clinical status
        for idx, r in zip(x_indices, valid_records):
            val = float(r["value"])
            status = r.get("status", "UNKNOWN").upper()
            marker_color = STATUS_COLORS.get(status, STATUS_COLORS["UNKNOWN"])

            ax.scatter(
                idx,
                val,
                color=marker_color,
                edgecolors="#ffffff",
                s=90,
                linewidths=1.5,
                zorder=4,
            )

            # Numerical value label above/below marker
            unit_label = f" {r.get('unit')}" if not units_consistent and r.get("unit") else ""
            ax.annotate(
                f"{val:g}{unit_label}",
                xy=(idx, val),
                textcoords="offset points",
                xytext=(0, 9),
                ha="center",
                fontsize=8.5,
                fontweight="bold",
                color="#1f2937",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    fc="#ffffff",
                    ec="#e5e7eb",
                    alpha=0.85,
                    linewidth=0.8
                ),
                zorder=5,
            )

        # Reference range overlay (only when units are consistent and bounds exist)
        if units_consistent and valid_records:
            first_rec = valid_records[0]
            lower = first_rec.get("lower_bound")
            upper = first_rec.get("upper_bound")

            if lower is not None and upper is not None:
                lower = float(lower)
                upper = float(upper)
                ax.axhspan(
                    lower,
                    upper,
                    color="#16a34a",
                    alpha=0.12,
                    zorder=1,
                    label=f"Reference Range ({lower:g} - {upper:g} {primary_unit})",
                )
                ax.axhline(lower, color="#16a34a", linestyle=":", linewidth=1.2, alpha=0.7)
                ax.axhline(upper, color="#16a34a", linestyle=":", linewidth=1.2, alpha=0.7)
            elif upper is not None:
                upper = float(upper)
                ax.axhline(
                    upper,
                    color="#dc2626",
                    linestyle=":",
                    linewidth=1.2,
                    alpha=0.7,
                    label=f"Upper Limit (< {upper:g} {primary_unit})",
                )
            elif lower is not None:
                lower = float(lower)
                ax.axhline(
                    lower,
                    color="#2563eb",
                    linestyle=":",
                    linewidth=1.2,
                    alpha=0.7,
                    label=f"Lower Limit (> {lower:g} {primary_unit})",
                )

        # Axes & Labels formatting
        ax.set_xticks(x_indices)
        ax.set_xticklabels(dates, rotation=20, ha="right", fontsize=9)
        ax.set_xlabel("Report Date", fontsize=9.5, fontweight="bold", labelpad=6)

        if units_consistent:
            ax.set_ylabel(
                f"Value ({primary_unit})" if primary_unit else "Observed Value",
                fontsize=9.5,
                fontweight="bold",
                labelpad=6
            )
        else:
            ax.set_ylabel("Observed Value (Mixed Units)", fontsize=9.5, fontweight="bold", labelpad=6)

        # Graph Title & Subtitle
        if units_consistent:
            ax.set_title(
                f"{test_name} - Historical Trend",
                fontsize=11.5,
                fontweight="bold",
                pad=10
            )
        else:
            ax.set_title(
                f"{test_name} - Historical Trend\n[Incompatible Units Detected: Values Not Connected Across Units]",
                fontsize=10.5,
                fontweight="bold",
                color="#b91c1c",
                pad=10
            )

        ax.grid(True, linestyle=":", alpha=0.6, zorder=0)

        # Subtle padding on Y-axis for annotations
        y_min, y_max = ax.get_ylim()
        margin = max((y_max - y_min) * 0.15, 0.5)
        ax.set_ylim(y_min - (margin * 0.5), y_max + margin)

        # Non-clinical educational watermark footer
        plt.figtext(
            0.5,
            0.01,
            "HealthLens Stage 7 | Objective Numerical Trend - Not a Clinical Diagnosis",
            ha="center",
            fontsize=7.5,
            color="#9ca3af",
            style="italic"
        )

        handles, labels = ax.get_legend_handles_labels()
        if handles and labels:
            ax.legend(handles, labels, loc="upper right", fontsize=8, framealpha=0.9)
        plt.tight_layout(rect=[0, 0.03, 1, 1])

        # Resolve destination path
        if output_path:
            dest = Path(output_path)
        else:
            safe_name = sanitize_filename(test_name)
            dest = self.output_dir / f"{safe_name}_trend.png"

        dest.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest, format="png", bbox_inches="tight")
        plt.close(fig)

        return dest

    def plot_from_dataframe(
        self,
        df: pd.DataFrame,
        test_name: str,
        output_path: Path | str | None = None
    ) -> Path | None:
        """
        Plots a trend chart for a specific test directly from a trend DataFrame.
        """
        matching = df[df["test_name"].str.strip().str.lower() == test_name.strip().lower()]
        if matching.empty:
            return None

        records = matching.to_dict(orient="records")
        return self.plot_test_trend(test_name, records, output_path)

    def generate_all_charts(
        self,
        trend_results_or_df: dict | pd.DataFrame,
        output_dir: Path | str | None = None
    ) -> list[Path]:
        """
        Generates individual line charts for all unique tests present in the analysis.

        Parameters:
            trend_results_or_df: Either the dictionary from analyze_trends() or the trend DataFrame.
            output_dir: Optional custom folder where charts should be saved.

        Returns:
            List of Paths to all saved charts.
        """
        saved_paths = []
        target_dir = Path(output_dir) if output_dir else self.output_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(trend_results_or_df, pd.DataFrame):
            df = trend_results_or_df
            unique_tests = df["test_name"].unique()
            for t_name in unique_tests:
                sub_df = df[df["test_name"] == t_name]
                records = sub_df.to_dict(orient="records")
                safe_name = sanitize_filename(t_name)
                out_path = target_dir / f"{safe_name}_trend.png"
                chart_path = self.plot_test_trend(t_name, records, out_path)
                if chart_path:
                    saved_paths.append(chart_path)

        elif isinstance(trend_results_or_df, dict):
            trends = trend_results_or_df.get("test_trends", {})
            for k, info in trends.items():
                t_name = info.get("test_name", k)
                records = info.get("records", [])
                safe_name = sanitize_filename(t_name)
                out_path = target_dir / f"{safe_name}_trend.png"
                chart_path = self.plot_test_trend(t_name, records, out_path)
                if chart_path:
                    saved_paths.append(chart_path)

        return saved_paths


def visualize_medical_trends(
    trend_results_or_df: dict | pd.DataFrame,
    output_dir: Path | str | None = None
) -> list[Path]:
    """
    Convenience function to generate charts for all tests in a trend dataset.
    """
    visualizer = MedicalTrendVisualizer(output_dir=output_dir)
    return visualizer.generate_all_charts(trend_results_or_df)
