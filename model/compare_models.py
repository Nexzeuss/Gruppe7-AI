"""Create model-comparison plots from per-model cross-validation CSV files."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from model_utils import PERFORMANCE_FOLDER


METRICS = ["rmse", "mae", "r2", "cv_rmse"]
METRIC_INFO = {
    "rmse": ("RMSE", "Electricity reading units (lower is better)"),
    "mae": ("MAE", "Electricity reading units (lower is better)"),
    "r2": ("R²", "Variance explained (higher is better; 0 = mean-prediction baseline)"),
    "cv_rmse": ("CV(RMSE)", "RMSE / fold mean (lower is better)"),
}


def main():
    files = sorted(PERFORMANCE_FOLDER.glob("*_performance.csv"))
    if not files:
        raise FileNotFoundError(f"No model performance CSV files found in {PERFORMANCE_FOLDER}")
    results = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    if not {"model", "fold", *METRICS}.issubset(results.columns):
        raise ValueError("Performance CSV files must contain model, fold, and all metric columns.")

    PERFORMANCE_FOLDER.mkdir(parents=True, exist_ok=True)
    summary = results.groupby("model")[METRICS].agg(["mean", "std"])
    summary.to_csv(PERFORMANCE_FOLDER / "model_performance_summary.csv")

    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(10, 6))
        models = sorted(results["model"].dropna().unique())
        colors = plt.get_cmap("tab10")
        for index, name in enumerate(models):
            values = results.loc[results["model"] == name, ["fold", metric]].dropna().sort_values("fold")
            x = index + 1
            color = colors(index % 10)
            ax.scatter(
                [x] * len(values), values[metric], color=color, alpha=0.8,
                edgecolor="white", linewidth=0.6, s=65,
                label="Individual fold" if index == 0 else None,
                zorder=3,
            )
            for _, row in values.iterrows():
                ax.annotate(f"F{int(row['fold'])}", (x, row[metric]), xytext=(6, 3),
                            textcoords="offset points", fontsize=8, color=color)
            mean = values[metric].mean()
            std = values[metric].std(ddof=1)
            ax.errorbar(
                x, mean, yerr=std, fmt="D", color="black", ecolor="black",
                capsize=5, markersize=7,
                label="Mean ± 1 standard deviation" if index == 0 else None,
                zorder=4,
            )

        title, ylabel = METRIC_INFO[metric]
        ax.set_xticks(range(1, len(models) + 1), models)
        ax.set_title(f"Five-fold grouped cross-validation: {title}")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Model (each F label is one held-out building fold)")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="best", frameon=True)
        ax.text(
            0.01, 0.99,
            "Dots = fold scores; black diamond/error bar = mean ± SD.\n"
            "For RMSE/MAE/CV(RMSE), lower is better; for R², higher is better.",
            transform=ax.transAxes, ha="left", va="top", fontsize=8,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.85},
        )
        fig.tight_layout()
        fig.savefig(PERFORMANCE_FOLDER / f"model_comparison_{metric}.png", dpi=160)
        plt.close(fig)

    # Diagnostic plots use out-of-fold predictions: every point comes from a
    # model that did not train on that point's building.
    prediction_files = sorted(PERFORMANCE_FOLDER.glob("*_oof_predictions.csv"))
    for path in prediction_files:
        predictions = pd.read_csv(path)
        required = {"model", "fold", "actual", "predicted"}
        if not required.issubset(predictions.columns):
            raise ValueError(f"{path.name} must include columns: {sorted(required)}")
        if predictions.empty:
            continue

        model_name = str(predictions["model"].iloc[0])
        residual = predictions["predicted"] - predictions["actual"]
        minimum = min(predictions["actual"].min(), predictions["predicted"].min())
        maximum = max(predictions["actual"].max(), predictions["predicted"].max())

        fig, ax = plt.subplots(figsize=(8, 7))
        scatter = ax.scatter(
            predictions["actual"], predictions["predicted"],
            c=predictions["fold"], cmap="viridis", s=9, alpha=0.28,
            linewidths=0, rasterized=True,
        )
        ax.plot([minimum, maximum], [minimum, maximum], "r--", linewidth=1.5,
                label="Perfect predictions (y = x)")
        colorbar = fig.colorbar(scatter, ax=ax)
        colorbar.set_label("Held-out building fold")
        ax.set_title(f"{model_name}: out-of-fold predicted vs actual")
        ax.set_xlabel("Actual electricity reading")
        ax.set_ylabel("Predicted electricity reading")
        ax.text(0.02, 0.98, "Points near the red line are more accurate.",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85})
        ax.legend(loc="lower right")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(PERFORMANCE_FOLDER / f"{path.stem}_actual_vs_predicted.png", dpi=170)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 6))
        scatter = ax.scatter(
            predictions["predicted"], residual,
            c=predictions["fold"], cmap="viridis", s=9, alpha=0.28,
            linewidths=0, rasterized=True,
        )
        ax.axhline(0, color="red", linestyle="--", linewidth=1.5, label="No error (residual = 0)")
        fig.colorbar(scatter, ax=ax, label="Held-out building fold")
        ax.set_title(f"{model_name}: out-of-fold residuals")
        ax.set_xlabel("Predicted electricity reading")
        ax.set_ylabel("Residual (predicted − actual)")
        ax.text(0.02, 0.98, "Above zero = overprediction; below zero = underprediction.\n"
                "A pattern or widening spread can indicate bias or uneven errors.",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85})
        ax.legend(loc="best")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(PERFORMANCE_FOLDER / f"{path.stem}_residuals.png", dpi=170)
        plt.close(fig)

    print(f"Saved comparison summary and plots in {PERFORMANCE_FOLDER}")
    if not prediction_files:
        print("No out-of-fold prediction CSVs found; rerun each model script to create prediction diagnostic plots.")


if __name__ == "__main__":
    main()