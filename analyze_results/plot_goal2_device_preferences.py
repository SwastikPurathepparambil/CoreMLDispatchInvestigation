from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def clean_model_name(model_name: str) -> str:
    return model_name.replace("_fp16", "")


def main():
    project_root = Path(__file__).resolve().parents[1]

    input_csv = project_root / "results" / "goal2_device_summary_clean.csv"
    output_dir = project_root / "plots"
    output_dir.mkdir(exist_ok=True)

    output_png = output_dir / "goal2_preferred_device_counts.png"
    output_pdf = output_dir / "goal2_preferred_device_counts.pdf"
    output_summary_csv = project_root / "results" / "goal2_preferred_device_counts.csv"

    df = pd.read_csv(input_csv)

    # Constants are weights/metadata, not runtime compute ops.
    df = df[df["operator"] != "const"].copy()

    df["model_clean"] = df["model"].apply(clean_model_name)

    model_order = [
        "MobileNetV2",
        "ResNet50",
        "SimpleMLP",
        "DistilBERT",
        "TinyGPT",
    ]

    device_order = ["CPU", "GPU", "ANE", "None", "Unknown"]

    grouped = (
        df.groupby(["model_clean", "preferred_device"], observed=False)["count"]
        .sum()
        .reset_index()
    )

    pivot = grouped.pivot(
        index="model_clean",
        columns="preferred_device",
        values="count",
    ).fillna(0)

    for device in device_order:
        if device not in pivot.columns:
            pivot[device] = 0

    pivot = pivot[device_order]
    pivot = pivot.reindex(model_order)

    # Drop columns with all zeros.
    pivot = pivot.loc[:, (pivot != 0).any(axis=0)]

    pivot.to_csv(output_summary_csv)

    ax = pivot.plot(
        kind="bar",
        stacked=True,
        figsize=(10, 6),
        width=0.75,
    )

    ax.set_title("Core ML Preferred Compute Device Counts by Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("Number of non-constant operations")
    ax.legend(title="Preferred device")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(output_png, dpi=300)
    plt.savefig(output_pdf)

    print(f"Saved plot to: {output_png}")
    print(f"Saved plot to: {output_pdf}")
    print(f"Saved summary CSV to: {output_summary_csv}")

    print("\nPreferred device count summary:")
    print(pivot.astype(int))


if __name__ == "__main__":
    main()