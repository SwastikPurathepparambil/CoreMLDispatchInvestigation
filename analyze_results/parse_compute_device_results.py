from pathlib import Path
import csv


def infer_preferred_device(line: str) -> str:
    if "preferred: <MLNeuralEngineComputeDevice" in line:
        return "ANE"
    if "preferred: <MLGPUComputeDevice" in line:
        return "GPU"
    if "preferred: <MLCPUComputeDevice" in line:
        return "CPU"
    if ",nil," in line or line.endswith(",nil"):
        return "None"
    return "Unknown"


def parse_operator_device_line(line: str):
    first_comma = line.find(",")
    last_comma = line.rfind(",")

    if first_comma == -1 or last_comma == -1 or first_comma == last_comma:
        return None

    operator = line[:first_comma].strip()
    count_str = line[last_comma + 1:].strip()

    if not count_str.isdigit():
        return None

    count = int(count_str)
    preferred_device = infer_preferred_device(line)

    return operator, preferred_device, count


def main():
    project_root = Path(__file__).resolve().parents[1]

    input_path = project_root / "results" / "compute_device_results.csv"
    output_path = project_root / "results" / "goal2_device_summary_clean.csv"

    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    rows = []
    current_model = None
    inside_device_summary = False

    with input_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()

            if line.startswith("MODEL,"):
                current_model = line.split(",", 1)[1].strip()

            elif line == "OPERATOR_DEVICE_SUMMARY_BEGIN":
                inside_device_summary = True

            elif line == "OPERATOR_DEVICE_SUMMARY_END":
                inside_device_summary = False

            elif inside_device_summary and current_model is not None:
                parsed = parse_operator_device_line(line)

                if parsed is None:
                    continue

                operator, preferred_device, count = parsed

                rows.append({
                    "model": current_model,
                    "operator": operator,
                    "count": count,
                    "preferred_device": preferred_device,
                })

    output_path.parent.mkdir(exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "operator", "count", "preferred_device"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved clean summary to: {output_path}")
    print(f"Rows written: {len(rows)}")


if __name__ == "__main__":
    main()