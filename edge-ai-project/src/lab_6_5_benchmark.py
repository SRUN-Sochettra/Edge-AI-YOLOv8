from pathlib import Path
import time

from ultralytics import YOLO
import numpy as np
import pandas as pd
import psutil
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

PT_MODEL_PATH = MODEL_DIR / "yolov8n.pt"
ONNX_MODEL_PATH = MODEL_DIR / "yolov8n.onnx"

CSV_OUTPUT_PATH = OUTPUT_DIR / "benchmark_results.csv"
CHART_OUTPUT_PATH = OUTPUT_DIR / "benchmark_chart.png"

N_ITERATIONS = 100
IMAGE_SIZE = 640


frame = np.random.randint(
    0,
    255,
    (IMAGE_SIZE, IMAGE_SIZE, 3),
    dtype=np.uint8
)


def benchmark_model(model_path):
    print(f"\nBenchmarking: {model_path}")

    model = YOLO(str(model_path))

    for _ in range(10):
        model(frame, verbose=False)

    latencies = []
    process = psutil.Process()

    cpu_before = psutil.cpu_percent(interval=0.1)

    t_total_start = time.perf_counter()

    for _ in range(N_ITERATIONS):
        t0 = time.perf_counter()
        model(frame, verbose=False)
        latency_ms = (time.perf_counter() - t0) * 1000
        latencies.append(latency_ms)

    t_total = time.perf_counter() - t_total_start

    cpu_after = psutil.cpu_percent(interval=0.1)
    ram_mb = process.memory_info().rss / (1024 ** 2)

    return {
        "Model": model_path.name,
        "Avg Latency (ms)": round(np.mean(latencies), 2),
        "P95 Latency (ms)": round(np.percentile(latencies, 95), 2),
        "FPS": round(N_ITERATIONS / t_total, 2),
        "CPU (%)": round((cpu_before + cpu_after) / 2, 1),
        "RAM (MB)": round(ram_mb, 1),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not PT_MODEL_PATH.exists():
        print(f"ERROR: PyTorch model not found: {PT_MODEL_PATH}")
        print("Fix: put yolov8n.pt inside the models folder.")
        return

    if not ONNX_MODEL_PATH.exists():
        print(f"ERROR: ONNX model not found: {ONNX_MODEL_PATH}")
        print("Fix: run python src\\lab_6_3_export_onnx.py first.")
        return

    results = [
        benchmark_model(PT_MODEL_PATH),
        benchmark_model(ONNX_MODEL_PATH)
    ]

    df = pd.DataFrame(results)

    print("\n" + df.to_string(index=False))

    df.to_csv(CSV_OUTPUT_PATH, index=False)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    labels = df["Model"]
    colors = ["#2E86AB", "#E8543E"]

    for ax, metric in zip(
        axes,
        ["Avg Latency (ms)", "FPS", "RAM (MB)"]
    ):
        ax.bar(labels, df[metric], color=colors)
        ax.set_title(metric, fontweight="bold")
        ax.set_ylabel(metric)

        for i, v in enumerate(df[metric]):
            ax.text(
                i,
                v + v * 0.02,
                str(v),
                ha="center",
                fontweight="bold"
            )

    plt.suptitle(
        "YOLOv8n: PyTorch vs ONNX Runtime",
        fontsize=14,
        fontweight="bold"
    )

    plt.tight_layout()
    plt.savefig(CHART_OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\nResults saved to:")
    print(f"- {CSV_OUTPUT_PATH}")
    print(f"- {CHART_OUTPUT_PATH}")


if __name__ == "__main__":
    main()