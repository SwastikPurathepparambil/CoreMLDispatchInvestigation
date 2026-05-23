Goal 1:
We benchmarked five FP16 Core ML models across four compute-unit configurations: automatic/all, CPU-only, CPU+GPU, and CPU+Neural Engine. The model suite included two CNNs, one MLP, one transformer encoder-style model, and one tiny GPT-style generative transformer.

The CNN models showed strong benefit from ANE-enabled execution. MobileNetV2 and ResNet50 achieved their best or near-best latency under either .all or .cpuAndNeuralEngine, suggesting that Core ML’s automatic dispatch selected an ANE-heavy path for these convolutional workloads.

The non-CNN models showed a different pattern. The SimpleMLP and DistilBERT-style model were fastest under CPU-only execution, and the TinyGPT-style model showed a large gap: CPU-only was about 6.25× faster than automatic dispatch. These results suggest that automatic dispatch is effective for FP16 CNNs but may be suboptimal for small dense or transformer-style workloads.