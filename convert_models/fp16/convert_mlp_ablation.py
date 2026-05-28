from pathlib import Path

import coremltools as ct
import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, num_layers=2, output_dim=10):
        super().__init__()

        layers = []

        in_dim = input_dim

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# Configurations

CONFIGS = [
    # width scaling (core experiment)
    {"hidden_dim": 64, "num_layers": 2},
    {"hidden_dim": 128, "num_layers": 2},
    {"hidden_dim": 256, "num_layers": 2},
    {"hidden_dim": 512, "num_layers": 2},
    {"hidden_dim": 1024, "num_layers": 2},
    {"hidden_dim": 2048, "num_layers": 2},
]


# Conversion Function

def convert_model(config):
    hidden_dim = config["hidden_dim"]
    num_layers = config["num_layers"]

    print("\n========================================")
    print("Converting MLP Variant")
    print("========================================")
    print(f"hidden_dim: {hidden_dim}")
    print(f"num_layers: {num_layers}")

    model = SimpleMLP(
        input_dim=784,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=10,
    )

    model.eval()

    example_input = torch.randn(1, 784)

    traced = torch.jit.trace(model, example_input)

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(
                name="input",
                shape=example_input.shape,
                dtype=float,
            )
        ],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS13,
    )

    model_name = f"MLP_h{hidden_dim}_l{num_layers}"

    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / f"{model_name}_fp16.mlpackage"

    mlmodel.save(str(output_path))

    print(f"Saved: {output_path}")


# Main

def main():
    for cfg in CONFIGS:
        convert_model(cfg)

    print("\nDone generating MLP ablation models.")


if __name__ == "__main__":
    main()
