import coremltools as ct
import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1000),
        )

    def forward(self, x):
        return self.net(x)


def main():
    model = SimpleMLP()
    model.eval()

    example_input = torch.rand(1, 1024)
    traced_model = torch.jit.trace(model, example_input)

    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(
                name="input",
                shape=example_input.shape,
            )
        ],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.macOS13,
    )

    output_path = "models/SimpleMLP_fp32.mlpackage"
    mlmodel.save(output_path)

    print(f"Saved Core ML model to {output_path}")


if __name__ == "__main__":
    main()