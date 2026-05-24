import coremltools as ct
import torch
import torchvision.models as models


def main():
    # 1. Load pretrained MobileNetV2 from torchvision
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.eval()

    # 2. Create example input: batch size 1, RGB image, 224x224
    example_input = torch.rand(1, 3, 224, 224)

    # 3. Trace the PyTorch model
    traced_model = torch.jit.trace(model, example_input)

    # 4. Convert to Core ML
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

    # 5. Save model
    output_path = "models/MobileNetV2_fp32.mlpackage"
    mlmodel.save(output_path)

    print(f"Saved Core ML model to {output_path}")


if __name__ == "__main__":
    main()