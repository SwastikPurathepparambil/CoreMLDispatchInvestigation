import coremltools as ct
import torch
import torchvision.models as models


def main():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.eval()

    example_input = torch.rand(1, 3, 224, 224)
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
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS13,
    )

    output_path = "models/ResNet50_fp16.mlpackage"
    mlmodel.save(output_path)

    print(f"Saved Core ML model to {output_path}")


if __name__ == "__main__":
    main()