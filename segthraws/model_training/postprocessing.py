"""
Copyright notice:
@author Cristopher Castro Traba, Ubotica Technologies
@copyright 2024 see license file for details
"""

import os
import warnings
import onnx
import torch
import onnxruntime
import numpy as np
import lightning as pl

from .train_constants import OPSET_VERSION


warnings.filterwarnings(category=FutureWarning, action="ignore")
warnings.filterwarnings(category=torch.jit.TracerWarning, action="ignore")


# MODEL CONVERSION TO ONNX
def convert_model_to_onnx(
    model: pl.LightningModule,
    model_checkpoint_path: str,
    input_shape: tuple = (1, 3, 256, 256),
    opset_version: int = OPSET_VERSION,
) -> None:
    """Convert the PyTorch model into ONNX format.

    Attributes
    ----------

    model : pl.LightningModule
            Input segmentation model in PyTorch format.

    model_checkpoint : str
            Path to the checkpoint model in ckpt format.

    input_shape : tuple
            Shape of the desired input image

    opset_version : int
        ONNX version to convert the model to.

    Outputs
    -------

    None
        ONNX model saved in its respective location

    Notes
    -----

    """

    onnx_path = os.path.join(os.path.dirname(model_checkpoint_path), "MYRIAD", "ONNX")
    os.makedirs(onnx_path, exist_ok=True)

    if len(input_shape) == 3:
        input_shape = input_shape.unsqueeze(0)
    elif len(input_shape) != 4 and len(input_shape) != 3:
        raise ValueError("Shape mismatch: The input shape must have ndim=3 or ndim=4.")

    x = torch.randn(input_shape).cpu()
    model_onnx = model.cpu()
    model_onnx.eval()
    torch_out = model_onnx(x)

    onnx_model_path = os.path.join(
        onnx_path, os.path.basename(model_checkpoint_path) + ".onnx"
    )

    # Export the model
    torch.onnx.export(
        model_onnx,  # model being run
        x,  # model input (or a tuple for multiple inputs)
        # where to save the model (can be a file or file-like object)
        onnx_model_path,
        export_params=True,  # store the trained parameter weights inside the model file
        # the ONNX version to export the model. 16 not supported by Myriad X.
        opset_version=opset_version,
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
    )
    # Checking that the model is correct
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)

    # Check that the ONNX model behave as expected
    ort_session = onnxruntime.InferenceSession(
        onnx_model_path, providers=["CPUExecutionProvider"]
    )

    def to_numpy(tensor):
        return (
            tensor.detach().cpu().numpy()
            if tensor.requires_grad
            else tensor.cpu().numpy()
        )

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    try:
        np.testing.assert_allclose(
            to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05
        )
        print("Exported model has been successfully tested with ONNXRuntime.")
    except:
        print("The ONNX model has not been converted successfully")
