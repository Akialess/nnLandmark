import argparse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../nnLandmark")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../nnUNet")))

import torch
import numpy as np
from nnlandmark.inference.predict_from_raw_data import nnUNetPredictor
import onnx
import onnxruntime as ort
import json

def export_to_onnx(model_folder, output_path, folds, checkpoint):
    
    # Load the trained model
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False
    )
 
    predictor.initialize_from_trained_model_folder(
        model_training_output_dir=model_folder,
        use_folds=folds,
        checkpoint_name=checkpoint
    )

    # Extract model information
    patch_size = predictor.configuration_manager.patch_size
    n_channels_in = len(predictor.dataset_json['channel_names'])
    n_channels_out = predictor.label_manager.num_segmentation_heads

    # Export to ONNX
    predictor.network.eval()
    device = torch.device('cuda')
    predictor.network.to(device)

    # Create a dummy input for exporting the model
    dummy_input = torch.randn(1, n_channels_in, *patch_size, device=device, dtype=torch.float16)

    print(f"Exporting to ONNX: {output_path}")

    torch.onnx.export(
        predictor.network,
        dummy_input,
        output_path,
        dynamo=False,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
    )
 
    print("ONNX export done")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export nnLandmark model to ONNX')
    parser.add_argument('--model_folder', type=str,
                        help='Path to nnLandmark trained model folder')
    parser.add_argument('--output', type=str, default='model.onnx',
                        help='Output ONNX file path')
    parser.add_argument('--folds', type=int, nargs='+', default=[0],
                        help='Folds to export, e.g. --folds 0 or --folds 0 1 2 3 4')
    parser.add_argument('--checkpoint', type=str, default='checkpoint_final.pth',
                        help='Checkpoint filename')

    args = parser.parse_args()
    export_to_onnx(args.model_folder, args.output, tuple(args.folds), args.checkpoint)