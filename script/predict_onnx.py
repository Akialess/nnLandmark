import time
_script_start = time.time()

import argparse
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../nnLandmark")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../nnUNet")))

import nibabel as nib

import torch
import torch.nn as nn
import onnxruntime as ort
from nnlandmark.inference.nnLandmark.predict_from_raw_data import nnUNetPredictor

# https://onnxruntime.ai/docs/api/python/api_summary.html
class ONNXWrapper(nn.Module):
    def __init__(self, session):
        super().__init__()
        self.session = session # ONNX Runtime inference session
        self.input_name = session.get_inputs()[0].name
        self.output_name = session.get_outputs()[0].name
        
        self.torch_dtype = torch.float16
        self.np_dtype = np.float16
        
    def forward(self, x: torch.Tensor):
        
        # Pytorch and ONNX works on different cuda streams, so we need to make them synchronize
        torch.cuda.current_stream().synchronize()

        x_mapped = x.to(self.torch_dtype).contiguous() # Cast tensor to float16

        out_channels = self.session.get_outputs()[0].shape[1]
        expected_batch = self.session.get_inputs()[0].shape[0]

        # If the model has a fixed batch size (e.g., 1) and we pass a different batch size, process iteratively
        if isinstance(expected_batch, int) and x_mapped.shape[0] != expected_batch:
            outputs = []
            for i in range(0, x_mapped.shape[0], expected_batch):
                batch_x = x_mapped[i:i+expected_batch]
                if batch_x.shape[0] < expected_batch:
                    # Pad if necessary (though usually we just process batch size 1)
                    pad_size = expected_batch - batch_x.shape[0]
                    batch_x = torch.cat([batch_x, torch.zeros((pad_size,) + batch_x.shape[1:], dtype=batch_x.dtype, device=batch_x.device)], dim=0)
                    out = self._forward_impl(batch_x)
                    outputs.append(out[: (expected_batch - pad_size)])
                else:
                    outputs.append(self._forward_impl(batch_x))
            return torch.cat(outputs, dim=0)
        else:
            return self._forward_impl(x_mapped)

    def _forward_impl(self, x_mapped: torch.Tensor):
        out_channels = self.session.get_outputs()[0].shape[1]

        # Allocate the empty tensor for output
        out_tensor = torch.empty(
            (x_mapped.shape[0], out_channels, x_mapped.shape[2], x_mapped.shape[3], x_mapped.shape[4]), 
            dtype=self.torch_dtype, 
            device=x_mapped.device
        ).contiguous()

        device_id = x_mapped.device.index if x_mapped.is_cuda and x_mapped.device.index is not None else 0
        device_type = 'cuda' if x_mapped.is_cuda else 'cpu'

        io_binding = self.session.io_binding()

        io_binding.bind_input(
            name=self.input_name,
            device_type=device_type,
            device_id=device_id,
            element_type=self.np_dtype,
            shape=x_mapped.shape,
            buffer_ptr=x_mapped.data_ptr()
        )

        io_binding.bind_output(
            name=self.output_name,
            device_type=device_type,
            device_id=device_id,
            element_type=self.np_dtype,
            shape=out_tensor.shape,
            buffer_ptr=out_tensor.data_ptr()
        )
        
        io_binding.synchronize_inputs()
        self.session.run_with_iobinding(io_binding)
        io_binding.synchronize_outputs()

        # Wait that the ONNX op have finished before pytorch can continue
        torch.cuda.current_stream().synchronize()

        return out_tensor.to(torch.float32) # ensure float32 to match previous behavior


def predict(onnx_model_path, model_folder, input_path, output_path, folds, checkpoint, fast=False, verbose=False):
    time_start = time.time()
    import_time = time_start - _script_start
    if verbose:
        print(f"Python imports took {import_time:.2f}s")

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    providers = []
    if 'CUDAExecutionProvider' in ort.get_available_providers():
        providers.append('CUDAExecutionProvider')
    providers.append('CPUExecutionProvider')

    session = ort.InferenceSession(onnx_model_path, sess_options=options, providers=providers)
    onnx_network = ONNXWrapper(session)
    wrapper_loading_time = time.time() - time_start
    if verbose:
        print(f"ONNX model loaded in {wrapper_loading_time:.2f}s")

    if verbose:
        print("Initializing nnUNetPredictor")
    use_cuda = torch.cuda.is_available()
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=use_cuda,
        device=torch.device('cuda') if use_cuda else torch.device('cpu'),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    predictor.wrapper_loading_time = wrapper_loading_time
    predictor.script_start_time = _script_start

    predictor.initialize_from_trained_model_folder(
        model_training_output_dir=model_folder,
        use_folds=folds,
        checkpoint_name=checkpoint,
        skip_network=True
    )

    # Overwrite the pytorch network with the ONNX wrapper
    predictor.network = onnx_network
    # Don't load pytorch state dicts by using empty params
    predictor.list_of_parameters = [{}]

    # Take an image folder or a single image
    if os.path.isdir(input_path):
        image_files = sorted([
            os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith('.nii.gz')
        ])
        image_list = [[f] for f in image_files]
        if verbose:
            print(f"Found {len(image_files)} images in {input_path}")
    else:
        image_files = [input_path]
        image_list = [[input_path]]

    if verbose:
        print("Running nnUNetPredictor")
    if fast:
        from nnlandmark.inference.nnLandmark.fast_predict import predict_fast
        predict_fast(
            predictor=predictor,
            image_files=image_list,
            output_folder=output_path,
            verbose=verbose
        )
    else:
        predictor.predict_from_files(
            list_of_lists_or_source_folder=image_list,
            output_folder_or_list_of_truncated_output_files=output_path,
            save_probabilities=False,
            overwrite=True
        )

    if verbose:
        print(f"\n========================================")
        print(f"Total processing time: {time.time() - _script_start:.2f}s")
        print(f"Output saved to: {output_path}")
        print(f"========================================")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict with ONNX model (native pipeline)')
    parser.add_argument('--onnx_model', type=str, required=True, help='Path to ONNX model')
    parser.add_argument('--model_folder', type=str, required=True, help='Path to model folder')
    parser.add_argument('--input', type=str, required=True, help='Input NIfTI file or directory of NIfTI files')
    parser.add_argument('--output', type=str, required=True, help='Output path')
    parser.add_argument('--folds', type=int, nargs='+', default=(0,), help='Fold')
    parser.add_argument('--checkpoint', type=str, default='checkpoint_final.pth', help='Checkpoint')
    parser.add_argument('--fast', action='store_true', required=False, default=False,
                        help='Use fast inference pipeline: inline preprocessing with GPU resampling, batched sliding window, no multiprocessing overhead.')
    parser.add_argument('--verbose', action='store_true', required=False, default=False,
                        help='Print progress and timing information.')

    args = parser.parse_args()
    predict(args.onnx_model, args.model_folder, args.input, args.output,
            tuple(args.folds), args.checkpoint, fast=args.fast, verbose=args.verbose)