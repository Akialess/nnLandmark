import argparse
import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../nnLandmark")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../nnUNet")))

import nibabel as nib
from vertebra_postprocessing import postprocess_json_file

import torch
import torch.nn as nn
import tensorrt as trt
from nnlandmark.inference.nnLandmark.predict_from_raw_data import nnUNetPredictor

class TRTWrapper(nn.Module):
    def __init__(self, engine_path):
        super().__init__()

        # Loads Tensor RT engine from file
        # https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/onnx/onnx_export.html
        # https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/python-api-docs.html

        self.logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(self.logger)

        with open(engine_path, "rb") as f:
            engine_data = f.read()

        self.engine = runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()
        
        # Dynamically map the input and output tensor names
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            print(f"name of tensor {name}", flush=True)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
            elif self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                self.output_name = name

    def forward(self, x):
        original_dtype = x.dtype
        x = x.half().contiguous()
        self.context.set_input_shape(self.input_name, tuple(x.shape))
        
        out_shape = tuple(self.context.get_tensor_shape(self.output_name))
        out_tensor = torch.empty(out_shape, dtype=torch.float16, device=x.device).contiguous()

        self.context.set_tensor_address(self.input_name, x.data_ptr())
        self.context.set_tensor_address(self.output_name, out_tensor.data_ptr())
        
        self.context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
        
        return out_tensor.to(original_dtype)


def predict_single_image(trt_engine_path, model_folder, input_path, output_path, folds, checkpoint, fast, postprocess=False, profile_memory=False):
    total_start = time.time()

    # Initialize pytorch cuda context 
    if torch.cuda.is_available():
        torch.cuda.init()
        _ = torch.zeros(1, device='cuda')

    print("Loading TensorRT model")
    load_start = time.time()
    trt_network = TRTWrapper(trt_engine_path)
    wrapper_loading_time = time.time() - load_start
    print(f"TensorRT model loaded in {wrapper_loading_time:.2f}s")

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
        allow_tqdm=True,
        
    )
    predictor.profile_memory = profile_memory
    predictor.wrapper_loading_time = wrapper_loading_time
    predictor.script_start_time = total_start

    predictor.initialize_from_trained_model_folder(
        model_training_output_dir=model_folder,
        use_folds=folds,
        checkpoint_name=checkpoint,
        skip_network=True
    )
    
    # Overwrite the pytorch network with the TRT wrapper
    predictor.network = trt_network
    # Don't load pytorch state dicts by using empty params
    predictor.list_of_parameters = [{}]

    # Take an image folder or a single image
    if os.path.isdir(input_path):
        image_files = sorted([
            os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith('.nii.gz')
        ])
        image_list = [[f] for f in image_files]
        print(f"Found {len(image_files)} images in {input_path}")
    else:
        image_files = [input_path]
        image_list = [[input_path]]

    print("Running nnUNetPredictor")
    if fast:
        from nnlandmark.inference.nnLandmark.fast_predict import predict_fast
        predict_fast(
            predictor=predictor,
            image_files=image_list,
            output_folder=output_path,
            batch_size=1,
            verbose=True
        )
    else:
        predictor.predict_from_files(
            list_of_lists_or_source_folder=[[input_path]],
            output_folder_or_list_of_truncated_output_files=output_path,
            save_probabilities=False,
            overwrite=True
        )

    if postprocess:
        for img_path in image_files:
            img = nib.load(img_path)
            spacing = list(img.header.get_zooms()[:3])
            case_id = os.path.basename(img_path).replace('_0000.nii.gz', '').replace('.nii.gz', '')
            json_path = os.path.join(output_path, f"{case_id}.json")
            if os.path.isfile(json_path):
                print(f"\nRunning vertebra postprocessing on {json_path}...")
                postprocess_json_file(json_path, spacing)
            else:
                print(f"\n[WARN] Postprocessing: output JSON not found at {json_path}")

    print(f"\n========================================")
    print(f"Total processing time: {time.time() - total_start:.2f}s")
    print(f"Output saved to: {output_path}")
    print(f"========================================")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict with TensorRT model (native pipeline)')
    parser.add_argument('--trt_model', type=str, required=True, help='Path to TensorRT engine')
    parser.add_argument('--model_folder', type=str, required=True, help='Path to model folder')
    parser.add_argument('--input', type=str, required=True, help='Input NIfTI')
    parser.add_argument('--output', type=str, required=True, help='Output path')
    parser.add_argument('--folds', type=int, nargs='+', default=(0,), help='Fold')
    parser.add_argument('--checkpoint', type=str, default='checkpoint_final.pth', help='Checkpoint')
    parser.add_argument('--fast', action='store_true', required=False, default=False,
                        help='Use fast inference pipeline: inline preprocessing with GPU resampling, batched sliding window, no multiprocessing overhead.')

    args = parser.parse_args()
    predict_single_image(args.trt_model, args.model_folder, args.input, args.output,
                         tuple(args.folds), args.checkpoint, args.fast, postprocess=args.postprocess,
                         profile_memory=args.profile_memory)