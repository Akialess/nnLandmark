"""Dispatch `nnLM_predict --onnx <name>` / `--tensorrt <name>` to the optimized
inference scripts under `script/`, resolving <name> against the workspace
`model/` folder."""

import argparse
import os
import runpy
import sys
from pathlib import Path


def _repo_root() -> Path:
    # nnlandmark/inference/optimized_entrypoints.py -> repo root is 3 levels up
    return Path(__file__).resolve().parents[2]


def _resolve_model_file(name: str, suffix: str) -> str:
    p = Path(name)
    if p.is_absolute() or p.exists():
        return str(p)
    model_dir = _repo_root() / "model"
    for candidate in (model_dir / name, model_dir / f"{name}{suffix}"):
        if candidate.exists():
            return str(candidate)
    return str(model_dir / (name if name.endswith(suffix) else f"{name}{suffix}"))


def _run_script(script_name: str, forwarded_argv: list) -> None:
    script_path = _repo_root() / "script" / script_name
    if not script_path.is_file():
        raise FileNotFoundError(f"Could not find optimized script: {script_path}")
    saved_argv = sys.argv
    sys.argv = [str(script_path), *forwarded_argv]
    try:
        runpy.run_path(str(script_path), run_name="__main__")
    finally:
        sys.argv = saved_argv


def _resolve_model_folder(args, parser: argparse.ArgumentParser) -> str:
    from nnlandmark.utilities.file_path_utilities import get_output_folder
    return get_output_folder(args.d, args.tr, args.p, args.c)


def predict_optimized_entry() -> None:
    parser = argparse.ArgumentParser(
        description="Run nnLandmark inference with an ONNX or TensorRT model "
                    "from the workspace `model/` folder."
    )
    backend = parser.add_mutually_exclusive_group(required=True)
    backend.add_argument('--onnx', type=str, default=None,
                         help='Name (or path) of the ONNX model in the model/ folder.')
    backend.add_argument('--tensorrt', type=str, default=None,
                         help='Name (or path) of the TensorRT engine in the model/ folder.')

    parser.add_argument('-i', dest='input', type=str, required=True,
                        help='Input NIfTI file or directory of NIfTI files.')
    parser.add_argument('-o', dest='output', type=str, required=True,
                        help='Output folder.')
    parser.add_argument('-d', type=str, required=True,
                        help='Dataset name/id (used to derive the trained model folder via $nnLM_results).')
    parser.add_argument('-p', type=str, default='nnUNetPlans',
                        help='Plans identifier. Default: nnUNetPlans')
    parser.add_argument('-tr', type=str, default='nnLandmark',
                        help='Trainer class name. Default: nnLandmark')
    parser.add_argument('-c', type=str, required=True,
                        help='Configuration (e.g. 3d_lowres_v11).')
    parser.add_argument('-f', dest='folds', nargs='+', type=int, default=(0,),
                        help='Folds to use. Default: (0,)')
    parser.add_argument('-chk', dest='checkpoint', type=str, default='checkpoint_final.pth',
                        help='Checkpoint filename. Default: checkpoint_final.pth')
    parser.add_argument('--fast', action='store_true', default=False,
                        help='Use the fast inference pipeline.')

    args = parser.parse_args()
    model_folder = _resolve_model_folder(args, parser)

    common = [
        '--model_folder', model_folder,
        '--input', args.input,
        '--output', args.output,
        '--checkpoint', args.checkpoint,
        '--folds', *[str(f) for f in args.folds],
    ]
    if args.fast:
        common.append('--fast')

    if args.onnx is not None:
        model_file = _resolve_model_file(args.onnx, '.onnx')
        _run_script('predict_onnx.py', ['--onnx_model', model_file, *common])
    else:
        model_file = _resolve_model_file(args.tensorrt, '.engine')
        _run_script('predict_tensorrt.py', ['--trt_model', model_file, *common])
