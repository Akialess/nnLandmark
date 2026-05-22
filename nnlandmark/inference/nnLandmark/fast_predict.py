"""
Fast single-image inference for nnLandmark.

Optimizations over the default nnUNet pipeline:
  1. Inline preprocessing — no multiprocessing spawn/queues for a single image
  2. GPU-accelerated resampling — torch F.interpolate instead of skimage.resize (CPU)
  3. No export pool — landmark coordinate extraction is lightweight, runs inline
  4. Batched sliding window — multiple patches in one GPU forward pass
  5. Removed segmentation-only code paths (cascade, one-hot, foreground sampling)
"""

import csv
import os
import time
import json
from typing import Tuple, Union, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from acvl_utils.cropping_and_padding.bounding_boxes import get_bbox_from_mask, bounding_box_to_slice
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from tqdm import tqdm

from nnlandmark.inference.nnLandmark.export_prediction import (
    export_prediction_from_logits,
    _extract_landmark_coord_and_likelihood,
)
from nnlandmark.inference.nnLandmark.sliding_window_prediction import (
    compute_gaussian,
    compute_steps_for_sliding_window,
)
from nnlandmark.utilities.helpers import empty_cache, dummy_context


# ---------------------------------------------------------------------------
# Fast preprocessing: GPU resampling, no segmentation, no multiprocessing
# ---------------------------------------------------------------------------

def _crop_to_nonzero_fast(data: np.ndarray):
    """
    Crop to non-zero bounding box without scipy binary_fill_holes.

    binary_fill_holes is expensive on large 3D volumes and only matters for
    training (mask-based normalization). For inference, a simple bounding box
    on the non-zero region is sufficient and much faster.
    """
    nonzero_mask = np.any(data != 0, axis=0)  # collapse channel dim
    bbox = get_bbox_from_mask(nonzero_mask)
    slicer = bounding_box_to_slice(bbox)
    data = data[(slice(None),) + slicer]
    return data, bbox


def _resample_gpu(data_tensor: torch.Tensor, new_shape: List[int],
                  device: torch.device) -> torch.Tensor:
    """
    Resample a 4D tensor (C, D, H, W) to new_shape using GPU trilinear interpolation.

    F.interpolate with trilinear is orders of magnitude faster than
    skimage.transform.resize(order=3) on CPU for large 3D medical volumes.
    """
    if list(data_tensor.shape[1:]) == list(new_shape):
        return data_tensor
    # F.interpolate expects (N, C, D, H, W)
    out = F.interpolate(
        data_tensor.unsqueeze(0).float().to(device),
        size=new_shape,
        mode='trilinear',
        align_corners=False,
    )
    return out.squeeze(0)  # back to (C, D, H, W)


def preprocess_fast(image_files: List[str],
                    plans_manager,
                    configuration_manager,
                    dataset_json: dict,
                    device: torch.device = torch.device('cuda'),
                    verbose: bool = False) -> Tuple[torch.Tensor, dict]:
    """
    Fast inline preprocessing for a single image. No multiprocessing.

    Replaces the full DefaultPreprocessor pipeline with:
      - Image loading (SimpleITK, unavoidable)
      - Transpose
      - Crop to nonzero (fast, no binary_fill_holes)
      - Normalization (CPU, usually fast)
      - GPU resampling (trilinear, replaces slow CPU bicubic)

    Returns:
      (data_tensor, properties_dict)
    """
    import nnlandmark
    from batchgenerators.utilities.file_and_folder_operations import join
    from nnlandmark.utilities.find_class_by_name import recursive_find_python_class
    from nnlandmark.preprocessing.resampling.default_resampling import compute_new_shape

    t0 = time.time()

    # --- Load image ---
    rw = plans_manager.image_reader_writer_class()
    data, properties = rw.read_images(image_files)
    t_load = time.time()

    # --- Transpose ---
    data = data.astype(np.float32, copy=False)
    data = data.transpose([0, *[i + 1 for i in plans_manager.transpose_forward]])
    original_spacing = [properties['spacing'][i] for i in plans_manager.transpose_forward]

    # --- Crop to nonzero (fast) ---
    properties['shape_before_cropping'] = data.shape[1:]
    data, bbox = _crop_to_nonzero_fast(data)
    properties['bbox_used_for_cropping'] = bbox
    properties['shape_after_cropping_and_before_resampling'] = data.shape[1:]
    t_crop = time.time()

    # --- Normalize (must happen before resampling) ---
    # Normalization needs a seg mask for mask-based normalization (ZScore).
    # seg >= 0 means "foreground" to the normalizer. Mark zero-valued voxels
    # as -1 (outside) so mask-based normalization ignores them.
    nonzero_mask = np.any(data != 0, axis=0)
    seg_dummy = np.where(nonzero_mask, np.int8(0), np.int8(-1))
    for c in range(data.shape[0]):
        scheme = configuration_manager.normalization_schemes[c]
        normalizer_class = recursive_find_python_class(
            join(nnlandmark.__path__[0], "preprocessing", "normalization"),
            scheme,
            'nnlandmark.preprocessing.normalization',
        )
        normalizer = normalizer_class(
            use_mask_for_norm=configuration_manager.use_mask_for_norm[c],
            intensityproperties=plans_manager.foreground_intensity_properties_per_channel[str(c)],
        )
        data[c] = normalizer.run(data[c], seg_dummy)
    t_norm = time.time()

    # --- GPU resampling ---
    target_spacing = configuration_manager.spacing
    if len(target_spacing) < len(data.shape[1:]):
        target_spacing = [original_spacing[0]] + list(target_spacing)
    new_shape = compute_new_shape(data.shape[1:], original_spacing, target_spacing)

    data_tensor = torch.from_numpy(data)
    data_tensor = _resample_gpu(data_tensor, [int(s) for s in new_shape], device)
    # Move back to CPU to match the rest of the pipeline expectations
    data_tensor = data_tensor.cpu()
    t_resample = time.time()

    preprocess_times = {
        'load_time_seconds': t_load - t0,
        'crop_time_seconds': t_crop - t_load,
        'norm_time_seconds': t_norm - t_crop,
        'resampling_time_seconds': t_resample - t_norm,
    }

    if verbose:
        print(f'[fast preprocess] load={t_load - t0:.3f}s  crop={t_crop - t_load:.3f}s  '
              f'norm={t_norm - t_crop:.3f}s  resample(GPU)={t_resample - t_norm:.3f}s  '
              f'total={t_resample - t0:.3f}s')
        print(f'  shape: {data.shape[1:]} -> {list(data_tensor.shape[1:])}  '
              f'spacing: {original_spacing} -> {list(target_spacing)}')

    return data_tensor, properties, preprocess_times


# ---------------------------------------------------------------------------
# Batched sliding window prediction
# ---------------------------------------------------------------------------

def _predict_sliding_window_batched(
    network: torch.nn.Module,
    data: torch.Tensor,
    patch_size: Tuple[int, ...],
    tile_step_size: float,
    num_output_channels: int,
    device: torch.device,
    use_gaussian: bool = True,
    batch_size: int = 2,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Sliding window prediction with batch processing of patches.

    Instead of feeding patches one at a time, groups them into mini-batches
    for a single GPU forward pass. This better utilizes GPU parallelism,
    especially for smaller patch sizes.
    """
    # Pad image to at least patch_size
    data_padded, slicer_revert = pad_nd_image(data, patch_size, 'constant', {'value': 0}, True, None)
    spatial_shape = data_padded.shape[1:]  # (D, H, W) or (H, W)

    # Compute slicer positions
    if len(patch_size) < len(spatial_shape):
        # 2D patches on 3D data
        steps = compute_steps_for_sliding_window(spatial_shape[1:], patch_size, tile_step_size)
        slicers = []
        for d in range(spatial_shape[0]):
            for sx in steps[0]:
                for sy in steps[1]:
                    slicers.append(
                        (slice(None), d,
                         slice(sx, sx + patch_size[0]),
                         slice(sy, sy + patch_size[1]))
                    )
    else:
        steps = compute_steps_for_sliding_window(spatial_shape, patch_size, tile_step_size)
        slicers = []
        for sx in steps[0]:
            for sy in steps[1]:
                for sz in steps[2]:
                    slicers.append(
                        (slice(None),
                         slice(sx, sx + patch_size[0]),
                         slice(sy, sy + patch_size[1]),
                         slice(sz, sz + patch_size[2]))
                    )

    n_patches = len(slicers)
    if verbose:
        print(f'[fast predict] {n_patches} patches, batch_size={batch_size}')

    # Preallocate output
    predicted_logits = torch.zeros(
        (num_output_channels, *spatial_shape), dtype=torch.half, device=device
    )
    n_predictions = torch.zeros(spatial_shape, dtype=torch.half, device=device)

    if use_gaussian:
        gaussian = compute_gaussian(
            tuple(patch_size), sigma_scale=1.0 / 8, value_scaling_factor=10, device=device
        )
    else:
        gaussian = 1

    # Move data to device once
    data_padded = data_padded.to(device)

    # Process in batches
    with torch.inference_mode():
        with tqdm(total=n_patches, desc='Inference', disable=not verbose) as pbar:
            for batch_start in range(0, n_patches, batch_size):
                batch_end = min(batch_start + batch_size, n_patches)
                batch_slicers = slicers[batch_start:batch_end]

                # Extract and stack patches: (B, C_in, *patch_size)
                patches = torch.stack(
                    [data_padded[s] for s in batch_slicers]
                )

                preds = network(patches)  # (B, C_out, *patch_size)

                # Apply gaussian weighting to entire batch at once
                if use_gaussian:
                    preds = preds * gaussian

                # Accumulate predictions
                for i, s in enumerate(batch_slicers):
                    predicted_logits[s] += preds[i]
                    n_predictions[s[1:]] += gaussian

                pbar.update(batch_end - batch_start)

    # Normalize by aggregation weights
    torch.div(predicted_logits, n_predictions, out=predicted_logits)

    if torch.any(torch.isinf(predicted_logits)):
        raise RuntimeError(
            'Encountered inf in predicted array. Reduce value_scaling_factor '
            'in compute_gaussian or increase dtype to fp32.'
        )

    # Revert padding
    predicted_logits = predicted_logits[(slice(None), *slicer_revert[1:])]
    return predicted_logits


# ---------------------------------------------------------------------------
# Fast landmark extraction (no resampling of full probability volume)
# ---------------------------------------------------------------------------

def _extract_landmarks_from_logits(
    predicted_logits: torch.Tensor,
    properties: dict,
    plans_manager,
    label_manager,
    output_file_truncated: Optional[str] = None,
    spine_process: bool = False,
) -> Tuple[dict, dict]:
    """
    Extract landmark coordinates directly from network-resolution logits.

    Instead of resampling the full C×D×H×W probability volume back to original
    space (expensive), we find the peak in network space and scale the
    coordinates back. This is what export_prediction_from_logits already does,
    but we call it inline without multiprocessing.

        When spine_process=True, uses spine centerline-based postprocessing instead
        of per-channel argmax.

        Returns:
            (output_json, postprocess_times)
    """
    post_times = {
        'resampling_time_seconds': 0.0,
        'mask_conversion_time_seconds': 0.0,
        'cropping_reverse_time_seconds': 0.0,
        'centroid_extraction_time_seconds': 0.0,
        'export_output_time_seconds': 0.0,
        'spine_postprocessing_time_seconds': 0.0,
    }

    t0_centroid = time.time()
    probs = torch.sigmoid(predicted_logits.float()).cpu().numpy()
    if probs.ndim == 3:
        probs = probs[:, None, ...]

    class_ids = list(label_manager.foreground_labels)
    shape_pred = probs.shape[1:]
    shape_after_crop = properties['shape_after_cropping_and_before_resampling']
    bbox = properties['bbox_used_for_cropping']
    transpose_backward = plans_manager.transpose_backward

    out_json = {}

    if spine_process:
        # --- Spine postprocessing path ---
        from nnlandmark.inference.nnLandmark.spine_processing import postprocess_nnlandmark_output
        print("[spine_process] Running spine postprocessing on predicted heatmaps...")
        t0_spine = time.time()
        spine_results = postprocess_nnlandmark_output(probs)
        post_times['spine_postprocessing_time_seconds'] = time.time() - t0_spine

        spine_lookup = {}
        for r in spine_results:
            spine_lookup[r["label"]] = (r["coordinate"], r["likelihood"])

        for ch, cls_id in enumerate(class_ids):
            if ch in spine_lookup:
                coord_pred, lik = spine_lookup[ch]
                cz_pred, cy_pred, cx_pred = coord_pred

                coord_crop_z = cz_pred * (shape_after_crop[0] / shape_pred[0])
                coord_crop_y = cy_pred * (shape_after_crop[1] / shape_pred[1])
                coord_crop_x = cx_pred * (shape_after_crop[2] / shape_pred[2])

                coord_trans = [
                    coord_crop_z + bbox[0][0],
                    coord_crop_y + bbox[1][0],
                    coord_crop_x + bbox[2][0],
                ]

                coord_orig = [0, 0, 0]
                for i in range(3):
                    coord_orig[i] = coord_trans[transpose_backward[i]]

                out_json[str(int(cls_id))] = {
                    "coordinates": [int(round(coord_orig[2])), int(round(coord_orig[1])), int(round(coord_orig[0]))],
                    "likelihood": float(lik),
                }
            else:
                out_json[str(int(cls_id))] = {
                    "coordinates": [None, None, None],
                    "likelihood": 0.0,
                }

        print(f"[spine_process] Detected {len(spine_results)} vertebrae via spine postprocessing.")
    else:
        # --- Default per-channel argmax path ---
        for ch, cls_id in enumerate(class_ids):
            coord_pred, lik = _extract_landmark_coord_and_likelihood(probs[ch])

            if coord_pred is None:
                out_json[str(int(cls_id))] = {
                    "coordinates": [None, None, None],
                    "likelihood": 0.0,
                }
            else:
                cx, cy, cz = coord_pred

                # Scale from network resolution back to cropped space
                coord_crop_z = cz * (shape_after_crop[0] / shape_pred[0])
                coord_crop_y = cy * (shape_after_crop[1] / shape_pred[1])
                coord_crop_x = cx * (shape_after_crop[2] / shape_pred[2])

                # Revert cropping
                coord_trans = [
                    coord_crop_z + bbox[0][0],
                    coord_crop_y + bbox[1][0],
                    coord_crop_x + bbox[2][0],
                ]

                # Revert transposition
                coord_orig = [0, 0, 0]
                for i in range(3):
                    coord_orig[i] = coord_trans[transpose_backward[i]]

                out_json[str(int(cls_id))] = {
                    "coordinates": [int(round(coord_orig[2])), int(round(coord_orig[1])), int(round(coord_orig[0]))],
                    "likelihood": float(lik),
                }

    centroid_time = time.time() - t0_centroid
    centroid_time -= post_times['spine_postprocessing_time_seconds']
    post_times['centroid_extraction_time_seconds'] = max(0.0, centroid_time)

    if output_file_truncated is not None:
        t0_export = time.time()
        with open(output_file_truncated + ".json", "w") as f:
            json.dump(out_json, f, indent=4)
        post_times['export_output_time_seconds'] = time.time() - t0_export

    print(f"fast postprocessing: {post_times}")
    return out_json, post_times


# ---------------------------------------------------------------------------
# Main fast prediction entry point
# ---------------------------------------------------------------------------

def predict_fast(
    predictor,
    image_files: Union[str, List[str], List[List[str]]],
    output_folder: Optional[str] = None,
    batch_size: int = 2,
    verbose: bool = False,
) -> Union[dict, List[dict]]:
    """
    Fast single-image (or few-image) landmark prediction.

    Bypasses all multiprocessing overhead of the default pipeline.
    Uses GPU resampling and batched patch inference.

    Writes timing data to the same ``pipeline_times.csv`` used by the default
    pipeline (appends rows with the same columns) so you can directly compare
    the two approaches side-by-side.

    Args:
        predictor: initialized nnUNetPredictor instance
        image_files: path(s) to input image(s).
            - str: single image file or folder
            - List[str]: list of channel files for one case
            - List[List[str]]: multiple cases, each a list of channel files
        output_folder: where to save .json results. If None, returns dicts.
        batch_size: number of patches per GPU forward pass (default 2,
            increase if GPU memory allows for faster inference)
        verbose: print timing breakdown

    Returns:
        dict or list of dicts with landmark coordinates per case.
    """
    from batchgenerators.utilities.file_and_folder_operations import (
        maybe_mkdir_p, subfiles
    )

    total_start = time.time()

    # Normalize input to List[List[str]]
    if isinstance(image_files, str):
        if os.path.isdir(image_files):
            # Folder: find all matching files
            from nnlandmark.utilities.utils import create_lists_from_splitted_dataset_folder
            cases = create_lists_from_splitted_dataset_folder(
                image_files, predictor.dataset_json['file_ending']
            )
        else:
            cases = [[image_files]]
    elif isinstance(image_files, list) and len(image_files) > 0:
        if isinstance(image_files[0], str):
            cases = [image_files]  # single case, multiple channels
        else:
            cases = image_files  # already List[List[str]]
    else:
        raise ValueError(f"Unsupported image_files type: {type(image_files)}")

    if output_folder is not None:
        maybe_mkdir_p(output_folder)

    device = predictor.device
    network = predictor.network
    network.to(device)
    network.eval()

    patch_size = predictor.configuration_manager.patch_size
    num_output_channels = predictor.label_manager.num_segmentation_heads - 1

    # ---- Timing accumulators (same columns as predict_from_data_iterator) ----
    # Columns that don't apply in the fast path are recorded as 0.
    wrapper_time = getattr(predictor, 'wrapper_loading_time', 0.0)
    model_time = getattr(predictor, 'model_loading_time', 0.0)
    weights_time = getattr(predictor, 'weights_loading_time', 0.0)
    # file_setup / data_iterator_setup / pool_creation are zero (skipped)
    file_setup_time = 0.0
    data_iterator_setup_time = 0.0
    pool_creation_time = 0.0

    per_case_times = {}  # data_id -> dict of column values
    post_subtimes_per_case = {}  # data_id -> postprocess subtimes

    spine_process_enabled = getattr(predictor, 'spine_process', False)

    results = []
    for case_idx, case_files in enumerate(cases):
        case_start = time.time()
        case_id = os.path.basename(case_files[0]).split('_0000')[0].split('.')[0]
        if verbose:
            print(f'\nPredicting {case_id}:')

        # --- Fast preprocessing (inline, GPU resampling) ---
        t_pre = time.time()
        data_tensor, properties, preprocess_subtimes = preprocess_fast(
            case_files,
            predictor.plans_manager,
            predictor.configuration_manager,
            predictor.dataset_json,
            device=device,
            verbose=verbose,
        )
        preprocess_time = time.time() - t_pre
        print(f'background_preprocessing_time_seconds for {case_id}: {preprocess_time:.4f}s')

        # In fast mode preprocessing is inline, so the main thread "wait" for
        # preprocessing equals the preprocessing time itself.
        wait_for_preprocessing_time = preprocess_time
        print(f'main_thread_wait_for_preprocessing_seconds for {case_id}: {wait_for_preprocessing_time:.4f}s')

        # No export pool → zero wait
        wait_for_export_pool_time = 0.0
        print(f'main_thread_wait_for_export_pool_seconds for {case_id}: {wait_for_export_pool_time:.4f}s')

        # --- Network inference (batched sliding window) ---
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t_inf = time.time()
        with torch.autocast(device.type, enabled=True) if device.type == 'cuda' else dummy_context():
            # Handle multi-fold ensemble
            prediction = None
            for params in predictor.list_of_parameters:
                from torch._dynamo import OptimizedModule
                if not isinstance(network, OptimizedModule):
                    network.load_state_dict(params)
                else:
                    network._orig_mod.load_state_dict(params)

                fold_pred = _predict_sliding_window_batched(
                    network, data_tensor, patch_size, predictor.tile_step_size,
                    num_output_channels, device,
                    use_gaussian=predictor.use_gaussian,
                    batch_size=batch_size,
                    verbose=verbose,
                )
                if prediction is None:
                    prediction = fold_pred
                else:
                    prediction += fold_pred

            if len(predictor.list_of_parameters) > 1:
                prediction /= len(predictor.list_of_parameters)

        if device.type == 'cuda':
            torch.cuda.synchronize()
        inference_time = time.time() - t_inf
        print(f'main_thread_inference_time_seconds for {case_id}: {inference_time:.4f}s')

        # --- Extract landmarks (inline, no export pool) ---
        t_post = time.time()
        ofile = None
        if output_folder is not None:
            ofile = os.path.join(output_folder, case_id)

        landmarks, postprocess_subtimes = _extract_landmarks_from_logits(
            prediction, properties, predictor.plans_manager,
            predictor.label_manager, ofile,
            spine_process=spine_process_enabled,
        )
        postprocess_time = time.time() - t_post
        print(f'background_postprocessing_time_seconds for {case_id}: {postprocess_time:.4f}s')

        # Inline → main thread "waits" for postprocessing = postprocessing time
        wait_for_postprocessing_time = postprocess_time
        print(f'main_thread_wait_for_postprocessing_seconds for {case_id}: {wait_for_postprocessing_time:.4f}s')

        results.append(landmarks)

        empty_cache(device)

        # Store per-case timing (for the original pipeline_times.csv)
        per_case_times[case_id] = {
            'background_preprocessing_time_seconds': preprocess_time,
            'main_thread_wait_for_preprocessing_seconds': wait_for_preprocessing_time,
            'main_thread_wait_for_export_pool_seconds': wait_for_export_pool_time,
            'main_thread_inference_time_seconds': inference_time,
            'main_thread_wait_for_postprocessing_seconds': wait_for_postprocessing_time,
            'background_postprocessing_time_seconds': postprocess_time,
            # Sequential sub-timings
            '_preprocess_subtimes': preprocess_subtimes,
            'spine_postprocessing_time_seconds': postprocess_subtimes.get('spine_postprocessing_time_seconds', 0.0),
        }
        post_subtimes_per_case[case_id] = postprocess_subtimes

        print(f'done with {case_id}')

    # ---- Compute total time ----
    total_time = time.time() - predictor.script_start_time \
        if hasattr(predictor, 'script_start_time') else time.time() - total_start
    print(f"\ntotal predict time : {total_time}", flush=True)

    # ---- Write CSVs ----
    if per_case_times:
        csv_dir = output_folder if output_folder is not None else os.getcwd()

        # ---- 1. Original pipeline_times.csv (same format as predict_from_data_iterator) ----
        csv_path = os.path.join(csv_dir, 'pipeline_times.csv')
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    'data_id',
                    'wrapper_loading_time_seconds',
                    'file_setup_time_seconds',
                    'data_iterator_setup_time_seconds',
                    'pool_creation_time_seconds',
                    'model_loading_time_seconds',
                    'weights_loading_time_seconds',
                    'main_thread_wait_for_preprocessing_seconds',
                    'background_preprocessing_time_seconds',
                    'main_thread_wait_for_export_pool_seconds',
                    'main_thread_inference_time_seconds',
                    'main_thread_wait_for_postprocessing_seconds',
                    'background_postprocessing_time_seconds',
                    'unaccounted_main_thread_time_seconds',
                    'case_pipeline_time_seconds', 
                    'pipeline_total_time_seconds',
                ])

            wrapper_v = float(wrapper_time) if wrapper_time else 0.0
            model_v = float(model_time) if model_time else 0.0
            weights_v = float(weights_time) if weights_time else 0.0

            sum_all_wait_prep = sum(t['main_thread_wait_for_preprocessing_seconds'] for t in per_case_times.values())
            sum_all_wait_pool = sum(t['main_thread_wait_for_export_pool_seconds'] for t in per_case_times.values())
            sum_all_inf = sum(t['main_thread_inference_time_seconds'] for t in per_case_times.values())
            sum_all_wait_post = sum(t['main_thread_wait_for_postprocessing_seconds'] for t in per_case_times.values())

            if isinstance(total_time, (float, int)):
                batch_accounted = (wrapper_v + file_setup_time + data_iterator_setup_time
                                   + pool_creation_time + model_v + weights_v
                                   + sum_all_wait_prep + sum_all_wait_pool + sum_all_inf + sum_all_wait_post)
                batch_unaccounted = total_time - batch_accounted
            else:
                batch_unaccounted = ''

            data_ids_list = list(per_case_times.keys())
            for idx, (data_id, t) in enumerate(per_case_times.items()):
                is_last = (idx == len(data_ids_list) - 1)
                writer.writerow([
                    data_id,
                    wrapper_time if wrapper_time else '',
                    file_setup_time if file_setup_time else '',
                    data_iterator_setup_time if data_iterator_setup_time else '',
                    pool_creation_time if pool_creation_time else '',
                    model_time if model_time else '',
                    weights_time if weights_time else '',
                    t['main_thread_wait_for_preprocessing_seconds'],
                    t['background_preprocessing_time_seconds'],
                    t['main_thread_wait_for_export_pool_seconds'],
                    t['main_thread_inference_time_seconds'],
                    t['main_thread_wait_for_postprocessing_seconds'],
                    t['background_postprocessing_time_seconds'],
                    batch_unaccounted if is_last else '',
                    t['background_preprocessing_time_seconds'] + t['main_thread_inference_time_seconds'] + t['background_postprocessing_time_seconds'],
                    total_time,
                ])
        print(f'\nPipeline times saved to {csv_path}')

        # ---- 2. Sequential timing CSV (granular, no wait/pool columns) ----
        seq_csv_path = os.path.join(csv_dir, 'sequential_times.csv')
        seq_file_exists = os.path.isfile(seq_csv_path)
        with open(seq_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)

            seq_header = [
                'data_id',
                'wrapper_loading_time_seconds',
                'model_loading_time_seconds',
                'weights_loading_time_seconds',
                'load_time_seconds',
                'crop_time_seconds',
                'norm_time_seconds',
                'pre_resampling_time_seconds',
                'preprocessing_total_time_seconds',
                'inference_time_seconds',
                'post_resampling_time_seconds',
                'mask_conversion_time_seconds',
                'cropping_reverse_time_seconds',
                'centroid_extraction_time_seconds',
                'export_output_time_seconds',
                'postprocessing_time_seconds',
            ]
            if spine_process_enabled:
                seq_header.append('spine_postprocessing_time_seconds')
            seq_header.extend([
                'unaccounted_pre_time_seconds',
                'unaccounted_post_time_seconds',
                'total_time_seconds',
            ])

            if not seq_file_exists:
                writer.writerow(seq_header)

            wrapper_v = float(wrapper_time) if wrapper_time else 0.0
            model_v = float(model_time) if model_time else 0.0
            weights_v = float(weights_time) if weights_time else 0.0

            for data_id, t in per_case_times.items():
                sub = t.get('_preprocess_subtimes', {})
                load_v = sub.get('load_time_seconds', 0.0)
                crop_v = sub.get('crop_time_seconds', 0.0)
                norm_v = sub.get('norm_time_seconds', 0.0)
                pre_resample_v = sub.get('resampling_time_seconds', 0.0)
                preprocess_total_v = float(t.get('background_preprocessing_time_seconds', 0.0))
                inf_v = float(t.get('main_thread_inference_time_seconds', 0.0))
                post_v = float(t.get('background_postprocessing_time_seconds', 0.0))

                post_sub = post_subtimes_per_case.get(data_id, {})
                post_resample_v = post_sub.get('resampling_time_seconds', 0.0)
                mask_v = post_sub.get('mask_conversion_time_seconds', 0.0)
                crop_rev_v = post_sub.get('cropping_reverse_time_seconds', 0.0)
                centroid_v = post_sub.get('centroid_extraction_time_seconds', 0.0)
                export_v = post_sub.get('export_output_time_seconds', 0.0)

                per_case_total = preprocess_total_v + inf_v + post_v
                per_case_unaccounted_pre = preprocess_total_v - (load_v + crop_v + norm_v + pre_resample_v)

                accounted_post = post_resample_v + mask_v + crop_rev_v + centroid_v + export_v
                spine_v = float(t.get('spine_postprocessing_time_seconds', 0.0))
                if spine_process_enabled:
                    accounted_post += spine_v
                per_case_unaccounted_post = post_v - accounted_post

                row = [
                    data_id,
                    wrapper_time if wrapper_time else '',
                    model_time if model_time else '',
                    weights_time if weights_time else '',
                    load_v,
                    crop_v,
                    norm_v,
                    pre_resample_v,
                    preprocess_total_v,
                    inf_v,
                    post_resample_v,
                    mask_v,
                    crop_rev_v,
                    centroid_v,
                    export_v,
                    post_v,
                ]
                if spine_process_enabled:
                    row.append(spine_v)
                row.extend([
                    per_case_unaccounted_pre,
                    per_case_unaccounted_post,
                    per_case_total,
                ])

                writer.writerow(row)
        print(f'Sequential times saved to {seq_csv_path}')

    # Clear caches
    compute_gaussian.cache_clear()
    empty_cache(device)

    return results[0] if len(results) == 1 else results
