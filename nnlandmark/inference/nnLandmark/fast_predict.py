import csv
import os
import time
import json
from typing import Tuple, Union, List, Optional

import nnlandmark

import numpy as np
import torch
import torch.nn.functional as F
from acvl_utils.cropping_and_padding.bounding_boxes import get_bbox_from_mask, bounding_box_to_slice
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from tqdm import tqdm

from nnlandmark.inference.nnLandmark.export_prediction import _extract_landmark_coord_and_likelihood
from nnlandmark.inference.nnLandmark.sliding_window_prediction import (
    compute_gaussian,
    compute_steps_for_sliding_window,
)
from nnlandmark.utilities.helpers import empty_cache, dummy_context
from batchgenerators.utilities.file_and_folder_operations import join
from nnlandmark.utilities.find_class_by_name import recursive_find_python_class
from nnlandmark.preprocessing.resampling.default_resampling import compute_new_shape
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p


def _crop_to_nonzero_fast(data: np.ndarray):
    """
    Crop to non-zero bounding box without scipy binary_fill_holes.
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
    """
    if list(data_tensor.shape[1:]) == list(new_shape):
        return data_tensor

    out = F.interpolate(
        data_tensor.unsqueeze(0).float().to(device),
        size=new_shape,
        mode='trilinear',
        align_corners=False,
    )
    return out.squeeze(0) # back to (C, D, H, W)


def preprocess_fast(image_files: List[str],
                    plans_manager,
                    configuration_manager,
                    dataset_json: dict,
                    device: torch.device = torch.device('cuda'),
                    verbose: bool = False) -> Tuple[torch.Tensor, dict]:
    """
    Fast inline preprocessing for a single image. No multiprocessing.

    Returns:
      (data_tensor, properties_dict)
    """

    t0 = time.time()

    # Load image
    rw = plans_manager.image_reader_writer_class()
    data, properties = rw.read_images(image_files)
    t_load = time.time()

    # Transpose
    data = data.astype(np.float32, copy=False)
    data = data.transpose([0, *[i + 1 for i in plans_manager.transpose_forward]])
    original_spacing = [properties['spacing'][i] for i in plans_manager.transpose_forward]

    # Crop to nonzero
    properties['shape_before_cropping'] = data.shape[1:]
    data, bbox = _crop_to_nonzero_fast(data)
    properties['bbox_used_for_cropping'] = bbox
    properties['shape_after_cropping_and_before_resampling'] = data.shape[1:]
    t_crop = time.time()

    # Normalization
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

    # GPU resampling
    target_spacing = configuration_manager.spacing
    if len(target_spacing) < len(data.shape[1:]):
        target_spacing = [original_spacing[0]] + list(target_spacing)
    new_shape = compute_new_shape(data.shape[1:], original_spacing, target_spacing)

    data_tensor = torch.from_numpy(data)
    data_tensor = _resample_gpu(data_tensor, [int(s) for s in new_shape], device)
    data_tensor = data_tensor.cpu() # Move back to CPU to match the rest of the pipeline expectations

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

def _extract_landmarks_from_logits(
    predicted_logits: torch.Tensor,
    properties: dict,
    plans_manager,
    label_manager,
    output_file_truncated: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[dict, dict]:
    """
    Extract landmark coordinates directly from network-resolution logits.

    Instead of resampling the full C×D×H×W probability volume back to original
    space (expensive), we find the peak in network space and scale the
    coordinates back. This is what export_prediction_from_logits already does,
    but we call it inline without multiprocessing.

    Returns:
        (output_json, postprocess_times)
    """

    # For measurement times
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

    # Extract centroid from network volume
    centroid_coords = {}
    for ch, cls_id in enumerate(class_ids):
        coord_pred, lik = _extract_landmark_coord_and_likelihood(probs[ch])
        if coord_pred is not None:
            centroid_coords[cls_id] = (coord_pred, lik)

    t1_centroid = time.time()

    # Reverse resampling
    t0_resample = time.time()
    coords_cropped = {}
    for cls_id, (coord_pred, lik) in centroid_coords.items():
        cx, cy, cz = coord_pred
        coords_cropped[cls_id] = (
            cz * (shape_after_crop[0] / shape_pred[0]),
            cy * (shape_after_crop[1] / shape_pred[1]),
            cx * (shape_after_crop[2] / shape_pred[2]),
            lik,
        )
    post_times['resampling_time_seconds'] = time.time() - t0_resample

    t0_crop_rev = time.time()
    # Reverse cropping and transposing
    for ch, cls_id in enumerate(class_ids):
        if cls_id in coords_cropped:
            coord_crop_z, coord_crop_y, coord_crop_x, lik = coords_cropped[cls_id]
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
    post_times['cropping_reverse_time_seconds'] = time.time() - t0_crop_rev

    post_times['centroid_extraction_time_seconds'] = t1_centroid - t0_centroid

    if output_file_truncated is not None:
        t0_export = time.time()
        with open(output_file_truncated + ".json", "w") as f:
            json.dump(out_json, f, indent=4)
        post_times['export_output_time_seconds'] = time.time() - t0_export

    if verbose:
        print(f"fast postprocessing: {post_times}")
    return out_json, post_times


# ---------------------------------------------------------------------------
# Main fast prediction entry point
# ---------------------------------------------------------------------------

def predict_fast(
    predictor,
    image_files: Union[str, List[str], List[List[str]]],
    output_folder: Optional[str] = None,
    verbose: bool = False,
) -> Union[dict, List[dict]]:
    """
    Fast single-image (or few-image) landmark prediction.

    Bypasses all multiprocessing overhead of the default pipeline.
    Uses GPU resampling and batched patch inference.

    Writes timing data to `pipeline_times.csv`

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
        likelihood_table_path: path to a calibration lookup table JSON
            (built by likelihood_calibration.py). When provided together with
            error_threshold_mm, predictions whose expected error exceeds the
            threshold are discarded.
        error_threshold_mm: maximum expected error (mm) to keep a prediction.
            Requires likelihood_table_path.
        error_metric: which error statistic to look up in the table.
            One of "p90_error_mm", "median_error_mm", "mean_error_mm".

    Returns:
        dict or list of dicts with landmark coordinates per case.
    """
    
    total_start = time.time()

    # Normalize input to list_of_lists_or_source_folder because we also accept only one image

    # Folder path
    if isinstance(image_files, str) and os.path.isdir(image_files):
        list_of_lists_or_source_folder = image_files
    # Single image with one channel
    elif isinstance(image_files, str):
        list_of_lists_or_source_folder = [[image_files]]

    elif isinstance(image_files, list) and len(image_files) > 0:
        # Single image with multiple channels
        if isinstance(image_files[0], str):
            list_of_lists_or_source_folder = [image_files]
        else:
            # Multiple images, each with multiple channels
            list_of_lists_or_source_folder = image_files
    else:
        raise ValueError(f"Unsupported image_files type: {type(image_files)}")

    if output_folder is not None:
        maybe_mkdir_p(output_folder)

    list_of_lists_or_source_folder, output_filename_truncated, _ = (
        predictor._manage_input_and_output_lists(
            list_of_lists_or_source_folder,
            output_folder,
            folder_with_segs_from_prev_stage=None,
            overwrite=True,
            part_id=0,
            num_parts=1,
            save_probabilities=False,
        )
    )

    if len(list_of_lists_or_source_folder) == 0:
        return []

    device = predictor.device
    network = predictor.network
    network.to(device)
    network.eval()

    wrapper_time = getattr(predictor, 'wrapper_loading_time', 0.0)
    model_time = getattr(predictor, 'model_loading_time', 0.0)
    weights_time = getattr(predictor, 'weights_loading_time', 0.0)
    file_setup_time = 0.0
    data_iterator_setup_time = 0.0
    pool_creation_time = 0.0
    per_case_times = {} 
    post_subtimes_per_case = {} 

    results = []
    for case_index, case_files in enumerate(list_of_lists_or_source_folder):
        case_start = time.time()
        if output_filename_truncated is not None:
            output_path = output_filename_truncated[case_index]
            case_id = os.path.basename(output_path)
        else:
            output_path = None
            case_id = os.path.basename(case_files[0]).split('_0000')[0].split('.')[0]
        if verbose:
            print(f'\nPredicting {case_id}:')

        # Fast preprocessing
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
        wait_for_preprocessing_time = preprocess_time # In fast mode preprocessing is inline, so the main thread wait for preprocessing equals the preprocessing time itself.
        wait_for_export_pool_time = 0.0 # No export pool so zero wait
        if verbose :
            print(f'background_preprocessing_time_seconds for {case_id}: {preprocess_time:.4f}s')
            print(f'main_thread_wait_for_preprocessing_seconds for {case_id}: {wait_for_preprocessing_time:.4f}s')
            print(f'main_thread_wait_for_export_pool_seconds for {case_id}: {wait_for_export_pool_time:.4f}s')

        # Network inference
        if device.type == 'cuda':
            empty_cache(device)
            torch.cuda.synchronize()

        t_inf = time.time()

        prediction = predictor.predict_logits_from_preprocessed_data(data_tensor)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        inference_time = time.time() - t_inf
        if verbose:
            print(f'main_thread_inference_time_seconds for {case_id}: {inference_time:.4f}s')

        # Extract landmarks
        t_post = time.time()
        ofile = None
        if output_folder is not None:
            ofile = os.path.join(output_folder, case_id)

        landmarks, postprocess_subtimes = _extract_landmarks_from_logits(
            prediction, properties, predictor.plans_manager,
            predictor.label_manager, ofile,
        )
        postprocess_time = time.time() - t_post
        wait_for_postprocessing_time = postprocess_time
        if verbose:
            print(f'background_postprocessing_time_seconds for {case_id}: {postprocess_time:.4f}s')
            print(f'main_thread_wait_for_postprocessing_seconds for {case_id}: {wait_for_postprocessing_time:.4f}s')

        results.append(landmarks)

        per_case_times[case_id] = {
            'main_thread_wait_for_preprocessing_seconds': wait_for_preprocessing_time,
            'background_preprocessing_time_seconds': preprocess_time,
            'main_thread_wait_for_export_pool_seconds': wait_for_export_pool_time,
            'main_thread_inference_time_seconds': inference_time,
            'main_thread_wait_for_postprocessing_seconds': wait_for_postprocessing_time,
            'background_postprocessing_time_seconds': postprocess_time,
            '_preprocess_subtimes': preprocess_subtimes,
        }
        post_subtimes_per_case[case_id] = postprocess_subtimes

        empty_cache(device)

        if verbose:
            print(f'done with {case_id}')

    # Compute total time
    total_time = time.time() - predictor.script_start_time \
        if hasattr(predictor, 'script_start_time') else time.time() - total_start
    if verbose:
        print(f"\ntotal predict time : {total_time}", flush=True)

    # Write CSV
    if per_case_times:
        csv_dir = output_folder if output_folder is not None else os.getcwd()

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

        if verbose:
            print(f'\nPipeline times saved to {csv_path}')

        # Sequential timing CSV
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

            if not seq_file_exists:
                writer.writerow(seq_header)

            wrapper_v = float(wrapper_time) if wrapper_time else 0.0
            model_v = float(model_time) if model_time else 0.0
            weights_v = float(weights_time) if weights_time else 0.0

            for seq_idx, (data_id, t) in enumerate(per_case_times.items()):
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

                writer.writerow(row)
        if verbose:
            print(f'Sequential times saved to {seq_csv_path}')

    # Clear caches
    compute_gaussian.cache_clear()
    empty_cache(device)

    return results[0] if len(results) == 1 else results
