"""
Spine postprocessing for nnLandmark vertebra detection.

Takes raw heatmaps (C, Z, Y, X) and applies:
1. Spine centerline extraction along Z (craniocaudal direction = dim 0)
2. Spine rectification (3D -> 1D)
3. Anatomically-constrained optimization
4. Mapping back to 3D coordinates
"""

import bisect

import numpy as np
from scipy.ndimage import gaussian_filter1d, map_coordinates
from scipy.signal import find_peaks


VERTEBRA_NAMES = [
    "C1", "C2", "C3", "C4", "C5", "C6", "C7",
    "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12",
    "L1", "L2", "L3", "L4", "L5", "L6"
]

def _get_vertebra_name(idx, num_vertebrae):
    """Return a vertebra name for a given channel index."""
    if idx < len(VERTEBRA_NAMES):
        return VERTEBRA_NAMES[idx]
    return f"V{idx + 1}"


def postprocess_nnlandmark_output(
    heatmaps,
    spacing=None,
    threshold=0.5,
    threshold_fallback=0.1,
    rectified_width=30,
    smooth_sigma=5.0,
    max_iterations=50,
    peak_min_distance=None,
    verbose=True,
):
    """
    Main entry point for spine postprocessing.

    Parameters
    ----------
    heatmaps : np.ndarray, shape (C, W, H, L)
        Per-vertebra activation maps. C=26, values in [0, 1].
    spacing : tuple of float, optional
        Voxel spacing (sx, sy, sz) in mm.
    threshold : float
        Activation threshold for centerline extraction.
    threshold_fallback : float
        Fallback threshold if too few centerline points found.
    rectified_width : int
        Half-width of the normal plane sampling grid.
    smooth_sigma : float
        Gaussian sigma for centerline smoothing.
    max_iterations : int
        Maximum optimization iterations.
    peak_min_distance : int, optional
        Minimum distance between peaks. Auto-computed if None.
    verbose : bool
        Print detailed debug output at each stage.

    Returns
    -------
    list of dict
        Each dict has "label" (int), "coordinate" (tuple), "likelihood" (float).
    """
    assert heatmaps.ndim == 4, f"Expected 4D heatmaps (C, W, H, L), got {heatmaps.ndim}D"
    num_vertebrae = heatmaps.shape[0]

    # Stage 1: Centerline extraction
    centerline = _extract_centerline(heatmaps, threshold, threshold_fallback, smooth_sigma)

    if verbose:
        print("=== STAGE 1: Centerline ===")
        print(f"Centerline points: {len(centerline)}")
        if len(centerline) >= 2:
            print(f"Z range: [{centerline[0, 0]:.1f}, {centerline[-1, 0]:.1f}]")

    if len(centerline) < 3:
        return _fallback_argmax(heatmaps, num_vertebrae)

    # Stage 2: Rectification
    Q_v, Q_hat = _rectify(heatmaps, centerline, rectified_width)

    M = Q_hat.shape[0]
    if peak_min_distance is None:
        peak_min_distance = max(3, M // 40)

    # Determine orientation along centerline.
    # Only use channels with significant signal (>5% of overall max)
    # to avoid noisy weak channels diluting the correlation.
    overall_max_orient = Q_v.max()
    sig_threshold = 0.05 * overall_max_orient
    t_indices = np.arange(M)
    centroids = np.zeros(num_vertebrae)
    for v in range(num_vertebrae):
        if Q_v[v].max() < sig_threshold:
            centroids[v] = np.nan
            continue
        mass = Q_v[v].sum()
        if mass > 1e-4:
            centroids[v] = np.sum(Q_v[v] * t_indices) / mass
        else:
            centroids[v] = np.nan

    valid = ~np.isnan(centroids)
    valid_indices = np.where(valid)[0]
    is_reversed = False
    if len(valid_indices) >= 2:
        # If correlation is negative, small v (head) is at large t.
        # We want small v to be at small t for the optimization to work.
        corr = np.corrcoef(valid_indices, centroids[valid_indices])[0, 1]
        if corr < 0:
            is_reversed = True

    if is_reversed:
        Q_v = Q_v[:, ::-1]
        Q_hat = Q_hat[::-1]
        centerline = centerline[::-1]

    if verbose:
        print(f"\n=== STAGE 2: Rectification ===")
        print(f"Q_hat shape: {M} points along centerline")
        print(f"Q_hat range: [{Q_hat.min():.2f}, {Q_hat.max():.2f}]")
        overall_max = Q_v.max()
        active_threshold = 0.1 * overall_max
        print(f"Per-channel max activation (Q_v):")
        for v in range(num_vertebrae):
            ch_max = Q_v[v].max()
            if ch_max > active_threshold:
                name = _get_vertebra_name(v, num_vertebrae)
                print(f"  {name} (ch {v}): max={ch_max:.2f} at t={Q_v[v].argmax()}")
        print(f"Orientation reversed: {is_reversed}")

    # Stage 3: LIS-based identification
    vertebrae = _identify_vertebrae(Q_v, Q_hat, num_vertebrae, centerline=centerline, spacing=spacing, verbose=verbose)

    if len(vertebrae) < 2:
        return _fallback_argmax(heatmaps, num_vertebrae)

    # Stage 4: Map back to 3D
    results = _map_to_3d(vertebrae, Q_v, centerline, heatmaps)

    if verbose:
        print(f"\n=== STAGE 4: Final detections ===")
        print(f"{len(results)} vertebrae detected:")
        for r in results:
            name = _get_vertebra_name(r['label'], num_vertebrae)
            c = r['coordinate']
            print(f"  {name}: coord=({c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f}), likelihood={r['likelihood']:.3f}")

    return results


# ---------- Stage 1: Centerline Extraction ----------


def _extract_centerline(heatmaps, threshold, threshold_fallback, smooth_sigma):
    G_hat = heatmaps.sum(axis=0)  # (D0, D1, D2) = (Z, Y, X) after nnLandmark preprocessing
    D0, D1, D2 = G_hat.shape

    centerline_points = []

    # Iterate along dim 0 (Z = craniocaudal direction, the spine axis)
    # For each axial slice, find the activation centroid in (Y, X)
    for z in range(D0):
        slice_2d = G_hat[z, :, :]  # axial slice, shape (D1, D2) = (Y, X)
        cy, cx = _slice_center(slice_2d, threshold)
        if cy is None:
            cy, cx = _slice_center(slice_2d, threshold_fallback)
        if cy is not None:
            centerline_points.append((float(z), cy, cx))  # (Z_pos, Y_center, X_center)

    if len(centerline_points) < 3:
        return np.array(centerline_points)

    centerline = np.array(centerline_points)

    # Smooth Y and X positions along the spine; keep Z as-is (integer parameterization)
    centerline[:, 1] = gaussian_filter1d(centerline[:, 1], sigma=smooth_sigma)
    centerline[:, 2] = gaussian_filter1d(centerline[:, 2], sigma=smooth_sigma)

    return centerline


def _slice_center(slice_2d, threshold):
    mask = slice_2d > threshold
    if mask.sum() < 2:
        return None, None
    coords = np.argwhere(mask)  # (N, 2) with columns (x, y)
    weights = slice_2d[mask]
    cx = np.average(coords[:, 0], weights=weights)
    cy = np.average(coords[:, 1], weights=weights)
    return cx, cy


# ---------- Stage 2: Rectification ----------


def _rectify(heatmaps, centerline, rectified_width):
    M = len(centerline)
    C = heatmaps.shape[0]

    # Compute local frames
    e1, e2, e3 = _compute_local_frames(centerline)

    # Build sampling grid offsets
    grid_range = np.arange(-rectified_width, rectified_width + 1)
    dx_grid, dy_grid = np.meshgrid(grid_range, grid_range, indexing="ij")
    dx_flat = dx_grid.ravel()
    dy_flat = dy_grid.ravel()
    num_plane_points = len(dx_flat)

    # Compute all sample coordinates at once: shape (M, num_plane_points, 3)
    # sample_point[t, p] = centerline[t] + e1[t]*dx_flat[p] + e2[t]*dy_flat[p]
    sample_coords = (
        centerline[:, np.newaxis, :]
        + e1[:, np.newaxis, :] * dx_flat[np.newaxis, :, np.newaxis]
        + e2[:, np.newaxis, :] * dy_flat[np.newaxis, :, np.newaxis]
    )  # (M, num_plane_points, 3)

    # Reshape for map_coordinates: need (3, N) where N = M * num_plane_points
    coords_flat = sample_coords.reshape(-1, 3).T  # (3, M*num_plane_points)

    # Sample each channel
    Q_v = np.zeros((C, M))
    for v in range(C):
        sampled = map_coordinates(heatmaps[v], coords_flat, order=1, mode="constant", cval=0.0)
        sampled_2d = sampled.reshape(M, num_plane_points)
        Q_v[v] = sampled_2d.sum(axis=1)

    Q_hat = Q_v.sum(axis=0)
    return Q_v, Q_hat


def _compute_local_frames(centerline):
    M = len(centerline)
    e3 = np.zeros((M, 3))

    # Tangent via finite differences
    for t in range(M):
        if t == 0:
            diff = centerline[1] - centerline[0]
        elif t == M - 1:
            diff = centerline[M - 1] - centerline[M - 2]
        else:
            diff = centerline[t + 1] - centerline[t - 1]
        norm = np.linalg.norm(diff)
        e3[t] = diff / norm if norm > 1e-8 else np.array([0, 0, 1])

    e1 = np.zeros((M, 3))
    e2 = np.zeros((M, 3))
    y_axis = np.array([0.0, 1.0, 0.0])
    x_axis = np.array([1.0, 0.0, 0.0])

    for t in range(M):
        ref = y_axis
        if abs(np.dot(e3[t], y_axis)) > 0.9:
            ref = x_axis
        proj = ref - np.dot(ref, e3[t]) * e3[t]
        norm = np.linalg.norm(proj)
        e2[t] = proj / norm if norm > 1e-8 else x_axis
        e1[t] = np.cross(e2[t], e3[t])

    return e1, e2, e3


# ---------- Stage 3: LIS-based Identification ----------


def _longest_increasing_subsequence(seq):
    """
    Return indices of the longest strictly increasing subsequence.
    Uses O(n log n) algorithm with backtracking.
    """
    n = len(seq)
    if n == 0:
        return []

    tails = []       # smallest tail value for LIS of each length
    tail_indices = [] # index in seq of that tail value
    predecessors = [-1] * n

    for i in range(n):
        pos = bisect.bisect_left(tails, seq[i])
        if pos == len(tails):
            tails.append(seq[i])
            tail_indices.append(i)
        else:
            tails[pos] = seq[i]
            tail_indices[pos] = i
        predecessors[i] = tail_indices[pos - 1] if pos > 0 else -1

    # Backtrack to recover the subsequence
    result = []
    idx = tail_indices[-1]
    while idx != -1:
        result.append(idx)
        idx = predecessors[idx]

    return list(reversed(result))


def _identify_vertebrae(Q_v, Q_hat, num_vertebrae, centerline, spacing=None, verbose=True,
                        confidence_threshold=0.3, extension_threshold=0.03):
    """
    Identify and localize vertebrae using True 3D physical distances.
    """
    M = Q_v.shape[1]
    overall_max = Q_v.max()
    if overall_max < 1e-8:
        return[]

    # Ensure spacing is valid for 3D math
    sp_arr = np.array(spacing) if spacing is not None else np.array([1.0, 1.0, 1.0])

    # Helper function: Gets the true physical 3D coordinate (in mm) for a position 't' on the centerline
    def get_phys_coord(t):
        idx = np.clip(int(round(t)), 0, len(centerline) - 1)
        return centerline[idx] * sp_arr

    if verbose:
        print(f"\n=== STAGE 3: LIS-based identification ===")

    # ------------------------------------------------------------------
    # Step 1: Extract candidate positions
    # ------------------------------------------------------------------
    candidates = []
    for v in range(num_vertebrae):
        peak_pos = int(Q_v[v].argmax())
        peak_val = float(Q_v[v].max())
        confidence = peak_val / overall_max
        candidates.append((v, peak_pos, confidence))

    if verbose:
        print(f"\nStep 1 - Candidates:")
        for v, pos, conf in candidates:
            if conf > confidence_threshold:
                name = _get_vertebra_name(v, num_vertebrae)
                print(f"  {name} (ch {v}): pos={pos}, confidence={conf:.2f}")

    # ------------------------------------------------------------------
    # Step 2: Filter weak channels
    # ------------------------------------------------------------------
    filtered = [(v, pos, conf) for v, pos, conf in candidates
                if conf > confidence_threshold]

    if verbose:
        excluded = [_get_vertebra_name(v, num_vertebrae)
                    for v, _, conf in candidates if conf <= confidence_threshold]
        print(f"\nStep 2 - Filtered: {len(filtered)} of {num_vertebrae} channels have signal")
        if excluded:
            print(f"  Excluded: {', '.join(excluded)}")

    if len(filtered) < 2:
        return []

    # ------------------------------------------------------------------
    # Step 3: Find Longest Increasing Subsequence
    # ------------------------------------------------------------------

    positions = [pos for _, pos, _ in filtered]
    lis_indices = _longest_increasing_subsequence(positions)

    filled = [(filtered[i][0], filtered[i][1]) for i in lis_indices]

    # --- Step 4: Spacing anomaly detection (3D distance, local reference) ---
    spacings_3d_mm =[]
    for i in range(len(filled) - 1):
        c_curr = get_phys_coord(filled[i][1])
        c_next = get_phys_coord(filled[i + 1][1])
        spacings_3d_mm.append(float(np.linalg.norm(c_next - c_curr)))
        
    global_med_sp = float(np.median(spacings_3d_mm)) if spacings_3d_mm else median_spacing_mm

    if verbose:
        print(f"\n  Step 4b - Spacing anomaly detection (3D physical distance):")
        print(f"    Global median 3D spacing (filled): {global_med_sp:.1f} mm")
        print(f"    3D spacings (mm): {[f'{s:.1f}' for s in spacings_3d_mm]}")

    spacing_insertions =[]
    for i in range(len(spacings_3d_mm)):
        neighbors =[]
        for j in range(max(0, i - 2), min(len(spacings_3d_mm), i + 3)):
            if j != i:
                neighbors.append(spacings_3d_mm[j])

        local_ref = float(np.mean(neighbors)) if len(neighbors) >= 2 else global_med_sp
        
        # Prevent division by zero or abnormally tiny references
        local_ref = max(local_ref, global_med_sp * 0.6) 

        gap_3d_mm = spacings_3d_mm[i]
        ratio = gap_3d_mm / local_ref
        
        v_curr, t_curr = filled[i]
        v_next, t_next = filled[i + 1]
        label_diff = v_next - v_curr

        if ratio > 3.5: n_total = 4
        elif ratio > 2.6: n_total = 3
        elif ratio > 2: n_total = 2  # Physical gap > 1.65x signals missing vertebra
        else: continue

        n_to_insert = n_total - label_diff
        if n_to_insert <= 0:
            continue

        if verbose:
            print(f"    Anomaly at gap {i}: 3D gap={gap_3d_mm:.1f}mm, local_ref={local_ref:.1f}mm, ratio={ratio:.2f}, inserting {n_to_insert}")

        t_span = t_next - t_curr
        for j in range(1, n_to_insert + 1):
            t_insert = t_curr + t_span * j / n_total
            spacing_insertions.append((-1, t_insert))

    if spacing_insertions:
        filled.extend(spacing_insertions)
        filled.sort(key=lambda x: x[1])

    # ------------------------------------------------------------------
    # Step 5: Extend at the ends
    # ------------------------------------------------------------------
    # Recompute median spacing from filled set
    if len(filled) >= 2:
        consec_diffs = [filled[i+1][1] - filled[i][1] for i in range(len(filled)-1)]
        median_spacing = float(np.median(consec_diffs))

    if verbose:
        print(f"\nStep 5 - End extension:")
        print(f"  Median spacing for extension: {median_spacing:.1f}")

    window_half = max(1, int(median_spacing / 3))

    # Extend toward head (lower channels)
    head_added = []
    v_first, t_first = filled[0]
    while v_first > 0 and t_first - median_spacing > 0:
        v_cand = v_first - 1
        t_cand = t_first - median_spacing
        w_start = max(0, int(t_cand) - window_half)
        w_end = min(M, int(t_cand) + window_half + 1)
        if w_end <= w_start:
            break
        window_vals = Q_v[v_cand, w_start:w_end]
        local_max = float(window_vals.max())
        if local_max > extension_threshold * overall_max:
            local_peak = w_start + int(window_vals.argmax())
            filled.insert(0, (v_cand, float(local_peak)))
            head_added.append(
                f"{_get_vertebra_name(v_cand, num_vertebrae)}@{local_peak}"
                f" (confidence={local_max/overall_max:.2f})")
            v_first, t_first = v_cand, float(local_peak)
        else:
            break

    # Extend toward tail (higher channels)
    tail_added = []
    v_last, t_last = filled[-1]
    while v_last < num_vertebrae - 1 and t_last + median_spacing < M:
        v_cand = v_last + 1
        t_cand = t_last + median_spacing
        w_start = max(0, int(t_cand) - window_half)
        w_end = min(M, int(t_cand) + window_half + 1)
        if w_end <= w_start:
            break
        window_vals = Q_v[v_cand, w_start:w_end]
        local_max = float(window_vals.max())
        if local_max > extension_threshold * overall_max:
            local_peak = w_start + int(window_vals.argmax())
            filled.append((v_cand, float(local_peak)))
            tail_added.append(
                f"{_get_vertebra_name(v_cand, num_vertebrae)}@{local_peak}"
                f" (confidence={local_max/overall_max:.2f})")
            v_last, t_last = v_cand, float(local_peak)
        else:
            break

    if verbose:
        if head_added:
            print(f"  Extended head: {', '.join(head_added)}")
        else:
            print(f"  No head extension")
        if tail_added:
            print(f"  Extended tail: {', '.join(tail_added)}")
        else:
            print(f"  No tail extension")

    # ------------------------------------------------------------------
    # Step 6: Final position refinement (gentle)
    # ------------------------------------------------------------------
    """
    reliable_set = set((v, p) for v, p in reliable)

    if verbose:
        print(f"\nStep 6 - Refinement:")

    for idx in range(len(filled)):
        v, t = filled[idx]
        if (v, t) in reliable_set:
            # Already at argmax, keep as is
            continue
        # Check for local peak near current position
        w_start = max(0, int(t) - window_half)
        w_end = min(M, int(t) + window_half + 1)
        if w_end <= w_start:
            continue
        window_vals = Q_v[v, w_start:w_end]
        local_max = float(window_vals.max())
        if local_max > confidence_threshold * overall_max:
            local_peak = w_start + int(window_vals.argmax())
            if verbose:
                name = _get_vertebra_name(v, num_vertebrae)
                print(f"  {name}: snapped to peak at {local_peak} (was {t:.1f})")
            filled[idx] = (v, float(local_peak))
        else:
            if verbose:
                name = _get_vertebra_name(v, num_vertebrae)
                print(f"  {name}: kept at geometric pos {t:.1f} (no strong peak nearby)")

    # Sort by position before returning
    filled.sort(key=lambda x: x[1])
    """
    # ------------------------------------------------------------------
    # Step 7: Final label reassignment
    # ------------------------------------------------------------------
    # Assign consecutive labels starting from the best vl offset
    all_positions = [t for _, t in filled]
    N = len(all_positions)
    best_vl = 0
    best_score = -np.inf
    all_scores = []

    for vl in range(max(0, num_vertebrae - N) + 1):
        score = 0.0
        valid = True
        for i in range(N):
            v = vl + i
            if v >= num_vertebrae:
                score = -1e12
                valid = False
                break
            t_idx = int(round(all_positions[i]))
            t_idx = np.clip(t_idx, 0, Q_v.shape[1] - 1)
            score += Q_v[v, t_idx]
        all_scores.append((vl, score))
        if score > best_score:
            best_score = score
            best_vl = vl

    # Rebuild filled with correct labels
    filled = [(best_vl + i, all_positions[i]) for i in range(N)]

    if verbose:
        print(f"\nStep 7 - Label reassignment:")
        all_scores.sort(key=lambda x: -x[1])
        print(f"  Top-3 vl candidates:")
        for rank, (vl_cand, sc) in enumerate(all_scores[:3]):
            first = _get_vertebra_name(vl_cand, num_vertebrae)
            last = _get_vertebra_name(min(vl_cand + N - 1, num_vertebrae - 1), num_vertebrae)
            marker = " <-- best" if vl_cand == best_vl else ""
            print(f"    vl={vl_cand} ({first}->{last}): score={sc:.2f}{marker}")
        first_name = _get_vertebra_name(filled[0][0], num_vertebrae)
        last_name = _get_vertebra_name(filled[-1][0], num_vertebrae)
        print(f"\nFinal: {len(filled)} vertebrae, {first_name} -> {last_name}")

    return filled


# ---------- Stage 4: Map Back to 3D ----------


def _map_to_3d(vertebrae, Q_v, centerline, heatmaps):
    """
    Map detected vertebrae back to 3D coordinates.

    Parameters
    ----------
    vertebrae : list of (int, float)
        Each tuple is (label, position_along_centerline).
    Q_v : np.ndarray
        Per-channel 1D signals (used for likelihood if heatmaps not available).
    centerline : np.ndarray
        3D centerline points.
    heatmaps : np.ndarray
        Original 4D heatmaps for reading likelihood at 3D coordinates.
    """
    results = []
    M = len(centerline)

    for label, t in vertebrae:
        # Interpolate 3D coordinate from centerline
        t_floor = int(np.floor(t))
        t_ceil = int(np.ceil(t))
        t_floor = np.clip(t_floor, 0, M - 1)
        t_ceil = np.clip(t_ceil, 0, M - 1)

        if t_floor == t_ceil:
            coord = centerline[t_floor]
        else:
            frac = t - t_floor
            coord = centerline[t_floor] * (1 - frac) + centerline[t_ceil] * frac

        # Likelihood from original heatmap at 3D coordinate
        coord_rounded = tuple(int(round(c)) for c in coord)
        coord_clipped = tuple(
            np.clip(coord_rounded[d], 0, heatmaps.shape[d + 1] - 1) for d in range(3)
        )
        likelihood = float(heatmaps[label, coord_clipped[0], coord_clipped[1], coord_clipped[2]])

        results.append({
            "label": int(label),
            "coordinate": tuple(coord),
            "likelihood": likelihood,
        })

    return results



# ---------- Fallback ----------


def _fallback_argmax(heatmaps, num_vertebrae=None):
    if num_vertebrae is None:
        num_vertebrae = heatmaps.shape[0]
    results = []
    for v in range(num_vertebrae):
        channel = heatmaps[v]
        max_val = channel.max()
        if max_val > 0.1:
            idx = np.unravel_index(channel.argmax(), channel.shape)
            results.append({
                "label": v,
                "coordinate": tuple(float(c) for c in idx),
                "likelihood": float(max_val),
            })
    return results
