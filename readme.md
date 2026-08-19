# nnLandmark — Vertebra Identification Fork

This repository is a fork of [nnLandmark](https://github.com/MIC-DKFZ/nnLandmark) specialized for **automatic identification and numbering of vertebrae on CT images**. It was developed as part of the master's thesis *"Optimizing deep learning-based vertebra identification on CT images for efficient clinical deployment"* (UCLouvain, 2025–2026), whose goal was to bring nnLandmark's accuracy into a real clinical workflow by making single-image inference fast enough for a PACS integration (target: a few seconds per image on a consumer GPU).

The full thesis is available in this repository as [`Master_thesis.pdf`](Master_thesis.pdf).

The upstream nnLandmark achieves excellent identification quality but its inference pipeline, inherited from nnU-Net, is optimized for batch/research use and is too slow for interactive clinical use on a single scan. This fork keeps the training procedure of nnLandmark and rewrites the inference path around a single-image scenario. On the VerSe 2019 test set, the resulting model achieves an identification rate of **93.62%**, with an average inference time of **1.40 s per image on a server-grade GPU** (A100) and **1.58 s on a consumer-grade GPU**.

## Optimizations

All optimizations are described in detail in Chapter 6 of the thesis. In summary:

1. **Single-threaded pipeline (`--fast`).** The upstream pipeline spawns two pools of background workers (preprocessing workers via `-npp`, export workers via `-nps`) that use Python's `multiprocessing` `spawn` start method. Each worker has to launch a fresh Python interpreter and re-import heavy libraries (`torch`, `numpy`, …), which is only worthwhile when many images are queued. For single-image inference — the clinical case — that spawn cost is pure overhead. The `--fast` mode runs preprocessing, inference and postprocessing sequentially on the main thread inside a single process.

2. **Bounding-box optimization.** The default cropping to the non-zero region calls `scipy.ndimage.binary_fill_holes` on the full 3D volume before computing the bounding box. Hole filling never changes the outermost non-zero voxels along any axis, so it cannot change the bounding box. It is removed and the bounding box is computed directly from the raw non-zero mask. The tiny impact on intensity statistics (a handful of interior zero voxels per image) is negligible in practice (average of 0.14 % zero voxels on VerSe 2019).

3. **GPU resampling.** The most expensive preprocessing step — resampling the cropped volume to the target spacing — is moved from CPU (Scikit-Image third-order spline) to GPU (PyTorch trilinear interpolation). This gives a large speed-up on 3D volumes, with a resampling accuracy difference that is invisible for heatmap-based landmark detection.

4. **New postprocessing (peak-in-network-space).** The original pipeline resamples the full `C × D × H × W` logits volume back to the original image shape on CPU with trilinear interpolation, then extracts the argmax per channel. This is the slowest part of the pipeline. Instead, we apply sigmoid directly on the network-space logits, extract the argmax per channel there, and then map the resulting coordinates back to the original image space with three cheap operations: (i) scaling by the ratio between cropped and predicted shapes, (ii) adding the crop bounding-box offset, (iii) inverting the axis transposition. This eliminates the resampling step entirely and, as shown in the thesis, is at least as accurate as the original method (and slightly more precise when the input resolution is coarser than the network grid).

5. **Model conversion to ONNX and TensorRT.** The trained PyTorch model is exported to ONNX and then compiled to a TensorRT engine (FP16). ONNX Runtime applies ahead-of-time graph optimizations (operator fusion, constant folding, graph pruning); TensorRT goes further with layer/tensor fusion, kernel auto-tuning for the target GPU and half-precision inference. Both are integrated via a thin `nn.Module` wrapper that overrides `forward()`, so nnLandmark's own prediction code is untouched.

6. **Minor training/inference tweaks.** Training batch size was tuned for the vertebra task, and centroids are extracted from the top-27 voxel patch of the heatmap using a **center-of-mass** rather than a plain argmax, which improves sub-voxel localization.

## New scripts

The `script/` folder contains the utilities added by this fork to support the ONNX/TensorRT deployment path:

- **`script/export_to_onnx.py`** — exports a trained nnLandmark model (a training output folder produced by `nnLM_train`) to an ONNX file. It re-uses `nnUNetPredictor` to load the checkpoint and infer the correct patch size and input/output channels, then calls `torch.onnx.export` with an FP16 dummy input.
  ```bash
  python script/export_to_onnx.py \
      --model_folder /path/to/nnLM_results/DatasetXXX/nnLandmark__nnUNetPlans__3d_lowres \
      --output nnLandmark_3d_lowres.onnx \
      --folds 0 \
      --checkpoint checkpoint_final.pth
  ```

- **`script/build_engine.py`** — builds a TensorRT engine from an ONNX file for the current GPU. FP16 is enabled when the platform supports it. TensorRT engines are hardware-specific and must be rebuilt on each target GPU.
  ```bash
  python script/build_engine.py \
      --onnx  nnLandmark_3d_lowres.onnx \
      --engine nnLandmark_3d_lowres.engine \
      --input_shape 1,1,96,160,160   # batch,channels,D,H,W — take D,H,W from the "patch_size" field of nnUNetPlans.json for the configuration you exported
  ```

- **`script/predict_onnx.py`** — runs prediction on a NIfTI file (or a folder of NIfTI files) using the ONNX model via ONNX Runtime. Internally it wraps the ONNX session in an `nn.Module` (`ONNXWrapper`) and plugs it into nnLandmark's predictor in place of the PyTorch network, so the rest of the pipeline (pre/postprocessing, JSON output) is identical.
  ```bash
  python script/predict_onnx.py \
      --onnx_model  model/nnLandmark_3d_lowres_v11.onnx \
      --model_folder /path/to/nnLM_results/DatasetXXX/nnLandmark__nnUNetPlans__3d_lowres \
      --input  /path/to/image.nii.gz \
      --output /path/to/predictions/ \
      --fast
  ```

- **`script/predict_tensorrt.py`** — same as `predict_onnx.py` but using a compiled TensorRT engine (via a `TRTWrapper`). This is the fastest inference path.
  ```bash
  python script/predict_tensorrt.py \
      --trt_model model/nnLandmark_3d_lowres_v11.engine \
      --model_folder /path/to/nnLM_results/DatasetXXX/nnLandmark__nnUNetPlans__3d_lowres \
      --input  /path/to/image.nii.gz \
      --output /path/to/predictions/ \
      --fast
  ```

All four scripts share `--folds` / `--checkpoint` for model loading and `--fast` / `--verbose` on the prediction side.

## New `nnLM_predict` flags

Two flags were added to the `nnLM_predict` entry point in addition to the upstream nnLandmark flags:

- **`--fast`** — enables the single-threaded, single-process inference pipeline described above. Preprocessing (with the new bounding-box optimization and GPU resampling), inference and postprocessing all run on the main thread, avoiding the cost of spawning `-npp` / `-nps` workers. **Recommended whenever you predict one image (or a few images) at a time**, typical of a clinical PACS use case.


Example:
```bash
nnLM_predict \
    -i /path/to/nnUNet_raw/DatasetXXX/imagesTs/ \
    -o /path/to/predictions/ \
    -d XXX \
    -c 3d_lowres \
    --fast \
```

## Pre-trained ONNX model

The [`model/`](model/) folder ships a pre-trained ONNX model, **`nnLandmark_3d_lowres_v11.onnx`**, trained on the **VerSe 2019 training set** with the `3d_lowres` configuration. It is exported in FP16 and can be used directly with `script/predict_onnx.py`, or compiled to a TensorRT engine with `script/build_engine.py` before use with `script/predict_tensorrt.py`.

---

The remaining sections below are the original nnLandmark documentation (installation, data format, planning, training, evaluation) and still apply to this fork.

## Installation
The upstream repository is itself a fork of [nnU-Net](https://github.com/MIC-DKFZ/nnUNet). Please head over there to read more about it.

We strongly recommend installing this in a dedicated virtual environment (for example conda).
We recommend using a Linux based operating system, for example Ubuntu. Windows should work as well but is not tested.

Some dependencies should be installed manually:
- Install python, we used 3.13.
- Install pytorch according to the instructions on the [pytorch website](https://pytorch.org/get-started/locally/). We recommend version 2.8.
- Pick the correct CUDA version for your system, we used 12.8.
- For the ONNX / TensorRT scripts, also install `onnx`, `onnxruntime-gpu` and (optionally) `tensorrt` matching your CUDA version.

Now you can just clone this repository and install it:

```commandline
git clone <this-fork-url>
cd nnLandmark
pip install -e .
```

## Data Format

### Path setup
We are using the same path system as nnU-Net, defined as environment variables pointing it to raw data, preprocessed data and results. Set them with

```
export nnLM_results=/home/isensee/nnLM_results
export nnLM_preprocessed=/home/isensee/nnLM_preprocessed
export nnLM_raw=/home/isensee/nnLM_raw
```
Make sure at least `$nnLM_preprocessed` (but ideally all of them) are on a fast storage such as a local SSD or very good network drive!

RECOMMENDED: Add these lines to your `.bashrc` file (or whatever you are using) so that the environment variables are set automatically. If you don't do this you need to export them every time you open a new terminal.

### Images and Labels
 Here we follow the nnU-Net format. The training data is stored in imagesTr and labelsTr folders. The labels are multi-label segmetnation maps. Each landmark class belongs to a specific label value, this must be consistent throughout the dataset! The landmark location is represented by a 3x3x3 cube round the target voxel. Generally the size is irrelevant, as during training the location will be extracted be the center of mass of the segmentation. However, it must be ensured that proximate labels do not overlap, as this would distort the location.

### Additional JSONs

- **dataset.json**: Follows the conventions of nnU-Net. The landmark locations are represented as multi-label segmentation map. Consequently each label corresponds to a specific landmark class. This must be consistent throught the entire dataset and experimentation and is defined in the dataset JSON. The label names are accessed in the evaluation to map from the predicted labels to the landmark class.

```bash
{
    "channel_names":
    {
        "0": "MRI"
    },
    "labels":
    {
        "background": 0,
        "landmark_1": 1,
        "landmark_2": 2,

    },
    "numTraining": 110,
    "file_ending": ".nii.gz",
    "name": "Dataset732_Afids"
}
```

- **spacing.json**: This spacing information is used in the evaluation. For each case it contains a image_spacing, taken from the image metadata, and annotation_spacing, taken from the landmark annotation files. This is because some datasets are published with no/wrong image spacing. nnLandmark defaults to look for image_spacing and, if it's null, falls back to annotation_spacing.

```bash
{
  "case_xyz":
  {
    "image_spacing": [
      0.5,
      0.5,
      0.5
    ],
    "annotation_spacing": null
  }
}
```

- **all_landmarks_voxel.json**: Voxel coordinate annotations for all cases (train and test).

```bash
{
  "case_xyz":
  {
    "landmark_1": [
      13,
      19,
      89
    ],
    "landmark_2": [
      19,
      75,
      85
    ],
  }
}
```

## Experiment Planning and Preprocessing

We use the experiment planning and preprocessing functionality of nnU-Net as is.

```bash
nnLM_plan_and_preprocess \
     -d DATASET_ID \
     -c 3d_fullres \
     --verify_dataset_integrity
```
To add the experiment plans for using the ResEncM architecture, our recommendation for the best results, :

```bash
nnLM_plan_experiment \
    -d DATASET \
    -pl nnUNetPlannerResEncM
```


## Training

Start a nnU-Net training with the nnLandmark trainer. For using the ResEncM architecture plans, add the respective flag:

```bash
nnLM_train \
    DATASET_NAME_OR_ID \
    3d_fullres \
    FOLD \
    -p nnUNetResEncUNetMPlans
```


## Predictions

Use the custom nnLandmark predict script to predict a raw image folder. See the *New `nnLM_predict` flags* section above for the vertebra-specific `--fast`.

```bash
nnLM_predict \
    -i /path/to/nnUNet_raw/DATASET_ID/imagesTs/ \
    -o /path/to/evaluation/DATASET_ID/predictions/ \
    -d DATASET_ID \
    -c 3d_fullres\
    -p nnUNetResEncUNetMPlans
```

This scrip will create:

- dataset.json, plans.json, predict_from_raw_data_args.json as in nnU-Net
- Multi-label segmentation .nii.gz for each case. Each landmark is represented by a label containing the top 27 voxels of the predicted heatmap.
- Prediction jsons for each case, containing voxel coordinates and a likelihood for each landmark.


## Evaluation

Use the custom nnLandmark evaluation script:

```bash
nnLM_evaluate \
    -d DATASET_ID \
    -pred /path/to/evaluation/DATASET_ID/predictions/
```

This script will create:

- prediction_all_landmark_voxel.json: Predictions of all cases in voxel coordinates.
- summary_voxel.py: Metrics in voxel
- summary_mm.py: Metrics in mm

## Citation

Upstream nnLandmark:

```bibtex
@misc{ertl2026nnlandmark,
      title={nnLandmark: A Self-Configuring Method for 3D Medical Landmark Detection},
      author={Alexandra Ertl and Stefan Denner and Robin Peretzke and Shuhan Xiao and David Zimmerer and Maximilian Fischer and Markus Bujotzek and Xin Yang and Peter Neher and Fabian Isensee and Klaus H. Maier-Hein},
      year={2026},
      eprint={2504.06742},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.06742},
}
```

If you use the optimizations or the pre-trained vertebra model from this fork, please also cite the associated master's thesis:

```
Giansante, A. (2026). Optimizing deep learning-based vertebra identification on CT images
for efficient clinical deployment. Master's thesis, École polytechnique de Louvain, UCLouvain.
```

## Acknowledgements

The upstream nnLandmark and nnU-Net frameworks are developed and maintained by the Applied Computer Vision Lab (ACVL) of [Helmholtz Imaging](http://helmholtz-imaging.de) and the [Division of Medical Image Computing](https://www.dkfz.de/en/mic/index.php) at the [German Cancer Research Center (DKFZ)](https://www.dkfz.de/en/index.html). Copyright DKFZ and contributors — please make sure your usage of this code is in compliance with its license.

The vertebra-identification adaptation and the optimizations described above were carried out at UCLouvain in collaboration with Telemis, using compute resources from the Consortium des Équipements de Calcul Intensif (CECI), funded by F.R.S.-FNRS (Grant No. 2.5020.11) and the Walloon Region.

<img src="documentation/assets/HI_Logo.png" height="100px" />

<img src="documentation/assets/dkfz_logo.png" height="100px" />
