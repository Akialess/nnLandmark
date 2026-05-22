import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from copy import deepcopy
from typing import Union, List, Tuple
from torch import autocast

from nnlandmark.utilities.helpers import dummy_context
from nnlandmark.training.nnUNetTrainer.project_specific.nnLandmark.nnLandmark_trainer import nnLandmark
from nnlandmark.utilities.plans_handling.plans_handler import PlansManager


# ---------------------------------------------------------------------------
# Distillation loss adapted from Fast-nnUNet for landmark heatmap regression
# ---------------------------------------------------------------------------
# In segmentation (nnUNet), KL divergence is applied over the CLASS dimension
# (dim=1) because classes are mutually exclusive (softmax over classes).
#
# In landmark identification (nnLandmark), each channel is an independent
# spatial heatmap predicting WHERE a landmark is.  The natural probability
# distribution is therefore over the SPATIAL dimensions (not channels).
#
# We apply a "spatial softmax" per channel: flatten spatial dims, divide by
# temperature, softmax -> probability map over spatial locations, then compute
# KL divergence.  Temperature softening lets the student learn from the full
# shape of the teacher's heatmap, not just the peak.
# ---------------------------------------------------------------------------

def spatial_kl_distillation_loss(student_logits, teacher_logits, temperature):
    """
    KL divergence distillation loss over spatial dimensions for landmark heatmaps.

    For each (batch, channel) pair, treats the spatial volume as a categorical
    distribution over locations, applies temperature-scaled softmax, and
    computes KL(teacher || student).

    Args:
        student_logits: [B, C, *spatial] raw student logits (before sigmoid)
        teacher_logits: [B, C, *spatial] raw teacher logits (before sigmoid)
        temperature: softness parameter (higher = softer distributions)

    Returns:
        Scalar loss, scaled by T^2 to keep gradient magnitudes consistent.
    """
    student_logits = student_logits.to(torch.float32)
    teacher_logits = teacher_logits.to(torch.float32).detach()

    B, C = student_logits.shape[:2]

    # Flatten spatial dimensions: [B, C, N] where N = product of spatial dims
    student_flat = student_logits.view(B, C, -1) / temperature
    teacher_flat = teacher_logits.view(B, C, -1) / temperature

    # Softmax over spatial dimension (dim=-1) -> probability over locations
    teacher_probs = F.softmax(teacher_flat, dim=-1)
    log_student_probs = F.log_softmax(student_flat, dim=-1)

    # KL divergence per (batch, channel), then average
    # kl_div expects input=log_probs, target=probs
    loss = F.kl_div(log_student_probs, teacher_probs, reduction='batchmean')
    # 'batchmean' divides by batch size. Also average over channels.
    loss = loss / C

    # Scale by T^2 to keep gradients on the same order of magnitude as T=1
    return loss * (temperature ** 2)


def _build_reduced_arch_kwargs(arch_init_kwargs, feature_reduction_factor):
    """Apply feature reduction to architecture kwargs."""
    arch_init_kwargs = deepcopy(arch_init_kwargs)
    if "features_per_stage" in arch_init_kwargs:
        original = arch_init_kwargs["features_per_stage"]
        reduced = [max(f // feature_reduction_factor, 8) for f in original]
        arch_init_kwargs["features_per_stage"] = reduced
    return arch_init_kwargs


class nnLandmarkTrainerKD(nnLandmark):
    """
    Knowledge distillation trainer for nnLandmark, adapted from Fast-nnUNet.

    Key differences from the naive MSE-based KD:
      1. Spatial KL divergence with temperature scaling (instead of raw MSE)
         - transfers the full spatial distribution, not just point values
      2. Lower alpha (0.3) - keeps more weight on ground truth
      3. Temperature = 3.0 with T^2 gradient scaling
      4. Multi-teacher ensemble support (average predictions from multiple folds)
      5. KD loss applied only on highest-resolution output (for deep supervision)
      6. Systematic feature reduction for the student network
    """

    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device=device)

        # --- KD hyperparameters (adapted from Fast-nnUNet) ---
        self.alpha = 0.2          # distillation weight (Fast-nnUNet default)
        self.temperature = 3.0    # temperature for soft distributions
        self.feature_reduction_factor = 2  # halve feature channels -> ~75% param reduction

        # Teacher configuration
        self.teacher_models = []
        # Set this to the folder containing fold_X/ subdirectories of the
        # trained teacher model. Example:
        #   /path/to/nnUNet_results/Dataset100_verse2019/nnLandmark__nnUNetPlans__3d_lowres_save
        self.teacher_model_folder = os.environ.get("TEACHER_MODEL_FOLDER")
        if self.teacher_model_folder is None:
            # Fallback based on the student's output folder
            # example self.output_folder: /path/Dataset100_verse2019/nnLandmarkTrainerKD__nnUNetPlans__3d_fullres/fold_0
            try:
                base = os.path.dirname(self.output_folder)
                base = base.replace(self.__class__.__name__, "nnLandmark")
                self.teacher_model_folder = base
            except (AttributeError, ValueError):
                pass
        # Which fold(s) to load as teacher(s). Use a list for multi-teacher ensemble.
        self.teacher_folds = [0]
        self.teacher_checkpoint_name = "checkpoint_final.pth"

    # ------------------------------------------------------------------
    # Override build_network_architecture to build a reduced student network
    # during inference when predicting from raw data.
    # ------------------------------------------------------------------
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        # Assuming feature_reduction_factor = 2 for KD student model
        feature_reduction_factor = 2
        reduced_kwargs = _build_reduced_arch_kwargs(
            arch_init_kwargs,
            feature_reduction_factor,
        )
        
        return nnLandmark.build_network_architecture(
            architecture_class_name,
            reduced_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision
        )

    # ------------------------------------------------------------------
    # Override initialize() to build a reduced-feature student network
    # ------------------------------------------------------------------
    def initialize(self):
        if not self.was_initialized:
            from nnlandmark.training.dataloading.nnunet_dataset import infer_dataset_class
            from nnlandmark.utilities.label_handling.label_handling import determine_num_input_channels

            self._set_batch_size_and_oversample()

            self.num_input_channels = determine_num_input_channels(
                self.plans_manager, self.configuration_manager, self.dataset_json
            )

            # Build REDUCED student network (feature_reduction_factor applied)
            reduced_kwargs = _build_reduced_arch_kwargs(
                self.configuration_manager.network_arch_init_kwargs,
                self.feature_reduction_factor,
            )

            original_features = self.configuration_manager.network_arch_init_kwargs.get(
                "features_per_stage", "N/A"
            )
            reduced_features = reduced_kwargs.get("features_per_stage", "N/A")
            self.print_to_log_file(
                f"Building STUDENT network with feature reduction factor "
                f"{self.feature_reduction_factor}:\n"
                f"  Original features: {original_features}\n"
                f"  Reduced features:  {reduced_features}"
            )

            self.network = self.build_network_architecture(
                self.configuration_manager.network_arch_class_name,
                self.configuration_manager.network_arch_init_kwargs,
                self.configuration_manager.network_arch_init_kwargs_req_import,
                self.num_input_channels,
                self.label_manager.num_segmentation_heads,
                self.enable_deep_supervision,
            ).to(self.device)

            student_params = sum(p.numel() for p in self.network.parameters())
            self.print_to_log_file(f"  Student parameters: {student_params:,}")

            if self._do_i_compile():
                self.print_to_log_file('Using torch.compile...')
                self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()

            if self.is_ddp:
                from torch.nn.parallel import DistributedDataParallel as DDP
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])

            self.loss = self._build_loss()
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)
            self.was_initialized = True
        else:
            raise RuntimeError(
                "You have called self.initialize even though the trainer was "
                "already initialized. That should not happen."
            )

    # ------------------------------------------------------------------
    # Teacher loading
    # ------------------------------------------------------------------
    def on_train_start(self):
        super().on_train_start()

        if self.teacher_model_folder is None:
            raise ValueError(
                "teacher_model_folder must be set before training. "
                "Point it to the folder containing fold_X/ subdirectories "
                "of your trained teacher model."
            )

        self.print_to_log_file(
            f"Initializing Teacher Model(s) for Knowledge Distillation...\n"
            f"  folder: {self.teacher_model_folder}\n"
            f"  folds:  {self.teacher_folds}\n"
            f"  alpha={self.alpha}, temperature={self.temperature}, "
            f"feature_reduction_factor={self.feature_reduction_factor}"
        )

        self.teacher_models = []
        for fold_idx in self.teacher_folds:
            checkpoint_path = os.path.join(
                self.teacher_model_folder,
                f"fold_{fold_idx}",
                self.teacher_checkpoint_name,
            )
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(
                    f"Teacher checkpoint not found: {checkpoint_path}"
                )

            teacher_checkpoint = torch.load(
                checkpoint_path, map_location=self.device, weights_only=False
            )

            # Reconstruct teacher architecture from its saved plans (FULL size)
            teacher_plans_manager = PlansManager(
                teacher_checkpoint['init_args']['plans']
            )
            teacher_config_manager = teacher_plans_manager.get_configuration(
                self.configuration_name
            )
            teacher_label_manager = teacher_plans_manager.get_label_manager(
                self.dataset_json
            )

            teacher_model = nnLandmark.build_network_architecture(
                teacher_config_manager.network_arch_class_name,
                teacher_config_manager.network_arch_init_kwargs,
                teacher_config_manager.network_arch_init_kwargs_req_import,
                self.num_input_channels,
                teacher_label_manager.num_segmentation_heads,
                self.enable_deep_supervision,
            ).to(self.device)

            teacher_model.load_state_dict(teacher_checkpoint['network_weights'])
            teacher_model.eval()
            teacher_model.float()
            for param in teacher_model.parameters():
                param.requires_grad = False

            self.teacher_models.append(teacher_model)
            teacher_params = sum(p.numel() for p in teacher_model.parameters())
            self.print_to_log_file(
                f"  Loaded teacher fold {fold_idx} ({teacher_params:,} params)"
            )

        self.print_to_log_file(
            f"Teacher ensemble ready: {len(self.teacher_models)} model(s)."
        )

    # ------------------------------------------------------------------
    # Training step with spatial KL distillation
    # ------------------------------------------------------------------
    def train_step(self, batch: dict) -> dict:
        data = batch['data'].to(self.device, non_blocking=True)
        target_structure = [
            i.to(self.device, non_blocking=True) for i in batch['target_struct']
        ]

        self.optimizer.zero_grad(set_to_none=True)

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            # --- Teacher forward (no gradients, fp32 for stability) ---
            all_teacher_logits = []
            with torch.no_grad():
                with autocast(self.device.type, enabled=False):
                    for teacher_model in self.teacher_models:
                        teacher_output = teacher_model(data.float())
                        # If deep supervision, take only the highest resolution
                        if isinstance(teacher_output, (list, tuple)):
                            teacher_output = teacher_output[0]
                        all_teacher_logits.append(teacher_output)

                # Ensemble: average teacher predictions
                if len(all_teacher_logits) > 1:
                    teacher_logits = torch.mean(
                        torch.stack(all_teacher_logits), dim=0
                    )
                else:
                    teacher_logits = all_teacher_logits[0]

            # --- Student forward ---
            student_outputs = self.network(data)

        # --- Ground truth loss (outside autocast for numerical stability) ---
        loss_gt = self.loss(student_outputs, target_structure, batch['bboxes'])

        # --- Spatial KL distillation loss ---
        # Use only highest-resolution output for distillation
        if isinstance(student_outputs, (list, tuple)):
            student_logits_for_distill = student_outputs[0]
        else:
            student_logits_for_distill = student_outputs

        loss_kd = spatial_kl_distillation_loss(
            student_logits_for_distill, teacher_logits, self.temperature
        )

        # --- Combined loss ---
        total_loss = (1 - self.alpha) * loss_gt + self.alpha * loss_kd

        # --- Backward pass ---
        if self.grad_scaler is not None:
            self.grad_scaler.scale(total_loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {'loss': total_loss.detach().cpu().numpy()}
