import torch
import torch.nn.functional as F
import os
from torch import autocast

from nnlandmark.utilities.helpers import dummy_context
from nnlandmark.training.nnUNetTrainer.project_specific.nnLandmark.nnLandmark_trainer import nnLandmark
from nnlandmark.utilities.plans_handling.plans_handler import PlansManager

class nnLandmarkTrainerKD(nnLandmark):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        
        # Hyperparameters 
        self.alpha = 0.5  # 50% Ground Truth Loss / 50% Teacher Distillation Loss
        self.teacher_model = None
        self.teacher_checkpoint_path = "/home/ucl/elen/agiansan/nnLandmark_workspace/nnUNet_results/Dataset100_verse2019/nnLandmark__nnUNetPlans__3d_lowres_save/fold_0/checkpoint_final.pth"

    def on_train_start(self):
        super().on_train_start()
        
        if not os.path.exists(self.teacher_checkpoint_path):
            raise FileNotFoundError(f"Teacher checkpoint not found at: {self.teacher_checkpoint_path}")
            
        self.print_to_log_file("Initializing Teacher Model for Regression Knowledge Distillation...")
        
        # Load Teacher Checkpoint
        teacher_checkpoint = torch.load(self.teacher_checkpoint_path, map_location=self.device, weights_only=False)
        
        # Reconstruct Teacher Architecture
        teacher_plans_manager = PlansManager(teacher_checkpoint['init_args']['plans'])
        teacher_config_manager = teacher_plans_manager.get_configuration(self.configuration_name)
        
        teacher_label_manager = teacher_plans_manager.get_label_manager(self.dataset_json)
        
        self.teacher_model = self.build_network_architecture(
            teacher_config_manager.network_arch_class_name,
            teacher_config_manager.network_arch_init_kwargs,
            teacher_config_manager.network_arch_init_kwargs_req_import,
            self.num_input_channels,
            teacher_label_manager.num_segmentation_heads,
            self.enable_deep_supervision
        ).to(self.device)
        
        # Load weights and freeze
        self.teacher_model.load_state_dict(teacher_checkpoint['network_weights'])
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
            
        self.print_to_log_file("Teacher Model successfully loaded and frozen.")

    def train_step(self, batch: dict) -> dict:
        data = batch['data'].to(self.device, non_blocking=True)
        # target_structure and bboxes are used by nnLandmark's custom loss
        target_structure = [i.to(self.device, non_blocking=True) for i in batch['target_struct']]

        self.optimizer.zero_grad(set_to_none=True)

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            # --- Teacher Forward Pass ---
            with torch.no_grad():
                teacher_outputs = self.teacher_model(data)

            # --- Student Forward Pass ---
            student_outputs = self.network(data)

        # compute loss outside autocast since Sigmoid/MSE might not be stable in fp16
        loss_gt = self.loss(student_outputs, target_structure, batch['bboxes'])

        # --- Knowledge Distillation Loss ---
        loss_kd = 0.0
        if self.enable_deep_supervision:
            for s_out, t_out in zip(student_outputs, teacher_outputs):
                loss_kd += F.mse_loss(s_out, t_out)
            loss_kd = loss_kd / len(student_outputs)
        else:
            loss_kd = F.mse_loss(student_outputs, teacher_outputs)

        # --- Combine Losses ---
        total_loss = (1 - self.alpha) * loss_gt + self.alpha * loss_kd

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
