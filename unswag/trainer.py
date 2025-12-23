import torch
import os
import json
from transformers import Trainer
from typing import Optional, Dict

class UnSwagTrainer(Trainer):
    """
    Custom Trainer optimized for UnSwag memory architecture.
    Enforces 8-bit AdamW and manages Gradient Checkpointing hooks.
    """
    
    def __init__(self, *args, **kwargs):
        # Force 8-bit AdamW to save VRAM
        if "args" in kwargs and kwargs["args"].optim == "adamw_hf":
            print("🦁 UnSwagTrainer: Upgrading to 'paged_adamw_8bit'")
            kwargs["args"].optim = "paged_adamw_8bit"
        
        super().__init__(*args, **kwargs)
        self.unswag_enabled = True

    def _create_optimizer(self):
        if self.args.optim in ["paged_adamw_8bit", "adamw_bnb_8bit"]:
            try:
                import bitsandbytes
            except ImportError:
                raise ImportError("Please install bitsandbytes for 8-bit optimization.")
        return super()._create_optimizer()

    def training_step(self, model, inputs):
        """Override to ensure Gradient Checkpointing is active."""
        model.train()
        inputs = self._prepare_inputs(inputs)

        if self.args.gradient_checkpointing and not model.is_gradient_checkpointing:
            model.gradient_checkpointing_enable()

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return loss.detach()

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        if output_dir is None: output_dir = self.args.output_dir
        super().save_model(output_dir, _internal_call)
        
        # Save UnSwag Metadata
        unswag_config = {
            "unswag_version": "0.2.0",
            "protocol": "delhi-lux"
        }
        
        if self.args.should_save:
            with open(os.path.join(output_dir, "unswag_config.json"), "w") as f:
                json.dump(unswag_config, f, indent=4)

    def log(self, logs: Dict[str, float]) -> None:
        if "epoch" in logs:
            mem = torch.cuda.max_memory_allocated() / 1e9
            logs["vram_gb"] = round(mem, 2)
        super().log(logs)
