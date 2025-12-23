import torch
import warnings
from transformers import AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from .surgery import apply_unswag_surgery

class UnSwagModel:
    """
    The UnSwag 'Steering Wheel'.
    Unified interface for loading, patching, and preparing models.
    """
    
    SUPPORTED_ARCHITECTURES = {
        "llama": ["LlamaForCausalLM", "TinyLlamaForCausalLM"],
        "mistral": ["MistralForCausalLM"],
    }

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        max_seq_length: int = 4096,
        load_in_4bit: bool = True,
        mode: str = "4bit",  # "2bit" (Delhi) or "4bit" (Delhi-Lux)
        device_map: str = "auto",
        use_gradient_checkpointing: bool = True,
        **kwargs
    ):
        # 1. Config Check
        config = AutoConfig.from_pretrained(model_name)
        architecture = config.architectures[0] if config.architectures else "Unknown"
        print(f"--- 🦁 UnSwagAI: Loading {model_name} ({architecture}) ---")

        # 2. Update Context Length
        if hasattr(config, "max_position_embeddings"):
            print(f"ℹ️ Scaling context from {config.max_position_embeddings} to {max_seq_length}")
            config.max_position_embeddings = max_seq_length
            
        # 3. Quantization (BNB NF4)
        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )

        # 4. Load Base Model (Force SDPA)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            quantization_config=quantization_config,
            device_map=device_map,
            attn_implementation="sdpa", 
            **kwargs
        )

        # 5. Apply UnSwag Surgery
        print(f"⚔️ Applying UnSwag Surgery (Mode: {mode})...")
        apply_unswag_surgery(model, mode=mode)
        
        # 6. Gradient Checkpointing
        if use_gradient_checkpointing:
            model.gradient_checkpointing_enable()
            print("☢️ Gradient Checkpointing Enabled.")

        return model, None

    @staticmethod
    def for_training(
        model,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: list = None,
        use_rslora: bool = False,
    ):
        print("--- 🔧 Preparing for UnSwag Training ---")
        model = prepare_model_for_kbit_training(model)
        
        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            
        peft_config = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, target_modules=target_modules,
            lora_dropout=lora_dropout, bias="none", task_type="CAUSAL_LM", use_rslora=use_rslora
        )
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        return model

    @staticmethod
    def for_inference(model):
        print("--- ⚡ Optimizing for Inference ---")
        model.eval()
        return model
