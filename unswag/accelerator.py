import os
import sys
import jax
import logging

def initialize_tpu():
    """
    Modern Handshake for JAX on Colab/Kaggle.
    Automatically handles TPU detection without legacy setup calls.
    """
    try:
        # 1. KAGGLE: Still needs a manual nudge to see the TPU
        if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
            print("⚡ UnSwag: Configuring for Kaggle TPU...")
            jax.config.update("jax_platforms", "tpu")

        # 2. COLAB: Auto-detects in JAX 0.4.x+ (No setup_tpu() needed)
        elif 'COLAB_RELEASE_TAG' in os.environ:
            print("⚡ UnSwag: Colab Environment Detected.")
        
        # 3. VERIFY HARDWARE
        # This is the moment of truth
        devices = jax.devices()
        dev_type = devices[0].device_kind.upper()
        
        print(f"✅ Hardware: {len(devices)} devices found.")
        print(f"   -> Type: {dev_type}")
        
        if "TPU" not in dev_type:
            logging.warning("⚠️ CRITICAL: TPU not found! Running on CPU/GPU.")
            
    except Exception as e:
        print(f"❌ Handshake Error: {e}")
        # Don't crash, just warn, so the user can debug
        print("   -> Tip: Ensure your Runtime is set to TPU v2-8 (Colab) or TPU VM v3-8 (Kaggle).")

if __name__ == "__main__":
    initialize_tpu()
