from .core import UnSwagActivations
from .layers import unswag_relu
from .ui import boot_sequence

# Run the boot sequence on import
# We check if it's the main process to avoid spamming in multi-host setups
if __name__ != "__main__":
    try:
        # Simple check to avoid running during build/install
        import os
        if "SETUP_PY" not in os.environ:
            boot_sequence()
    except Exception:
        pass

__all__ = ["UnSwagActivations", "unswag_relu", "boot_sequence"]
