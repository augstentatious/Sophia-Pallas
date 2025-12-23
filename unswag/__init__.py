# unswag/__init__.py

from .core import UnSwagActivations
# FIX: Use a comma, not a second 'import'
#from .layers import unswag_relu, UnSwagLinear
from .ui import boot_sequence
from .patcher import unswag_model

# Run the boot sequence on import
# We check if it's the main process to avoid spamming in multi-host setups
if __name__ != "__main__":
    try:
        import os
        # Simple check to avoid running during build/install or CI
        if "SETUP_PY" not in os.environ and "CI" not in os.environ:
            boot_sequence()
    except Exception:
        pass

# Export the public API
__all__ = [
    "UnSwagActivations", 
    "unswag_relu", 
    "UnSwagLinear",   # <--- Added this
    "unswag_model",   # <--- Added this
    "boot_sequence"
]
