
from .model import UnSwagModel
from .trainer import UnSwagTrainer
from .surgery import apply_unswag_surgery
from .kv import UnSwagKV
from .data import StreamingContextDataLoader

# Optional Import: KV Cache (Handle missing dependency gracefully)
try:
    from .kv import UnSwagKV
except ImportError:
    UnSwagKV = None

__all__ = [
    "UnSwagModel",
    "UnSwagTrainer",
    "apply_unswag_surgery",
    "UnSwagKV",
    "StreamingContextDataLoader",
]

__version__ = "0.2.0"

# UnSwag Attention (Protocol C)
from .unswag_attention import (
    UnSwagAttentionLayer,
    UnSwagAttentionConfig,
    SemanticRouter,
    LocalTetherCNN,
    AdaptiveSummaryRegister,
    SparseGlobalAttention
)
