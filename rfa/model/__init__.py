from rfa import RFAConfig, MultiheadIsotropicRFA, ComplexRMSNorm, convert_legacy_state_dict, load_legacy_state_dict
from .config import AttentionConfig, BackboneConfig
from .layers import MultiHeadAttentionLayer
from .blocks import Block, build_attention
from .networks import Backbone, LanguageModel
from .pos_encoding import RoPE, SCRoPE, get_alibi_slopes
from .model_utils import resolve_multihead_dims, autoregressive_sample, get_model_sde_snapshots, print_layer_diagnostics

# Constructors taking the notebook-style `args` namespace (paper notebooks)
from .compat import (TransformerBlock, RFATransformerBlock, TransformerNetwork, RFATransformerNetwork,
                     SelfAttentionBlock, RFA_Block, build_rfa_layer)
