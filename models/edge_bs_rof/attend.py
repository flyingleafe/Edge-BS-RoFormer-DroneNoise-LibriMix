from functools import wraps  # Import wraps decorator to preserve decorated function's original metadata (e.g., function name, docstring)
from packaging import version  # Import version module from packaging library for version comparison operations
from collections import namedtuple  # Import namedtuple from collections module to create named tuples for storing configuration data

import os  # Import OS interface module for system information (e.g., detecting OS type)
import torch  # Import PyTorch library for tensor computation and deep learning operations
from torch import nn, einsum  # Import neural network module nn and Einstein summation function einsum for concise tensor operations
import torch.nn.functional as F  # Import torch.nn.functional module providing various functional neural network operations (e.g., softmax, convolution)

from einops import rearrange, reduce  # Import rearrange and reduce functions from einops library for flexible tensor shape manipulation and reduction

# Constants definition section

# Define FlashAttentionConfig named tuple containing three boolean configuration parameters:
# enable_flash: Whether to enable Flash Attention (accelerated attention computation)
# enable_math: Whether to use mathematical computation for attention
# enable_mem_efficient: Whether to enable memory optimization strategy
FlashAttentionConfig = namedtuple('FlashAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])

# Helper function definitions

def exists(val):
    # Check if the input variable exists (i.e., is not None)
    return val is not None

def default(v, d):
    # If variable v exists, return v; otherwise return default value d
    return v if exists(v) else d

def once(fn):
    # Decorator: ensures the decorated function fn is called only once, avoiding repeated execution (e.g., preventing multiple print messages)
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            # If function has already been called, return directly without execution
            return
        called = True  # Mark that the function has been called
        return fn(x)   # Call the original function fn
    return inner

# Create a print function that only outputs once, to prevent duplicate messages
print_once = once(print)

# Core attention mechanism class implementation
class Attend(nn.Module):
    """
    The Attend class implements the core attention mechanism functionality, supporting two computation methods:
      1. Standard dot-product attention
      2. Flash Attention (optimized attention computation), which can significantly reduce memory consumption and speed up computation,
         but requires PyTorch 2.0 or higher.

    Parameters:
      dropout - The dropout ratio used in attention computation, for randomly dropping some attention weights to prevent overfitting
      flash   - Boolean flag indicating whether to enable Flash Attention optimization mode
      scale   - Scaling factor for attention scores; if not specified, defaults to 1/sqrt(d) (where d is the feature dimension)
    """
    def __init__(
        self,
        dropout = 0.,      # Dropout ratio in attention mechanism
        flash = False,     # Whether to enable Flash Attention optimized computation mode
        scale = None       # Scaling factor for attention scores; if None, uses default 1/sqrt(d) in computation
    ):
        super().__init__()  # Initialize parent class nn.Module
        self.scale = scale  # Store scaling factor
        self.dropout = dropout  # Store Dropout ratio
        self.attn_dropout = nn.Dropout(dropout)  # Create Dropout layer based on given dropout ratio for random dropping of attention weights

        self.flash = flash  # Store whether Flash Attention is enabled
        # If Flash Attention is enabled, require PyTorch version at least 2.0, otherwise trigger assertion error
        assert not (flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        # Set default CPU attention configuration, all parameters set to True
        self.cpu_config = FlashAttentionConfig(True, True, True)
        # Initialize CUDA configuration to None, will be set later based on CUDA device properties
        self.cuda_config = None

        # If no CUDA device available or Flash Attention not enabled, no need to set CUDA configuration, return directly
        if not torch.cuda.is_available() or not flash:
            return

        # Get current CUDA device properties to determine GPU compute capability
        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))
        # Construct GPU device version number including major and minor for comparing compute capability (e.g., 8.0)
        device_version = version.parse(f'{device_properties.major}.{device_properties.minor}')

        if device_version >= version.parse('8.0'):
            if os.name == 'nt':
                # If running on Windows OS, even with GPU compute capability >= 8.0, use math or memory efficient attention (not enabling Flash Attention)
                print_once('Windows OS detected, using math or mem efficient attention if input tensor is on cuda')
                self.cuda_config = FlashAttentionConfig(False, True, True)
            else:
                # Non-Windows system with GPU compute capability >= 8.0, enable Flash Attention to fully utilize hardware acceleration
                print_once('GPU Compute Capability equal or above 8.0, using flash attention if input tensor is on cuda')
                self.cuda_config = FlashAttentionConfig(True, False, False)
        else:
            # If GPU compute capability below 8.0, use math or memory efficient attention to avoid potential Flash Attention incompatibility
            print_once('GPU Compute Capability below 8.0, using math or mem efficient attention if input tensor is on cuda')
            self.cuda_config = FlashAttentionConfig(False, True, True)

    def flash_attn(self, q, k, v):
        """Implements optimized Flash Attention computation method"""
        # Unpack query vector q's shape and get number of attention heads, query sequence length, etc.
        # Also get key sequence length from key vector k, and detect if tensor is on CUDA and its device
        _, heads, q_len, _, k_len, is_cuda, device = *q.shape, k.shape[-2], q.is_cuda, q.device

        # If user specified custom scaling factor, adjust query vector q accordingly
        if exists(self.scale):
            default_scale = q.shape[-1] ** -0.5  # Default scaling factor, typically 1/sqrt(feature_dimension)
            q = q * (self.scale / default_scale)  # Adjust q by custom scale ratio

        # Use PyTorch's scaled_dot_product_attention with automatic kernel selection
        # This allows PyTorch to choose the best available kernel (flash, math, or memory efficient)
        # based on input tensor properties and hardware capabilities
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.  # Apply dropout during training, disable during inference
        )

        return out

    def forward(self, q, k, v):
        """
        Forward propagation function, computes attention output.
        Uses Einstein summation convention where dimension meanings are:
          b - batch size
          h - number of heads
          n, i, j - sequence-related dimensions (e.g., query and key sequence lengths)
          d - feature dimension
        Parameters:
          q - query vector
          k - key vector
          v - value vector
        """
        # Get query and key sequence lengths, and determine device of query
        q_len, k_len, device = q.shape[-2], k.shape[-2], q.device

        # If custom scaling factor not provided, use default value 1/sqrt(feature_dimension)
        scale = default(self.scale, q.shape[-1] ** -0.5)

        # When Flash Attention mode is enabled, call flash_attn function to get attention output
        if self.flash:
            return self.flash_attn(q, k, v)

        # Compute standard dot-product attention similarity scores:
        # Use Einstein summation to compute inner product of query q and key k vectors, multiply by scaling factor
        sim = einsum(f"b h i d, b h j d -> b h i j", q, k) * scale

        # Apply softmax normalization to similarity scores to get attention weights
        attn = sim.softmax(dim=-1)
        # Use Dropout layer to randomly drop attention weights to prevent overfitting
        attn = self.attn_dropout(attn)

        # Use Einstein summation to compute weighted sum of value vector v using normalized attention weights
        out = einsum(f"b h i j, b h j d -> b h i d", attn, v)

        return out
