from functools import partial

import math  # Added for PositionalEncoding
import torch
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from models.edge_bs_rof.attend import Attend
from torch.utils.checkpoint import checkpoint

from beartype.typing import Tuple, Optional, List, Callable
from beartype import beartype

from rotary_embedding_torch import RotaryEmbedding

from einops import rearrange, pack, unpack
from einops.layers.torch import Rearrange

# Helper functions

def exists(val):
    """
    Check if input value exists (is not None)
    Args:
        val: Any input value
    Returns:
        bool: True if val is not None, otherwise False
    """
    return val is not None


def default(v, d):
    """
    Return default value, returns v if v exists, otherwise returns default value d
    Args:
        v: Primary value
        d: Default value
    Returns:
        v if v exists, otherwise d
    """
    return v if exists(v) else d


def pack_one(t, pattern):
    """
    Pack a single tensor according to specified pattern
    Args:
        t: Input tensor
        pattern: Packing pattern string
    Returns:
        Packed tensor
    """
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    """
    Unpack a packed tensor and return the first element
    Args:
        t: Packed tensor
        ps: Original shape information
        pattern: Unpacking pattern string
    Returns:
        First unpacked tensor
    """
    return unpack(t, ps, pattern)[0]


# Normalization layers

def l2norm(t):
    """
    Apply L2 normalization to input tensor
    Args:
        t: Input tensor
    Returns:
        Normalized tensor
    """
    return F.normalize(t, dim = -1, p = 2)


class RMSNorm(Module):
    """
    RMS (Root Mean Square) normalization layer
    Simpler computation than LayerNorm with better performance

    Args:
        dim: Dimension size for normalization
    """
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.gamma


class PositionalEncoding(Module):
    """
    Absolute positional encoding

    Args:
        dim: Encoding dimension
        max_seq_len: Maximum sequence length
    """
    def __init__(self, dim, max_seq_len=1000):
        super().__init__()
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_seq_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [batch_size, seq_len, dim]
        """
        return x + self.pe[:x.size(1), :].unsqueeze(0)


# Attention mechanism related modules

class FeedForward(Module):
    """
    Feed-forward neural network
    Contains two linear layers and GELU activation function for feature transformation

    Args:
        dim: Input dimension
        mult: Hidden layer dimension expansion multiplier
        dropout: Dropout ratio
    """
    def __init__(
            self,
            dim,
            mult=4,
            dropout=0.
    ):
        super().__init__()
        dim_inner = int(dim * mult)
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_inner, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(Module):
    """
    Multi-head attention mechanism
    Supports rotary positional encoding and Flash Attention optimization

    Args:
        dim: Input dimension
        heads: Number of attention heads
        dim_head: Dimension of each attention head
        dropout: Dropout ratio
        rotary_embed: Rotary positional encoding instance
        flash: Whether to use Flash Attention optimization
        use_rotary_pos: Whether to use rotary positional encoding
    """
    def __init__(
            self,
            dim,
            heads=8,
            dim_head=64,
            dropout=0.,
            rotary_embed=None,
            flash=True,
            use_rotary_pos=False
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        dim_inner = heads * dim_head

        self.rotary_embed = rotary_embed  # Rotary positional encoding
        self.use_rotary_pos = use_rotary_pos  # Whether to use rotary positional encoding

        self.attend = Attend(flash=flash, dropout=dropout)  # Core attention computation module

        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias=False)  # Linear layer for generating Query(Q), Key(K), Value(V)

        self.to_gates = nn.Linear(dim, heads)  # Linear layer for generating attention gate weights

        self.to_out = nn.Sequential(
            nn.Linear(dim_inner, dim, bias=False),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.norm(x)

        # Generate Q,K,V and rearrange dimensions
        q, k, v = rearrange(self.to_qkv(x), 'b n (qkv h d) -> qkv b h n d', qkv=3, h=self.heads)

        # Apply rotary positional encoding
        if self.use_rotary_pos and exists(self.rotary_embed):
            q = self.rotary_embed.rotate_queries_or_keys(q)
            k = self.rotary_embed.rotate_queries_or_keys(k)

        # Compute attention
        out = self.attend(q, k, v)

        # Apply gating mechanism
        gates = self.to_gates(x)
        out = out * rearrange(gates, 'b n h -> b h n 1').sigmoid()

        # Output projection
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class LinearAttention(Module):
    """
    Linear attention mechanism
    Reduces traditional attention computation complexity from O(n^2) to O(n)
    Based on paper https://arxiv.org/abs/2106.09681

    Args:
        dim: Input dimension
        dim_head: Dimension of each attention head
        heads: Number of attention heads
        scale: Scaling factor
        flash: Whether to use Flash Attention optimization
        dropout: Dropout ratio
    """

    @beartype
    def __init__(
            self,
            *,
            dim,
            dim_head=32,
            heads=8,
            scale=8,
            flash=False,
            dropout=0.
    ):
        super().__init__()
        dim_inner = dim_head * heads  # Calculate inner dimension
        self.norm = RMSNorm(dim)      # Layer normalization

        # Linear transformation and dimension rearrangement for generating Q,K,V
        self.to_qkv = nn.Sequential(
            nn.Linear(dim, dim_inner * 3, bias=False),  # Linear projection to Q,K,V space
            Rearrange('b n (qkv h d) -> qkv b h d n', qkv=3, h=heads)  # Rearrange dimensions for multi-head attention
        )

        # Learnable temperature parameter for scaling attention scores
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))

        # Attention computation module
        self.attend = Attend(
            scale=scale,
            dropout=dropout,
            flash=flash
        )

        # Output projection layer
        self.to_out = nn.Sequential(
            Rearrange('b h d n -> b n (h d)'),  # Rearrange dimensions
            nn.Linear(dim_inner, dim, bias=False)  # Linear projection back to original dimension
        )

    def forward(
            self,
            x
    ):
        x = self.norm(x)  # Input normalization

        # Generate Q,K,V vectors
        q, k, v = self.to_qkv(x)

        # Apply L2 normalization to Q,K for enhanced stability
        q, k = map(l2norm, (q, k))
        # Apply temperature scaling
        q = q * self.temperature.exp()

        # Compute attention
        out = self.attend(q, k, v)

        # Output projection
        return self.to_out(out)


class Transformer(Module):
    """
    Transformer encoder, composed of multiple attention layers and feed-forward networks

    Args:
        dim: Input dimension
        depth: Number of layers
        dim_head: Attention head dimension
        heads: Number of attention heads
        attn_dropout: Attention dropout rate
        ff_dropout: Feed-forward network dropout rate
        ff_mult: Feed-forward network hidden layer dimension multiplier
        norm_output: Whether to normalize output
        rotary_embed: Rotary positional encoding
        flash_attn: Whether to use Flash Attention
        linear_attn: Whether to use linear attention
        use_rotary_pos: Whether to use rotary positional encoding (RoPE)
        max_seq_len: Maximum sequence length
    """
    def __init__(
            self,
            *,
            dim,
            depth,
            dim_head=64,
            heads=8,
            attn_dropout=0.,
            ff_dropout=0.,
            ff_mult=4,
            norm_output=True,
            rotary_embed=None,
            flash_attn=True,
            linear_attn=False,
            use_rotary_pos=False,  # Whether to use rotary positional encoding
            max_seq_len=1000       # Maximum sequence length
    ):
        super().__init__()
        self.layers = ModuleList([])  # Store all layers
        self.use_rotary_pos = use_rotary_pos
        self.max_seq_len = max_seq_len

        # Initialize absolute positional encoding
        if not self.use_rotary_pos:
            self.positional_encoding = PositionalEncoding(dim=dim, max_seq_len=max_seq_len)
        else:
            self.positional_encoding = None

        # Build multi-layer structure
        for _ in range(depth):
            if linear_attn:
                # Use linear attention layer
                attn = LinearAttention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout, flash=flash_attn)
            else:
                # Use standard attention layer
                attn = Attention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout,
                                 rotary_embed=rotary_embed, flash=flash_attn, use_rotary_pos=use_rotary_pos)
            # Each layer contains attention module and feed-forward network
            self.layers.append(ModuleList([
                attn,
                FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
            ]))

        # Output normalization layer
        self.norm = RMSNorm(dim) if norm_output else nn.Identity()

    def forward(self, x):
        if not self.use_rotary_pos:
            x = self.positional_encoding(x)
        # Pass through each layer sequentially with residual connections
        for attn, ff in self.layers:
            x = attn(x) + x  # Attention layer + residual
            x = ff(x) + x    # Feed-forward network + residual

        return self.norm(x)  # Output normalization


# Band splitting module: This module splits the input feature tensor according to predefined band dimensions,
# performs feature extraction mapping for each band, and stacks all band outputs for subsequent processing.
class BandSplit(Module):
    """
    Split input by frequency bands and map to specified dimension

    Parameters:
        dim: Target dimension of output features for each band after mapping.
        dim_inputs: A tuple where each element represents the dimension occupied by the corresponding band in the input data.
                    For example, if dim_inputs=(a, b, c), it means the last dimension of input tensor is split into a, b, c parts,
                    corresponding to three bands respectively.
    """
    @beartype
    def __init__(self, dim, dim_inputs: Tuple[int, ...]):
        super().__init__()
        self.dim_inputs = dim_inputs  # Store input dimension info for each band
        self.to_features = ModuleList([])  # Store feature extraction network for each band

        # Build separate feature extraction network for each band
        for dim_in in dim_inputs:
            # Build a sequential model:
            # 1. First use RMSNorm to normalize input data, stabilizing numerical distribution;
            # 2. Then use nn.Linear layer to map input from original dimension dim_in to target dimension dim.
            net = nn.Sequential(
                RMSNorm(dim_in),   # Normalize current band input
                nn.Linear(dim_in, dim)  # Linear transformation to map input to target dimension
            )
            # Add the constructed network to ModuleList
            self.to_features.append(net)

    def forward(self, x):
        """
        Forward propagation:
        1. Split input tensor x on the last dimension according to predefined dim_inputs, obtaining multiple band data.
        2. Process each band through its corresponding feature extraction network to extract target features.
        3. Stack all processed band results on the newly added second-to-last dimension,
           so subsequent modules can identify each band's information.
        """
        # Split input tensor along last dimension according to dim_inputs values, each split corresponds to one band
        x = x.split(self.dim_inputs, dim=-1)

        outs = []  # Store processed outputs for each band
        # Iterate through each band's input and corresponding feature extraction network
        for split_input, to_feature in zip(x, self.to_features):
            # Perform feature extraction on single band data
            split_output = to_feature(split_input)
            outs.append(split_output)

        # Stack all band outputs on new dimension (second-to-last) to generate final output
        return torch.stack(outs, dim=-2)


def MLP(dim_in, dim_out, dim_hidden=None, depth=1, activation=nn.Tanh):
    """
    Multi-Layer Perceptron (MLP) module:
    Uses multiple linear layers and activation functions stacked to achieve non-linear mapping,
    for transforming input features to specified output dimension.

    Parameters:
        dim_in: Input feature dimension.
        dim_out: Target output feature dimension.
        dim_hidden: Hidden layer dimension, defaults to dim_in if not specified.
        depth: Network depth, i.e., total number of linear layers (at least 1).
        activation: Activation function, defaults to nn.Tanh.
    """
    # If hidden layer size not specified, use input feature size as default
    dim_hidden = default(dim_hidden, dim_in)

    net = []  # Store constructed layers
    # Generate layer dimension sequence: input layer, depth-1 hidden layers, and output layer
    dims = (dim_in, *((dim_hidden,) * (depth - 1)), dim_out)

    # Build each layer: add linear layers sequentially, add activation function after non-final layers
    for ind, (layer_dim_in, layer_dim_out) in enumerate(zip(dims[:-1], dims[1:])):
        is_last = (ind == (len(dims) - 2))  # Check if current layer is the last

        # Add linear mapping layer: transform input from layer_dim_in to layer_dim_out
        net.append(nn.Linear(layer_dim_in, layer_dim_out))

        # If not the last layer, add activation function to introduce non-linearity
        if not is_last:
            net.append(activation())

    # Wrap all constructed layers into a Sequential module and return
    return nn.Sequential(*net)


class MaskEstimator(Module):
    """
    Mask estimator module:
    This module uses multi-layer perceptron and GLU gating mechanism to process input features,
    estimating the audio separation mask for each frequency band.
    """
    @beartype
    def __init__(self, dim, dim_inputs: Tuple[int, ...], depth, mlp_expansion_factor=4):
        super().__init__()
        self.dim_inputs = dim_inputs  # Store input feature dimension info for each band
        self.to_freqs = ModuleList([])  # Store frequency mapping network for each band
        # Calculate hidden layer dimension: typically base dimension dim times expansion factor
        dim_hidden = dim * mlp_expansion_factor

        # Build mapping network for each band, each network consists of MLP module and GLU activation
        for dim_in in dim_inputs:
            # Construct Sequential network:
            # a. Use MLP to map input from dim to (dim_in * 2) dimension,
            # b. Apply nn.GLU gating operation on last dimension to enhance mapping complexity
            mlp = nn.Sequential(
                MLP(dim, dim_in * 2, dim_hidden=dim_hidden, depth=depth),
                nn.GLU(dim=-1)
            )
            self.to_freqs.append(mlp)

    def forward(self, x):
        """
        Forward propagation:
        1. Split input tensor x along second-to-last dimension into data for each frequency band.
        2. Pass each band through corresponding mapping network to generate separation mask info.
        3. Concatenate all band outputs along last dimension into unified output.
        """
        # Use unbind to split x, each split element represents one band's features
        x = x.unbind(dim=-2)

        outs = []  # Store network mapping results for each band
        # Iterate through each band's data and corresponding network
        for band_features, mlp in zip(x, self.to_freqs):
            # Use current network to estimate frequency mapping output for current band
            freq_out = mlp(band_features)
            outs.append(freq_out)

        # Concatenate all band outputs along last dimension into a whole
        return torch.cat(outs, dim=-1)


# Main model

# Default frequency band partition configuration:
# Defines the number of frequencies contained in each band, providing multi-scale frequency resolution info,
# e.g., the first 24 bands each contain 2 frequencies, subsequent bands have increasing numbers.
DEFAULT_FREQS_PER_BANDS = (
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    12, 12, 12, 12, 12, 12, 12, 12,
    24, 24, 24, 24, 24, 24, 24, 24,
    48, 48, 48, 48, 48, 48, 48, 48,
    128, 129,
)


class BSRoformer(Module):
    """
    Roformer-based audio source separation model
    Uses multi-layer Transformer to model time-frequency representations
    Supports mono/stereo input
    Can separate multiple audio sources
    """

    @beartype
    def __init__(
            self,
            dim,                        # Model's base feature dimension
            *,
            depth,                      # Number of Transformer layers
            stereo=False,               # Whether to process stereo audio (True=stereo, 2 channels; False=mono, 1 channel)
            num_stems=1,                # Number of audio sources to separate (e.g., single or multiple speakers)
            time_transformer_depth=2,   # Depth of Transformer module for processing time info
            freq_transformer_depth=2,   # Depth of Transformer module for processing frequency info
            linear_transformer_depth=0, # Depth of linear attention module (0 means not used)
            freqs_per_bands: Tuple[int, ...] = DEFAULT_FREQS_PER_BANDS,  # Frequency count configuration per band
            dim_head=64,                # Dimension of each head in multi-head attention
            heads=8,                    # Number of Transformer attention heads
            attn_dropout=0.,            # Dropout probability in attention layers
            ff_dropout=0.,              # Dropout probability in feed-forward layers (MLP)
            flash_attn=True,            # Whether to use flash attention to speed up attention computation
            dim_freqs_in=1025,          # Number of frequencies from STFT (typically determined by STFT params)
            stft_n_fft=2048,            # FFT window size for STFT
            stft_hop_length=512,        # STFT hop length (interval between frames)
            stft_win_length=2048,       # STFT window length
            stft_normalized=False,      # Whether to normalize STFT results
            stft_window_fn: Optional[Callable] = None,  # Method for generating STFT window function
            mask_estimator_depth=2,     # Transformer depth in mask estimator
            multi_stft_resolution_loss_weight=1.,  # Weight for multi-resolution STFT loss
            multi_stft_resolutions_window_sizes: Tuple[int, ...] = (4096, 2048, 1024, 512, 256),  # Window size config for multi-resolution
            multi_stft_hop_size=147,    # Hop length for multi-resolution STFT
            multi_stft_normalized=False,  # Whether to normalize multi-resolution STFT
            multi_stft_window_fn: Callable = torch.hann_window,  # Multi-resolution STFT window function, default Hann window
            mlp_expansion_factor=4,     # MLP expansion factor, controls hidden layer width
            use_torch_checkpoint=False, # Whether to use torch checkpoint to reduce intermediate memory usage
            skip_connection=False,      # Whether to use skip connections (residual) between Transformer modules
            use_rotary_pos=False,       # Whether to use rotary positional encoding (RoPE)
            max_seq_len=1000            # Maximum sequence length
    ):
        super().__init__()  # Initialize parent class Module

        # Determine audio channels based on stereo parameter, stereo=2 channels, mono=1 channel
        self.stereo = stereo
        self.audio_channels = 2 if stereo else 1
        self.num_stems = num_stems  # Store number of sources to separate
        self.use_torch_checkpoint = use_torch_checkpoint
        self.skip_connection = skip_connection
        self.use_rotary_pos = use_rotary_pos
        self.max_seq_len = max_seq_len

        # Initialize ModuleList to store multi-layer Transformer modules
        self.layers = ModuleList([])

        # Define common parameters for Transformer, passed to each layer's Transformer module
        transformer_kwargs = dict(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            flash_attn=flash_attn,
            norm_output=False,  # Output not normalized, handled by final_norm
            use_rotary_pos=use_rotary_pos,
            max_seq_len=max_seq_len
        )

        # Initialize rotary positional encoders for time and frequency axes to provide position info to Transformer
        time_rotary_embed = RotaryEmbedding(dim=dim_head) if use_rotary_pos else None
        freq_rotary_embed = RotaryEmbedding(dim=dim_head) if use_rotary_pos else None

        # Construct multiple Transformer layers based on depth, each layer can contain different attention modules
        for _ in range(depth):
            tran_modules = []
            if linear_transformer_depth > 0:
                # If linear Transformer layers are specified, add linear attention module
                tran_modules.append(Transformer(depth=linear_transformer_depth, linear_attn=True, **transformer_kwargs))
            # Add Transformer for time dimension to capture temporal correlations
            tran_modules.append(
                Transformer(depth=time_transformer_depth, rotary_embed=time_rotary_embed, **transformer_kwargs)
            )
            # Add Transformer for frequency dimension to capture frequency correlations
            tran_modules.append(
                Transformer(depth=freq_transformer_depth, rotary_embed=freq_rotary_embed, **transformer_kwargs)
            )
            # Wrap current layer's module list as ModuleList and add to overall layers
            self.layers.append(nn.ModuleList(tran_modules))

        # Normalize all Transformer layer outputs
        self.final_norm = RMSNorm(dim)

        # Configure STFT transform parameters to convert time-domain audio to frequency domain
        self.stft_kwargs = dict(
            n_fft=stft_n_fft,
            hop_length=stft_hop_length,
            win_length=stft_win_length,
            normalized=stft_normalized
        )

        # Set STFT window function generator, use default torch.hann_window if not provided, with fixed window length
        self.stft_window_fn = partial(default(stft_window_fn, torch.hann_window), stft_win_length)

        # Calculate number of frequencies after STFT by applying STFT to random signal
        freqs = torch.stft(torch.randn(1, 4096), **self.stft_kwargs, window=torch.ones(stft_win_length), return_complex=True).shape[1]

        # Ensure band configuration has at least two bands, and total frequencies equals STFT output frequencies
        assert len(freqs_per_bands) > 1
        assert sum(freqs_per_bands) == freqs, f'Band count must equal frequency count {freqs} from STFT settings, but got {sum(freqs_per_bands)}'

        # Process for complex data representation and audio channels, multiply each band's frequency count by 2 (real and imaginary) and audio channels
        freqs_per_bands_with_complex = tuple(2 * f * self.audio_channels for f in freqs_per_bands)

        # Initialize band split module to divide input spectrum data into multiple preset bands
        self.band_split = BandSplit(
            dim=dim,
            dim_inputs=freqs_per_bands_with_complex
        )

        # Build mask estimator for each source, used to estimate each source's mask
        self.mask_estimators = nn.ModuleList([])

        for _ in range(num_stems):
            mask_estimator = MaskEstimator(
                dim=dim,
                dim_inputs=freqs_per_bands_with_complex,
                depth=mask_estimator_depth,
                mlp_expansion_factor=mlp_expansion_factor,
            )
            self.mask_estimators.append(mask_estimator)

        # Set multi-resolution STFT loss parameters to measure reconstruction vs target audio differences at multiple resolutions
        self.multi_stft_resolution_loss_weight = multi_stft_resolution_loss_weight
        self.multi_stft_resolutions_window_sizes = multi_stft_resolutions_window_sizes
        self.multi_stft_n_fft = stft_n_fft
        self.multi_stft_window_fn = multi_stft_window_fn

        # Configure other multi-resolution STFT parameters like hop length and normalization options
        self.multi_stft_kwargs = dict(
            hop_length=multi_stft_hop_size,
            normalized=multi_stft_normalized
        )

    def forward(
            self,
            raw_audio,  # Input raw time-domain audio, shape can be [b, t] or [b, s, t]
            target=None,  # Target audio, used for computing loss during training
            return_loss_breakdown=False  # Whether to return loss breakdown components
    ):
        """
        Forward propagation process

        Dimension notation:
        b - batch size
        f - number of frequencies
        t - number of time frames
        s - audio channels (mono=1, stereo=2)
        n - number of sources (separation targets)
        c - complex representation dimension (2, representing real and imaginary parts)
        d - Transformer internal feature dimension
        """
        device = raw_audio.device  # Get device of input audio (e.g., CPU, GPU)

        # Check if running on MacOS MPS device to handle potential FFT compatibility issues
        x_is_mps = True if device.type == "mps" else False

        # If input audio is 2D, it's missing channel dimension, so expand it (e.g., [b, t] -> [b, 1, t])
        if raw_audio.ndim == 2:
            raw_audio = rearrange(raw_audio, 'b t -> b 1 t')

        channels = raw_audio.shape[1]  # Get number of input audio channels
        # Check if input channels match model configuration: mono should be 1 channel, stereo should be 2 channels
        assert (not self.stereo and channels == 1) or (self.stereo and channels == 2), 'Stereo setting must match input audio channels'

        # Apply STFT transform to convert time-domain audio to frequency domain for processing
        raw_audio, batch_audio_channel_packed_shape = pack_one(raw_audio, '* t')
        stft_window = self.stft_window_fn(device=device)  # Create STFT window function based on device

        # Execute STFT operation, use try/except for MacOS MPS platform compatibility
        try:
            stft_repr = torch.stft(raw_audio, **self.stft_kwargs, window=stft_window, return_complex=True)
        except:
            stft_repr = torch.stft(raw_audio.cpu() if x_is_mps else raw_audio, **self.stft_kwargs,
                                   window=stft_window.cpu() if x_is_mps else stft_window, return_complex=True).to(device)
        # Convert complex STFT output to real tensor (last dimension is real and imaginary parts)
        stft_repr = torch.view_as_real(stft_repr)

        # Restore original packed shape of STFT output
        stft_repr = unpack_one(stft_repr, batch_audio_channel_packed_shape, '* f t c')

        # Merge different audio channels (e.g., stereo's two channels) with frequency axis for combined processing
        stft_repr = rearrange(stft_repr, 'b s f t c -> b (f s) t c')

        # Adjust tensor shape so time axis becomes first dimension, convenient for Transformer processing
        x = rearrange(stft_repr, 'b f t c -> b t (f c)')

        # Split spectrum data into multiple preset bands through band split module
        if self.use_torch_checkpoint:
            x = checkpoint(self.band_split, x, use_reentrant=False)
        else:
            x = self.band_split(x)

        # Enter multi-layer Transformer modules for axial (time/frequency) attention processing
        store = [None] * len(self.layers)  # Store each layer's output (used if skip connection enabled)
        for i, transformer_block in enumerate(self.layers):
            if len(transformer_block) == 3:
                # If current layer has three sub-modules, it contains linear attention module
                linear_transformer, time_transformer, freq_transformer = transformer_block

                # Pack x with shape info for later shape restoration
                x, ft_ps = pack([x], 'b * d')
                if self.use_torch_checkpoint:
                    # Use checkpoint mechanism to save memory
                    x = checkpoint(linear_transformer, x, use_reentrant=False)
                else:
                    x = linear_transformer(x)
                # Unpack to restore original shape
                x, = unpack(x, ft_ps, 'b * d')
            else:
                # Otherwise current layer only contains time and frequency Transformers
                time_transformer, freq_transformer = transformer_block

            if self.skip_connection:
                # If skip connection enabled, accumulate previous layers' outputs to current output
                for j in range(i):
                    x = x + store[j]

            # Apply Transformer on time dimension: swap dimensions so time dimension is in proper position
            x = rearrange(x, 'b t f d -> b f t d')
            x, ps = pack([x], '* t d')
            if self.use_torch_checkpoint:
                x = checkpoint(time_transformer, x, use_reentrant=False)
            else:
                x = time_transformer(x)
            x, = unpack(x, ps, '* t d')

            # Apply Transformer on frequency dimension: rearrange dimensions with frequency as target
            x = rearrange(x, 'b f t d -> b t f d')
            x, ps = pack([x], '* f d')
            if self.use_torch_checkpoint:
                x = checkpoint(freq_transformer, x, use_reentrant=False)
            else:
                x = freq_transformer(x)
            x, = unpack(x, ps, '* f d')

            if self.skip_connection:
                # Save current layer output for subsequent skip connections
                store[i] = x

        # Final normalization on all Transformer layer outputs
        x = self.final_norm(x)

        # Record actual number of sources used, i.e., number of mask estimators
        num_stems = len(self.mask_estimators)

        # Estimate separation mask for each source, using checkpoint (if enabled) to reduce memory usage
        if self.use_torch_checkpoint:
            mask = torch.stack([checkpoint(fn, x, use_reentrant=False) for fn in self.mask_estimators], dim=1)
        else:
            mask = torch.stack([fn(x) for fn in self.mask_estimators], dim=1)
        # Adjust mask tensor shape, decompose last dimension into frequency and complex (real, imaginary) parts
        mask = rearrange(mask, 'b n t (f c) -> b n f t c', c=2)

        # Add source dimension to original STFT representation for subsequent modulation
        stft_repr = rearrange(stft_repr, 'b f t c -> b 1 f t c')

        # Convert real form of original STFT and mask to complex form for frequency domain operations
        stft_repr = torch.view_as_complex(stft_repr)
        mask = torch.view_as_complex(mask)

        # Modulate original spectrum with estimated mask to complete frequency-domain source separation
        stft_repr = stft_repr * mask

        # Adjust separated spectrum shape for inverse STFT, restoring independent dimensions for each source and audio channel
        stft_repr = rearrange(stft_repr, 'b n (f s) t -> (b n s) f t', s=self.audio_channels)

        # Execute inverse STFT to convert frequency-domain data back to time-domain audio, considering MacOS MPS compatibility
        try:
            recon_audio = torch.istft(stft_repr, **self.stft_kwargs, window=stft_window, return_complex=False, length=raw_audio.shape[-1])
        except:
            recon_audio = torch.istft(stft_repr.cpu() if x_is_mps else stft_repr, **self.stft_kwargs, window=stft_window.cpu() if x_is_mps else stft_window, return_complex=False, length=raw_audio.shape[-1]).to(device)
        # Adjust inverse transformed audio shape to separate multiple sources and channels
        recon_audio = rearrange(recon_audio, '(b n s) t -> b n s t', s=self.audio_channels, n=num_stems)

        # If only one source, remove source dimension
        if num_stems == 1:
            recon_audio = rearrange(recon_audio, 'b 1 s t -> b s t')

        # If no target audio provided, directly return reconstructed audio
        if not exists(target):
            return recon_audio

        # When model separates multiple sources, check target audio dimensions (4D with first dim matching source count)
        if self.num_stems > 1:
            assert target.ndim == 4 and target.shape[1] == self.num_stems

        # If target audio is only 2D, expand to 3D (add source dimension)
        if target.ndim == 2:
            target = rearrange(target, '... t -> ... 1 t')

        # Truncate target audio time length to match inverse STFT generated audio length
        target = target[..., :recon_audio.shape[-1]]

        # Compute base L1 loss to measure absolute difference between reconstructed and target audio
        loss = F.l1_loss(recon_audio, target)

        # Initialize multi-resolution STFT loss to capture subtle errors at different frequency resolutions
        multi_stft_resolution_loss = 0.
        for window_size in self.multi_stft_resolutions_window_sizes:
            # Configure STFT parameters for current window size, ensure FFT length is not smaller than window size
            res_stft_kwargs = dict(
                n_fft=max(window_size, self.multi_stft_n_fft),
                win_length=window_size,
                return_complex=True,
                window=self.multi_stft_window_fn(window_size, device=device),
                **self.multi_stft_kwargs,
            )
            # Apply STFT transform to both reconstructed and target audio
            recon_Y = torch.stft(rearrange(recon_audio, '... s t -> (... s) t'), **res_stft_kwargs)
            target_Y = torch.stft(rearrange(target, '... s t -> (... s) t'), **res_stft_kwargs)
            # Accumulate L1 loss across resolutions
            multi_stft_resolution_loss = multi_stft_resolution_loss + F.l1_loss(recon_Y, target_Y)

        # Multiply multi-resolution STFT loss by preset weight and add to base L1 loss for total loss
        weighted_multi_resolution_loss = multi_stft_resolution_loss * self.multi_stft_resolution_loss_weight
        total_loss = loss + weighted_multi_resolution_loss

        # If detailed loss breakdown not needed, directly return total loss
        if not return_loss_breakdown:
            return total_loss

        # Return total loss and detailed breakdown (L1 loss, multi-resolution STFT loss)
        return total_loss, (loss, multi_stft_resolution_loss)