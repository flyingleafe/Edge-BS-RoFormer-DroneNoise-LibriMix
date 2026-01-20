import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import torch.utils.benchmark as benchmark

# Ensure CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16
batch_size, num_heads, seq_len, embed_dim = 32, 8, 512, 64

# Create sample inputs
query = torch.rand(batch_size, num_heads, seq_len, embed_dim, device=device, dtype=dtype)
key = torch.rand(batch_size, num_heads, seq_len, embed_dim, device=device, dtype=dtype)
value = torch.rand(batch_size, num_heads, seq_len, embed_dim, device=device, dtype=dtype)

# Benchmark function
def benchmark_op(op, *args, **kwargs):
    t0 = benchmark.Timer(stmt="op(*args, **kwargs)", globals={"op": op, "args": args, "kwargs": kwargs})
    return t0.blocked_autorange().mean * 1e6  # microseconds

# Test each backend explicitly
backends = {
    SDPBackend.MATH: "Math (emulated)",
    SDPBackend.FLASH_ATTENTION: "FlashAttention",
    SDPBackend.EFFICIENT_ATTENTION: "Memory-Efficient Attention"
}

print("Runtime of each SDPA backend:")
for backend, name in backends.items():
    try:
        with sdpa_kernel(backend):
            t = benchmark_op(F.scaled_dot_product_attention, query, key, value)
            print(f"{name}: {t:.2f} μs")
    except RuntimeError as e:
        print(f"{name}: Not supported ({e})")