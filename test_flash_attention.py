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

# Memory measurement function
def measure_memory(op, *args, **kwargs):
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated()
        result = op(*args, **kwargs)
        torch.cuda.synchronize()
        mem_peak = torch.cuda.max_memory_allocated()
        return (mem_peak - mem_before) / (1024 ** 2)  # MB
    return 0.0

# Test each backend explicitly
backends = {
    SDPBackend.MATH: "Math (emulated)",
    SDPBackend.FLASH_ATTENTION: "FlashAttention",
    SDPBackend.EFFICIENT_ATTENTION: "Memory-Efficient Attention"
}

print("=" * 62)
print("TEST 1: Raw F.scaled_dot_product_attention with each backend")
print("=" * 62)
print(f"Config: batch={batch_size}, heads={num_heads}, seq_len={seq_len}, embed_dim={embed_dim}")
print(f"Device: {device}, dtype: {dtype}\n")
print(f"{'Backend':<30} {'Runtime (μs)':>15} {'Memory (MB)':>15}")
print("-" * 62)
for backend, name in backends.items():
    try:
        with sdpa_kernel(backend):
            t = benchmark_op(F.scaled_dot_product_attention, query, key, value)
            mem = measure_memory(F.scaled_dot_product_attention, query, key, value)
            print(f"{name:<30} {t:>15.2f} {mem:>15.2f}")
    except RuntimeError as e:
        print(f"{name:<30} Not supported ({e})")

# ============================================================================
# TEST 2: Test the Attend module from models/edge_bs_rof/attend.py
# ============================================================================
print("\n" + "=" * 62)
print("TEST 2: Attend module (models/edge_bs_rof/attend.py)")
print("=" * 62)

from models.edge_bs_rof.attend import Attend

# Test Attend with flash=True
print("\n--- Attend(flash=True) ---")
attend_flash = Attend(dropout=0., flash=True).to(device)
attend_flash.eval()

# Benchmark Attend with flash=True
t_flash = benchmark_op(attend_flash, query, key, value)
mem_flash = measure_memory(attend_flash, query, key, value)
print(f"{'Attend(flash=True)':<30} {t_flash:>15.2f} {mem_flash:>15.2f}")

# Test Attend with flash=False (standard attention)
print("\n--- Attend(flash=False) ---")
attend_standard = Attend(dropout=0., flash=False).to(device)
attend_standard.eval()

t_standard = benchmark_op(attend_standard, query, key, value)
mem_standard = measure_memory(attend_standard, query, key, value)
print(f"{'Attend(flash=False)':<30} {t_standard:>15.2f} {mem_standard:>15.2f}")

# Verify which backend Attend(flash=True) is actually using by forcing backends
print("\n--- Attend(flash=True) with forced backends ---")
print(f"{'Backend':<30} {'Runtime (μs)':>15} {'Memory (MB)':>15}")
print("-" * 62)
for backend, name in backends.items():
    try:
        with sdpa_kernel(backend):
            t = benchmark_op(attend_flash, query, key, value)
            mem = measure_memory(attend_flash, query, key, value)
            print(f"{name:<30} {t:>15.2f} {mem:>15.2f}")
    except RuntimeError as e:
        print(f"{name:<30} Not supported ({e})")

# Summary analysis
print("\n" + "=" * 62)
print("ANALYSIS")
print("=" * 62)
speedup = t_standard / t_flash if t_flash > 0 else 0
mem_reduction = mem_standard / mem_flash if mem_flash > 0 else 0
print(f"Attend(flash=True) vs Attend(flash=False):")
print(f"  - Speedup: {speedup:.2f}x")
print(f"  - Memory reduction: {mem_reduction:.2f}x")

# Check if flash attention is truly being used
if t_flash < t_standard * 0.5 and mem_flash < mem_standard * 0.5:
    print("\n[PASS] Attend(flash=True) is using an optimized attention kernel (Flash or Memory-Efficient)")
else:
    print("\n[WARN] Attend(flash=True) may not be using an optimized kernel")