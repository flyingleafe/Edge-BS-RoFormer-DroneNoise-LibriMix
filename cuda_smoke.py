import hashlib
import json
from pathlib import Path
import time

import torch
import pytest
from models.jhtr import JHTR

source = json.loads(Path('source_revision.json').read_text())
for filename, digest in source['sha256'].items():
    assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == digest, filename
assert torch.cuda.is_available(), 'CUDA allocation required, not a CPU-only skip pass'
Path('results').mkdir(exist_ok=True)
print('CUDA:', torch.cuda.get_device_name(), 'torch:', torch.__version__, flush=True)
exit_code = pytest.main(['tests/tracking/test_jhtr_dsp.py', 'tests/models/test_jhtr.py', '-q', '--junitxml=results/cuda-tests.xml'])
if exit_code:
    raise SystemExit(exit_code)
torch.manual_seed(0)
model = JHTR().cuda().train()
audio = torch.randn(1, 16000, device='cuda') * 0.1
cond = torch.full((1, 4, 32), 80.0, device='cuda')
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
start = time.perf_counter()
with torch.autocast('cuda', dtype=torch.float16):
    output = model(audio, cond)
    loss = (output - (cond + 0.5)).square().mean()
loss.backward()
torch.cuda.synchronize()
grads = [p.grad for p in model.parameters() if p.grad is not None]
assert output.shape == cond.shape and torch.isfinite(output).all()
assert grads and all(torch.isfinite(g).all() for g in grads)
result = dict(source_commit=source['source_commit'], torch=torch.__version__, gpu=torch.cuda.get_device_name(), parameters=sum(p.numel() for p in model.parameters()), batch=1, seconds=1, amp='float16', elapsed_forward_backward_s=time.perf_counter()-start, peak_allocated_bytes=torch.cuda.max_memory_allocated(), pass_small_cuda_smoke=True, matched_batch_profile=False)
Path('results/cuda-smoke.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)
