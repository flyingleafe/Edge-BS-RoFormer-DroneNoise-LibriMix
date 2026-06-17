from __future__ import annotations

import torch
from torch.utils.data import DataLoader, IterableDataset, TensorDataset

from utils.dataloader_benchmark import benchmark_dataloader, infer_num_audio_clips


def test_benchmark_dataloader_exhausts_finite_loader():
    ds = TensorDataset(torch.zeros(5, 16000), torch.zeros(5, 4, 32))
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)

    result = benchmark_dataloader(loader, seconds=60.0)

    assert result["exhausted"] is True
    assert result["batches"] == 3
    assert result["examples"] == 5
    assert result["audio_clips"] == 5
    assert result["batches_per_s"] > 0


def test_benchmark_dataloader_respects_max_batches_for_infinite_loader():
    class InfiniteAudio(IterableDataset):
        def __iter__(self):
            while True:
                yield torch.zeros(2, 16000), torch.zeros(4, 32)

    loader = DataLoader(InfiniteAudio(), batch_size=3, num_workers=0)

    result = benchmark_dataloader(loader, seconds=60.0, max_batches=4)

    assert result["exhausted"] is False
    assert result["batches"] == 4
    assert result["examples"] == 12
    assert result["audio_clips"] == 12 * 2


def test_infer_num_audio_clips_counts_multichannel_as_effective_clips():
    batch = (torch.zeros(4, 8, 16000), torch.zeros(4, 4, 32))

    assert infer_num_audio_clips(batch) == 32
