"""
Distributed training harness used for the A03/A04 pilot evidence and the
ARE-II Sprint 1 kernel-attribution baseline.

This is a cleaned version of the script used in the Kaggle 2x T4 notebook for
Assignment 4. It supports two attention paths via PyTorch's SDPA selector:
"math" (standard attention) and "flash" (FlashAttention SDPA backend).

Important caveat (documented in the A05 bundle and the ARE-II plan):
On Turing-class GPUs (sm_75, including the Tesla T4 used by Kaggle and Colab
free tier), PyTorch's SDPBackend.FLASH_ATTENTION path is not supported. When
this script is run with use_flash=True on a T4, PyTorch will most likely
substitute a different attention method silently. Verifying which kernel is
actually executed is the first sprint of ARE-II.

Usage (single node, two GPUs):
    torchrun --nproc_per_node=2 src/dist_train.py PATCH_SIZE USE_FLASH OUTPUT_FILE

Arguments:
    PATCH_SIZE   integer (4 or 16) — controls sequence length
                 (32 / patch_size)^2 + 1 = sequence length
    USE_FLASH    "1" to request the FlashAttention SDPA backend, "0" for math
    OUTPUT_FILE  path where the rank-0 process writes the JSON results
"""

import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torchvision
import torchvision.transforms as transforms

SEED = 42
BATCH_SIZE = 64
NUM_STEPS = 100
WARMUP_STEPS = 10
LEARNING_RATE = 1e-3
IMG_SIZE = 32
EMBED_DIM = 384
NUM_HEADS = 6
NUM_LAYERS = 12
NUM_CLASSES = 10


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_cifar10_dataloader(batch_size, rank, world_size, data_root):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=False, transform=transform
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        sampler=sampler, num_workers=2, pin_memory=True, drop_last=True,
    )
    return loader


class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, embed_dim, in_channels=3):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))

    def forward(self, x):
        b = x.shape[0]
        x = self.proj(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, use_flash=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.use_flash = use_flash

    def forward(self, x):
        normed = self.norm1(x)
        if self.use_flash:
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
                attn_out, _ = self.attn(normed, normed, normed)
        else:
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class ViTSmall(nn.Module):
    def __init__(self, img_size=32, patch_size=16, embed_dim=384,
                 num_heads=6, num_layers=12, num_classes=10, use_flash=False):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, embed_dim)
        self.blocks = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, use_flash) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.blocks(x)
        x = self.norm(x[:, 0])
        x = self.head(x)
        return x


def main():
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group("nccl")
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    patch_size = int(sys.argv[1])
    use_flash = sys.argv[2] == "1"
    output_file = sys.argv[3]
    data_root = os.environ.get("DATA_ROOT", "/kaggle/working/data")

    set_seed(SEED + rank)

    seq_len = (IMG_SIZE // patch_size) ** 2 + 1
    if rank == 0:
        print(
            f"Config: patch_size={patch_size}, seq_len={seq_len}, "
            f"flash={use_flash}, world_size={world_size}"
        )

    model = ViTSmall(
        img_size=IMG_SIZE, patch_size=patch_size, embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS, num_layers=NUM_LAYERS, num_classes=NUM_CLASSES,
        use_flash=use_flash,
    ).to(device)
    model = DDP(model, device_ids=[rank])

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    loader = get_cifar10_dataloader(BATCH_SIZE, rank, world_size, data_root)

    model.train()
    step_times, fwd_times, bwd_times, losses = [], [], [], []
    data_iter = iter(loader)

    for _ in range(NUM_STEPS):
        try:
            images, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            images, labels = next(data_iter)
        images, labels = images.to(device), labels.to(device)

        start_event = torch.cuda.Event(enable_timing=True)
        fwd_end = torch.cuda.Event(enable_timing=True)
        bwd_end = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        fwd_end.record()
        loss.backward()
        bwd_end.record()
        optimizer.step()
        end_event.record()
        torch.cuda.synchronize()

        step_times.append(start_event.elapsed_time(end_event) / 1000.0)
        fwd_times.append(start_event.elapsed_time(fwd_end) / 1000.0)
        bwd_times.append(fwd_end.elapsed_time(bwd_end) / 1000.0)
        losses.append(loss.item())

    if rank == 0:
        m_times = step_times[WARMUP_STEPS:]
        m_fwd = fwd_times[WARMUP_STEPS:]
        m_bwd = bwd_times[WARMUP_STEPS:]
        m_losses = losses[WARMUP_STEPS:]
        median_time = float(np.median(m_times))
        results = {
            "patch_size": patch_size,
            "seq_len": seq_len,
            "use_flash": use_flash,
            "median_time_per_step": median_time,
            "median_fwd_time": float(np.median(m_fwd)),
            "median_bwd_time": float(np.median(m_bwd)),
            "throughput": (BATCH_SIZE * world_size) / median_time,
            "final_loss": float(m_losses[-1]),
            "all_step_times": m_times,
        }
        with open(output_file, "w") as f:
            json.dump(results, f)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
