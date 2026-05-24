#!/usr/bin/env python3
"""Evaluate a saved RPS predictor checkpoint on the validation set."""
import argparse
import sys
sys.path.insert(0, ".")

from train_rps_predictor import DREGONRPSDataset, get_model, evaluate
import torch
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--data_root", default="datasets/DREGON-LM")
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--batch_size", type=int, default=16)
args = parser.parse_args()

device = torch.device(args.device)
model = get_model(args.model).to(device)
model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))

valid_ds = DREGONRPSDataset(f"{args.data_root}/valid")
valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

metrics = evaluate(model, valid_loader, device, len(valid_ds))
print(f"MSE: {metrics['mse']:.4f}")
print(f"MAE/frame: {metrics['mae_frame']:.4f}")
print(f"MAE/clip: {metrics['mae_clip']:.4f}")
print(f"R²: {metrics['r2']:.4f}")
print(f"R² median: {metrics['r2_median']:.4f}")
