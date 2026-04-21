#!/usr/bin/env python3
"""
Helper script to generate Slidev presentation from slide descriptions.

This script assists in creating Slidev presentations by:
1. Generating comparison plots using generate_comparison.py
2. Creating slide markdown structure
3. Organizing assets

Usage:
    python generate_slides.py --help
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Available models
AVAILABLE_MODELS = [
    'Edge-BS-RoFormer',
    'DCUNet',
    'DPTNet',
    'HTDemucs',
    'Diffusion-Buffer-BBED',
]


def run_command(cmd, check=True):
    """Run a shell command."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 and check:
        print(f"Error: {result.stderr}")
        sys.exit(1)
    return result


def generate_comparison_plot(models, output_dir, plot_name='comparison.png'):
    """Generate comparison plot using generate_comparison.py."""
    cmd = [
        'python', 'generate_comparison.py',
        '--models'] + models + [
        '--output_dir', str(output_dir),
        '--plot_name', plot_name
    ]
    run_command(cmd)


def generate_audio_comparison(models, samples, output_dir):
    """Generate audio comparison plots."""
    cmd = [
        'python', 'generate_comparison.py',
        '--models'] + models + [
        '--samples'] + samples + [
        '--output_dir', str(output_dir)
    ]
    run_command(cmd)


def main():
    parser = argparse.ArgumentParser(
        description='Generate Slidev presentation assets',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--sync-results',
        action='store_true',
        help='Sync results from vast-server before generating plots'
    )
    
    parser.add_argument(
        '--models',
        nargs='+',
        help='Models to compare for results slides'
    )
    
    parser.add_argument(
        '--samples',
        nargs='+',
        help='Sample IDs for audio comparison'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='slides/assets',
        help='Output directory for assets (default: slides/assets)'
    )
    
    args = parser.parse_args()
    
    # Sync results if requested
    if args.sync_results:
        print("Syncing results from vast-server...")
        run_command(['./sync_results.sh'])
    
    # Generate comparison plots if models specified
    if args.models:
        output_dir = Path(args.output_dir) / 'comparison'
        generate_comparison_plot(args.models, output_dir)
        print(f"✓ Comparison plots generated in {output_dir}")
    
    # Generate audio comparisons if samples specified
    if args.samples and args.models:
        output_dir = Path(args.output_dir) / 'audio'
        generate_audio_comparison(args.models, args.samples, output_dir)
        print(f"✓ Audio comparison plots generated in {output_dir}")


if __name__ == '__main__':
    main()
