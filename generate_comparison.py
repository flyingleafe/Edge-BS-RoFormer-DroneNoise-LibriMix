#!/usr/bin/env python3
"""
Generate Model Comparison Plots and Tables

This script creates comparison plots (SI-SDR, STOI, PESQ) and summary tables
from evaluation results, suitable for presentation preparation.

Usage:
    python generate_comparison.py --models Edge-BS-RoFormer DCUNet --output_dir presentations/fig1
    python generate_comparison.py --models all --output_dir results/comparison
    python generate_comparison.py --models Edge-BS-RoFormer DCUNet --samples 00000 00001 --output_dir presentations/audio_comparison
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import sys
import librosa
import librosa.display
import soundfile as sf

warnings.filterwarnings('ignore')

# Model file mapping
MODEL_FILES = {
    '3_FA_RoPE(64)_validation.xlsx': 'Edge-BS-RoFormer',
    '5_Baseline_dcunet_validation.xlsx': 'DCUNet',
    '7_Baseline_dptnet_validation.xlsx': 'DPTNet',
    '8_Baseline_htdemucs_validation.xlsx': 'HTDemucs',
    '9_Diffusion_Buffer_BBED_validation.xlsx': 'Diffusion-Buffer-BBED',
}

# Color mapping (matching paper colors)
MODEL_COLORS = {
    'Edge-BS-RoFormer': '#FF7F00',  # Orange
    'DCUNet': '#377EB8',              # Blue
    'DPTNet': '#E41A1C',              # Red
    'HTDemucs': '#4DAF4A',             # Green
    'Diffusion-Buffer-BBED': '#984EA3', # Purple
}

# Model directory mapping (display name -> directory name)
MODEL_DIRS = {
    'Edge-BS-RoFormer': '3_FA_RoPE(64)',
    'DCUNet': '5_Baseline_dcunet',
    'DPTNet': '7_Baseline_dptnet',
    'HTDemucs': '8_Baseline_htdemucs',
    'Diffusion-Buffer-BBED': '9_Diffusion_Buffer_BBED',
}

# Default evaluation directory
DEFAULT_EVAL_DIR = Path('results/evaluation')
DEFAULT_AUDIO_DIR = Path('results/evaluation')


def load_model_data(evaluation_dir, model_names):
    """Load data for specified models."""
    dataframes = {}
    
    # Reverse mapping: display name -> filename
    name_to_file = {v: k for k, v in MODEL_FILES.items()}
    
    for model_name in model_names:
        if model_name not in name_to_file:
            print(f"Warning: Model '{model_name}' not found. Available models: {list(name_to_file.keys())}")
            continue
        
        filename = name_to_file[model_name]
        filepath = evaluation_dir / filename
        
        if not filepath.exists():
            print(f"Warning: File not found: {filepath}")
            continue
        
        try:
            df = pd.read_excel(filepath)
            df['Model'] = model_name
            dataframes[model_name] = df
            print(f"✓ Loaded {model_name}: {len(df)} samples")
        except Exception as e:
            print(f"✗ Error loading {filename}: {e}")
    
    return dataframes


def prepare_plot_data(dataframes):
    """Prepare data for plotting by grouping by SNR bins."""
    metrics_to_plot = ['si_sdr', 'stoi', 'pesq']
    plot_data = {}
    
    # Combine all dataframes for binning
    combined_df = pd.concat(dataframes.values(), ignore_index=True)
    
    # Create SNR bins (similar to paper: -30 to 0 dB in 5 dB steps)
    snr_bins = np.arange(-30, 5, 5)  # -30, -25, -20, -15, -10, -5, 0
    
    # Filter out inf values before binning
    combined_df = combined_df[combined_df['Input_SNR'] != np.inf].copy()
    
    # Create SNR bin labels
    combined_df['SNR_bin'] = pd.cut(
        combined_df['Input_SNR'],
        bins=snr_bins,
        labels=[int(b) for b in snr_bins[:-1]],
        include_lowest=True
    )
    
    for metric in metrics_to_plot:
        plot_data[metric] = {}
        
        for model_name in combined_df['Model'].unique():
            model_data = combined_df[combined_df['Model'] == model_name].copy()
            
            if metric not in model_data.columns:
                continue
            
            # Group by SNR bin
            grouped = model_data.groupby('SNR_bin', observed=True)[metric].agg(['mean', 'std', 'count'])
            
            # Extract SNR values and metrics
            snr_values = []
            metric_means = []
            metric_stds = []
            
            for snr_bin, row in grouped.iterrows():
                if row['count'] > 0 and pd.notna(row['mean']):
                    snr_values.append(int(snr_bin))
                    metric_means.append(row['mean'])
                    metric_stds.append(row['std'] if pd.notna(row['std']) else 0)
            
            if snr_values:
                # Sort by SNR
                sorted_indices = np.argsort(snr_values)
                plot_data[metric][model_name] = {
                    'snr': [snr_values[i] for i in sorted_indices],
                    'mean': [metric_means[i] for i in sorted_indices],
                    'std': [metric_stds[i] for i in sorted_indices]
                }
            else:
                plot_data[metric][model_name] = {
                    'snr': [],
                    'mean': [],
                    'std': []
                }
    
    return plot_data


def create_comparison_plot(plot_data, model_names, output_path, figsize=(18, 8)):
    """Create 3-panel comparison plot (SI-SDR, STOI, PESQ)."""
    # Set style
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        plt.style.use('seaborn-darkgrid')
    
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    metrics_config = [
        ('si_sdr', 'SI-SDR (dB)', '(a) Scale-Invariant Signal-to-Distortion Ratio', axes[0]),
        ('stoi', 'STOI', '(b) Short-Time Objective Intelligibility', axes[1]),
        ('pesq', 'PESQ', '(c) Perceptual Evaluation of Speech Quality', axes[2]),
    ]
    
    for metric, ylabel, title, ax in metrics_config:
        for model_name in model_names:
            if model_name not in plot_data.get(metric, {}):
                continue
            
            data = plot_data[metric][model_name]
            if len(data['snr']) == 0 or len(data['mean']) == 0:
                continue
            
            # Filter out NaN values
            valid_mask = [pd.notna(m) for m in data['mean']]
            if not any(valid_mask):
                continue
            
            snr_vals = [data['snr'][i] for i in range(len(data['snr'])) if valid_mask[i]]
            mean_vals = [data['mean'][i] for i in range(len(data['mean'])) if valid_mask[i]]
            std_vals = [data['std'][i] if pd.notna(data['std'][i]) else 0
                       for i in range(len(data['std'])) if valid_mask[i]]
            
            color = MODEL_COLORS.get(model_name, None)
            ax.plot(snr_vals, mean_vals,
                   marker='o', linewidth=2, markersize=8,
                   label=model_name, color=color)
            
            # Add error bands (standard deviation)
            ax.fill_between(snr_vals,
                           np.array(mean_vals) - np.array(std_vals),
                           np.array(mean_vals) + np.array(std_vals),
                           alpha=0.2, color=color)
        
        ax.set_xlabel('Input SNR (dB)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {output_path}")
    plt.close()


def create_summary_table(dataframes, output_path):
    """Create comprehensive summary table."""
    if not dataframes:
        print("No data to create summary table")
        return
    
    # Combine all dataframes
    combined_df = pd.concat(dataframes.values(), ignore_index=True)
    
    # Calculate statistics
    summary = combined_df.groupby('Model').agg({
        'si_sdr': ['mean', 'std', 'min', 'max'],
        'pesq': ['mean', 'std', 'min', 'max'],
        'stoi': ['mean', 'std', 'min', 'max'],
        'Input_SNR': ['mean', 'min', 'max']
    }).round(3)
    
    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path)
    print(f"✓ Summary table saved to: {output_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    print(summary)
    
    return summary


def load_audio_file(audio_dir, model_name, sample_id):
    """Load audio file for a given model and sample ID."""
    model_dir = MODEL_DIRS.get(model_name)
    if not model_dir:
        return None, None
    
    audio_path = audio_dir / model_dir / f"sample_{sample_id}_vocals.wav"
    
    if not audio_path.exists():
        return None, None
    
    try:
        audio, sr = librosa.load(str(audio_path), sr=None, mono=True)
        return audio, sr
    except Exception as e:
        print(f"Warning: Could not load {audio_path}: {e}")
        return None, None


def normalize_audio(audio):
    """Normalize audio to [-1, 1] range."""
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        return audio / max_val
    return audio


def save_normalized_audio(audio, sr, output_path):
    """Save audio normalized to [-1, 1] range."""
    normalized = normalize_audio(audio)
    sf.write(str(output_path), normalized, sr)
    return normalized


def create_audio_comparison_plots(model_names, sample_ids, audio_dir, output_dir):
    """Create waveform and spectrogram comparison plots for selected samples."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for sample_id in sample_ids:
        print(f"\nProcessing sample_{sample_id}...")
        
        # Load audio files for all models
        audio_data = {}
        sample_rates = {}
        
        for model_name in model_names:
            audio, sr = load_audio_file(audio_dir, model_name, sample_id)
            if audio is not None:
                audio_data[model_name] = audio
                sample_rates[model_name] = sr
                print(f"  ✓ Loaded {model_name}: {len(audio)/sr:.2f}s @ {sr}Hz")
            else:
                print(f"  ✗ Could not load audio for {model_name}")
        
        if not audio_data:
            print(f"  Warning: No audio files found for sample_{sample_id}")
            continue
        
        # Create comparison plot: waveforms on top, spectrograms on bottom
        n_models = len(audio_data)
        fig, axes = plt.subplots(2, n_models, figsize=(5*n_models, 10))
        
        if n_models == 1:
            axes = axes.reshape(2, 1)
        
        # Plot waveforms (top row)
        for idx, (model_name, audio) in enumerate(audio_data.items()):
            sr = sample_rates[model_name]
            time_axis = np.arange(len(audio)) / sr
            
            ax_wav = axes[0, idx]
            ax_wav.plot(time_axis, audio, linewidth=0.5, color=MODEL_COLORS.get(model_name, 'black'))
            ax_wav.set_title(f'{model_name}\nWaveform', fontsize=12, fontweight='bold')
            ax_wav.set_xlabel('Time (s)', fontsize=10)
            ax_wav.set_ylabel('Amplitude', fontsize=10)
            ax_wav.grid(True, alpha=0.3)
            ax_wav.set_xlim(0, time_axis[-1] if len(time_axis) > 0 else 1)
        
        # Plot spectrograms (bottom row)
        for idx, (model_name, audio) in enumerate(audio_data.items()):
            sr = sample_rates[model_name]
            
            ax_spec = axes[1, idx]
            
            # Compute spectrogram
            D = librosa.stft(audio)
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            
            # Display spectrogram
            img = librosa.display.specshow(
                S_db,
                x_axis='time',
                y_axis='hz',
                sr=sr,
                ax=ax_spec,
                cmap='plasma'
            )
            
            ax_spec.set_title(f'{model_name}\nSpectrogram', fontsize=12, fontweight='bold')
            ax_spec.set_xlabel('Time (s)', fontsize=10)
            ax_spec.set_ylabel('Frequency (Hz)', fontsize=10)
            
            # Add colorbar
            plt.colorbar(img, ax=ax_spec, format='%+2.0f dB')
        
        plt.tight_layout()
        
        # Save plot
        output_path = output_dir / f"sample_{sample_id}_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved comparison plot: {output_path}")
        
        # Save normalized audio files for each model
        for model_name, audio in audio_data.items():
            sr = sample_rates[model_name]
            audio_output_path = output_dir / f"{model_name}_{sample_id}.wav"
            save_normalized_audio(audio, sr, audio_output_path)
            print(f"  ✓ Saved normalized audio: {audio_output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate model comparison plots and tables',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare specific models (metrics only)
  python generate_comparison.py --models Edge-BS-RoFormer DCUNet --output_dir presentations/fig1
  
  # Compare all available models
  python generate_comparison.py --models all --output_dir results/comparison
  
  # Generate audio comparison plots for specific samples
  python generate_comparison.py --models Edge-BS-RoFormer DCUNet HTDemucs --samples 00000 00001 --output_dir presentations/audio_comparison
  
  # Both metrics and audio comparisons
  python generate_comparison.py --models Edge-BS-RoFormer DCUNet --samples 00000 --output_dir presentations/full_comparison
        """
    )
    
    parser.add_argument(
        '--models',
        nargs='+',
        required=True,
        help='Model names to compare (use "all" for all available models)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for plots and tables'
    )
    
    parser.add_argument(
        '--eval_dir',
        type=str,
        default=None,
        help=f'Evaluation results directory (default: {DEFAULT_EVAL_DIR})'
    )
    
    parser.add_argument(
        '--plot_name',
        type=str,
        default='model_comparison.png',
        help='Name for the comparison plot (default: model_comparison.png)'
    )
    
    parser.add_argument(
        '--table_name',
        type=str,
        default='performance_summary.csv',
        help='Name for the summary table (default: performance_summary.csv)'
    )
    
    parser.add_argument(
        '--samples',
        nargs='+',
        default=None,
        help='Sample IDs to generate audio comparison plots for (e.g., "00000 00001"). If not provided, only metric plots are generated.'
    )
    
    parser.add_argument(
        '--audio_dir',
        type=str,
        default=None,
        help=f'Audio files directory (default: {DEFAULT_AUDIO_DIR})'
    )
    
    args = parser.parse_args()
    
    # Determine evaluation directory
    eval_dir = Path(args.eval_dir) if args.eval_dir else DEFAULT_EVAL_DIR
    
    if not eval_dir.exists():
        print(f"Error: Evaluation directory not found: {eval_dir}")
        print("Hint: Run './sync_results.sh' first to sync results from vast-server")
        sys.exit(1)
    
    # Determine models to compare
    if args.models == ['all']:
        model_names = list(MODEL_FILES.values())
    else:
        model_names = args.models
    
    # Validate model names
    available_models = list(MODEL_FILES.values())
    invalid_models = [m for m in model_names if m not in available_models]
    if invalid_models:
        print(f"Error: Invalid model names: {invalid_models}")
        print(f"Available models: {available_models}")
        sys.exit(1)
    
    print(f"Comparing models: {', '.join(model_names)}")
    print(f"Evaluation directory: {eval_dir}")
    print(f"Output directory: {args.output_dir}\n")
    
    # Load data
    dataframes = load_model_data(eval_dir, model_names)
    
    if not dataframes:
        print("Error: No data loaded. Check evaluation directory and model names.")
        sys.exit(1)
    
    # Prepare plot data
    plot_data = prepare_plot_data(dataframes)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    
    # Generate plot
    plot_path = output_dir / args.plot_name
    create_comparison_plot(plot_data, model_names, plot_path)
    
    # Generate summary table
    table_path = output_dir / args.table_name
    create_summary_table(dataframes, table_path)
    
    # Generate audio comparison plots if samples are specified
    if args.samples:
        audio_dir = Path(args.audio_dir) if args.audio_dir else DEFAULT_AUDIO_DIR
        
        if not audio_dir.exists():
            print(f"\nWarning: Audio directory not found: {audio_dir}")
            print("Skipping audio comparison plots.")
        else:
            print(f"\nGenerating audio comparison plots for samples: {', '.join(args.samples)}")
            create_audio_comparison_plots(model_names, args.samples, audio_dir, output_dir)
    
    print(f"\n✓ All outputs saved to: {output_dir}")


if __name__ == '__main__':
    main()
