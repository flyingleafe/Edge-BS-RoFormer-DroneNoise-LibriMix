# Import necessary libraries
import numpy as np
import torch
import librosa
import torch.nn.functional as F
from typing import Dict, List, Tuple

def sdr(references: np.ndarray, estimates: np.ndarray) -> np.ndarray:
    """
    Calculate Signal-to-Distortion Ratio (SDR).

    SDR measures how well a predicted source matches a reference source. It evaluates separation quality
    by computing the ratio of the reference signal energy to the error energy (difference between reference and prediction).
    Returns SDR values in decibels (dB).

    Parameters:
    ----------
    references : np.ndarray
        3D array of shape (num_sources, num_channels, num_samples) representing reference source signals

    estimates : np.ndarray
        3D array of shape (num_sources, num_channels, num_samples) representing predicted source signals

    Returns:
    -------
    np.ndarray
        1D array containing SDR value for each source
    """
    eps = 1e-8  # Small value to avoid numerical errors
    num = np.sum(np.square(references), axis=(1, 2))  # Calculate reference signal energy
    den = np.sum(np.square(references - estimates), axis=(1, 2))  # Calculate error energy
    num += eps
    den += eps
    return 10 * np.log10(num / den)  # Convert to decibels


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """
    Calculate Scale-Invariant Signal-to-Distortion Ratio (SI-SDR).

    SI-SDR is a variant of SDR that is insensitive to scaling of the predicted signal relative to the reference.
    It calculates SDR after scaling the predicted signal to match the reference signal.

    Parameters:
    ----------
    reference : np.ndarray
        3D array of shape (num_sources, num_channels, num_samples) representing reference source signals

    estimate : np.ndarray
        3D array of shape (num_sources, num_channels, num_samples) representing predicted source signals

    Returns:
    -------
    float
        SI-SDR scalar value in decibels (dB)
    """
    eps = 1e-8  # Avoid numerical errors
    # Calculate optimal scaling factor
    scale = np.sum(estimate * reference + eps, axis=(0, 1)) / np.sum(reference ** 2 + eps, axis=(0, 1))
    scale = np.expand_dims(scale, axis=(0, 1))  # Reshape to [num_sources, 1]

    # Apply scaling and calculate SI-SDR
    reference = reference * scale
    si_sdr = np.mean(10 * np.log10(
        np.sum(reference ** 2, axis=(0, 1)) / (np.sum((reference - estimate) ** 2, axis=(0, 1)) + eps) + eps))

    return si_sdr


def L1Freq_metric(
        reference: np.ndarray,
        estimate: np.ndarray,
        fft_size: int = 2048,
        hop_size: int = 1024,
        device: str = 'cpu'
) -> float:
    """
    Calculate L1 frequency metric between reference and estimated signals.

    This metric compares the magnitude spectra of reference and estimated audio signals
    using Short-Time Fourier Transform (STFT), computing L1 loss between them.
    Results are scaled to [0, 100] range, where higher values indicate better performance.

    Parameters:
    ----------
    reference : np.ndarray
        2D array of shape (num_channels, num_samples) representing reference audio signal

    estimate : np.ndarray
        2D array of shape (num_channels, num_samples) representing estimated audio signal

    fft_size : int, optional
        Window size for STFT, default 2048

    hop_size : int, optional
        Hop size between STFT frames, default 1024

    device : str, optional
        Computation device ('cpu' or 'cuda'), default 'cpu'

    Returns:
    -------
    float
        L1 frequency metric value in range [0, 100]
    """

    # Convert numpy arrays to torch tensors and move to specified device
    reference = torch.from_numpy(reference).to(device)
    estimate = torch.from_numpy(estimate).to(device)

    # Compute STFT to get complex spectra
    reference_stft = torch.stft(reference, fft_size, hop_size, return_complex=True)
    estimated_stft = torch.stft(estimate, fft_size, hop_size, return_complex=True)

    # Compute magnitude spectra (absolute value of complex numbers)
    reference_mag = torch.abs(reference_stft)
    estimate_mag = torch.abs(estimated_stft)

    # Calculate L1 loss and scale
    # Multiply by 10 to adjust loss scale for better discrimination
    loss = 10 * F.l1_loss(estimate_mag, reference_mag)

    # Convert loss value to [0,100] range, smaller loss means closer to 100
    ret = 100 / (1. + float(loss.cpu().numpy()))

    return ret


def NegLogWMSE_metric(
        reference: np.ndarray,
        estimate: np.ndarray,
        mixture: np.ndarray,
        device: str = 'cpu',
) -> float:
    """
    Calculate Log-Weighted Mean Squared Error (Log-WMSE) between reference, estimated, and mixture signals.

    This metric evaluates the quality of estimated signals relative to reference signals in audio source separation.
    Using logarithmic scale helps evaluate signals with large amplitude differences.
    Result is negative, where larger values indicate better separation.

    Parameters:
    ----------
    reference : np.ndarray
        2D array of shape (num_channels, num_samples) representing reference audio signal

    estimate : np.ndarray
        2D array of shape (num_channels, num_samples) representing estimated audio signal

    mixture : np.ndarray
        2D array of shape (num_channels, num_samples) representing mixture audio signal

    device : str, optional
        Computation device ('cpu' or 'cuda'), default 'cpu'

    Returns:
    -------
    float
        Negative log-weighted mean squared error value
    """
    # Import LogWMSE loss function
    from torch_log_wmse import LogWMSE

    # Initialize LogWMSE calculator
    log_wmse = LogWMSE(
        audio_length=reference.shape[-1] / 44100,  # Audio length in seconds
        sample_rate=44100,  # Sample rate 44100Hz
        return_as_loss=False,  # Return as evaluation metric rather than loss
        bypass_filter=False,  # Enable frequency filtering
    )

    # Expand dimensions and convert to torch tensors
    # Add batch dimension and extra channel dimension to match model input requirements
    reference = torch.from_numpy(reference).unsqueeze(0).unsqueeze(0).to(device)
    estimate = torch.from_numpy(estimate).unsqueeze(0).unsqueeze(0).to(device)
    mixture = torch.from_numpy(mixture).unsqueeze(0).to(device)

    # Calculate LogWMSE and return negative value (convert smaller-is-better to larger-is-better)
    res = log_wmse(mixture, reference, estimate)
    return -float(res.cpu().numpy())


def AuraSTFT_metric(
        reference: np.ndarray,
        estimate: np.ndarray,
        device: str = 'cpu',
) -> float:
    """
    Calculate spectral difference between reference and estimated signals using STFT loss.

    This metric considers both log and linear magnitude STFT losses, commonly used to evaluate
    audio separation task quality. Results are scaled to [0, 100] range, where higher values
    indicate better separation.

    Parameters:
    ----------
    reference : np.ndarray
        2D array of shape (num_channels, num_samples) representing reference audio signal

    estimate : np.ndarray
        2D array of shape (num_channels, num_samples) representing estimated audio signal

    device : str, optional
        Computation device ('cpu' or 'cuda'), default 'cpu'

    Returns:
    -------
    float
        STFT metric value in range [0, 100]
    """

    # Import STFT loss function
    from auraloss.freq import STFTLoss

    # Initialize STFT loss calculator
    stft_loss = STFTLoss(
        w_log_mag=1.0,  # Log magnitude weight
        w_lin_mag=0.0,  # Linear magnitude weight
        w_sc=1.0,       # Spectral convergence weight
        device=device,
    )

    # Convert to torch tensors and add batch dimension
    reference = torch.from_numpy(reference).unsqueeze(0).to(device)
    estimate = torch.from_numpy(estimate).unsqueeze(0).to(device)

    # Calculate loss and scale to [0,100] range
    res = 100 / (1. + 10 * stft_loss(reference, estimate))
    return float(res.cpu().numpy())


def AuraMRSTFT_metric(
        reference: np.ndarray,
        estimate: np.ndarray,
        device: str = 'cpu',
) -> float:
    """
    Calculate spectral difference between reference and estimated signals using Multi-Resolution STFT (MRSTFT) loss.

    This metric uses multiple resolution STFT analysis, which better represents both low and high frequency
    components of audio signals. Results are scaled to [0, 100] range, where higher values indicate better separation.

    Parameters:
    ----------
    reference : np.ndarray
        2D array of shape (num_channels, num_samples) representing reference audio signal

    estimate : np.ndarray
        2D array of shape (num_channels, num_samples) representing estimated audio signal

    device : str, optional
        Computation device ('cpu' or 'cuda'), default 'cpu'

    Returns:
    -------
    float
        MRSTFT metric value in range [0, 100]
    """

    # Import multi-resolution STFT loss function
    from auraloss.freq import MultiResolutionSTFTLoss

    # Initialize multi-resolution STFT loss calculator
    mrstft_loss = MultiResolutionSTFTLoss(
        fft_sizes=[1024, 2048, 4096],  # Use three different FFT window sizes
        hop_sizes=[256, 512, 1024],    # Corresponding hop sizes
        win_lengths=[1024, 2048, 4096], # Window lengths
        scale="mel",  # Use mel frequency scale
        n_bins=128,   # Number of mel frequency bins
        sample_rate=44100,
        perceptual_weighting=True,  # Use perceptual weighting
        device=device
    )

    # Convert to torch tensors and add batch dimension
    reference = torch.from_numpy(reference).unsqueeze(0).float().to(device)
    estimate = torch.from_numpy(estimate).unsqueeze(0).float().to(device)

    # Calculate loss and scale to [0,100] range
    res = 100 / (1. + 10 * mrstft_loss(reference, estimate))
    return float(res.cpu().numpy())


def bleed_full(
        reference: np.ndarray,
        estimate: np.ndarray,
        sr: int = 44100,
        n_fft: int = 4096,
        hop_length: int = 1024,
        n_mels: int = 512,
        device: str = 'cpu',
) -> Tuple[float, float]:
    """
    Calculate 'bleedless' and 'fullness' metrics between reference and estimated signals.

    'bleedless' metric measures the degree of leakage from estimated signal to reference,
    'fullness' metric measures the completeness of estimated signal relative to reference,
    both calculated using mel spectrograms and decibel scale.

    Parameters:
    ----------
    reference : np.ndarray
        2D array of shape (num_channels, num_samples) representing reference audio signal

    estimate : np.ndarray
        2D array of shape (num_channels, num_samples) representing estimated audio signal

    sr : int, optional
        Sample rate, default 44100Hz

    n_fft : int, optional
        FFT size for STFT, default 4096

    hop_length : int, optional
        Hop length for STFT, default 1024

    n_mels : int, optional
        Number of mel frequency bins, default 512

    device : str, optional
        Computation device ('cpu' or 'cuda'), default 'cpu'

    Returns:
    -------
    tuple
        Contains two values:
        - bleedless (float): Bleed metric score (higher is better)
        - fullness (float): Fullness metric score (higher is better)
    """

    # Import amplitude to dB conversion function
    from torchaudio.transforms import AmplitudeToDB

    # Convert to torch tensors
    reference = torch.from_numpy(reference).float().to(device)
    estimate = torch.from_numpy(estimate).float().to(device)

    # Create Hann window
    window = torch.hann_window(n_fft).to(device)

    # Compute STFT using Hann window
    D1 = torch.abs(torch.stft(reference, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True,
                              pad_mode="constant"))
    D2 = torch.abs(torch.stft(estimate, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True,
                              pad_mode="constant"))

    # Create mel filter bank
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    mel_filter_bank = torch.from_numpy(mel_basis).to(device)

    # Compute mel spectrograms
    S1_mel = torch.matmul(mel_filter_bank, D1)
    S2_mel = torch.matmul(mel_filter_bank, D2)

    # Convert to decibel scale
    S1_db = AmplitudeToDB(stype="magnitude", top_db=80)(S1_mel)
    S2_db = AmplitudeToDB(stype="magnitude", top_db=80)(S2_mel)

    # Calculate spectral difference
    diff = S2_db - S1_db

    # Separately handle positive (bleed) and negative (incompleteness) differences
    positive_diff = diff[diff > 0]
    negative_diff = diff[diff < 0]

    # Calculate average differences
    average_positive = torch.mean(positive_diff) if positive_diff.numel() > 0 else torch.tensor(0.0).to(device)
    average_negative = torch.mean(negative_diff) if negative_diff.numel() > 0 else torch.tensor(0.0).to(device)

    # Calculate final scores
    bleedless = 100 * 1 / (average_positive + 1)  # Bleed score
    fullness = 100 * 1 / (-average_negative + 1)  # Fullness score

    return bleedless.cpu().numpy(), fullness.cpu().numpy()


def get_metrics(
        metrics: List[str],
        reference: np.ndarray,
        estimate: np.ndarray,
        mix: np.ndarray,
        device: str = 'cpu',
) -> Dict[str, float]:
    """
    Calculate audio source separation model performance evaluation metrics.

    Computes various evaluation metrics between reference, estimated, and mixture signals
    based on the specified list of metrics.

    Parameters:
    ----------
    metrics : List[str]
        List of metric names to calculate (e.g., ['sdr', 'si_sdr', 'l1_freq'])

    reference : np.ndarray
        2D array of shape (num_channels, num_samples) representing reference audio signal

    estimate : np.ndarray
        2D array of shape (num_channels, num_samples) representing estimated audio signal

    mix : np.ndarray
        2D array of shape (num_channels, num_samples) representing mixture audio signal

    device : str, optional
        Computation device ('cpu' or 'cuda'), default 'cpu'

    Returns:
    -------
    Dict[str, float]
        Dictionary containing all calculated metrics
    """
    result = dict()

    # Adjust all input signal lengths to be the same
    min_length = min(reference.shape[1], estimate.shape[1])
    reference = reference[..., :min_length]
    estimate = estimate[..., :min_length]
    mix = mix[..., :min_length]

    # Calculate metrics based on the metric list
    if 'sdr' in metrics:
        # Signal-to-Distortion Ratio
        references = np.expand_dims(reference, axis=0)
        estimates = np.expand_dims(estimate, axis=0)
        result['sdr'] = sdr(references, estimates)[0]

    if 'si_sdr' in metrics:
        # Scale-Invariant Signal-to-Distortion Ratio
        result['si_sdr'] = si_sdr(reference, estimate)

    if 'l1_freq' in metrics:
        # L1 frequency metric
        result['l1_freq'] = L1Freq_metric(reference, estimate, device=device)

    if 'neg_log_wmse' in metrics:
        # Negative log-weighted mean squared error
        result['neg_log_wmse'] = NegLogWMSE_metric(reference, estimate, mix, device)

    if 'aura_stft' in metrics:
        # STFT loss metric
        result['aura_stft'] = AuraSTFT_metric(reference, estimate, device)

    if 'aura_mrstft' in metrics:
        # Multi-resolution STFT loss metric
        result['aura_mrstft'] = AuraMRSTFT_metric(reference, estimate, device)

    if 'bleedless' in metrics or 'fullness' in metrics:
        # Calculate bleedless and fullness metrics
        bleedless, fullness = bleed_full(reference, estimate, device=device)
        if 'bleedless' in metrics:
            result['bleedless'] = bleedless
        if 'fullness' in metrics:
            result['fullness'] = fullness

    return result