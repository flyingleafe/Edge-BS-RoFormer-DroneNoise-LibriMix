# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'

# Import necessary libraries
import os
import random
import numpy as np
import torch
import soundfile as sf
import pickle
import time
import itertools
import multiprocessing
from tqdm.auto import tqdm
from glob import glob
import audiomentations as AU  # Audio augmentation library
import pedalboard as PB      # Audio effects processing library
import warnings
warnings.filterwarnings("ignore")


def load_chunk(path, length, chunk_size, offset=None):
    """
    Load audio chunk.
    Args:
        path: Audio file path
        length: Total audio length
        chunk_size: Size of chunk to load
        offset: Starting position offset
    Returns:
        Audio data with shape (channels, chunk_size)
    """
    if chunk_size <= length:
        # If chunk size is less than audio length, randomly select a position to start reading
        if offset is None:
            offset = np.random.randint(length - chunk_size + 1)
        x = sf.read(path, dtype='float32', start=offset, frames=chunk_size)[0]
    else:
        # If chunk size is greater than audio length, need to pad with zeros
        x = sf.read(path, dtype='float32')[0]
        if len(x.shape) == 1:
            # Mono case
            pad = np.zeros((chunk_size - length))
        else:
            pad = np.zeros([chunk_size - length, x.shape[-1]])
        x = np.concatenate([x, pad], axis=0)
    # Convert mono to stereo
    if len(x.shape) == 1:
        x = np.expand_dims(x, axis=1)
    return x.T


def get_track_set_length(params):
    """
    Get the length of a set of tracks.
    Args:
        params: Tuple containing path, instrument list, and file types
    Returns:
        Path and minimum track length
    """
    path, instruments, file_types = params
    # Check length of all instrument tracks (may differ)
    lengths_arr = []
    for instr in instruments:
        length = -1
        for extension in file_types:
            path_to_audio_file = path + '/{}.{}'.format(instr, extension)
            if os.path.isfile(path_to_audio_file):
                length = len(sf.read(path_to_audio_file)[0])
                break
        if length == -1:
            print('Cant find file "{}" in folder {}'.format(instr, path))
            continue
        lengths_arr.append(length)
    lengths_arr = np.array(lengths_arr)
    if lengths_arr.min() != lengths_arr.max():
        print('Warning: lengths of stems are different for path: {}. ({} != {})'.format(
            path,
            lengths_arr.min(),
            lengths_arr.max())
        )
    # Use minimum length to avoid overflow
    return path, lengths_arr.min()


# For multiprocessing
def get_track_length(params):
    """Get length of a single track"""
    path = params
    length = len(sf.read(path)[0])
    return (path, length)


class MSSDataset(torch.utils.data.Dataset):
    """
    Music source separation dataset class.
    Supports multiple dataset formats:
    - type 1: Each folder contains tracks for all instruments
    - type 2: Each instrument has its own folder
    - type 3: Dataset organized using CSV files
    - type 4: Similar to type 1, but ensures track alignment
    """
    def __init__(self, config, data_path, metadata_path="metadata.pkl", dataset_type=1, batch_size=None, verbose=True):
        """
        Initialize dataset.
        Args:
            config: Configuration object containing training parameters
            data_path: Dataset path
            metadata_path: Metadata cache path
            dataset_type: Dataset type (1-4)
            batch_size: Batch size
            verbose: Whether to print detailed information
        """
        self.verbose = verbose
        self.config = config
        self.dataset_type = dataset_type
        self.data_path = data_path
        self.instruments = instruments = config.training.instruments  # List of instruments to separate
        if batch_size is None:
            batch_size = config.training.batch_size
        self.batch_size = batch_size
        self.file_types = ['wav', 'flac']  # Supported audio formats
        self.metadata_path = metadata_path

        # Audio augmentation settings
        self.aug = False
        if 'augmentations' in config:
            if config['augmentations'].enable is True:
                if self.verbose:
                    print('Use augmentation for training')
                self.aug = True
        else:
            if self.verbose:
                print('There is no augmentations block in config. Augmentations disabled for training...')

        # Load or generate metadata
        metadata = self.get_metadata()

        # Check if dataset is valid
        if self.dataset_type in [1, 4]:
            if len(metadata) > 0:
                if self.verbose:
                    print('Found tracks in dataset: {}'.format(len(metadata)))
            else:
                print('No tracks found for training. Check paths you provided!')
                exit()
        else:
            for instr in self.instruments:
                if self.verbose:
                    print('Found tracks for {} in dataset: {}'.format(instr, len(metadata[instr])))
        self.metadata = metadata
        self.chunk_size = config.audio.chunk_size  # Audio chunk size
        self.min_mean_abs = config.audio.min_mean_abs  # Minimum mean absolute value (for filtering silent chunks)

    def __len__(self):
        """Return dataset size"""
        return self.config.training.num_steps * self.batch_size

    def read_from_metadata_cache(self, track_paths, instr=None):
        """
        Read metadata from cache.
        Args:
            track_paths: List of track paths
            instr: Instrument name
        Returns:
            List of uncached paths and cached metadata
        """
        metadata = []
        if os.path.isfile(self.metadata_path):
            if self.verbose:
                print('Found metadata cache file: {}'.format(self.metadata_path))
            old_metadata = pickle.load(open(self.metadata_path, 'rb'))
        else:
            return track_paths, metadata

        if instr:
            old_metadata = old_metadata[instr]

        # Only re-read uncached tracks
        track_paths_set = set(track_paths)
        for old_path, file_size in old_metadata:
            if old_path in track_paths_set:
                metadata.append([old_path, file_size])
                track_paths_set.remove(old_path)
        track_paths = list(track_paths_set)
        if len(metadata) > 0:
            print('Old metadata was used for {} tracks.'.format(len(metadata)))
        return track_paths, metadata

    def get_metadata(self):
        """
        Get or generate dataset metadata.
        Returns:
            Metadata dictionary or list containing track paths and length information
        """
        # Set number of processes
        read_metadata_procs = multiprocessing.cpu_count()
        if 'read_metadata_procs' in self.config['training']:
            read_metadata_procs = int(self.config['training']['read_metadata_procs'])

        if self.verbose:
            print(
                'Dataset type:', self.dataset_type,
                'Processes to use:', read_metadata_procs,
                '\nCollecting metadata for', str(self.data_path),
            )

        # Handle different dataset types
        if self.dataset_type in [1, 4]:
            # Type 1 and 4: Each folder contains tracks for all instruments
            track_paths = []
            if type(self.data_path) == list:
                for tp in self.data_path:
                    tracks_for_folder = sorted(glob(tp + '/*'))
                    if len(tracks_for_folder) == 0:
                        print('Warning: no tracks found in folder \'{}\'. Please check it!'.format(tp))
                    track_paths += tracks_for_folder
            else:
                track_paths += sorted(glob(self.data_path + '/*'))

            track_paths = [path for path in track_paths if os.path.basename(path)[0] != '.' and os.path.isdir(path)]
            track_paths, metadata = self.read_from_metadata_cache(track_paths, None)

            # Single process or multi-process handling
            if read_metadata_procs <= 1:
                for path in tqdm(track_paths):
                    track_path, track_length = get_track_set_length((path, self.instruments, self.file_types))
                    metadata.append((track_path, track_length))
            else:
                p = multiprocessing.Pool(processes=read_metadata_procs)
                with tqdm(total=len(track_paths)) as pbar:
                    track_iter = p.imap(
                        get_track_set_length,
                        zip(track_paths, itertools.repeat(self.instruments), itertools.repeat(self.file_types))
                    )
                    for track_path, track_length in track_iter:
                        metadata.append((track_path, track_length))
                        pbar.update()
                p.close()

        elif self.dataset_type == 2:
            # Type 2: Each instrument has its own folder
            metadata = dict()
            for instr in self.instruments:
                metadata[instr] = []
                track_paths = []
                if type(self.data_path) == list:
                    for tp in self.data_path:
                        track_paths += sorted(glob(tp + '/{}/*.wav'.format(instr)))
                        track_paths += sorted(glob(tp + '/{}/*.flac'.format(instr)))
                else:
                    track_paths += sorted(glob(self.data_path + '/{}/*.wav'.format(instr)))
                    track_paths += sorted(glob(self.data_path + '/{}/*.flac'.format(instr)))

                track_paths, metadata[instr] = self.read_from_metadata_cache(track_paths, instr)

                if read_metadata_procs <= 1:
                    for path in tqdm(track_paths):
                        length = len(sf.read(path)[0])
                        metadata[instr].append((path, length))
                else:
                    p = multiprocessing.Pool(processes=read_metadata_procs)
                    for out in tqdm(p.imap(get_track_length, track_paths), total=len(track_paths)):
                        metadata[instr].append(out)

        elif self.dataset_type == 3:
            # Type 3: Dataset organized using CSV files
            import pandas as pd
            if type(self.data_path) != list:
                data_path = [self.data_path]

            metadata = dict()
            for i in range(len(self.data_path)):
                if self.verbose:
                    print('Reading tracks from: {}'.format(self.data_path[i]))
                df = pd.read_csv(self.data_path[i])

                skipped = 0
                for instr in self.instruments:
                    part = df[df['instrum'] == instr].copy()
                    print('Tracks found for {}: {}'.format(instr, len(part)))
                for instr in self.instruments:
                    part = df[df['instrum'] == instr].copy()
                    metadata[instr] = []
                    track_paths = list(part['path'].values)
                    track_paths, metadata[instr] = self.read_from_metadata_cache(track_paths, instr)

                    for path in tqdm(track_paths):
                        if not os.path.isfile(path):
                            print('Cant find track: {}'.format(path))
                            skipped += 1
                            continue
                        try:
                            length = len(sf.read(path)[0])
                        except:
                            print('Problem with path: {}'.format(path))
                            skipped += 1
                            continue
                        metadata[instr].append((path, length))
                if skipped > 0:
                    print('Missing tracks: {} from {}'.format(skipped, len(df)))
        else:
            print('Unknown dataset type: {}. Must be 1, 2, 3 or 4'.format(self.dataset_type))
            exit()

        # Save metadata cache
        pickle.dump(metadata, open(self.metadata_path, 'wb'))
        return metadata

    def load_source(self, metadata, instr):
        """
        Load a single audio source.
        Args:
            metadata: Metadata
            instr: Instrument name
        Returns:
            Loaded audio data tensor
        """
        while True:
            if self.dataset_type in [1, 4]:
                track_path, track_length = random.choice(metadata)
                for extension in self.file_types:
                    path_to_audio_file = track_path + '/{}.{}'.format(instr, extension)
                    if os.path.isfile(path_to_audio_file):
                        try:
                            source = load_chunk(path_to_audio_file, track_length, self.chunk_size)
                        except Exception as e:
                            print('Error: {} Path: {}'.format(e, path_to_audio_file))
                            source = np.zeros((2, self.chunk_size), dtype=np.float32)
                        break
            else:
                track_path, track_length = random.choice(metadata[instr])
                try:
                    source = load_chunk(track_path, track_length, self.chunk_size)
                except Exception as e:
                    print('Error: {} Path: {}'.format(e, track_path))
                    source = np.zeros((2, self.chunk_size), dtype=np.float32)

            # Filter out silent chunks
            if np.abs(source).mean() >= self.min_mean_abs:
                break
        # Apply audio augmentation
        if self.aug:
            source = self.augm_data(source, instr)
        return torch.tensor(source, dtype=torch.float32)

    def load_random_mix(self):
        """
        Load random audio mixture.
        Returns:
            Tensor containing all instrument tracks
        """
        res = []
        for instr in self.instruments:
            s1 = self.load_source(self.metadata, instr)
            # Mixup augmentation
            if self.aug:
                if 'mixup' in self.config['augmentations']:
                    if self.config['augmentations'].mixup:
                        mixup = [s1]
                        for prob in self.config.augmentations.mixup_probs:
                            if random.uniform(0, 1) < prob:
                                s2 = self.load_source(self.metadata, instr)
                                mixup.append(s2)
                        mixup = torch.stack(mixup, dim=0)
                        # Random volume adjustment
                        loud_values = np.random.uniform(
                            low=self.config.augmentations.loudness_min,
                            high=self.config.augmentations.loudness_max,
                            size=(len(mixup),)
                        )
                        loud_values = torch.tensor(loud_values, dtype=torch.float32)
                        mixup *= loud_values[:, None, None]
                        s1 = mixup.mean(dim=0, dtype=torch.float32)
            res.append(s1)
        res = torch.stack(res)
        return res

    def load_aligned_data(self):
        """
        Load aligned audio data (for dataset type 4).
        Returns:
            Aligned audio data tensor
        """
        track_path, track_length = random.choice(self.metadata)
        attempts = 10
        while attempts:
            if track_length >= self.chunk_size:
                common_offset = np.random.randint(track_length - self.chunk_size + 1)
            else:
                common_offset = None
            res = []
            silent_chunks = 0
            for i in self.instruments:
                for extension in self.file_types:
                    path_to_audio_file = track_path + '/{}.{}'.format(i, extension)
                    if os.path.isfile(path_to_audio_file):
                        try:
                            source = load_chunk(path_to_audio_file, track_length, self.chunk_size, offset=common_offset)
                        except Exception as e:
                            print('Error: {} Path: {}'.format(e, path_to_audio_file))
                            source = np.zeros((2, self.chunk_size), dtype=np.float32)
                        break
                res.append(source)
                if np.abs(source).mean() < self.min_mean_abs:
                    silent_chunks += 1
            if silent_chunks == 0:
                break

            attempts -= 1
            if attempts <= 0:
                print('Attempts max!', track_path)
            if common_offset is None:
                break

        res = np.stack(res, axis=0)
        if self.aug:
            for i, instr in enumerate(self.instruments):
                res[i] = self.augm_data(res[i], instr)
        return torch.tensor(res, dtype=torch.float32)

    def augm_data(self, source, instr):
        """
        Apply augmentation to audio data.
        Args:
            source: Source audio data
            instr: Instrument name
        Returns:
            Augmented audio data
        """
        source_shape = source.shape
        applied_augs = []
        if 'all' in self.config['augmentations']:
            augs = self.config['augmentations']['all']
        else:
            augs = dict()

        # Add instrument-specific augmentations
        if instr in self.config['augmentations']:
            for el in self.config['augmentations'][instr]:
                augs[el] = self.config['augmentations'][instr][el]

        # Apply various audio augmentations
        # 1. Channel shuffle
        if 'channel_shuffle' in augs:
            if augs['channel_shuffle'] > 0:
                if random.uniform(0, 1) < augs['channel_shuffle']:
                    source = source[::-1].copy()
                    applied_augs.append('channel_shuffle')

        # 2. Random inverse
        if 'random_inverse' in augs:
            if augs['random_inverse'] > 0:
                if random.uniform(0, 1) < augs['random_inverse']:
                    source = source[:, ::-1].copy()
                    applied_augs.append('random_inverse')

        # 3. Random polarity (multiply by -1)
        if 'random_polarity' in augs:
            if augs['random_polarity'] > 0:
                if random.uniform(0, 1) < augs['random_polarity']:
                    source = -source.copy()
                    applied_augs.append('random_polarity')

        # 4. Pitch shift
        if 'pitch_shift' in augs:
            if augs['pitch_shift'] > 0:
                if random.uniform(0, 1) < augs['pitch_shift']:
                    apply_aug = AU.PitchShift(
                        min_semitones=augs['pitch_shift_min_semitones'],
                        max_semitones=augs['pitch_shift_max_semitones'],
                        p=1.0
                    )
                    source = apply_aug(samples=source, sample_rate=44100)
                    applied_augs.append('pitch_shift')

        # 5. Seven band parametric equalizer
        if 'seven_band_parametric_eq' in augs:
            if augs['seven_band_parametric_eq'] > 0:
                if random.uniform(0, 1) < augs['seven_band_parametric_eq']:
                    apply_aug = AU.SevenBandParametricEQ(
                        min_gain_db=augs['seven_band_parametric_eq_min_gain_db'],
                        max_gain_db=augs['seven_band_parametric_eq_max_gain_db'],
                        p=1.0
                    )
                    source = apply_aug(samples=source, sample_rate=44100)
                    applied_augs.append('seven_band_parametric_eq')

        # 6. Tanh distortion
        if 'tanh_distortion' in augs:
            if augs['tanh_distortion'] > 0:
                if random.uniform(0, 1) < augs['tanh_distortion']:
                    apply_aug = AU.TanhDistortion(
                        min_distortion=augs['tanh_distortion_min'],
                        max_distortion=augs['tanh_distortion_max'],
                        p=1.0
                    )
                    source = apply_aug(samples=source, sample_rate=44100)
                    applied_augs.append('tanh_distortion')

        # 7. MP3 compression
        if 'mp3_compression' in augs:
            if augs['mp3_compression'] > 0:
                if random.uniform(0, 1) < augs['mp3_compression']:
                    apply_aug = AU.Mp3Compression(
                        min_bitrate=augs['mp3_compression_min_bitrate'],
                        max_bitrate=augs['mp3_compression_max_bitrate'],
                        backend=augs['mp3_compression_backend'],
                        p=1.0
                    )
                    source = apply_aug(samples=source, sample_rate=44100)
                    applied_augs.append('mp3_compression')

        # 8. Gaussian noise
        if 'gaussian_noise' in augs:
            if augs['gaussian_noise'] > 0:
                if random.uniform(0, 1) < augs['gaussian_noise']:
                    apply_aug = AU.AddGaussianNoise(
                        min_amplitude=augs['gaussian_noise_min_amplitude'],
                        max_amplitude=augs['gaussian_noise_max_amplitude'],
                        p=1.0
                    )
                    source = apply_aug(samples=source, sample_rate=44100)
                    applied_augs.append('gaussian_noise')

        # 9. Time stretch
        if 'time_stretch' in augs:
            if augs['time_stretch'] > 0:
                if random.uniform(0, 1) < augs['time_stretch']:
                    apply_aug = AU.TimeStretch(
                        min_rate=augs['time_stretch_min_rate'],
                        max_rate=augs['time_stretch_max_rate'],
                        leave_length_unchanged=True,
                        p=1.0
                    )
                    source = apply_aug(samples=source, sample_rate=44100)
                    applied_augs.append('time_stretch')

        # Fix shape
        if source_shape != source.shape:
            source = source[..., :source_shape[-1]]

        # 10. Reverb effect
        if 'pedalboard_reverb' in augs:
            if augs['pedalboard_reverb'] > 0:
                if random.uniform(0, 1) < augs['pedalboard_reverb']:
                    room_size = random.uniform(
                        augs['pedalboard_reverb_room_size_min'],
                        augs['pedalboard_reverb_room_size_max'],
                    )
                    damping = random.uniform(
                        augs['pedalboard_reverb_damping_min'],
                        augs['pedalboard_reverb_damping_max'],
                    )
                    wet_level = random.uniform(
                        augs['pedalboard_reverb_wet_level_min'],
                        augs['pedalboard_reverb_wet_level_max'],
                    )
                    dry_level = random.uniform(
                        augs['pedalboard_reverb_dry_level_min'],
                        augs['pedalboard_reverb_dry_level_max'],
                    )
                    width = random.uniform(
                        augs['pedalboard_reverb_width_min'],
                        augs['pedalboard_reverb_width_max'],
                    )
                    board = PB.Pedalboard([PB.Reverb(
                        room_size=room_size,  # 0.1 - 0.9
                        damping=damping,  # 0.1 - 0.9
                        wet_level=wet_level,  # 0.1 - 0.9
                        dry_level=dry_level,  # 0.1 - 0.9
                        width=width,  # 0.9 - 1.0
                        freeze_mode=0.0,
                    )])
                    source = board(source, 44100)
                    applied_augs.append('pedalboard_reverb')

        # 11. Chorus effect
        if 'pedalboard_chorus' in augs:
            if augs['pedalboard_chorus'] > 0:
                if random.uniform(0, 1) < augs['pedalboard_chorus']:
                    rate_hz = random.uniform(
                        augs['pedalboard_chorus_rate_hz_min'],
                        augs['pedalboard_chorus_rate_hz_max'],
                    )
                    depth = random.uniform(
                        augs['pedalboard_chorus_depth_min'],
                        augs['pedalboard_chorus_depth_max'],
                    )
                    centre_delay_ms = random.uniform(
                        augs['pedalboard_chorus_centre_delay_ms_min'],
                        augs['pedalboard_chorus_centre_delay_ms_max'],
                    )
                    feedback = random.uniform(
                        augs['pedalboard_chorus_feedback_min'],
                        augs['pedalboard_chorus_feedback_max'],
                    )
                    mix = random.uniform(
                        augs['pedalboard_chorus_mix_min'],
                        augs['pedalboard_chorus_mix_max'],
                    )
                    board = PB.Pedalboard([PB.Chorus(
                        rate_hz=rate_hz,
                        depth=depth,
                        centre_delay_ms=centre_delay_ms,
                        feedback=feedback,
                        mix=mix,
                    )])
                    source = board(source, 44100)
                    applied_augs.append('pedalboard_chorus')

        # 12. Phaser effect
        if 'pedalboard_phazer' in augs:
            if augs['pedalboard_phazer'] > 0:
                if random.uniform(0, 1) < augs['pedalboard_phazer']:
                    rate_hz = random.uniform(
                        augs['pedalboard_phazer_rate_hz_min'],
                        augs['pedalboard_phazer_rate_hz_max'],
                    )
                    depth = random.uniform(
                        augs['pedalboard_phazer_depth_min'],
                        augs['pedalboard_phazer_depth_max'],
                    )
                    centre_frequency_hz = random.uniform(
                        augs['pedalboard_phazer_centre_frequency_hz_min'],
                        augs['pedalboard_phazer_centre_frequency_hz_max'],
                    )
                    feedback = random.uniform(
                        augs['pedalboard_phazer_feedback_min'],
                        augs['pedalboard_phazer_feedback_max'],
                    )
                    mix = random.uniform(
                        augs['pedalboard_phazer_mix_min'],
                        augs['pedalboard_phazer_mix_max'],
                    )
                    board = PB.Pedalboard([PB.Phaser(
                        rate_hz=rate_hz,
                        depth=depth,
                        centre_frequency_hz=centre_frequency_hz,
                        feedback=feedback,
                        mix=mix,
                    )])
                    source = board(source, 44100)
                    applied_augs.append('pedalboard_phazer')

        # 13. Distortion effect - adds distortion and overdrive to audio, enhances expressiveness
        if 'pedalboard_distortion' in augs:
            if augs['pedalboard_distortion'] > 0:
                if random.uniform(0, 1) < augs['pedalboard_distortion']:
                    # Control distortion intensity in decibels
                    drive_db = random.uniform(
                        augs['pedalboard_distortion_drive_db_min'],
                        augs['pedalboard_distortion_drive_db_max'],
                    )
                    board = PB.Pedalboard([PB.Distortion(
                        drive_db=drive_db,
                    )])
                    source = board(source, 44100)
                    applied_augs.append('pedalboard_distortion')

        # 14. Pitch shift effect - changes audio pitch, increases data diversity
        if 'pedalboard_pitch_shift' in augs:
            if augs['pedalboard_pitch_shift'] > 0:
                if random.uniform(0, 1) < augs['pedalboard_pitch_shift']:
                    # Control pitch shift in semitones
                    semitones = random.uniform(
                        augs['pedalboard_pitch_shift_semitones_min'],
                        augs['pedalboard_pitch_shift_semitones_max'],
                    )
                    board = PB.Pedalboard([PB.PitchShift(
                        semitones=semitones
                    )])
                    source = board(source, 44100)
                    applied_augs.append('pedalboard_pitch_shift')

        # Resample effect - simulates different audio quality by changing sample rate
        if 'pedalboard_resample' in augs:
            if augs['pedalboard_resample'] > 0:
                if random.uniform(0, 1) < augs['pedalboard_resample']:
                    # Target sample rate, simulates audio characteristics at different sample rates
                    target_sample_rate = random.uniform(
                        augs['pedalboard_resample_target_sample_rate_min'],
                        augs['pedalboard_resample_target_sample_rate_max'],
                    )
                    board = PB.Pedalboard([PB.Resample(
                        target_sample_rate=target_sample_rate
                    )])
                    source = board(source, 44100)
                    applied_augs.append('pedalboard_resample')

        # Bit depth reduction effect - simulates low quality audio by reducing bit depth
        if 'pedalboard_bitcrash' in augs:
            if augs['pedalboard_bitcrash'] > 0:
                if random.uniform(0, 1) < augs['pedalboard_bitcrash']:
                    # Control bit depth, smaller value means lower quality
                    bit_depth = random.uniform(
                        augs['pedalboard_bitcrash_bit_depth_min'],
                        augs['pedalboard_bitcrash_bit_depth_max'],
                    )
                    board = PB.Pedalboard([PB.Bitcrush(
                        bit_depth=bit_depth
                    )])
                    source = board(source, 44100)
                    applied_augs.append('pedalboard_bitcrash')

        # MP3 compression effect - simulates MP3 compression impact on audio
        if 'pedalboard_mp3_compressor' in augs:
            if augs['pedalboard_mp3_compressor'] > 0:
                if random.uniform(0, 1) < augs['pedalboard_mp3_compressor']:
                    # VBR (variable bit rate) compression quality, affects MP3 compression level
                    vbr_quality = random.uniform(
                        augs['pedalboard_mp3_compressor_pedalboard_mp3_compressor_min'],
                        augs['pedalboard_mp3_compressor_pedalboard_mp3_compressor_max'],
                    )
                    board = PB.Pedalboard([PB.MP3Compressor(
                        vbr_quality=vbr_quality
                    )])
                    source = board(source, 44100)
                    applied_augs.append('pedalboard_mp3_compressor')

        # Return augmented audio source
        return source

    def __getitem__(self, index):
        # Load data based on dataset type
        if self.dataset_type in [1, 2, 3]:
            res = self.load_random_mix()  # Random mix loading
        else:
            res = self.load_aligned_data()  # Load aligned data

        # Randomly adjust loudness for each track
        if self.aug:
            if 'loudness' in self.config['augmentations']:
                if self.config['augmentations']['loudness']:
                    # Generate random loudness values for each track
                    loud_values = np.random.uniform(
                        low=self.config['augmentations']['loudness_min'],
                        high=self.config['augmentations']['loudness_max'],
                        size=(len(res),)
                    )
                    loud_values = torch.tensor(loud_values, dtype=torch.float32)
                    res *= loud_values[:, None, None]  # Apply loudness adjustment

        # Mix all tracks into one mixture
        mix = res.sum(0)

        # Apply MP3 compression augmentation to mixture
        if self.aug:
            if 'mp3_compression_on_mixture' in self.config['augmentations']:
                apply_aug = AU.Mp3Compression(
                    min_bitrate=self.config['augmentations']['mp3_compression_on_mixture_bitrate_min'],
                    max_bitrate=self.config['augmentations']['mp3_compression_on_mixture_bitrate_max'],
                    backend=self.config['augmentations']['mp3_compression_on_mixture_backend'],
                    p=self.config['augmentations']['mp3_compression_on_mixture']
                )
                mix_conv = mix.cpu().numpy().astype(np.float32)
                required_shape = mix_conv.shape
                mix = apply_aug(samples=mix_conv, sample_rate=44100)
                # If audio length changes after compression, trim
                if mix.shape != required_shape:
                    mix = mix[..., :required_shape[-1]]
                mix = torch.tensor(mix, dtype=torch.float32)

        # If target instrument is specified (for roformer models), only return that instrument's track and mixture
        if self.config.training.target_instrument is not None:
            index = self.config.training.instruments.index(self.config.training.target_instrument)
            return res[index], mix

        # Return all tracks and mixture
        return res, mix
