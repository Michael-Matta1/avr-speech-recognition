"""
ATmega32A Isolated-Word Speech Recognition Training Pipeline.

Copyright (c) 2026 Michael Matta
GitHub: https://github.com/Michael-Matta1

Licensed under the MIT License. See LICENSE for details.

This script trains and evaluates an 8-word keyword-spotting pipeline and exports
word_templates.h for deployment with classify_snippet.c on ATmega32A.
"""

import argparse
import hashlib as _hashlib
import inspect
import multiprocessing as _mp
import os
import random
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
from scipy.fftpack import dct
from scipy.signal import butter, lfilter
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import check_random_state as _crs


# =============================================================================
# Configuration
# =============================================================================

WORDS = ["on", "off", "go", "stop", "left", "right", "up", "down"]
WORD_TO_INDEX = {w: i for i, w in enumerate(WORDS)}

SAMPLE_RATE = 8000
DURATION = 1.0
N_SAMPLES = int(SAMPLE_RATE * DURATION)

# 20 ms frame / 10 ms hop at 8 kHz (standard speech setup)
FRAME_SIZE = 160
HOP_SIZE = 80
N_FFT = 256
PRE_EMPHASIS = 0.97

# MFCC feature stack
N_MEL_FILTERS = 24
N_MFCC = 20
CEPSTRAL_LIFTER = 22
DELTA_WINDOW = 2
USE_LIBROSA_MEL = True
USE_CMVN = False
USE_PCEN = True
PCEN_GAIN = 0.98
PCEN_BIAS = 2.0
PCEN_POWER = 0.5
PCEN_TIME_CONSTANT = 0.06
PCEN_EPS = 1e-6
PCEN_INPUT_SCALE = float(2 ** 31)
TEMPORAL_BINS = 5
TEMPORAL_MFCC_COEFFS = 10
TEMPORAL_DELTA_COEFFS = 8
TEMPORAL_LOGMEL_COEFFS = 8
N_LFCC = 12
N_LFCC_FILTERS = 24
LPCC_ORDER = 12
LPCC_N_CEP = 12

FEATURES_STATIC_DIMS = (6 * N_MFCC)
FEATURES_SPECTRAL_STATS_DIMS = 6
FEATURES_TEMPORAL_MFCC_DIMS = TEMPORAL_BINS * TEMPORAL_MFCC_COEFFS
FEATURES_TEMPORAL_DELTA_DIMS = TEMPORAL_BINS * TEMPORAL_DELTA_COEFFS
FEATURES_TEMPORAL_LOGMEL_DIMS = TEMPORAL_BINS * TEMPORAL_LOGMEL_COEFFS
FEATURES_ENERGY_DIMS = 3
FEATURES_ROBUST_DIMS = 6
FEATURES_LOGMEL_STATS_DIMS = 2 * N_MEL_FILTERS
FEATURES_CONTRAST_DIMS = 12
FEATURES_LFCC_DIMS = 2 * N_LFCC
FEATURES_FORMANT_DIMS = 4

# Feature vector:
# [mfcc_mean(20), mfcc_std(20),
#  delta_mean(20), delta_std(20),
#  delta2_mean(20), delta2_std(20),
#  spectral flux/rolloff/bandwidth stats (6),
#  temporal MFCC bins (5 x 10), temporal delta bins (5 x 8),
#  temporal logmel bins (5 x 8), energy stats (3), robust extras (6),
#  logmel mean/std (48), spectral contrast (12),
#  LFCC mean/std (24), formants (4)] => 353 dims
N_FEATURES = (
    FEATURES_STATIC_DIMS
    + FEATURES_SPECTRAL_STATS_DIMS
    + FEATURES_TEMPORAL_MFCC_DIMS
    + FEATURES_TEMPORAL_DELTA_DIMS
    + FEATURES_TEMPORAL_LOGMEL_DIMS
    + FEATURES_ENERGY_DIMS
    + FEATURES_ROBUST_DIMS
    + FEATURES_LOGMEL_STATS_DIMS
    + FEATURES_CONTRAST_DIMS
    + FEATURES_LFCC_DIMS
    + FEATURES_FORMANT_DIMS
)

# Classifier dimensions required by fixed AVR classifier schema
N_WORDS = len(WORDS)
LDA_DIMS = N_WORDS - 1  # must stay 7
K_TEMPLATES = 16

RECOMMENDED_SAMPLES = 25
DATASET_DIR = "."
TEMPLATES_DIR = "templates"

# Keep fallback split as default to preserve historical benchmark comparability.
USE_OFFICIAL_SPLIT_LISTS = False

# VAD and confidence settings
VAD_TRAIN_THRESHOLD = 0.005
VAD_EVAL_THRESHOLD = 0.005
VAD_LIVE_FALLBACK = 0.002
NOISE_MEASURE_MS = 300
NOISE_MULTIPLIER = 4.0
CONFIDENCE_THRESHOLD = 3.0

# CI pass/fail target
TARGET_ACCURACY = 90.0
BASELINE_ACCURACY = 38.6

# Deterministic seed setup.
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
_sklearn_rng = _crs(RANDOM_SEED)

# Feature-set versioning for cache invalidation and run traceability.
FEAT_VERSION = 10
FEATURE_CACHE_VERSION = "v13_realbg_ratio_mix"

# Data augmentation settings
ENABLE_TRAIN_AUGMENTATION = True
AUG_SPEED_FACTORS = (0.85, 0.92, 1.08, 1.15)
AUG_NOISE_COPIES = 2
AUG_NOISE_RMS_RATIO = 0.018
AUG_REVERB_COPIES = 1
AUG_REVERB_RT60 = 0.08

# Strategy #3: blend clean and noisy samples with a tunable ratio.
ENABLE_REAL_BG_NOISE = True
BG_NOISE_DIR_NAME = "_background_noise_"
BG_NOISE_SNR_DB_RANGE = (18.0, 30.0)
NOISE_MIX_NOISY_RATIO = 0.50

# Confusion-focused augmentation for hardest classes from recent evals.
AUG_HARD_CLASS_WORDS = ("up", "off", "stop")
AUG_HARD_EXTRA_SPEED_FACTORS = (0.95, 1.05)
AUG_HARD_EXTRA_NOISE_COPIES = 3
AUG_HARD_EXTRA_REVERB_COPIES = 2
AUG_HARD_EXTRA_SHIFT_MS = (-120, 120)
_AUG_HARD_LABELS = {
    WORD_TO_INDEX[w]
    for w in AUG_HARD_CLASS_WORDS
    if w in WORD_TO_INDEX
}

# Template refinement and optimizer settings
USE_LVQ_REFINEMENT = True
LVQ_EPOCHS = 30
LVQ_LR0 = 0.04
LVQ_OUTLIER_PERCENTILE = 97.0
LVQ_MAX_SAMPLES = 12000

OPT_EPOCHS = 220
OPT_LR = 0.0009
OPT_TAU = 0.22
OPT_REG = 1e-6
OPT_EARLY_STOP_PATIENCE = 999
ENABLE_FINAL_TRAINVAL_FINETUNE = False


# =============================================================================
# Optional dependencies and accelerators
# =============================================================================

try:
    import librosa  # type: ignore
    LIBROSA_AVAILABLE = True
except Exception:  # pragma: no cover
    librosa = None
    LIBROSA_AVAILABLE = False

_IS_MAIN_PROCESS = (_mp.current_process().name == "MainProcess")

try:
    import cupy as cp  # type: ignore

    # Find the NVIDIA GPU and avoid binding to non-CUDA integrated adapters.
    _CUPY_AVAILABLE = False
    _CUPY_DEVICE_ID = 0
    _CUPY_DEVICE_NAME = ""
    for _dev_id in range(cp.cuda.runtime.getDeviceCount()):
        _dev_props = cp.cuda.runtime.getDeviceProperties(_dev_id)
        _raw_name = _dev_props["name"]
        _dev_name = (
            _raw_name.decode("utf-8", errors="ignore")
            if isinstance(_raw_name, (bytes, bytearray))
            else str(_raw_name)
        )
        if any(k in _dev_name for k in ("NVIDIA", "GeForce", "RTX", "GTX", "Tesla", "Quadro")):
            _CUPY_DEVICE_ID = _dev_id
            _CUPY_DEVICE_NAME = _dev_name
            _CUPY_AVAILABLE = True
            break

    if _CUPY_AVAILABLE:
        cp.cuda.Device(_CUPY_DEVICE_ID).use()
        _ = (cp.arange(4, dtype=cp.float32) + 1).sum().item()
        if _IS_MAIN_PROCESS:
            print(
                f"  [GPU] CuPy available on Device {_CUPY_DEVICE_ID + 1} "
                f"(CUDA {_CUPY_DEVICE_ID}, {_CUPY_DEVICE_NAME}) - using GPU for optimizer"
            )
    else:
        if _IS_MAIN_PROCESS:
            print("  [GPU] No NVIDIA GPU found via CuPy - falling back to CPU")
        cp = None  # type: ignore[assignment]
except Exception as _cupy_exc:
    cp = None  # type: ignore[assignment]
    _CUPY_AVAILABLE = False
    if _IS_MAIN_PROCESS:
        print(f"  [GPU] CuPy unavailable ({_cupy_exc}); using CPU")


# =============================================================================
# Dataset structures
# =============================================================================

@dataclass
class SampleEntry:
    word: str
    label: int
    file_path: str
    rel_path: str
    speaker_id: str


@dataclass
class PreparedDataset:
    entries: List[SampleEntry]
    split_indices: Dict[str, np.ndarray]
    features: List[Optional[np.ndarray]]
    skipped: int


@dataclass
class EvalMetrics:
    overall_accuracy: float
    total_correct: int
    total_samples: int
    skipped: int
    confusion: Dict[str, Dict[str, int]]
    per_word: Dict[str, Dict[str, int]]
    threshold: float


# =============================================================================
# Optional interactive dependency
# =============================================================================

def _require_sounddevice():
    try:
        import sounddevice as sd  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "sounddevice is required for record/test modes. "
            "Install with: pip install sounddevice"
        ) from exc
    return sd


# =============================================================================
# Signal preprocessing and features
# =============================================================================

def butter_lowpass(cutoff: float = 3500.0, fs: int = SAMPLE_RATE, order: int = 4):
    b, a = butter(order, cutoff / (fs / 2.0), btype="low", analog=False)
    return b.astype(np.float32), a.astype(np.float32)


_LPF_B, _LPF_A = butter_lowpass()
_HAMMING_WINDOW = np.hamming(FRAME_SIZE).astype(np.float32)
_FFT_FREQS = np.fft.rfftfreq(N_FFT, d=1.0 / SAMPLE_RATE).astype(np.float32)
_LIFTER_VEC = (
    1.0 + (CEPSTRAL_LIFTER / 2.0)
    * np.sin(np.pi * np.arange(N_MFCC) / CEPSTRAL_LIFTER)
).astype(np.float32)



def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + (hz / 700.0))


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _build_mel_filterbank(
    sr: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    n_mels: int = N_MEL_FILTERS,
    fmin: float = 20.0,
    fmax: float = 3800.0,
) -> np.ndarray:
    if USE_LIBROSA_MEL and LIBROSA_AVAILABLE and librosa is not None:
        try:
            fb = librosa.filters.mel(
                sr=sr,
                n_fft=n_fft,
                n_mels=n_mels,
                fmin=fmin,
                fmax=fmax,
                htk=True,
                norm="slaney",
            ).astype(np.float32)

            denom = fb.sum(axis=1, keepdims=True)
            denom[denom == 0.0] = 1.0
            return fb / denom
        except Exception:
            pass

    n_bins = (n_fft // 2) + 1
    mel_points = np.linspace(_hz_to_mel(np.array([fmin]))[0], _hz_to_mel(np.array([fmax]))[0], n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(np.int32)
    bins = np.clip(bins, 0, n_bins - 1)

    fb = np.zeros((n_mels, n_bins), dtype=np.float32)

    for m in range(1, n_mels + 1):
        left = bins[m - 1]
        center = bins[m]
        right = bins[m + 1]

        if center <= left:
            center = min(left + 1, n_bins - 1)
        if right <= center:
            right = min(center + 1, n_bins - 1)

        if center > left:
            fb[m - 1, left:center] = (
                np.arange(left, center, dtype=np.float32) - left
            ) / (center - left)
        if right > center:
            fb[m - 1, center:right] = (
                right - np.arange(center, right, dtype=np.float32)
            ) / (right - center)

    # Normalize each filter to unit area for stable magnitudes.
    denom = fb.sum(axis=1, keepdims=True)
    denom[denom == 0.0] = 1.0
    return fb / denom


_MEL_FILTERBANK = _build_mel_filterbank()


def _build_linear_filterbank(
    sr: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    n_filters: int = N_LFCC_FILTERS,
    f_min: float = 0.0,
    f_max: float = 4000.0,
) -> np.ndarray:
    """Uniform triangular filterbank for LFCC extraction."""
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr).astype(np.float32)
    centers = np.linspace(f_min, f_max, n_filters + 2, dtype=np.float32)
    bank = np.zeros((n_filters, len(freqs)), dtype=np.float32)

    for m in range(n_filters):
        lo, mid, hi = float(centers[m]), float(centers[m + 1]), float(centers[m + 2])
        rising = (freqs >= lo) & (freqs <= mid)
        falling = (freqs > mid) & (freqs <= hi)
        bank[m, rising] = (freqs[rising] - lo) / (mid - lo + 1e-10)
        bank[m, falling] = (hi - freqs[falling]) / (hi - mid + 1e-10)

    row_sums = bank.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return (bank / row_sums).astype(np.float32)


_LINEAR_FILTERBANK = _build_linear_filterbank()


def _pcen_transform(
    mel_spec: np.ndarray,
    sr: int = SAMPLE_RATE,
    hop_length: int = HOP_SIZE,
    gain: float = PCEN_GAIN,
    bias: float = PCEN_BIAS,
    power: float = PCEN_POWER,
    time_constant: float = PCEN_TIME_CONSTANT,
    eps: float = PCEN_EPS,
    input_scale: float = PCEN_INPUT_SCALE,
) -> np.ndarray:
    """Per-channel energy normalization as a robust alternative to log compression."""
    s = np.maximum(mel_spec.astype(np.float32) * float(input_scale), 1e-10)
    n_frames, n_ch = s.shape
    if n_frames == 0 or n_ch == 0:
        return s

    t_frames = max(1e-6, float(time_constant) * float(sr) / float(hop_length))
    b = (np.sqrt(1.0 + 4.0 * (t_frames ** 2)) - 1.0) / (2.0 * (t_frames ** 2))
    b = float(np.clip(b, 1e-5, 1.0))

    m = np.empty_like(s, dtype=np.float32)
    m[0] = s[0]
    one_minus_b = 1.0 - b
    for t in range(1, n_frames):
        m[t] = one_minus_b * m[t - 1] + b * s[t]

    pcen = ((s / np.power(eps + m, gain)) + bias) ** power - (bias ** power)
    return pcen.astype(np.float32)


def _compute_lpcc(signal: np.ndarray, order: int = LPCC_ORDER,
                  n_cep: int = LPCC_N_CEP) -> np.ndarray:
    """
    Compute Linear Predictive Cepstral Coefficients via Levinson-Durbin.
    Returns n_cep coefficients (c1..cn, skipping c0 which is log energy).
    order: LPC order (12 is standard for 8kHz speech)
    n_cep: number of cepstral coefficients to return
    """
    if len(signal) <= order + 1:
        return np.zeros(n_cep, dtype=np.float32)

    # Autocorrelation method
    r = np.zeros(order + 1, dtype=np.float64)
    n = len(signal)
    for k in range(order + 1):
        r[k] = float(np.dot(signal[:n - k], signal[k:]))
    if r[0] < 1e-10:
        return np.zeros(n_cep, dtype=np.float32)

    # Levinson-Durbin recursion
    a = np.zeros(order + 1, dtype=np.float64)
    a[0] = 1.0
    e = float(r[0])
    for i in range(1, order + 1):
        lam = -float(np.dot(a[:i], r[i:0:-1])) / max(e, 1e-10)
        a_new = a.copy()
        for j in range(1, i + 1):
            a_new[j] = a[j] + lam * a[i - j]
        a_new[i] = lam
        a = a_new
        e *= (1.0 - lam * lam)
        if e < 1e-10:
            break

    # Convert LPC -> LPCC via recursion (standard formula)
    lpc = a[1:]
    c = np.zeros(n_cep, dtype=np.float64)
    for m in range(1, n_cep + 1):
        if m <= order:
            c[m - 1] = -lpc[m - 1]
            for k in range(1, m):
                c[m - 1] += (k / m) * c[k - 1] * (
                    -lpc[m - k - 1] if (m - k) <= order else 0.0
                )
        else:
            for k in range(1, order + 1):
                c[m - 1] += (k / m) * c[k - 1] * (
                    -lpc[m - k - 1] if (m - k) <= order else 0.0
                )
    return c.astype(np.float32)


def _extract_formants(signal: np.ndarray, sr: int = SAMPLE_RATE,
                      lpc_order: int = 10) -> np.ndarray:
    """
    Estimate first 2 formant frequencies from LPC spectrum peaks.
    Returns [F1_mean, F1_std, F2_mean, F2_std] normalized by Nyquist.
    Uses voiced-frame selection (energy + ZCR threshold).
    """
    hop = HOP_SIZE
    fsize = FRAME_SIZE
    n_frames = max(1, (len(signal) - fsize) // hop + 1)

    f1_vals = []
    f2_vals = []

    win = np.hamming(fsize).astype(np.float64)
    freqs = np.fft.rfftfreq(512, 1.0 / sr)
    f_mask = (freqs >= 200.0) & (freqs <= 3500.0)
    freq_f = freqs[f_mask]

    for i in range(n_frames):
        frame = signal[i * hop: i * hop + fsize].astype(np.float64)
        if len(frame) < fsize:
            break

        energy = float(np.mean(frame ** 2))
        zcr = float(np.sum(np.diff(np.sign(frame)) != 0) / fsize)
        if energy < 1e-6 or zcr > 0.35:
            continue

        frame_pe = np.append(frame[0], frame[1:] - 0.95 * frame[:-1])
        frame_w = frame_pe * win

        r = np.array(
            [np.dot(frame_w[:fsize - k], frame_w[k:]) for k in range(lpc_order + 1)],
            dtype=np.float64,
        )
        if r[0] < 1e-10:
            continue

        try:
            from scipy.linalg import solve_toeplitz
            a = solve_toeplitz(r[:lpc_order], -r[1:lpc_order + 1])
            lpc_coeffs = np.concatenate([[1.0], a])
        except Exception:
            continue

        # LPC envelope in frequency domain from denominator polynomial.
        lpc_resp = np.zeros(len(freqs), dtype=np.complex128)
        for k, coef in enumerate(lpc_coeffs):
            lpc_resp += coef * np.exp(-2j * np.pi * freqs / sr * k)
        spectrum = 1.0 / (np.abs(lpc_resp) + 1e-10)
        spec_f = spectrum[f_mask]

        peaks = []
        for pi in range(1, len(spec_f) - 1):
            if spec_f[pi] > spec_f[pi - 1] and spec_f[pi] > spec_f[pi + 1]:
                peaks.append((spec_f[pi], float(freq_f[pi])))
        peaks.sort(reverse=True)

        if len(peaks) >= 2:
            freq_peaks = sorted([p[1] for p in peaks[:4]])
            f1_vals.append(freq_peaks[0] / (sr / 2.0))
            if len(freq_peaks) >= 2:
                f2_vals.append(freq_peaks[1] / (sr / 2.0))

    if len(f1_vals) >= 2:
        return np.array(
            [
                float(np.mean(f1_vals)),
                float(np.std(f1_vals)),
                float(np.mean(f2_vals)) if f2_vals else 0.0,
                float(np.std(f2_vals)) if len(f2_vals) >= 2 else 0.0,
            ],
            dtype=np.float32,
        )
    return np.zeros(4, dtype=np.float32)


def preprocess(signal: np.ndarray) -> np.ndarray:
    """Low-pass, peak normalize, pre-emphasize, and length-normalize."""
    signal = lfilter(_LPF_B, _LPF_A, signal.astype(np.float32))

    peak = float(np.max(np.abs(signal))) if len(signal) else 0.0
    if peak > 0.0:
        signal = signal / peak

    if len(signal) > 1:
        signal = np.append(signal[0], signal[1:] - PRE_EMPHASIS * signal[:-1]).astype(np.float32)

    if len(signal) < N_SAMPLES:
        signal = np.pad(signal, (0, N_SAMPLES - len(signal)))

    return signal[:N_SAMPLES].astype(np.float32)


def _frame_signal(signal: np.ndarray) -> np.ndarray:
    starts = np.arange(0, N_SAMPLES - FRAME_SIZE + 1, HOP_SIZE)
    frames = np.stack([signal[s:s + FRAME_SIZE] for s in starts], axis=0)
    return frames * _HAMMING_WINDOW


def _compute_delta(feat: np.ndarray, win: int = DELTA_WINDOW) -> np.ndarray:
    """
    Vectorized ETSI-standard delta (first-difference) features.
    Identical output to the loop version; ~40x faster for 85 frames.

    Formula:  d_t = sum_{n=1}^{win} n*(c_{t+n} - c_{t-n})
                    / (2 * sum_{n=1}^{win} n^2)

    Replaces the double Python for-loop with numpy slice arithmetic:
    instead of iterating t=0..T-1, we use shifted views of the padded
    matrix, which is a pure C-level numpy operation.
    """
    n_frames = feat.shape[0]
    if n_frames <= 1:
        return np.zeros_like(feat, dtype=np.float32)

    denom = float(2.0 * sum(i * i for i in range(1, win + 1)))
    # Edge-pad so boundary frames use nearest valid frame (ETSI ES 201 108)
    padded = np.pad(feat, ((win, win), (0, 0)), mode="edge")

    # Accumulate weighted symmetric differences using numpy slicing.
    # Each slice operation is an O(T*D) numpy call - no Python loop over T.
    delta = np.zeros_like(feat, dtype=np.float32)
    for n in range(1, win + 1):
        delta += n * (
            padded[win + n: win + n + n_frames]
            - padded[win - n: win - n + n_frames]
        )
    return (delta / denom).astype(np.float32)


def extract_features(signal: np.ndarray) -> np.ndarray:
    """
    Returns an N_FEATURES-D feature vector containing
        MFCC / delta / delta2 summary stats, spectral stats,
        coarse temporal MFCC trajectory bins, and energy contour stats.
    """
    signal = preprocess(signal)
    frames = _frame_signal(signal)

    spec = np.fft.rfft(frames, n=N_FFT, axis=1)
    power_spec = (np.abs(spec) ** 2).astype(np.float32)

    mel_energies = power_spec @ _MEL_FILTERBANK.T
    mel_energies = np.maximum(mel_energies, 1e-10)
    mel_repr = _pcen_transform(mel_energies) if USE_PCEN else np.log(mel_energies)

    mfcc = dct(mel_repr, type=2, axis=1, norm="ortho")[:, :N_MFCC].astype(np.float32)
    mfcc = mfcc * _LIFTER_VEC

    if USE_CMVN:
        mfcc_mean = mfcc.mean(axis=0, keepdims=True)
        mfcc_std = mfcc.std(axis=0, keepdims=True)
        mfcc_std[mfcc_std < 1e-8] = 1e-8
        mfcc = (mfcc - mfcc_mean) / mfcc_std

    delta = _compute_delta(mfcc).astype(np.float32)
    delta2 = _compute_delta(delta).astype(np.float32)

    # Spectral flux in mel-log domain captures transient dynamics.
    flux_seq = np.sqrt(
        np.sum(np.diff(mel_repr, axis=0, prepend=mel_repr[:1]) ** 2, axis=1)
    ).astype(np.float32)

    ps = power_spec + 1e-10
    ps_sum = np.sum(ps, axis=1)
    centroid = np.sum(ps * _FFT_FREQS[None, :], axis=1) / ps_sum
    bandwidth = np.sqrt(
        np.sum(ps * ((_FFT_FREQS[None, :] - centroid[:, None]) ** 2), axis=1) / ps_sum
    ) / (SAMPLE_RATE / 2.0)

    csum = np.cumsum(ps, axis=1)
    roll_thresh = 0.85 * ps_sum[:, None]
    roll_idx = np.argmax(csum >= roll_thresh, axis=1)
    rolloff = _FFT_FREQS[roll_idx] / (SAMPLE_RATE / 2.0)

    frame_energy = np.sum(power_spec, axis=1).astype(np.float32)

    energy_thr = float(np.percentile(frame_energy, 45.0))
    speech_mask = frame_energy > max(energy_thr, 1e-10)

    min_keep = max(5, int(0.25 * len(frame_energy)))
    if int(np.sum(speech_mask)) < min_keep:
        top_idx = np.argsort(frame_energy)[-min_keep:]
        speech_mask = np.zeros_like(frame_energy, dtype=bool)
        speech_mask[top_idx] = True

    mfcc_use = mfcc[speech_mask]
    delta_use = delta[speech_mask]
    delta2_use = delta2[speech_mask]
    flux_use = flux_seq[speech_mask]
    rolloff_use = rolloff[speech_mask]
    bandwidth_use = bandwidth[speech_mask]

    mfcc_head = mfcc_use[:, :TEMPORAL_MFCC_COEFFS]
    temporal_blocks = np.array_split(mfcc_head, TEMPORAL_BINS, axis=0)
    temporal_mfcc = []
    for blk in temporal_blocks:
        if len(blk) == 0:
            temporal_mfcc.append(np.zeros((TEMPORAL_MFCC_COEFFS,), dtype=np.float32))
        else:
            temporal_mfcc.append(blk.mean(axis=0).astype(np.float32))
    temporal_mfcc_vec = np.concatenate(temporal_mfcc, axis=0).astype(np.float32)

    delta_head = delta_use[:, :TEMPORAL_DELTA_COEFFS]
    temporal_delta_blocks = np.array_split(delta_head, TEMPORAL_BINS, axis=0)
    temporal_delta = []
    for blk in temporal_delta_blocks:
        if len(blk) == 0:
            temporal_delta.append(np.zeros((TEMPORAL_DELTA_COEFFS,), dtype=np.float32))
        else:
            temporal_delta.append(blk.mean(axis=0).astype(np.float32))
    temporal_delta_vec = np.concatenate(temporal_delta, axis=0).astype(np.float32)

    log_energy = np.log(ps_sum + 1e-10).astype(np.float32)
    log_energy_use = log_energy[speech_mask]
    if len(log_energy_use) >= 2:
        energy_slope = float((log_energy_use[-1] - log_energy_use[0]) / (len(log_energy_use) - 1))
    else:
        energy_slope = 0.0

    mfcc0 = mfcc_use[:, 0]
    delta0 = delta_use[:, 0]
    robust_extra = np.array(
        [
            float(np.min(mfcc0)),
            float(np.max(mfcc0)),
            float(np.min(delta0)),
            float(np.max(delta0)),
            float(np.percentile(flux_use, 90.0)),
            float(np.percentile(rolloff_use, 90.0)),
        ],
        dtype=np.float32,
    )

    # mel_repr shape is (n_frames, N_MEL_FILTERS)
    logmel_use = mel_repr[speech_mask]
    logmel_mean = logmel_use.mean(axis=0).astype(np.float32)
    logmel_std = logmel_use.std(axis=0).astype(np.float32)

    logmel_head = logmel_use[:, :TEMPORAL_LOGMEL_COEFFS]
    temporal_logmel_blocks = np.array_split(logmel_head, TEMPORAL_BINS, axis=0)
    temporal_logmel = []
    for blk in temporal_logmel_blocks:
        if len(blk) == 0:
            temporal_logmel.append(np.zeros((TEMPORAL_LOGMEL_COEFFS,), dtype=np.float32))
        else:
            temporal_logmel.append(blk.mean(axis=0).astype(np.float32))
    temporal_logmel_vec = np.concatenate(temporal_logmel, axis=0).astype(np.float32)

    # LFCC features: complementary linear-frequency cepstra for fricatives/stops.
    lfb = np.maximum(power_spec @ _LINEAR_FILTERBANK.T, 1e-10)
    log_lfb = np.log(lfb).astype(np.float32)
    lfcc = dct(log_lfb, type=2, axis=1, norm="ortho")[:, :N_LFCC].astype(np.float32)
    lfcc = lfcc * _LIFTER_VEC[:N_LFCC]
    lfcc_use = lfcc[speech_mask]

    if len(lfcc_use) >= 2:
        lfcc_mean = lfcc_use.mean(axis=0).astype(np.float32)
        lfcc_std = lfcc_use.std(axis=0).astype(np.float32)
    else:
        lfcc_mean = np.zeros(N_LFCC, dtype=np.float32)
        lfcc_std = np.zeros(N_LFCC, dtype=np.float32)
    lfcc_vec = np.concatenate([lfcc_mean, lfcc_std]).astype(np.float32)

    formant_vec = _extract_formants(signal).astype(np.float32)

    def _spectral_contrast_vec(pspec: np.ndarray, n_bands: int = 6,
                                alpha: float = 0.02) -> np.ndarray:
        """Returns mean and std of spectral contrast across speech frames."""
        freqs = _FFT_FREQS
        edges = np.linspace(0.0, float(SAMPLE_RATE) / 2.0, n_bands + 2)
        results = []
        for lo, hi in zip(edges[:-2], edges[2:]):
            mask = (freqs >= lo) & (freqs < hi)
            band = pspec[:, mask]
            if band.shape[1] < 2:
                results.extend([0.0, 0.0])
                continue
            n_top = max(1, int(alpha * band.shape[1]))
            sorted_b = np.sort(band, axis=1)
            peak = sorted_b[:, -n_top:].mean(axis=1)
            valley = sorted_b[:, :n_top].mean(axis=1)
            sc = np.log(peak + 1e-10) - np.log(valley + 1e-10)
            results.extend([float(sc.mean()), float(sc.std())])
        return np.array(results, dtype=np.float32)

    contrast_vec = _spectral_contrast_vec(power_spec[speech_mask])

    fv = np.concatenate(
        [
            mfcc_use.mean(axis=0),
            mfcc_use.std(axis=0),
            delta_use.mean(axis=0),
            delta_use.std(axis=0),
            delta2_use.mean(axis=0),
            delta2_use.std(axis=0),
            np.array([flux_use.mean(), flux_use.std()], dtype=np.float32),
            np.array([rolloff_use.mean(), rolloff_use.std()], dtype=np.float32),
            np.array([bandwidth_use.mean(), bandwidth_use.std()], dtype=np.float32),
            temporal_mfcc_vec,
            temporal_delta_vec,
            temporal_logmel_vec,
            np.array([log_energy_use.mean(), log_energy_use.std(), energy_slope], dtype=np.float32),
            robust_extra,
            logmel_mean,
            logmel_std,
            contrast_vec,
            lfcc_vec,
            formant_vec,
        ],
        axis=0,
    ).astype(np.float32)

    if fv.shape[0] != N_FEATURES:
        raise RuntimeError(f"Feature length mismatch: expected {N_FEATURES}, got {fv.shape[0]}")
    return fv


def _speed_perturb(signal: np.ndarray, speed_factor: float) -> np.ndarray:
    if len(signal) < 2 or speed_factor <= 0.0:
        return signal.astype(np.float32, copy=True)

    out_len = max(2, int(round(len(signal) / speed_factor)))
    x_old = np.linspace(0.0, 1.0, num=len(signal), endpoint=False, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, num=out_len, endpoint=False, dtype=np.float32)
    return np.interp(x_new, x_old, signal).astype(np.float32)


def _add_reverb(
    signal: np.ndarray,
    sr: int = SAMPLE_RATE,
    rt60: float = AUG_REVERB_RT60,
) -> np.ndarray:
    """Mild synthetic reverb using an exponential-decay impulse response."""
    decay_samples = int(rt60 * sr)
    if decay_samples < 2:
        return signal.astype(np.float32, copy=True)

    t = np.arange(decay_samples, dtype=np.float32) / float(sr)
    ir = np.exp(-6.9 * t / max(rt60, 1e-4)).astype(np.float32)
    ir /= (ir.sum() + 1e-10)

    reverbed = np.convolve(signal, ir, mode="full")[:len(signal)].astype(np.float32)
    orig_rms = max(_rms(signal), 1e-8)
    rev_rms = max(_rms(reverbed), 1e-8)
    reverbed = reverbed * (orig_rms / rev_rms)
    return np.clip(reverbed, -1.0, 1.0).astype(np.float32)


_BG_NOISE_BANK: Optional[List[np.ndarray]] = None


def _resample_to_sample_rate(signal: np.ndarray, src_sr: int, dst_sr: int = SAMPLE_RATE) -> np.ndarray:
    if src_sr == dst_sr or len(signal) < 2:
        return signal.astype(np.float32, copy=True)
    out_len = max(2, int(round(len(signal) * float(dst_sr) / float(src_sr))))
    x_old = np.linspace(0.0, 1.0, num=len(signal), endpoint=False, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, num=out_len, endpoint=False, dtype=np.float32)
    return np.interp(x_new, x_old, signal).astype(np.float32)


def _load_bg_noise_bank() -> List[np.ndarray]:
    global _BG_NOISE_BANK
    if _BG_NOISE_BANK is not None:
        return _BG_NOISE_BANK

    if not ENABLE_REAL_BG_NOISE:
        _BG_NOISE_BANK = []
        return _BG_NOISE_BANK

    noise_dir = os.path.join(DATASET_DIR, BG_NOISE_DIR_NAME)
    if not os.path.isdir(noise_dir):
        _BG_NOISE_BANK = []
        return _BG_NOISE_BANK

    bank: List[np.ndarray] = []
    for root, _, files in os.walk(noise_dir):
        for fn in sorted(files):
            if not fn.lower().endswith(".wav"):
                continue
            wav_path = os.path.join(root, fn)
            try:
                noise_sig, noise_sr = sf.read(wav_path)
            except Exception:
                continue

            if noise_sig.ndim > 1:
                noise_sig = noise_sig[:, 0]
            noise_sig = noise_sig.astype(np.float32)
            if noise_sr != SAMPLE_RATE:
                noise_sig = _resample_to_sample_rate(noise_sig, noise_sr, SAMPLE_RATE)
            if len(noise_sig) < FRAME_SIZE:
                continue

            noise_sig = noise_sig - float(np.mean(noise_sig))
            if _rms(noise_sig) < 1e-6:
                continue
            bank.append(np.clip(noise_sig, -1.0, 1.0).astype(np.float32))

    _BG_NOISE_BANK = bank
    return _BG_NOISE_BANK


def _sample_bg_noise_segment(n_samples: int, rng: np.random.Generator) -> Optional[np.ndarray]:
    bank = _load_bg_noise_bank()
    if not bank:
        return None

    clip = bank[int(rng.integers(0, len(bank)))]
    if len(clip) >= n_samples:
        max_start = len(clip) - n_samples
        start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
        seg = clip[start:start + n_samples]
    else:
        reps = int(np.ceil(float(n_samples) / float(len(clip))))
        seg = np.tile(clip, reps)[:n_samples]

    seg = seg.astype(np.float32)
    seg = seg - float(np.mean(seg))
    return seg


def _mix_signal_with_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    sig_rms = max(_rms(signal), 1e-6)
    noise_rms = max(_rms(noise), 1e-6)
    target_noise_rms = sig_rms / (10.0 ** (snr_db / 20.0))
    noise_scale = target_noise_rms / noise_rms
    mixed = signal + (noise * noise_scale)
    return np.clip(mixed, -1.0, 1.0).astype(np.float32)


# Must be at module level (not nested) for ProcessPoolExecutor pickling.
# Each worker process imports this module fresh; all globals (_MEL_FILTERBANK,
# _LPF_B, _LPF_A, _HAMMING_WINDOW, etc.) are re-created on import - safe.
def _augment_worker(args: tuple) -> tuple:
    """
    (file_path, label, vad_threshold, seed_offset)
    -> (List[np.ndarray], List[int])

    Reads one WAV file, applies all 8 augmentation variants, returns
    feature vectors. Exceptions are caught per-variant; bad files return
    empty lists (they are silently skipped, same as the serial version).
    """
    file_path, label, vad_threshold, seed_offset = args
    rng = np.random.default_rng(RANDOM_SEED + 101 + seed_offset)

    x_out: List[np.ndarray] = []
    y_out: List[int] = []

    try:
        signal, sr = sf.read(file_path)
    except Exception:
        return x_out, y_out
    if sr != SAMPLE_RATE:
        return x_out, y_out
    if signal.ndim > 1:
        signal = signal[:, 0]
    signal = signal.astype(np.float32)
    if not has_speech(signal, vad_threshold):
        return x_out, y_out

    is_hard_class = int(label) in _AUG_HARD_LABELS
    speed_factors = AUG_SPEED_FACTORS + (AUG_HARD_EXTRA_SPEED_FACTORS if is_hard_class else ())
    n_noise = AUG_NOISE_COPIES + (AUG_HARD_EXTRA_NOISE_COPIES if is_hard_class else 0)
    n_reverb = AUG_REVERB_COPIES + (AUG_HARD_EXTRA_REVERB_COPIES if is_hard_class else 0)
    shift_values = [-80, 80]
    if is_hard_class:
        shift_values.extend([int(v) for v in AUG_HARD_EXTRA_SHIFT_MS])
    shift_values = list(dict.fromkeys(shift_values))

    # Speed perturbation (4 variants: 0.85, 0.92, 1.08, 1.15)
    for speed in speed_factors:
        try:
            sig_s = _speed_perturb(signal, float(speed))
            if has_speech(sig_s, vad_threshold * 0.8):
                x_out.append(extract_features(sig_s))
                y_out.append(label)
        except Exception:
            continue

    # Noise branch: blend clean/noisy examples at a controlled ratio.
    rms = max(_rms(signal), 1e-5)
    for _ in range(max(0, n_noise)):
        try:
            make_noisy = (rng.random() < float(NOISE_MIX_NOISY_RATIO))
            sig_n: Optional[np.ndarray]

            if make_noisy:
                sig_n = None
                if ENABLE_REAL_BG_NOISE:
                    bg_seg = _sample_bg_noise_segment(len(signal), rng)
                    if bg_seg is not None:
                        snr_db = float(rng.uniform(BG_NOISE_SNR_DB_RANGE[0], BG_NOISE_SNR_DB_RANGE[1]))
                        sig_n = _mix_signal_with_snr(signal, bg_seg, snr_db)
                if sig_n is None:
                    noise_std = AUG_NOISE_RMS_RATIO * rms * float(rng.uniform(0.8, 1.2))
                    noise = rng.normal(0.0, noise_std, size=signal.shape).astype(np.float32)
                    sig_n = np.clip(signal + noise, -1.0, 1.0)
            else:
                sig_n = signal.copy()

            if has_speech(sig_n, vad_threshold * 0.8):
                x_out.append(extract_features(sig_n))
                y_out.append(label)
        except Exception:
            continue

    # Synthetic reverb augmentation (class-targeted copies for hard words)
    for _ in range(max(0, n_reverb)):
        try:
            rt60 = float(AUG_REVERB_RT60 * rng.uniform(0.85, 1.15))
            sig_r = _add_reverb(signal, sr=SAMPLE_RATE, rt60=rt60)
            if has_speech(sig_r, vad_threshold * 0.8):
                x_out.append(extract_features(sig_r))
                y_out.append(label)
        except Exception:
            continue

    # Time-shift with stronger offsets for hard words.
    for shift_ms in shift_values:
        shift_n = int(SAMPLE_RATE * abs(shift_ms) / 1000)
        if shift_ms < 0:
            shifted = np.concatenate([signal[shift_n:],
                                      np.zeros(shift_n, np.float32)])
        else:
            shifted = np.concatenate([np.zeros(shift_n, np.float32),
                                      signal[:len(signal) - shift_n]])
        try:
            if has_speech(shifted, vad_threshold * 0.8):
                x_out.append(extract_features(shifted))
                y_out.append(label)
        except Exception:
            continue

    return x_out, y_out


def _extract_worker(args: tuple) -> tuple:
    """
    (file_path, label, vad_threshold) -> (np.ndarray | None, int)
    Reads one WAV file and extracts features. Returns None if extraction fails.
    """
    file_path, label, vad_threshold = args
    try:
        signal, sr = sf.read(file_path)
    except Exception:
        return None, label
    if sr != SAMPLE_RATE:
        return None, label
    if signal.ndim > 1:
        signal = signal[:, 0]
    signal = signal.astype(np.float32)
    if not has_speech(signal, vad_threshold):
        return None, label
    try:
        fv = extract_features(signal)
        return fv, label
    except Exception:
        return None, label


def _base_cache_key(entries: Sequence[SampleEntry], vad_threshold: float) -> str:
    """Stable hash for the base (non-augmented) feature matrix."""
    h = _hashlib.md5()
    for e in entries:
        h.update(e.file_path.encode())
        try:
            h.update(str(os.path.getmtime(e.file_path)).encode())
        except OSError:
            pass
    config = (
        f"BASE|{vad_threshold}|{N_FEATURES}|{RANDOM_SEED}|"
        f"{FRAME_SIZE}|{N_MFCC}|{N_MEL_FILTERS}|"
        f"PCEN={USE_PCEN}_{PCEN_GAIN}_{PCEN_BIAS}_{PCEN_POWER}_{PCEN_TIME_CONSTANT}_{PCEN_EPS}_{PCEN_INPUT_SCALE}|"
        f"FEAT_VERSION={FEAT_VERSION}"
    )
    h.update(config.encode())
    return h.hexdigest()[:16]


def _aug_cache_key(train_entries: Sequence[SampleEntry],
                   vad_threshold: float) -> str:
    """Stable hash of all inputs that determine the augmented feature matrix."""
    h = _hashlib.md5()
    for e in train_entries:
        h.update(e.file_path.encode())
        try:
            h.update(str(os.path.getmtime(e.file_path)).encode())
        except OSError:
            pass
    if ENABLE_REAL_BG_NOISE:
        noise_dir = os.path.join(DATASET_DIR, BG_NOISE_DIR_NAME)
        if os.path.isdir(noise_dir):
            for root, _, files in os.walk(noise_dir):
                for fn in sorted(files):
                    if not fn.lower().endswith(".wav"):
                        continue
                    npth = os.path.join(root, fn)
                    h.update(npth.encode())
                    try:
                        h.update(str(os.path.getmtime(npth)).encode())
                    except OSError:
                        pass
    config_str = (f"{FEATURE_CACHE_VERSION}|{vad_threshold}|{AUG_SPEED_FACTORS}|{AUG_NOISE_COPIES}|"
                  f"{AUG_NOISE_RMS_RATIO}|{AUG_REVERB_COPIES}|{AUG_REVERB_RT60}|"
                  f"REALBG={ENABLE_REAL_BG_NOISE}|{BG_NOISE_DIR_NAME}|SNR={BG_NOISE_SNR_DB_RANGE}|"
                  f"NOISY_RATIO={NOISE_MIX_NOISY_RATIO}|"
                  f"HARD={AUG_HARD_CLASS_WORDS}|{AUG_HARD_EXTRA_SPEED_FACTORS}|"
                  f"{AUG_HARD_EXTRA_NOISE_COPIES}|{AUG_HARD_EXTRA_REVERB_COPIES}|{AUG_HARD_EXTRA_SHIFT_MS}|"
                  f"{N_FEATURES}|{RANDOM_SEED}|"
                  f"{FRAME_SIZE}|{N_MFCC}|{N_MEL_FILTERS}|"
                  f"{TEMPORAL_BINS}|{TEMPORAL_MFCC_COEFFS}|"
                  f"{TEMPORAL_DELTA_COEFFS}|{TEMPORAL_LOGMEL_COEFFS}|"
                  f"{USE_CMVN}|{USE_LIBROSA_MEL}|{CEPSTRAL_LIFTER}|{PRE_EMPHASIS}|"
                  f"PCEN_{USE_PCEN}_{PCEN_GAIN}_{PCEN_BIAS}_{PCEN_POWER}_{PCEN_TIME_CONSTANT}_{PCEN_EPS}_{PCEN_INPUT_SCALE}|"
                  f"LFCC_{N_LFCC}_{N_LFCC_FILTERS}|FORMANT_{FEATURES_FORMANT_DIMS}|"
                  f"FEAT_VERSION_{FEAT_VERSION}")
    h.update(config_str.encode())
    try:
        feature_src = (
            inspect.getsource(extract_features)
            + inspect.getsource(_compute_delta)
            + inspect.getsource(_augment_worker)
        )
    except (OSError, TypeError):
        feature_src = "feature_source_unavailable"
    h.update(feature_src.encode("utf-8"))
    return h.hexdigest()[:16]


def _build_train_augmentations(
    train_entries: Sequence[SampleEntry],
    vad_threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parallel augmentation using ProcessPoolExecutor.
    Each entry is processed independently in a worker process,
    bypassing the GIL for true CPU parallelism.

    Worker count: CPU_count - 1, capped at 8 to avoid RAM pressure.
    chunksize=8: batches work items to reduce IPC overhead.
    """
    import concurrent.futures
    import os as _os

    n_workers = max(1, min(8, (_os.cpu_count() or 1) - 1))

    args_list = [
        (e.file_path, e.label, vad_threshold, i)
        for i, e in enumerate(train_entries)
    ]

    x_aug: List[np.ndarray] = []
    y_aug: List[int] = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
        for fvs, labels in pool.map(_augment_worker, args_list, chunksize=8):
            x_aug.extend(fvs)
            y_aug.extend(labels)

    if not x_aug:
        return (
            np.zeros((0, N_FEATURES), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
        )
    return np.vstack(x_aug).astype(np.float32), np.array(y_aug, dtype=np.int32)


# =============================================================================
# VAD helpers
# =============================================================================

def _rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(signal.astype(np.float64) ** 2)))


def has_speech(signal: np.ndarray, threshold: float) -> bool:
    return _rms(signal) > threshold


def measure_noise_floor() -> float:
    sd = _require_sounddevice()
    n_samples = int(SAMPLE_RATE * NOISE_MEASURE_MS / 1000)
    print(f"  [VAD] Measuring ambient noise ({NOISE_MEASURE_MS} ms) - stay quiet ...")
    try:
        rec = sd.rec(n_samples, samplerate=SAMPLE_RATE, channels=1, dtype="float32")
        sd.wait()
        noise_rms = _rms(rec.flatten())
        print(f"  [VAD] Noise floor RMS = {noise_rms:.5f}")
        return max(noise_rms, 1e-6)
    except Exception as exc:
        print(f"  [VAD] Noise measurement failed ({exc}), using fallback threshold.")
        return VAD_LIVE_FALLBACK / NOISE_MULTIPLIER


# =============================================================================
# Dataset and split helpers
# =============================================================================

def _normalize_rel_path(rel_path: str) -> str:
    return rel_path.replace("\\", "/").strip().lower()


def _speaker_id_from_name(filename: str) -> str:
    base = os.path.basename(filename)
    base = os.path.splitext(base)[0]
    m = re.match(r"(.+?)_nohash_", base)
    if m:
        return m.group(1)
    if "_" in base:
        return base.split("_", 1)[0]
    return base


def collect_entries(data_dir: str) -> List[SampleEntry]:
    entries: List[SampleEntry] = []

    for word in WORDS:
        word_dir = os.path.join(data_dir, word)
        if not os.path.isdir(word_dir):
            continue

        files = sorted(f for f in os.listdir(word_dir) if f.lower().endswith(".wav"))
        for fname in files:
            rel = _normalize_rel_path(os.path.join(word, fname))
            entries.append(
                SampleEntry(
                    word=word,
                    label=WORD_TO_INDEX[word],
                    file_path=os.path.join(word_dir, fname),
                    rel_path=rel,
                    speaker_id=_speaker_id_from_name(fname),
                )
            )

    entries.sort(key=lambda e: e.rel_path)
    return entries


def _read_official_split_file(path: str) -> set:
    if not os.path.exists(path):
        return set()
    out = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.add(_normalize_rel_path(line))
    return out


def build_split_indices(entries: Sequence[SampleEntry], data_dir: str) -> Dict[str, np.ndarray]:
    test_path = os.path.join(data_dir, "testing_list.txt")
    val_path = os.path.join(data_dir, "validation_list.txt")
    if USE_OFFICIAL_SPLIT_LISTS:
        test_set = _read_official_split_file(test_path)
        val_set = _read_official_split_file(val_path)
    else:
        test_set = set()
        val_set = set()

    if len(test_set) > 0:
        if len(val_set) > 0:
            val_msg = f"official validation list at {val_path}"
        else:
            val_msg = f"missing/empty validation list at {val_path}; will carve 10% from train"
        print(
            "\n  [SPLIT] Using OFFICIAL split lists: "
            f"test={test_path}, val={val_msg}"
        )
    else:
        print(
            "\n  [SPLIT] Official testing list missing/empty "
            f"(expected {test_path}); using deterministic stratified fallback"
        )

    idx_all = np.arange(len(entries), dtype=np.int32)
    y_all = np.array([e.label for e in entries], dtype=np.int32)

    if test_set:
        test_idx = [i for i, e in enumerate(entries) if e.rel_path in test_set]
        val_idx = [i for i, e in enumerate(entries) if e.rel_path in val_set]
        train_idx = [
            i for i, e in enumerate(entries)
            if e.rel_path not in test_set and e.rel_path not in val_set
        ]

        if len(train_idx) > 0 and len(test_idx) > 0:
            train_idx = np.array(train_idx, dtype=np.int32)
            val_idx = np.array(val_idx, dtype=np.int32)
            test_idx = np.array(test_idx, dtype=np.int32)

            # If official validation list is missing, carve out 10% stratified from training.
            if len(val_idx) == 0:
                splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=RANDOM_SEED)
                y_train = y_all[train_idx]
                train_sub, val_sub = next(splitter.split(train_idx, y_train))
                val_idx = train_idx[val_sub]
                train_idx = train_idx[train_sub]

            return {"train": train_idx, "val": val_idx, "test": test_idx}

    # Fallback split: deterministic stratified shuffle.
    splitter_test = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
    train_val_loc, test_loc = next(splitter_test.split(idx_all, y_all))

    train_val_idx = idx_all[train_val_loc]
    test_idx = idx_all[test_loc]

    splitter_val = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=RANDOM_SEED)
    y_train_val = y_all[train_val_idx]
    train_loc, val_loc = next(splitter_val.split(train_val_idx, y_train_val))

    train_idx = train_val_idx[train_loc]
    val_idx = train_val_idx[val_loc]

    return {"train": train_idx, "val": val_idx, "test": test_idx}


def extract_feature_list(entries: Sequence[SampleEntry], vad_threshold: float) -> Tuple[List[Optional[np.ndarray]], int]:
    os.makedirs(TEMPLATES_DIR, exist_ok=True)

    base_key = _base_cache_key(entries, vad_threshold)
    base_cache_path = os.path.join(TEMPLATES_DIR, f"base_cache_{base_key}.npz")

    if os.path.exists(base_cache_path):
        try:
            base_npz = np.load(base_cache_path, allow_pickle=True)
            features_arr = base_npz["features"]
            if len(features_arr) != len(entries):
                raise ValueError(
                    f"base cache size mismatch ({len(features_arr)} != {len(entries)})"
                )

            features: List[Optional[np.ndarray]] = []
            skipped = 0
            for fv in features_arr:
                if fv is None:
                    features.append(None)
                    skipped += 1
                else:
                    features.append(np.asarray(fv, dtype=np.float32))

            print(f"  [CACHE] Loaded base features from {base_cache_path}")
            return features, skipped
        except Exception as exc:
            try:
                os.remove(base_cache_path)
            except OSError:
                pass
            print(f"  [CACHE] Base cache invalid ({exc}); rebuilding")

    import concurrent.futures

    n_workers = max(1, min(8, (os.cpu_count() or 1) - 1))
    args_list = [(e.file_path, e.label, vad_threshold) for e in entries]
    features = [None] * len(entries)
    skipped = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
        for i, (fv, _lbl) in enumerate(pool.map(_extract_worker, args_list, chunksize=16)):
            if fv is None:
                skipped += 1
            else:
                features[i] = np.asarray(fv, dtype=np.float32)
            if _IS_MAIN_PROCESS and (i + 1) % 2000 == 0:
                print(f"  [BASE] extracted {i + 1}/{len(args_list)} files...")

    feat_obj = np.empty(len(features), dtype=object)
    for i, fv in enumerate(features):
        feat_obj[i] = fv

    try:
        np.savez_compressed(base_cache_path, features=feat_obj)
        print(f"  [CACHE] Saved base features to {base_cache_path}")
    except Exception as exc:
        print(f"  [CACHE] Could not save base cache: {exc}")

    return features, skipped


def prepare_dataset(data_dir: str, vad_threshold: float = VAD_TRAIN_THRESHOLD) -> PreparedDataset:
    entries = collect_entries(data_dir)
    if not entries:
        raise RuntimeError(
            "No dataset files found. Expected folders: " + ", ".join(WORDS)
        )

    split_indices = build_split_indices(entries, data_dir)
    features, skipped = extract_feature_list(entries, vad_threshold)
    return PreparedDataset(entries=entries, split_indices=split_indices, features=features, skipped=skipped)


def _subset_from_indices(
    prepared: PreparedDataset,
    indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[SampleEntry], int]:
    x_list: List[np.ndarray] = []
    y_list: List[int] = []
    kept_entries: List[SampleEntry] = []
    skipped = 0

    for idx in indices:
        fv = prepared.features[int(idx)]
        if fv is None:
            skipped += 1
            continue
        x_list.append(fv)
        y_list.append(prepared.entries[int(idx)].label)
        kept_entries.append(prepared.entries[int(idx)])

    if x_list:
        X = np.vstack(x_list).astype(np.float32)
        y = np.array(y_list, dtype=np.int32)
    else:
        X = np.zeros((0, N_FEATURES), dtype=np.float32)
        y = np.zeros((0,), dtype=np.int32)

    return X, y, kept_entries, skipped


# =============================================================================
# Normalizer and LDA projector
# =============================================================================

class FeatureNormalizer:
    def __init__(self):
        self.mean = np.zeros(N_FEATURES, dtype=np.float32)
        self.std = np.ones(N_FEATURES, dtype=np.float32)

    def fit(self, matrix: np.ndarray) -> "FeatureNormalizer":
        self.mean = matrix.mean(axis=0).astype(np.float32)
        self.std = matrix.std(axis=0).astype(np.float32)
        self.std[self.std == 0.0] = 1.0

        os.makedirs(TEMPLATES_DIR, exist_ok=True)
        np.save(os.path.join(TEMPLATES_DIR, "norm_mean.npy"), self.mean)
        np.save(os.path.join(TEMPLATES_DIR, "norm_std.npy"), self.std)
        return self

    def load(self) -> "FeatureNormalizer":
        self.mean = np.load(os.path.join(TEMPLATES_DIR, "norm_mean.npy")).astype(np.float32)
        self.std = np.load(os.path.join(TEMPLATES_DIR, "norm_std.npy")).astype(np.float32)
        self.std[self.std == 0.0] = 1.0
        return self

    def transform(self, fv: np.ndarray) -> np.ndarray:
        return ((fv - self.mean) / self.std).astype(np.float32)

    def transform_batch(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean) / self.std).astype(np.float32)


class FeatureWarper:
    """
    Per-dimension histogram equalization (Gaussian warping).
    Maps each feature dimension's empirical CDF to N(0,1).
    Applied after z-normalization to improve class covariance conditioning.
    """
    def __init__(self, n_quantiles: int = 1000):
        self.n_quantiles = n_quantiles
        self.quantile_vals: Optional[np.ndarray] = None  # (n_quantiles, n_features)
        self.target_vals: np.ndarray = np.array([])

    def fit(self, X: np.ndarray) -> "FeatureWarper":
        from scipy.stats import norm

        probs = np.linspace(0.01, 0.99, self.n_quantiles)
        self.target_vals = norm.ppf(probs).astype(np.float32)
        self.quantile_vals = np.quantile(X, probs, axis=0).astype(np.float32)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.quantile_vals is None:
            return X.astype(np.float32)

        X_2d = X if X.ndim == 2 else X[None, :]
        X_out = np.empty_like(X_2d, dtype=np.float32)
        for j in range(X_2d.shape[1]):
            X_out[:, j] = np.interp(
                X_2d[:, j],
                self.quantile_vals[:, j],
                self.target_vals,
            )

        if X.ndim == 1:
            return X_out[0]
        return X_out

    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            quantile_vals=self.quantile_vals,
            target_vals=self.target_vals,
            n_quantiles=np.array([self.n_quantiles], dtype=np.int32),
        )

    @classmethod
    def load(cls, path: str) -> "FeatureWarper":
        npz = np.load(path)
        nq_arr = npz.get("n_quantiles")
        n_quantiles = int(nq_arr[0]) if nq_arr is not None else 1000
        obj = cls(n_quantiles=n_quantiles)
        obj.quantile_vals = npz["quantile_vals"].astype(np.float32)
        obj.target_vals = npz["target_vals"].astype(np.float32)
        return obj


class LDAProjector:
    def __init__(self):
        self.W: Optional[np.ndarray] = None
        self.xbar: Optional[np.ndarray] = None

    def fit(self, X_norm: np.ndarray, y: np.ndarray) -> "LDAProjector":
        from sklearn.decomposition import PCA

        n_samples, n_feats = X_norm.shape
        # Cap PCA components: enough to explain variance but keep covariance
        # matrix well-conditioned for LDA. 60 is the practical sweet spot
        # for ~177 features and ~12k samples.
        n_pca = min(80, n_feats, n_samples - 1)

        pca = PCA(n_components=n_pca, whiten=False, random_state=RANDOM_SEED)
        X_pca = pca.fit_transform(X_norm).astype(np.float32)

        lda = LinearDiscriminantAnalysis(
            n_components=LDA_DIMS,
            solver="eigen",
            shrinkage="auto",
        )
        try:
            lda.fit(X_pca, y)
            Z = lda.transform(X_pca).astype(np.float32)
        except Exception:
            lda = LinearDiscriminantAnalysis(n_components=LDA_DIMS, solver="svd")
            lda.fit(X_pca, y)
            Z = lda.transform(X_pca).astype(np.float32)

        # Collapse PCA + LDA into one affine map X_norm -> Z via lstsq.
        # This is the only form that classify_snippet.c can use.
        # Z ~= X_norm @ W - xbar
        # Subsample for lstsq: 10k rows is sufficient to fit the affine map.
        # The PCA+LDA projection Z is already computed on the full matrix above;
        # we only need a representative sample to recover W and xbar via regression.
        LSTSQ_MAX = 10_000
        if n_samples > LSTSQ_MAX:
            rng_ls = np.random.default_rng(RANDOM_SEED + 7)
            sel = rng_ls.choice(n_samples, size=LSTSQ_MAX, replace=False)
            A_ls = np.hstack([
                X_norm[sel].astype(np.float32),
                -np.ones((LSTSQ_MAX, 1), dtype=np.float32),
            ])
            Z_ls = Z[sel]
        else:
            A_ls = np.hstack([
                X_norm.astype(np.float32),
                -np.ones((n_samples, 1), dtype=np.float32),
            ])
            Z_ls = Z
        B, _, _, _ = np.linalg.lstsq(A_ls, Z_ls, rcond=None)
        self.W = B[:-1, :].astype(np.float32)   # (N_FEATURES, LDA_DIMS)
        self.xbar = B[-1,  :].astype(np.float32)   # (LDA_DIMS,)

        os.makedirs(TEMPLATES_DIR, exist_ok=True)
        np.save(os.path.join(TEMPLATES_DIR, "lda_W.npy"), self.W)
        np.save(os.path.join(TEMPLATES_DIR, "lda_xbar.npy"), self.xbar)
        return self

    def load(self) -> "LDAProjector":
        self.W = np.load(os.path.join(TEMPLATES_DIR, "lda_W.npy")).astype(np.float32)
        self.xbar = np.load(os.path.join(TEMPLATES_DIR, "lda_xbar.npy")).astype(np.float32)
        return self

    def transform(self, fv_norm: np.ndarray) -> np.ndarray:
        if self.W is None or self.xbar is None:
            raise RuntimeError("LDA projector not loaded")
        return ((fv_norm @ self.W) - self.xbar).astype(np.float32)

    def transform_batch(self, X_norm: np.ndarray) -> np.ndarray:
        if self.W is None or self.xbar is None:
            raise RuntimeError("LDA projector not loaded")
        return ((X_norm @ self.W) - self.xbar).astype(np.float32)


# =============================================================================
# Classification helpers
# =============================================================================

def _euclidean(v1: np.ndarray, v2: np.ndarray) -> float:
    d = v1 - v2
    return float(np.sqrt(np.dot(d, d)))


def _templates_to_stack(lda_templates_dict: Dict[str, np.ndarray]) -> np.ndarray:
    return np.stack([lda_templates_dict[w] for w in WORDS], axis=0).astype(np.float32)


def classify_from_lda(
    fv_lda: np.ndarray,
    templates_stack: np.ndarray,
    threshold: float,
) -> Tuple[Optional[str], float, Dict[str, float]]:
    diff = templates_stack - fv_lda[None, None, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))   # (N_WORDS, K_TEMPLATES)
    per_word = np.min(dists, axis=1)               # (N_WORDS,)

    best_idx = int(np.argmin(per_word))
    best_dist = float(per_word[best_idx])

    dist_dict = {WORDS[i]: float(per_word[i]) for i in range(N_WORDS)}
    if best_dist > threshold:
        return None, best_dist, dist_dict
    return WORDS[best_idx], best_dist, dist_dict


def run_pipeline(
    signal: np.ndarray,
    normalizer: FeatureNormalizer,
    projector: LDAProjector,
    templates_stack: np.ndarray,
    threshold: float,
    warper: Optional[FeatureWarper] = None,
) -> Tuple[Optional[str], float, Dict[str, float]]:
    fv = extract_features(signal)
    fv_norm = normalizer.transform(fv)
    if warper is not None:
        fv_norm = warper.transform(fv_norm)
    fv_lda = projector.transform(fv_norm)
    return classify_from_lda(fv_lda, templates_stack, threshold)


# =============================================================================
# Training and evaluation core
# =============================================================================

def _select_templates_svm_guided(
    X_train_lda: np.ndarray,
    y_train: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Use LinearSVC on a stratified subsample of 7-D LDA projections
    to identify easy vs hard samples per class, then place K_TEMPLATES
    templates to cover both regions.
    LinearSVC is O(n) via LIBLINEAR - runs in seconds vs minutes for RBF.
    """
    from sklearn.svm import LinearSVC

    # Stratified subsample: at most 750 samples per class = 6000 total.
    # Enough to get stable decision boundaries in 7-D space.
    SVM_MAX_PER_CLASS = 750
    sel_idx: List[int] = []
    rng_svm = np.random.default_rng(RANDOM_SEED + 13)
    for wi in range(N_WORDS):
        class_idx = np.where(y_train == wi)[0]
        if len(class_idx) > SVM_MAX_PER_CLASS:
            class_idx = rng_svm.choice(
                class_idx, size=SVM_MAX_PER_CLASS, replace=False
            )
        sel_idx.extend(class_idx.tolist())

    sel_idx_arr = np.array(sel_idx, dtype=np.int32)
    X_sub = X_train_lda[sel_idx_arr]
    y_sub = y_train[sel_idx_arr]

    svm = LinearSVC(C=1.0, max_iter=2000, random_state=RANDOM_SEED)
    svm.fit(X_sub, y_sub)

    # Predict on the FULL training set using the subsample-trained SVM
    # so template coverage reflects the full distribution.
    svm_pred = svm.predict(X_train_lda)

    result: Dict[str, np.ndarray] = {}

    for wi, word in enumerate(WORDS):
        mask = (y_train == wi)
        X_word = X_train_lda[mask]
        pred_word = svm_pred[mask]

        if len(X_word) == 0:
            result[word] = np.zeros((K_TEMPLATES, LDA_DIMS), np.float32)
            continue

        easy_mask = (pred_word == wi)
        X_easy = X_word[easy_mask]
        X_hard = X_word[~easy_mask]

        # 70% of templates from easy (cluster-representative) region
        # 30% from hard (boundary) region for better margin coverage
        n_hard = max(1, int(K_TEMPLATES * 0.30))
        n_easy = K_TEMPLATES - n_hard

        selected: List[np.ndarray] = []

        # Easy templates via k-means medoids
        for X_pool, n_want in [(X_easy, n_easy), (X_hard, n_hard)]:
            if len(X_pool) == 0:
                # fall back to full set if one region is empty
                X_pool = X_word
            k = min(n_want, len(X_pool))
            km = KMeans(n_clusters=k, n_init=15, random_state=RANDOM_SEED)
            labels = km.fit_predict(X_pool)
            for ci in range(k):
                pts = X_pool[labels == ci]
                if len(pts) == 0:
                    pts = X_pool
                idx = int(np.argmin(
                    np.linalg.norm(pts - km.cluster_centers_[ci], axis=1)
                ))
                selected.append(pts[idx])

        # Pad to K_TEMPLATES if needed
        while len(selected) < K_TEMPLATES:
            selected.append(selected[-1])

        result[word] = np.vstack(selected[:K_TEMPLATES]).astype(np.float32)

    return result


def _refine_templates_lvq(
    X_train_lda: np.ndarray,
    y_train: np.ndarray,
    lda_templates_dict: Dict[str, np.ndarray],
    epochs: int = LVQ_EPOCHS,
    lr0: float = LVQ_LR0,
) -> Dict[str, np.ndarray]:
    """
    Refine class prototypes with a lightweight LVQ-style update.
    This preserves the fixed AVR inference path (1-NN over templates)
    while making templates more discriminative.
    """
    if len(X_train_lda) == 0:
        return lda_templates_dict

    rng = np.random.default_rng(RANDOM_SEED)

    # Keep LVQ runtime bounded when augmentation significantly enlarges training data.
    if len(X_train_lda) > LVQ_MAX_SAMPLES:
        sel = rng.permutation(len(X_train_lda))[:LVQ_MAX_SAMPLES]
        X_work = X_train_lda[sel]
        y_work = y_train[sel]
    else:
        X_work = X_train_lda
        y_work = y_train

    class_centers = np.zeros((N_WORDS, LDA_DIMS), dtype=np.float32)
    class_thr = np.full(N_WORDS, np.inf, dtype=np.float32)
    for wi in range(N_WORDS):
        Xi = X_work[y_work == wi]
        if len(Xi) == 0:
            continue
        center = Xi.mean(axis=0).astype(np.float32)
        class_centers[wi] = center
        dist = np.sqrt(np.sum((Xi - center[None, :]) ** 2, axis=1))
        class_thr[wi] = float(np.percentile(dist, LVQ_OUTLIER_PERCENTILE))

    protos = []
    proto_labels = []
    for wi, word in enumerate(WORDS):
        for t in lda_templates_dict[word]:
            protos.append(t.astype(np.float32))
            proto_labels.append(wi)

    proto = np.vstack(protos).astype(np.float32)
    proto_labels_arr = np.array(proto_labels, dtype=np.int32)

    for epoch in range(max(1, epochs)):
        lr = lr0 * (1.0 - (epoch / max(1, epochs)))
        lr = max(lr, 0.01)
        order = rng.permutation(len(X_work))

        for idx in order:
            x = X_work[int(idx)]
            y = int(y_work[int(idx)])

            # Skip class outliers to keep LVQ updates conservative.
            d_center = float(np.linalg.norm(x - class_centers[y]))
            if d_center > float(class_thr[y]):
                continue

            d2 = np.sum((proto - x) ** 2, axis=1)
            bmu = int(np.argmin(d2))

            correct_mask = (proto_labels_arr == y)
            wrong_mask = ~correct_mask

            d_correct = float(np.min(d2[correct_mask])) if correct_mask.any() else np.inf
            d_wrong = float(np.min(d2[wrong_mask])) if wrong_mask.any() else np.inf

            MARGIN = 0.30

            # Skip update if already well-separated (avoid over-tuning)
            if d_correct < d_wrong - MARGIN:
                continue

            # Otherwise apply the standard LVQ-2 update:
            if int(proto_labels_arr[bmu]) == y:
                proto[bmu] += lr * (x - proto[bmu])
            else:
                proto[bmu] -= lr * (x - proto[bmu])
                same = np.where(proto_labels_arr == y)[0]
                if len(same) > 0:
                    same_idx = int(same[np.argmin(d2[same])])
                    proto[same_idx] += lr * (x - proto[same_idx])

    refined: Dict[str, np.ndarray] = {}
    offset = 0
    for word in WORDS:
        refined[word] = proto[offset:offset + K_TEMPLATES].astype(np.float32)
        offset += K_TEMPLATES

    return refined


def _optimize_projection_and_templates(
    X_train_norm: np.ndarray,
    y_train: np.ndarray,
    X_val_norm: np.ndarray,
    y_val: np.ndarray,
    projector: LDAProjector,
    lda_templates_dict: Dict[str, np.ndarray],
    epochs: int = OPT_EPOCHS,
    batch_size: int = 256,
    lr: float = OPT_LR,
    tau: float = OPT_TAU,
    reg: float = OPT_REG,
    shuffle_seed: Optional[int] = None,
) -> Tuple[LDAProjector, Dict[str, np.ndarray]]:
    """
    GPU-accelerated Adam optimizer for joint projection+template training.
    """
    if projector.W is None or projector.xbar is None:
        return projector, lda_templates_dict

    def _scalar_to_float(v) -> float:
        if hasattr(v, "item"):
            return float(v.item())
        return float(v)

    # Move data to GPU when CuPy is available; otherwise stay on CPU.
    if _CUPY_AVAILABLE:
        xp_lib = cp
        print("  [OPT-GPU] using CuPy backend")
        X_tr = cp.asarray(X_train_norm, dtype=cp.float32)
        y_tr = cp.asarray(y_train, dtype=cp.int32)
        X_vl = cp.asarray(X_val_norm, dtype=cp.float32) if len(X_val_norm) > 0 else None
        y_vl = cp.asarray(y_val, dtype=cp.int32) if len(y_val) > 0 else None
        W = cp.asarray(projector.W, dtype=cp.float32)
        b = cp.asarray(projector.xbar, dtype=cp.float32)
        proto = cp.asarray(_templates_to_stack(lda_templates_dict), dtype=cp.float32)
    else:
        xp_lib = np
        print("  [OPT] using NumPy CPU backend")
        X_tr = X_train_norm.astype(np.float32)
        y_tr = y_train.astype(np.int32)
        X_vl = X_val_norm.astype(np.float32) if len(X_val_norm) > 0 else None
        y_vl = y_val.astype(np.int32) if len(y_val) > 0 else None
        W = projector.W.copy().astype(np.float32)
        b = projector.xbar.copy().astype(np.float32)
        proto = _templates_to_stack(lda_templates_dict).astype(np.float32)

    n_train = len(X_tr)
    if n_train == 0:
        return projector, lda_templates_dict

    def _acc_gpu(X_g, y_g, W_g, b_g, p_g) -> float:
        if X_g is None or len(X_g) == 0:
            return 0.0
        z = X_g @ W_g - b_g
        d2 = xp_lib.sum((z[:, None, None, :] - p_g[None, :, :, :]) ** 2, axis=3)
        pred = xp_lib.argmin(xp_lib.min(d2, axis=2), axis=1)
        acc = xp_lib.mean(pred == y_g)
        return _scalar_to_float(acc)

    def _loss_and_grads_gpu(Xb, yb, W_g, b_g, p_g):
        B = Xb.shape[0]
        z = Xb @ W_g - b_g
        diff = z[:, None, None, :] - p_g[None, :, :, :]
        d2 = xp_lib.sum(diff * diff, axis=3)

        m = xp_lib.min(d2, axis=2, keepdims=True)
        expv = xp_lib.exp(-(d2 - m) / tau)
        sumexp = xp_lib.sum(expv, axis=2, keepdims=True) + 1e-12
        alpha = expv / sumexp

        d_class = m[:, :, 0] - tau * xp_lib.log(sumexp[:, :, 0])
        scores = -d_class

        smax = xp_lib.max(scores, axis=1, keepdims=True)
        pexp = xp_lib.exp(scores - smax)
        probs = pexp / (xp_lib.sum(pexp, axis=1, keepdims=True) + 1e-12)

        ce_terms = xp_lib.log(probs[xp_lib.arange(B), yb] + 1e-12)
        ce = -xp_lib.mean(ce_terms)
        loss = ce + (0.5 * reg * (xp_lib.sum(W_g * W_g) + xp_lib.sum(p_g * p_g)))

        g_scores = probs.copy()
        g_scores[xp_lib.arange(B), yb] -= 1.0
        g_scores /= B

        g_d = -g_scores
        g_d2 = g_d[:, :, None] * alpha
        grad_z = 2.0 * xp_lib.sum(g_d2[:, :, :, None] * diff, axis=(1, 2))
        grad_W = Xb.T @ grad_z + (reg * W_g)
        grad_b = -xp_lib.sum(grad_z, axis=0)
        grad_p = -2.0 * xp_lib.sum(g_d2[:, :, :, None] * diff, axis=0) + (reg * p_g)

        return _scalar_to_float(loss), grad_W, grad_b, grad_p

    mW = xp_lib.zeros_like(W)
    vW = xp_lib.zeros_like(W)
    mb = xp_lib.zeros_like(b)
    vb = xp_lib.zeros_like(b)
    mp = xp_lib.zeros_like(proto)
    vp = xp_lib.zeros_like(proto)

    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
    step = 0
    rng_opt = np.random.default_rng(
        shuffle_seed if shuffle_seed is not None else RANDOM_SEED
    )

    best_metric = -1.0
    best_W = W.copy()
    best_b = b.copy()
    best_p = proto.copy()
    no_improve = 0
    lr_min = lr * 0.01

    for epoch in range(1, epochs + 1):
        order = rng_opt.permutation(n_train)
        epoch_loss = 0.0
        n_batches = 0
        lr_e = lr_min + 0.5 * (lr - lr_min) * (
            1.0 + np.cos(np.pi * float(epoch - 1) / float(max(epochs - 1, 1)))
        )

        for start in range(0, n_train, batch_size):
            idx_np = order[start:start + batch_size]
            Xb = X_tr[idx_np]
            yb = y_tr[idx_np]

            loss_val, gW, gb, gp = _loss_and_grads_gpu(Xb, yb, W, b, proto)

            gnorm = _scalar_to_float(
                xp_lib.sqrt(xp_lib.sum(gW * gW) + xp_lib.sum(gb * gb) + xp_lib.sum(gp * gp))
            )
            if gnorm > 5.0:
                scale = 5.0 / (gnorm + 1e-12)
                gW *= scale
                gb *= scale
                gp *= scale

            step += 1
            mW = beta1 * mW + (1 - beta1) * gW
            vW = beta2 * vW + (1 - beta2) * (gW * gW)
            mb = beta1 * mb + (1 - beta1) * gb
            vb = beta2 * vb + (1 - beta2) * (gb * gb)
            mp = beta1 * mp + (1 - beta1) * gp
            vp = beta2 * vp + (1 - beta2) * (gp * gp)

            bc1 = 1 - (beta1 ** step)
            bc2 = 1 - (beta2 ** step)

            W -= lr_e * (mW / bc1) / (xp_lib.sqrt(vW / bc2) + eps_adam)
            b -= lr_e * (mb / bc1) / (xp_lib.sqrt(vb / bc2) + eps_adam)
            proto -= lr_e * (mp / bc1) / (xp_lib.sqrt(vp / bc2) + eps_adam)

            epoch_loss += loss_val
            n_batches += 1

        tr_acc = _acc_gpu(X_tr, y_tr, W, b, proto)
        val_acc = _acc_gpu(X_vl, y_vl, W, b, proto) if X_vl is not None else tr_acc
        metric = val_acc

        if metric > best_metric:
            best_metric = metric
            best_W = W.copy()
            best_b = b.copy()
            best_p = proto.copy()
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 4 == 0 or epoch == 1 or epoch == epochs:
            avg_l = epoch_loss / max(1, n_batches)
            print(
                f"  [OPT] epoch {epoch:02d}/{epochs} "
                f"loss={avg_l:.4f} train={100*tr_acc:.1f}% val={100*val_acc:.1f}%"
            )

        if no_improve >= OPT_EARLY_STOP_PATIENCE:
            print(f"  [OPT] early stop at epoch {epoch}")
            break

    # Move best results back to CPU numpy for export.
    if _CUPY_AVAILABLE:
        projector.W = cp.asnumpy(best_W).astype(np.float32)
        projector.xbar = cp.asnumpy(best_b).astype(np.float32)
        refined = {
            w: cp.asnumpy(best_p[wi]).astype(np.float32)
            for wi, w in enumerate(WORDS)
        }
    else:
        projector.W = np.array(best_W, dtype=np.float32)
        projector.xbar = np.array(best_b, dtype=np.float32)
        refined = {
            w: np.array(best_p[wi], dtype=np.float32)
            for wi, w in enumerate(WORDS)
        }

    return projector, refined


def _calibrate_threshold(
    X_val_lda: np.ndarray,
    y_val: np.ndarray,
    templates_stack: np.ndarray,
) -> float:
    if len(X_val_lda) == 0:
        return CONFIDENCE_THRESHOLD

    genuine = []
    impostor = []

    for fv_lda, yi in zip(X_val_lda, y_val):
        diff = templates_stack - fv_lda[None, None, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        per_word = np.min(dists, axis=1)

        genuine.append(float(per_word[int(yi)]))
        mask = np.ones(N_WORDS, dtype=bool)
        mask[int(yi)] = False
        impostor.append(float(np.min(per_word[mask])))

    genuine_arr = np.array(genuine, dtype=np.float32)
    impostor_arr = np.array(impostor, dtype=np.float32)

    # Keep almost all genuine samples accepted; bound by a conservative impostor quantile.
    t_genuine = float(np.percentile(genuine_arr, 99.5) * 1.10)
    t_impostor = float(np.percentile(impostor_arr, 20.0) * 0.98)

    threshold = max(t_genuine, CONFIDENCE_THRESHOLD)
    if t_impostor > threshold:
        threshold = t_impostor

    return float(threshold)


def _estimate_flash_bytes() -> int:
    return (
        (2 * N_FEATURES * 4)
        + (N_FEATURES * LDA_DIMS * 4)
        + (LDA_DIMS * 4)
        + (N_WORDS * K_TEMPLATES * LDA_DIMS * 4)
    )


def _assert_flash_budget(program_reserve: int = 3500) -> Tuple[int, int, int]:
    model_flash = _estimate_flash_bytes()
    n_fft_bins = (N_FFT // 2) + 1
    mel_flash = N_MEL_FILTERS * n_fft_bins * 4
    total = model_flash + mel_flash
    if total + program_reserve > (32 * 1024):
        raise AssertionError(
            "OVER BUDGET: "
            f"model={model_flash} mel={mel_flash} total={total} reserve={program_reserve} "
            f"limit={32 * 1024}"
        )
    return model_flash, mel_flash, total


def _print_flash_budget_report():
    model_flash, mel_flash, total_static = _assert_flash_budget(program_reserve=3500)
    n_fft_bins = (N_FFT // 2) + 1
    flash_limit = 32 * 1024
    remaining = flash_limit - total_static
    remaining_after_reserve = remaining - 3500

    print("\n  Flash budget check")
    print(f"  - Model arrays Flash               : {model_flash} bytes")
    print(f"  - Mel filterbank runtime Flash est : {mel_flash} bytes  ({N_MEL_FILTERS} x {n_fft_bins} x 4)")

    if total_static <= flash_limit:
        print(
            "  - 32KB fit check                  : PASS "
            f"(model+mel={total_static} bytes, remaining for program code~{remaining} bytes)"
        )
        print(
            "  - Reserve-aware fit (3.5KB code)   : PASS "
            f"(remaining after reserve~{remaining_after_reserve} bytes)"
        )
    else:
        print(
            "  - 32KB fit check                  : FAIL "
            f"(model+mel={total_static} bytes exceeds limit by {abs(remaining)} bytes)"
        )


def _print_feature_breakdown() -> None:
    print("\n  Feature vector breakdown")
    print(f"  - N_FEATURES total           : {N_FEATURES}")
    print(f"  - MFCC/delta/delta2 stats    : {FEATURES_STATIC_DIMS}")
    print(f"  - Spectral stats             : {FEATURES_SPECTRAL_STATS_DIMS}")
    print(f"  - Temporal MFCC bins         : {FEATURES_TEMPORAL_MFCC_DIMS}")
    print(f"  - Temporal delta bins        : {FEATURES_TEMPORAL_DELTA_DIMS}")
    print(f"  - Temporal logmel bins       : {FEATURES_TEMPORAL_LOGMEL_DIMS}")
    print(f"  - Energy stats               : {FEATURES_ENERGY_DIMS}")
    print(f"  - Robust extras              : {FEATURES_ROBUST_DIMS}")
    print(f"  - Logmel mean/std            : {FEATURES_LOGMEL_STATS_DIMS}")
    print(f"  - Spectral contrast          : {FEATURES_CONTRAST_DIMS}")
    print(f"  - LFCC mean/std              : {FEATURES_LFCC_DIMS}")
    print(f"  - Formant stats              : {FEATURES_FORMANT_DIMS}")


def train_from_prepared(prepared: PreparedDataset) -> Dict[str, object]:
    train_idx = prepared.split_indices["train"]
    val_idx = prepared.split_indices["val"]

    X_train, y_train, train_entries, skipped_train = _subset_from_indices(prepared, train_idx)
    X_val, y_val, val_entries, skipped_val = _subset_from_indices(prepared, val_idx)

    if len(X_train) == 0:
        raise RuntimeError("No usable training samples after filtering.")

    print(f"\n  Train samples (usable): {len(X_train)}")
    print(f"  Val samples   (usable): {len(X_val)}")
    print(f"  Skipped in train split : {skipped_train}")
    print(f"  Skipped in val split   : {skipped_val}")

    X_aug = np.zeros((0, N_FEATURES), dtype=np.float32)
    y_aug = np.zeros((0,), dtype=np.int32)

    if ENABLE_TRAIN_AUGMENTATION and len(train_entries) > 0:
        import time as _time

        base_aug_per_sample = (
            len(AUG_SPEED_FACTORS)
            + max(0, AUG_NOISE_COPIES)
            + max(0, AUG_REVERB_COPIES)
            + 2
        )
        hard_extra_aug = (
            len(AUG_HARD_EXTRA_SPEED_FACTORS)
            + max(0, AUG_HARD_EXTRA_NOISE_COPIES)
            + max(0, AUG_HARD_EXTRA_REVERB_COPIES)
            + len(AUG_HARD_EXTRA_SHIFT_MS)
        )
        aug_per_sample = base_aug_per_sample + hard_extra_aug
        print(
            "  [AUG] Variants/sample (max): "
            f"speed={len(AUG_SPEED_FACTORS)} + noise={AUG_NOISE_COPIES} + "
            f"reverb={AUG_REVERB_COPIES} + shift=2 + hard_extra={hard_extra_aug} "
            f"for {AUG_HARD_CLASS_WORDS} => {aug_per_sample}"
        )
        if ENABLE_REAL_BG_NOISE:
            bg_bank = _load_bg_noise_bank()
            print(
                "  [AUG] real background noise: "
                f"dir={BG_NOISE_DIR_NAME}, snr_db={BG_NOISE_SNR_DB_RANGE}, "
                f"noisy_ratio={NOISE_MIX_NOISY_RATIO:.2f}, clips={len(bg_bank)}"
            )

        t_aug0 = _time.perf_counter()
        os.makedirs(TEMPLATES_DIR, exist_ok=True)
        _cache_key = _aug_cache_key(train_entries, VAD_TRAIN_THRESHOLD)
        _cache_path = os.path.join(TEMPLATES_DIR, f"aug_cache_{_cache_key}.npz")

        if os.path.exists(_cache_path):
            try:
                print(f"  [CACHE] Loading augmented features from {_cache_path}")
                _npz = np.load(_cache_path)
                X_aug = _npz["X_aug"]
                y_aug = _npz["y_aug"]
            except Exception as exc:
                print(f"  [CACHE] Cache load failed ({exc}); rebuilding cache")
                try:
                    os.remove(_cache_path)
                except OSError:
                    pass
                X_aug, y_aug = _build_train_augmentations(train_entries, VAD_TRAIN_THRESHOLD)
                np.savez_compressed(_cache_path, X_aug=X_aug, y_aug=y_aug)
                print(f"  [CACHE] Saved augmented features to {_cache_path}")
        else:
            X_aug, y_aug = _build_train_augmentations(train_entries, VAD_TRAIN_THRESHOLD)
            np.savez_compressed(_cache_path, X_aug=X_aug, y_aug=y_aug)
            print(f"  [CACHE] Saved augmented features to {_cache_path}")

        t_aug1 = _time.perf_counter()
        print(f"  Augmentation phase wall-clock: {t_aug1 - t_aug0:.2f}s")

        if len(X_aug) > 0:
            X_train = np.vstack([X_train, X_aug]).astype(np.float32)
            y_train = np.concatenate([y_train, y_aug]).astype(np.int32)
            print(
                f"  [AUG] Effective generated/sample: {len(X_aug) / max(1, len(train_entries)):.2f} "
                f"({len(X_aug)} generated / {len(train_entries)} source train utterances)"
            )
            print(f"  Augmented train samples: +{len(X_aug)} ({len(X_train)} total)")
        else:
            print("  Augmented train samples: +0 (augmentation generated no usable samples)")

    # Keep exported mean/std from pre-warp z-normalization for AVR compatibility.
    normalizer = FeatureNormalizer().fit(X_train)
    X_train_norm = normalizer.transform_batch(X_train)
    X_val_norm = normalizer.transform_batch(X_val) if len(X_val) else np.zeros((0, N_FEATURES), dtype=np.float32)

    warper = FeatureWarper(n_quantiles=1000).fit(X_train_norm)
    X_train_norm = warper.transform(X_train_norm)
    if len(X_val_norm):
        X_val_norm = warper.transform(X_val_norm)
    warper.save(os.path.join(TEMPLATES_DIR, "feature_warper.npz"))

    projector = LDAProjector().fit(X_train_norm, y_train)
    X_train_lda = projector.transform_batch(X_train_norm)
    X_val_lda = projector.transform_batch(X_val_norm) if len(X_val_norm) else np.zeros((0, LDA_DIMS), dtype=np.float32)

    lda_templates_dict = _select_templates_svm_guided(X_train_lda, y_train)

    if USE_LVQ_REFINEMENT:
        lda_templates_dict = _refine_templates_lvq(
            X_train_lda,
            y_train,
            lda_templates_dict,
            epochs=LVQ_EPOCHS,
            lr0=LVQ_LR0,
        )

    if OPT_EPOCHS > 0:
        if projector.W is None or projector.xbar is None:
            raise RuntimeError("LDA projector not initialized before optimization")

        init_W = projector.W.copy().astype(np.float32)
        init_xbar = projector.xbar.copy().astype(np.float32)
        init_templates = {
            w: lda_templates_dict[w].copy().astype(np.float32)
            for w in WORDS
        }

        opt_seeds = [RANDOM_SEED, RANDOM_SEED + 1000, RANDOM_SEED + 2000]
        all_Ws: List[np.ndarray] = []
        all_bs: List[np.ndarray] = []
        all_protos: List[np.ndarray] = []

        for si, opt_seed in enumerate(opt_seeds, start=1):
            print(f"  [OPT-ENSEMBLE] run {si}/{len(opt_seeds)} with shuffle_seed={opt_seed}")

            proj_seed = LDAProjector()
            proj_seed.W = init_W.copy()
            proj_seed.xbar = init_xbar.copy()
            tmpl_seed = {
                w: init_templates[w].copy()
                for w in WORDS
            }

            proj_seed, tmpl_seed = _optimize_projection_and_templates(
                X_train_norm,
                y_train,
                X_val_norm,
                y_val,
                proj_seed,
                tmpl_seed,
                epochs=OPT_EPOCHS,
                lr=OPT_LR,
                tau=OPT_TAU,
                reg=OPT_REG,
                shuffle_seed=opt_seed,
            )

            if proj_seed.W is None or proj_seed.xbar is None:
                raise RuntimeError("Optimizer returned invalid projection for ensemble run")

            all_Ws.append(proj_seed.W.copy().astype(np.float32))
            all_bs.append(proj_seed.xbar.copy().astype(np.float32))
            all_protos.append(_templates_to_stack(tmpl_seed).astype(np.float32))

        projector.W = np.mean(np.stack(all_Ws, axis=0), axis=0).astype(np.float32)
        projector.xbar = np.mean(np.stack(all_bs, axis=0), axis=0).astype(np.float32)
        avg_proto = np.mean(np.stack(all_protos, axis=0), axis=0).astype(np.float32)
        lda_templates_dict = {
            w: avg_proto[wi].astype(np.float32)
            for wi, w in enumerate(WORDS)
        }

        if len(X_val_norm) > 0 and len(y_val) > 0:
            if ENABLE_FINAL_TRAINVAL_FINETUNE:
                print("  [FINAL] Retraining on train+val combined...")
                X_tv = np.vstack([X_train_norm, X_val_norm]).astype(np.float32)
                y_tv = np.concatenate([y_train, y_val]).astype(np.int32)
                fine_epochs = max(20, OPT_EPOCHS // 4)
                print(f"  [FINAL] fine-tune epochs={fine_epochs}, lr={OPT_LR * 0.5:.6f}")
                projector, lda_templates_dict = _optimize_projection_and_templates(
                    X_tv,
                    y_tv,
                    np.zeros((0, X_tv.shape[1]), dtype=np.float32),
                    np.zeros((0,), dtype=np.int32),
                    projector,
                    lda_templates_dict,
                    epochs=fine_epochs,
                    lr=OPT_LR * 0.5,
                    tau=OPT_TAU,
                    reg=0.0,
                )
            else:
                print("  [FINAL] train+val fine-tune disabled (using best val-selected ensemble average)")
    else:
        print("  [OPT] disabled (epochs=0) -> using clean LDA + k-medoids templates")

    # Recompute projected sets with optimized projection.
    X_train_lda = projector.transform_batch(X_train_norm)
    X_val_lda = projector.transform_batch(X_val_norm) if len(X_val_norm) else np.zeros((0, LDA_DIMS), dtype=np.float32)

    if projector.W is None or projector.xbar is None:
        raise RuntimeError("Projection optimizer produced invalid parameters")

    np.save(os.path.join(TEMPLATES_DIR, "lda_W.npy"), projector.W)
    np.save(os.path.join(TEMPLATES_DIR, "lda_xbar.npy"), projector.xbar)

    for word in WORDS:
        np.save(os.path.join(TEMPLATES_DIR, f"{word}_lda_templates.npy"), lda_templates_dict[word])

    templates_stack = _templates_to_stack(lda_templates_dict)
    threshold = _calibrate_threshold(X_val_lda, y_val, templates_stack)
    np.save(os.path.join(TEMPLATES_DIR, "confidence_threshold.npy"), np.array([threshold], dtype=np.float32))

    _generate_header_v5(normalizer, projector, lda_templates_dict, threshold)
    _print_flash_budget_report()

    return {
        "normalizer": normalizer,
        "warper": warper,
        "projector": projector,
        "lda_templates_dict": lda_templates_dict,
        "templates_stack": templates_stack,
        "threshold": threshold,
        "flash_bytes": _estimate_flash_bytes(),
        "train_count": len(train_entries),
        "val_count": len(val_entries),
    }


def _load_all() -> Tuple[
    FeatureNormalizer,
    LDAProjector,
    Optional[FeatureWarper],
    Dict[str, np.ndarray],
    np.ndarray,
    float,
]:
    required = [
        "norm_mean.npy",
        "norm_std.npy",
        "lda_W.npy",
        "lda_xbar.npy",
        "confidence_threshold.npy",
    ]

    missing = [f for f in required if not os.path.exists(os.path.join(TEMPLATES_DIR, f))]
    if missing:
        raise RuntimeError(
            "Missing model files: " + ", ".join(missing) + ". Run training first."
        )

    normalizer = FeatureNormalizer().load()
    projector = LDAProjector().load()
    warper_path = os.path.join(TEMPLATES_DIR, "feature_warper.npz")
    warper = FeatureWarper.load(warper_path) if os.path.exists(warper_path) else None

    lda_templates_dict: Dict[str, np.ndarray] = {}
    for word in WORDS:
        p = os.path.join(TEMPLATES_DIR, f"{word}_lda_templates.npy")
        if os.path.exists(p):
            lda_templates_dict[word] = np.load(p).astype(np.float32)
        else:
            lda_templates_dict[word] = np.zeros((K_TEMPLATES, LDA_DIMS), dtype=np.float32)

    templates_stack = _templates_to_stack(lda_templates_dict)
    threshold = float(np.load(os.path.join(TEMPLATES_DIR, "confidence_threshold.npy"))[0])

    return normalizer, projector, warper, lda_templates_dict, templates_stack, threshold


def evaluate_from_prepared(
    prepared: PreparedDataset,
    normalizer: FeatureNormalizer,
    projector: LDAProjector,
    templates_stack: np.ndarray,
    threshold: float,
    warper: Optional[FeatureWarper] = None,
    split_name: str = "test",
) -> EvalMetrics:
    indices = prepared.split_indices[split_name]
    X_test, y_test, _, skipped_split = _subset_from_indices(prepared, indices)

    confusion = {w: {w2: 0 for w2 in WORDS + ["unknown"]} for w in WORDS}
    per_word = {w: {"correct": 0, "total": 0} for w in WORDS}

    total_correct = 0
    total_samples = 0

    if len(X_test) == 0:
        return EvalMetrics(
            overall_accuracy=0.0,
            total_correct=0,
            total_samples=0,
            skipped=prepared.skipped + skipped_split,
            confusion=confusion,
            per_word=per_word,
            threshold=threshold,
        )

    X_test_norm = normalizer.transform_batch(X_test)
    if warper is not None:
        X_test_norm = warper.transform(X_test_norm)
    X_test_lda = projector.transform_batch(X_test_norm)

    for fv_lda, yi in zip(X_test_lda, y_test):
        true_word = WORDS[int(yi)]
        pred, _, _ = classify_from_lda(fv_lda, templates_stack, threshold)

        label = pred if pred is not None else "unknown"
        confusion[true_word][label] += 1
        per_word[true_word]["total"] += 1
        total_samples += 1

        if pred == true_word:
            per_word[true_word]["correct"] += 1
            total_correct += 1

    overall = (100.0 * total_correct / total_samples) if total_samples else 0.0

    return EvalMetrics(
        overall_accuracy=overall,
        total_correct=total_correct,
        total_samples=total_samples,
        skipped=prepared.skipped + skipped_split,
        confusion=confusion,
        per_word=per_word,
        threshold=threshold,
    )


# =============================================================================
# Reporting helpers
# =============================================================================

def print_split_summary(prepared: PreparedDataset):
    print(f"\n  Total discovered files: {len(prepared.entries)}")
    for split in ("train", "val", "test"):
        print(f"  {split:>5} split files    : {len(prepared.split_indices[split])}")


def print_eval_report(metrics: EvalMetrics):
    print(f"\n  Threshold used : {metrics.threshold:.4f}")
    print(f"  {'Word':<10} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print(f"  {'-' * 40}")

    for word in WORDS:
        c = metrics.per_word[word]["correct"]
        t = metrics.per_word[word]["total"]
        acc = f"{(100.0 * c / t):.1f}%" if t > 0 else "N/A"
        print(f"  {word:<10} {c:>8} {t:>8} {acc:>10}")

    print(
        f"\n  Overall accuracy : {metrics.overall_accuracy:.1f}% "
        f"({metrics.total_correct}/{metrics.total_samples})"
    )
    print(f"  Skipped files    : {metrics.skipped}")

    print("\n  Confusion summary (true -> mistakes):")
    for word in WORDS:
        wrong = [(pred, n) for pred, n in metrics.confusion[word].items() if pred != word and n > 0]
        if wrong:
            detail = ", ".join(f"{pred}x{n}" for pred, n in sorted(wrong, key=lambda x: -x[1]))
            print(f"    {word:<8} -> {detail}")


def print_pass_summary(metrics: EvalMetrics, flash_bytes: int):
    delta = metrics.overall_accuracy - BASELINE_ACCURACY
    print("\n  Summary vs baseline")
    print(f"  - Baseline overall accuracy : {BASELINE_ACCURACY:.1f}%")
    print(f"  - New overall accuracy      : {metrics.overall_accuracy:.1f}% ({delta:+.1f} pp)")
    print(
        "  - Features                  : "
        f"N_FEATURES={N_FEATURES} (MFCC + delta + delta2 mean/std + flux/rolloff/bandwidth stats)"
    )
    print("  - LDA tuning                : solver='eigen', shrinkage='auto' (svd fallback)")
    print(f"  - Template selection        : k-means medoid representatives, K={K_TEMPLATES}")
    print(f"  - Flash usage estimate      : ~{flash_bytes} bytes")


# =============================================================================
# Header generation (AVR schema compatible)
# =============================================================================

def _generate_header_v5(
    normalizer: FeatureNormalizer,
    projector: LDAProjector,
    lda_templates_dict: Dict[str, np.ndarray],
    threshold: float,
    path: str = "word_templates.h",
):
    flash_bytes = _estimate_flash_bytes()

    if projector.W is None or projector.xbar is None:
        raise RuntimeError("Projector is not trained")

    with open(path, "w", encoding="utf-8") as f:
        f.write("/*\n")
        f.write(" * word_templates.h - auto-generated by main-pipeline.py\n")
        f.write(
            f" * Pipeline: {N_FEATURES}-D features -> z-norm -> {LDA_DIMS}-D LDA -> 1-NN\n"
        )
        f.write(f" * {N_WORDS} words x {K_TEMPLATES} templates x {LDA_DIMS} LDA dims\n")
        f.write(f" * Estimated Flash usage : ~{flash_bytes} bytes\n")
        f.write(f" * Confidence threshold  : {threshold:.4f} (LDA space)\n")
        f.write(" */\n\n")

        f.write("#ifndef WORD_TEMPLATES_H\n")
        f.write("#define WORD_TEMPLATES_H\n\n")
        f.write("#include <avr/pgmspace.h>\n")
        f.write("#include <math.h>\n\n")

        f.write(f"#define N_WORDS              {N_WORDS}\n")
        f.write(f"#define N_FEATURES           {N_FEATURES}\n")
        f.write(f"#define LDA_DIMS             {LDA_DIMS}\n")
        f.write(f"#define K_TEMPLATES          {K_TEMPLATES}\n")
        f.write(f"#define CONFIDENCE_THRESHOLD {threshold:.4f}f\n\n")

        f.write("/* Word labels */\n")
        for i, word in enumerate(WORDS):
            f.write(f'static const char _word_{i}[] PROGMEM = "{word}";\n')
        f.write("const char* const word_labels[N_WORDS] PROGMEM = {\n")
        for i in range(N_WORDS):
            comma = "," if i < N_WORDS - 1 else ""
            f.write(f"    _word_{i}{comma}\n")
        f.write("};\n\n")

        mean_s = ", ".join(f"{v:.8f}f" for v in normalizer.mean)
        std_s = ", ".join(f"{v:.8f}f" for v in normalizer.std)
        f.write("/* Z-score normalizer */\n")
        f.write(f"const float feature_mean[N_FEATURES] PROGMEM = {{{mean_s}}};\n")
        f.write(f"const float feature_std[N_FEATURES]  PROGMEM = {{{std_s}}};\n\n")

        f.write("/* LDA projection matrix: lda_W[feat][dim] */\n")
        f.write("const float lda_W[N_FEATURES][LDA_DIMS] PROGMEM = {\n")
        for i in range(N_FEATURES):
            row = ", ".join(f"{v:11.7f}f" for v in projector.W[i])
            comma = "," if i < N_FEATURES - 1 else ""
            f.write(f"    /* f{i:02d} */ {{{row}}}{comma}\n")
        f.write("};\n\n")

        xbar_s = ", ".join(f"{v:.8f}f" for v in projector.xbar)
        f.write("/* LDA projected-space mean (centering offset) */\n")
        f.write(f"const float lda_xbar[LDA_DIMS] PROGMEM = {{{xbar_s}}};\n\n")

        f.write("/* LDA-space templates: lda_templates[word][k][dim] */\n")
        f.write("const float lda_templates[N_WORDS][K_TEMPLATES][LDA_DIMS] PROGMEM = {\n")
        for wi, word in enumerate(WORDS):
            f.write(f"    /* {word} */ {{\n")
            tmpl = lda_templates_dict[word]
            for ki in range(K_TEMPLATES):
                vals = ", ".join(f"{v:11.7f}f" for v in tmpl[ki])
                comma = "," if ki < K_TEMPLATES - 1 else ""
                f.write(f"        /* k{ki} */ {{{vals}}}{comma}\n")
            comma = "," if wi < N_WORDS - 1 else ""
            f.write(f"    }}{comma}\n")

        f.write("};\n\n")
        f.write("#endif /* WORD_TEMPLATES_H */\n")

    print(f"\n  Header written: {path}")


# =============================================================================
# Interactive options
# =============================================================================

def _record_one(word: str, idx: int, vad_thresh: float) -> np.ndarray:
    sd = _require_sounddevice()

    print(f"\n  Say '{word}' (sample {idx + 1}/{RECOMMENDED_SAMPLES})")
    input("  Press Enter, then speak immediately ...")
    print("  Recording ...")

    rec = sd.rec(N_SAMPLES, samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    signal = rec.flatten()

    sig_rms = _rms(signal)
    if sig_rms <= vad_thresh:
        print(f"  No speech detected (RMS={sig_rms:.5f} <= threshold={vad_thresh:.5f})")
        return _record_one(word, idx, vad_thresh)

    return signal


def option_record(data_dir: str):
    noise_rms = measure_noise_floor()
    vad_thresh = max(VAD_LIVE_FALLBACK, noise_rms * NOISE_MULTIPLIER)
    print(f"  Live VAD threshold set to {vad_thresh:.5f}\n")

    n = input(f"  Samples per word [{RECOMMENDED_SAMPLES}]: ").strip()
    n = int(n) if n.isdigit() else RECOMMENDED_SAMPLES

    if n < 20:
        print("  Warning: sample count is below project recommendation (20+).")

    for word in WORDS:
        print(f"\n  Word: {word}")
        out_dir = os.path.join(data_dir, word)
        os.makedirs(out_dir, exist_ok=True)

        for i in range(n):
            signal = _record_one(word, i, vad_thresh)
            out_path = os.path.join(out_dir, f"{word}_{i:03d}.wav")
            sf.write(out_path, signal, SAMPLE_RATE)
            print(f"  Saved: {out_path}")


def option_extract(data_dir: str):
    print(f"\n  Dataset dir    : {os.path.abspath(data_dir)}")
    print(f"  Feature vector : {N_FEATURES}-D")
    print(f"  LDA            : {N_FEATURES}-D -> {LDA_DIMS}-D")
    print(f"  Templates      : {K_TEMPLATES} per word")

    os.makedirs(TEMPLATES_DIR, exist_ok=True)
    prepared = prepare_dataset(data_dir, vad_threshold=VAD_TRAIN_THRESHOLD)
    print_split_summary(prepared)

    artifacts = train_from_prepared(prepared)
    print(f"  Confidence threshold: {float(artifacts['threshold']):.4f}")
    print(f"  Flash estimate      : ~{int(artifacts['flash_bytes'])} bytes")


def option_test(data_dir: str):
    sd = _require_sounddevice()

    normalizer, projector, warper, _, templates_stack, threshold = _load_all()
    print("\n  Real-time word detection")
    print(f"  Threshold: {threshold:.4f} (Ctrl+C to exit)\n")

    noise_rms = measure_noise_floor()
    vad_thresh = max(VAD_LIVE_FALLBACK, noise_rms * NOISE_MULTIPLIER)
    print(f"  Live VAD threshold: {vad_thresh:.5f}\n")

    while True:
        try:
            input("  Press Enter, then say a word ...")
        except KeyboardInterrupt:
            print("\n  Exiting real-time test.")
            break

        rec = sd.rec(N_SAMPLES, samplerate=SAMPLE_RATE, channels=1, dtype="float32")
        sd.wait()
        signal = rec.flatten().astype(np.float32)

        sig_rms = _rms(signal)
        if sig_rms <= vad_thresh:
            print(f"  [VAD] No speech (RMS={sig_rms:.5f} <= {vad_thresh:.5f})\n")
            continue

        pred, dist, per_word = run_pipeline(
            signal,
            normalizer,
            projector,
            templates_stack,
            threshold,
            warper=warper,
        )

        if pred is None:
            print(f"  Result: UNKNOWN (dist={dist:.4f} > threshold={threshold:.4f})")
        else:
            print(f"  Result: {pred.upper()} (dist={dist:.4f})")

        top3 = sorted(per_word.items(), key=lambda x: x[1])[:3]
        print("  Top-3 : " + "  ".join(f"{w}={d:.3f}" for w, d in top3) + "\n")


def option_evaluate(data_dir: str):
    prepared = prepare_dataset(data_dir, vad_threshold=VAD_EVAL_THRESHOLD)
    print_split_summary(prepared)

    normalizer, projector, warper, _, templates_stack, threshold = _load_all()
    metrics = evaluate_from_prepared(
        prepared,
        normalizer,
        projector,
        templates_stack,
        threshold,
        warper=warper,
        split_name="test",
    )

    print_eval_report(metrics)


# =============================================================================
# Headless CLI mode
# =============================================================================

def run_headless(mode: str, data_dir: str) -> int:
    os.makedirs(TEMPLATES_DIR, exist_ok=True)
    _print_feature_breakdown()

    if mode not in {"train", "eval", "all"}:
        raise RuntimeError(f"Unsupported mode: {mode}")

    prepared = prepare_dataset(data_dir, vad_threshold=VAD_TRAIN_THRESHOLD)
    print_split_summary(prepared)

    flash_bytes = _estimate_flash_bytes()

    if mode == "train":
        artifacts = train_from_prepared(prepared)
        flash_bytes = int(artifacts["flash_bytes"])
        print(f"  Training done. Header regenerated. Flash estimate: ~{flash_bytes} bytes")
        print("ACCURACY: N/A")
        return 0

    if mode == "eval":
        normalizer, projector, warper, _, templates_stack, threshold = _load_all()
        metrics = evaluate_from_prepared(
            prepared,
            normalizer,
            projector,
            templates_stack,
            threshold,
            warper=warper,
            split_name="test",
        )
        print_eval_report(metrics)

        if metrics.overall_accuracy >= TARGET_ACCURACY:
            print_pass_summary(metrics, flash_bytes)

        print(f"ACCURACY: {metrics.overall_accuracy:.1f}%")
        return 0 if metrics.overall_accuracy >= TARGET_ACCURACY else 1

    # mode == "all"
    artifacts = train_from_prepared(prepared)
    normalizer = artifacts["normalizer"]  # type: ignore[assignment]
    warper = artifacts["warper"]          # type: ignore[assignment]
    projector = artifacts["projector"]    # type: ignore[assignment]
    templates_stack = artifacts["templates_stack"]  # type: ignore[assignment]
    threshold = float(artifacts["threshold"])       # type: ignore[arg-type]
    flash_bytes = int(artifacts["flash_bytes"])

    metrics = evaluate_from_prepared(
        prepared,
        normalizer,
        projector,
        templates_stack,
        threshold,
        warper=warper,
        split_name="test",
    )
    print_eval_report(metrics)

    if metrics.overall_accuracy >= TARGET_ACCURACY:
        print_pass_summary(metrics, flash_bytes)

    print(f"ACCURACY: {metrics.overall_accuracy:.1f}%")
    return 0 if metrics.overall_accuracy >= TARGET_ACCURACY else 1


# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Speech command training/eval pipeline")
    parser.add_argument(
        "--mode",
        choices=["train", "eval", "all"],
        default=None,
        help="Headless mode. If omitted, interactive menu is shown.",
    )
    parser.add_argument(
        "--data_dir",
        default=".",
        help="Dataset root containing on/off/go/stop/left/right/up/down folders.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = os.path.abspath(args.data_dir)

    if args.mode is not None:
        # No interactive prompts in headless mode.
        return run_headless(args.mode, data_dir)

    print("=" * 64)
    print("  Speech Recognition Pipeline")
    print("=" * 64)
    print(f"\n  Dataset dir  : {data_dir}")
    print(f"  Words        : {', '.join(WORDS)}")
    print(f"  Features     : {N_FEATURES}-D")
    print(f"  Pipeline     : z-norm -> LDA({LDA_DIMS}-D) -> 1-NN")
    print(f"  Templates    : {K_TEMPLATES} per word")
    print(f"  Flash budget : ~{_estimate_flash_bytes()} bytes")
    print()
    print("  1 - Record voice")
    print("  2 - Extract/train/generate header")
    print("  3 - Real-time test")
    print("  4 - Evaluate accuracy")
    print()

    choice = input("  Choose (1/2/3/4): ").strip()

    try:
        if choice == "1":
            option_record(data_dir)
        elif choice == "2":
            option_extract(data_dir)
        elif choice == "3":
            option_test(data_dir)
        elif choice == "4":
            option_evaluate(data_dir)
        else:
            print("  Invalid choice.")
            return 1
    except Exception as exc:
        print(f"\n  ERROR: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
