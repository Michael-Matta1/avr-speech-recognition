"""
Real-time AVR Inference Emulator for ATmega32A Speech Commands.

Copyright (c) 2026 Michael Matta
GitHub: https://github.com/Michael-Matta1

Licensed under the MIT License. See LICENSE for details.

This script performs PC-side inference that mirrors the deployment classifier
defined in classify_snippet.c. Model parameters are parsed directly from
word_templates.h to ensure deployment consistency.
"""

import os
import sys
import re
import time
import csv

import numpy as np
from scipy.fftpack import dct
from scipy.signal import butter, lfilter
from scipy.io import wavfile

try:
    import sounddevice as sd
except Exception:
    sd = None


SAMPLE_RATE = 8000
DURATION = 1.0
N_SAMPLES = int(SAMPLE_RATE * DURATION)

FRAME_SIZE = 160
HOP_SIZE = 80
N_FFT = 256
PRE_EMPHASIS = 0.97

N_MEL_FILTERS = 24
N_MFCC = 20
CEPSTRAL_LIFTER = 22
DELTA_WINDOW = 2
USE_LIBROSA_MEL = False
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

N_FEATURES_EXPECTED = 353


def butter_lowpass(cutoff=3500.0, fs=SAMPLE_RATE, order=4):
    b, a = butter(order, cutoff / (fs / 2.0), btype="low", analog=False)
    return b.astype(np.float32), a.astype(np.float32)


_LPF_B, _LPF_A = butter_lowpass()
_HAMMING_WINDOW = np.hamming(FRAME_SIZE).astype(np.float32)
_FFT_FREQS = np.fft.rfftfreq(N_FFT, d=1.0 / SAMPLE_RATE).astype(np.float32)
_LIFTER_VEC = (
    1.0 + (CEPSTRAL_LIFTER / 2.0)
    * np.sin(np.pi * np.arange(N_MFCC) / CEPSTRAL_LIFTER)
).astype(np.float32)


def _hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + (hz / 700.0))


def _mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _build_mel_filterbank(
    sr=SAMPLE_RATE,
    n_fft=N_FFT,
    n_mels=N_MEL_FILTERS,
    fmin=20.0,
    fmax=3800.0,
):
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

    denom = fb.sum(axis=1, keepdims=True)
    denom[denom == 0.0] = 1.0
    return fb / denom


_MEL_FILTERBANK = _build_mel_filterbank()


def _build_linear_filterbank(
    sr=SAMPLE_RATE,
    n_fft=N_FFT,
    n_filters=N_LFCC_FILTERS,
    f_min=0.0,
    f_max=4000.0,
):
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
    mel_spec,
    sr=SAMPLE_RATE,
    hop_length=HOP_SIZE,
    gain=PCEN_GAIN,
    bias=PCEN_BIAS,
    power=PCEN_POWER,
    time_constant=PCEN_TIME_CONSTANT,
    eps=PCEN_EPS,
    input_scale=PCEN_INPUT_SCALE,
):
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


def _compute_lpcc(signal, order=LPCC_ORDER, n_cep=LPCC_N_CEP):
    if len(signal) <= order + 1:
        return np.zeros(n_cep, dtype=np.float32)

    r = np.zeros(order + 1, dtype=np.float64)
    n = len(signal)
    for k in range(order + 1):
        r[k] = float(np.dot(signal[:n - k], signal[k:]))
    if r[0] < 1e-10:
        return np.zeros(n_cep, dtype=np.float32)

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


def _extract_formants(signal, sr=SAMPLE_RATE, lpc_order=10):
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


def preprocess(signal):
    signal = lfilter(_LPF_B, _LPF_A, signal.astype(np.float32))

    peak = float(np.max(np.abs(signal))) if len(signal) else 0.0
    if peak > 0.0:
        signal = signal / peak

    if len(signal) > 1:
        signal = np.append(signal[0], signal[1:] - PRE_EMPHASIS * signal[:-1]).astype(np.float32)

    if len(signal) < N_SAMPLES:
        signal = np.pad(signal, (0, N_SAMPLES - len(signal)))

    return signal[:N_SAMPLES].astype(np.float32)


def _frame_signal(signal):
    starts = np.arange(0, N_SAMPLES - FRAME_SIZE + 1, HOP_SIZE)
    frames = np.stack([signal[s:s + FRAME_SIZE] for s in starts], axis=0)
    return frames * _HAMMING_WINDOW


def _compute_delta(feat, win=DELTA_WINDOW):
    n_frames = feat.shape[0]
    if n_frames <= 1:
        return np.zeros_like(feat, dtype=np.float32)

    denom = float(2.0 * sum(i * i for i in range(1, win + 1)))
    padded = np.pad(feat, ((win, win), (0, 0)), mode="edge")

    delta = np.zeros_like(feat, dtype=np.float32)
    for n in range(1, win + 1):
        delta += n * (
            padded[win + n: win + n + n_frames]
            - padded[win - n: win - n + n_frames]
        )
    return (delta / denom).astype(np.float32)


def extract_features(signal):
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

    def _spectral_contrast_vec(pspec, n_bands=6, alpha=0.02):
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

    if fv.shape[0] != N_FEATURES_EXPECTED:
        raise RuntimeError(f"Feature length mismatch: expected {N_FEATURES_EXPECTED}, got {fv.shape[0]}")
    return fv


def _resample_to_sample_rate(signal, src_sr, dst_sr=SAMPLE_RATE):
    if src_sr == dst_sr or len(signal) < 2:
        return signal.astype(np.float32, copy=True)
    out_len = max(2, int(round(len(signal) * float(dst_sr) / float(src_sr))))
    x_old = np.linspace(0.0, 1.0, num=len(signal), endpoint=False, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, num=out_len, endpoint=False, dtype=np.float32)
    return np.interp(x_new, x_old, signal).astype(np.float32)


def _normalize_loaded_wav(data):
    if data.dtype == np.uint8:
        sig = (data.astype(np.float32) - 128.0) / 128.0
    elif data.dtype == np.int16:
        sig = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        sig = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.float32 or data.dtype == np.float64:
        sig = data.astype(np.float32)
    else:
        sig = data.astype(np.float32)

    if sig.ndim > 1:
        sig = sig[:, 0]
    return np.clip(sig, -1.0, 1.0).astype(np.float32)


def _find_array_block(text, marker):
    idx = text.find(marker)
    if idx < 0:
        raise RuntimeError(f"Array marker not found: {marker}")

    start = text.find("{", idx)
    if start < 0:
        raise RuntimeError(f"Opening brace not found for marker: {marker}")

    depth = 0
    end = -1
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end < 0:
        raise RuntimeError(f"Closing brace not found for marker: {marker}")

    return text[start + 1:end]


def _parse_float_values(c_block):
    block = re.sub(r"/\*.*?\*/", " ", c_block, flags=re.S)
    block = re.sub(r"//.*", " ", block)
    toks = re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?[fF]?", block)

    vals = []
    for t in toks:
        if t.endswith("f") or t.endswith("F"):
            t = t[:-1]
        vals.append(float(t))
    return np.array(vals, dtype=np.float32)


def _parse_define_number(value_text):
    v = value_text.strip()
    if v.endswith("f") or v.endswith("F"):
        v = v[:-1]
    if "." in v or "e" in v.lower():
        return float(v)
    return int(v)


def load_model_from_header(header_path):
    """Parse all deployment constants and arrays from word_templates.h."""
    with open(header_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    defines = {}
    for m in re.finditer(r"^\s*#define\s+([A-Z_][A-Z0-9_]*)\s+([^\s/]+)", text, flags=re.M):
        key = m.group(1)
        raw = m.group(2)
        try:
            defines[key] = _parse_define_number(raw)
        except Exception:
            continue

    n_words = int(defines.get("N_WORDS", 8))
    n_features = int(defines.get("N_FEATURES", N_FEATURES_EXPECTED))
    lda_dims = int(defines.get("LDA_DIMS", 7))
    k_templates = int(defines.get("K_TEMPLATES", 16))
    threshold = float(defines.get("CONFIDENCE_THRESHOLD", 0.0))

    words = re.findall(r'static\s+const\s+char\s+_word_\d+\[\]\s+PROGMEM\s*=\s*"([^"]+)"\s*;', text)
    if len(words) != n_words:
        words = ["on", "off", "go", "stop", "left", "right", "up", "down"][:n_words]

    feature_mean_vals = _parse_float_values(_find_array_block(text, "const float feature_mean"))
    feature_std_vals = _parse_float_values(_find_array_block(text, "const float feature_std"))
    lda_w_vals = _parse_float_values(_find_array_block(text, "const float lda_W"))
    lda_xbar_vals = _parse_float_values(_find_array_block(text, "const float lda_xbar"))
    lda_templates_vals = _parse_float_values(_find_array_block(text, "const float lda_templates"))

    if feature_mean_vals.size != n_features:
        raise RuntimeError(f"feature_mean size mismatch: expected {n_features}, got {feature_mean_vals.size}")
    if feature_std_vals.size != n_features:
        raise RuntimeError(f"feature_std size mismatch: expected {n_features}, got {feature_std_vals.size}")
    if lda_w_vals.size != n_features * lda_dims:
        raise RuntimeError(f"lda_W size mismatch: expected {n_features * lda_dims}, got {lda_w_vals.size}")
    if lda_xbar_vals.size != lda_dims:
        raise RuntimeError(f"lda_xbar size mismatch: expected {lda_dims}, got {lda_xbar_vals.size}")
    if lda_templates_vals.size != n_words * k_templates * lda_dims:
        raise RuntimeError(
            f"lda_templates size mismatch: expected {n_words * k_templates * lda_dims}, got {lda_templates_vals.size}"
        )

    model = {
        "N_WORDS": n_words,
        "N_FEATURES": n_features,
        "LDA_DIMS": lda_dims,
        "K_TEMPLATES": k_templates,
        "CONFIDENCE_THRESHOLD": threshold,
        "WORDS": words,
        "feature_mean": feature_mean_vals.astype(np.float32),
        "feature_std": np.maximum(feature_std_vals.astype(np.float32), 1e-12),
        "lda_W": lda_w_vals.reshape(n_features, lda_dims).astype(np.float32),
        "lda_xbar": lda_xbar_vals.astype(np.float32),
        "lda_templates": lda_templates_vals.reshape(n_words, k_templates, lda_dims).astype(np.float32),
    }
    return model


def classify_word(signal, model, warper=None):
    """Run z-norm -> optional warper -> LDA projection -> template 1-NN."""
    fv = extract_features(signal)
    if fv is None:
        return "unknown", float("inf"), -1

    fv = (fv - model["feature_mean"]) / model["feature_std"]
    if warper is not None:
        fv = _apply_feature_warper(fv, warper)
    proj = fv @ model["lda_W"] - model["lda_xbar"]

    best_dist = float("inf")
    best_word_idx = -1
    for w in range(model["N_WORDS"]):
        for k in range(model["K_TEMPLATES"]):
            diff = proj - model["lda_templates"][w, k]
            dist = float(np.sqrt(np.sum(diff * diff)))
            if dist < best_dist:
                best_dist = dist
                best_word_idx = w

    if best_dist > model["CONFIDENCE_THRESHOLD"] or best_word_idx < 0:
        return "unknown", best_dist, -1

    return model["WORDS"][best_word_idx], best_dist, best_word_idx


def _collect_wavs(folder):
    if not os.path.isdir(folder):
        return []
    return [
        os.path.join(folder, fn)
        for fn in sorted(os.listdir(folder))
        if fn.lower().endswith(".wav")
    ]


def _load_feature_warper(path):
    """Load optional training-side feature warper used for parity checks."""
    if not os.path.isfile(path):
        return None
    npz = np.load(path)
    return {
        "quantile_vals": npz["quantile_vals"].astype(np.float32),
        "target_vals": npz["target_vals"].astype(np.float32),
    }


def _apply_feature_warper(fv_norm, warper):
    qv = warper["quantile_vals"]
    tv = warper["target_vals"]
    out = np.empty_like(fv_norm, dtype=np.float32)
    for j in range(fv_norm.shape[0]):
        out[j] = np.interp(fv_norm[j], qv[:, j], tv)
    return out


def _approximate_mode(class_counts, n_draws, rng):
    continuous = class_counts / class_counts.sum() * float(n_draws)
    floored = np.floor(continuous)
    need_to_add = int(n_draws - floored.sum())

    if need_to_add > 0:
        remainder = continuous - floored
        unique_vals = np.sort(np.unique(remainder))[::-1]
        for val in unique_vals:
            inds = np.where(remainder == val)[0]
            if len(inds) == 0:
                continue
            add_now = min(len(inds), need_to_add)
            picked = rng.choice(inds, size=add_now, replace=False)
            floored[picked] += 1
            need_to_add -= add_now
            if need_to_add == 0:
                break

    return floored.astype(np.int64)


def _build_deterministic_test_entries(words, data_root=".", test_fraction=0.2, seed=42):
    entries = []
    word_to_idx = {w: i for i, w in enumerate(words)}

    for w in words:
        folder = os.path.join(data_root, w)
        wavs = _collect_wavs(folder)
        for wav_path in wavs:
            rel = (w + "/" + os.path.basename(wav_path)).replace("\\", "/").strip().lower()
            entries.append((w, wav_path, rel, word_to_idx[w]))

    entries.sort(key=lambda x: x[2])
    if len(entries) == 0:
        return []

    y = np.array([e[3] for e in entries], dtype=np.int32)
    n_samples = len(entries)
    n_test = int(round(float(test_fraction) * float(n_samples)))
    n_test = max(1, min(n_samples - 1, n_test))
    n_train = n_samples - n_test

    classes, y_indices = np.unique(y, return_inverse=True)
    class_counts = np.bincount(y_indices)
    rng = np.random.RandomState(seed)

    n_i = _approximate_mode(class_counts, n_train, rng)
    class_counts_remaining = class_counts - n_i
    t_i = _approximate_mode(class_counts_remaining, n_test, rng)

    class_indices = np.split(
        np.argsort(y_indices, kind="mergesort"),
        np.cumsum(class_counts)[:-1],
    )

    test = []
    for i in range(len(classes)):
        permutation = rng.permutation(class_counts[i])
        perm_indices_class_i = class_indices[i].take(permutation, mode="clip")
        test.extend(perm_indices_class_i[n_i[i]: n_i[i] + t_i[i]])

    test = rng.permutation(test)
    return [(entries[int(i)][0], entries[int(i)][1]) for i in test]


def run_test_mode(model, data_root=".", expected_accuracy=89.7):
    """
    Execute two checks:
    1) quick sanity set (5 files per class, AVR-exact path),
    2) deterministic parity set (training-equivalent path with optional warper).
    """
    quick_total = 0
    quick_correct = 0

    print("Running --test_mode quick check on 5 wav files per keyword folder...\n")

    for word in model["WORDS"]:
        folder = os.path.join(data_root, word)
        if not os.path.isdir(folder):
            print(f"[WARN] missing folder: {folder}")
            continue

        wavs = _collect_wavs(folder)[:5]
        if len(wavs) == 0:
            print(f"[WARN] no wav files found in {folder}")
            continue

        w_total = 0
        w_correct = 0
        for wav_path in wavs:
            sr, data = wavfile.read(wav_path)
            sig = _normalize_loaded_wav(data)
            if sr != SAMPLE_RATE:
                sig = _resample_to_sample_rate(sig, sr, SAMPLE_RATE)

            pred, dist, _ = classify_word(sig, model)
            ok = int(pred == word)
            quick_total += 1
            quick_correct += ok
            w_total += 1
            w_correct += ok
            print(f"{word:>5} | {os.path.basename(wav_path):<28} -> {pred:<8} dist={dist:7.4f}")

        w_acc = (100.0 * w_correct / max(1, w_total))
        print(f"  [{word}] {w_correct}/{w_total} = {w_acc:.1f}%\n")

    quick_acc = (100.0 * quick_correct / max(1, quick_total))
    print(f"QUICK-SET: {quick_acc:.1f}% over {quick_total} samples")

    warper = _load_feature_warper(os.path.join("templates", "feature_warper.npz"))
    if warper is not None:
        print("Parity mode: loaded templates/feature_warper.npz for training-equivalent evaluation.")
    else:
        print("Parity mode: feature warper missing; using AVR-exact path only.")

    test_entries = _build_deterministic_test_entries(
        model["WORDS"],
        data_root=data_root,
        test_fraction=0.2,
        seed=42,
    )
    parity_total = 0
    parity_correct = 0
    for true_word, wav_path in test_entries:
        sr, data = wavfile.read(wav_path)
        sig = _normalize_loaded_wav(data)
        if sr != SAMPLE_RATE:
            sig = _resample_to_sample_rate(sig, sr, SAMPLE_RATE)

        pred, _, _ = classify_word(sig, model, warper=warper)
        parity_total += 1
        parity_correct += int(pred == true_word)

    acc = (100.0 * parity_correct / max(1, parity_total))
    print(f"SELF-TEST: {acc:.1f}% (expected ~{expected_accuracy:.1f}%)")

    delta = abs(acc - expected_accuracy)
    if delta > 1.0:
        print(f"[FAIL] mismatch is {delta:.2f}% (> 1.0%).")
        return 1

    print(f"[PASS] mismatch is {delta:.2f}% (<= 1.0%).")
    return 0


def _led_indicator(pred_word, words):
    chunks = []
    for w in words:
        state = "ON" if pred_word == w else "--"
        chunks.append(f"{w}:{state}")
    return " | ".join(chunks)


def _record_one_second(duration=1.0):
    if sd is None:
        raise RuntimeError("sounddevice is not installed. Install with: pip install sounddevice")

    n = int(duration * SAMPLE_RATE)
    rec = sd.rec(n, samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    sig = rec[:, 0].astype(np.float32)
    return np.clip(sig, -1.0, 1.0)


def run_live_mode(model, duration=1.0):
    log_path = f"realtime_predictions_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "prediction", "distance", "threshold"])

        print("Header loaded. Real-time AVR-emulation mode started.")
        print(f"Threshold: {model['CONFIDENCE_THRESHOLD']:.4f}")
        print(f"CSV log: {log_path}")

        while True:
            user_in = input("Press ENTER to record 1 second (q to quit): ").strip().lower()
            if user_in == "q":
                break

            print("Recording...")
            sig = _record_one_second(duration=duration)
            pred, dist, _ = classify_word(sig, model)

            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([ts, pred, f"{dist:.6f}", f"{model['CONFIDENCE_THRESHOLD']:.6f}"])
            f.flush()

            print(f"Detected: {pred} | distance={dist:.4f}")
            print(_led_indicator(pred, model["WORDS"]))
            print("-")

    print(f"Session finished. Predictions saved to {log_path}")
    return 0


def _parse_duration(argv):
    duration = 1.0
    for arg in argv:
        if arg.startswith("--duration="):
            try:
                duration = float(arg.split("=", 1)[1])
            except Exception:
                pass
    return max(0.2, min(5.0, duration))


def main():
    if "--help" in sys.argv or "-h" in sys.argv:
        print("Usage:")
        print("  python realtime_test.py              # live mic mode")
        print("  python realtime_test.py --test_mode  # dataset self-test")
        print("  python realtime_test.py --duration=1.0")
        return 0

    header_path = "word_templates.h"
    if not os.path.isfile(header_path):
        print("word_templates.h not found in current directory.")
        return 1

    model = load_model_from_header(header_path)

    if model["N_FEATURES"] != N_FEATURES_EXPECTED:
        print(
            f"Header N_FEATURES={model['N_FEATURES']} but extractor expects {N_FEATURES_EXPECTED}."
        )
        return 1

    if "--test_mode" in sys.argv:
        return run_test_mode(model, data_root=".", expected_accuracy=89.7)

    duration = _parse_duration(sys.argv[1:])
    return run_live_mode(model, duration=duration)


if __name__ == "__main__":
    sys.exit(main())
