"""Signal-processing feature extraction for VoiceOrigin AI.

Pure numpy/scipy implementation (no librosa/ffmpeg) so the service builds
fast and reliably on Render's free tier. Input must be 16-bit/32-bit PCM WAV,
mono or stereo (the client encodes WAV before upload).
"""

import io
import wave
from dataclasses import dataclass, field

import numpy as np

FRAME_MS = 30
HOP_MS = 10
MIN_F0 = 70.0
MAX_F0 = 400.0
VOICING_THRESHOLD = 0.35
SILENCE_ENERGY_PERCENTILE = 30


@dataclass
class FeatureSet:
    duration_sec: float
    voiced_ratio: float
    mean_f0: float
    f0_std: float
    jitter: float
    shimmer: float
    mean_formants: tuple
    hf_energy_ratio: float
    hf_energy_std: float
    silence_rms_std: float
    voiced_frame_count: int
    sample_rate: int


def read_wav_mono(data: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(data), "rb") as wf:
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        sampwidth = wf.getsampwidth()
        channels = wf.getnchannels()
        raw = wf.readframes(n_frames)

    if sampwidth == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sampwidth == 1:
        audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128.0
    else:
        raise ValueError(f"unsupported sample width: {sampwidth}")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    return audio, sr


def _frame_starts(n_samples: int, frame_len: int, hop_len: int):
    return range(0, max(n_samples - frame_len, 0), hop_len)


def _autocorr_f0(frame: np.ndarray, sr: int, fmin=MIN_F0, fmax=MAX_F0):
    frame = frame - np.mean(frame)
    if np.max(np.abs(frame)) < 1e-6:
        return 0.0, 0.0
    windowed = frame * np.hanning(len(frame))
    corr = np.correlate(windowed, windowed, mode="full")
    corr = corr[len(corr) // 2:]
    if corr[0] <= 0:
        return 0.0, 0.0

    min_lag = int(sr / fmax)
    max_lag = min(int(sr / fmin), len(corr) - 1)
    if max_lag <= min_lag:
        return 0.0, 0.0

    segment = corr[min_lag:max_lag]
    peak_idx = int(np.argmax(segment))
    peak_val = segment[peak_idx]
    voicing = float(peak_val / (corr[0] + 1e-9))
    lag = min_lag + peak_idx
    f0 = sr / lag if lag > 0 else 0.0
    return f0, voicing


def _lpc_coeffs(frame: np.ndarray, order: int):
    windowed = frame * np.hamming(len(frame))
    autocorr = np.correlate(windowed, windowed, mode="full")[len(windowed) - 1:]
    autocorr = autocorr[: order + 1]
    if autocorr[0] == 0:
        return None

    a = np.zeros(order + 1)
    a[0] = 1.0
    e = autocorr[0]
    for i in range(1, order + 1):
        acc = autocorr[i] - np.sum(a[1:i] * autocorr[i - 1:0:-1])
        k = acc / e if e != 0 else 0.0
        a_new = a.copy()
        a_new[i] = k
        a_new[1:i] = a[1:i] - k * a[i - 1:0:-1]
        a = a_new
        e *= 1 - k**2
        if e <= 0:
            break
    return a


def _formants_from_frame(frame: np.ndarray, sr: int):
    order = 2 + sr // 1000
    a = _lpc_coeffs(frame, order)
    if a is None:
        return []

    roots = np.roots(a)
    roots = roots[np.imag(roots) >= 0]
    angs = np.arctan2(np.imag(roots), np.real(roots))
    freqs = angs * (sr / (2 * np.pi))
    bandwidths = -0.5 * (sr / (2 * np.pi)) * np.log(np.abs(roots) + 1e-9)

    formants = sorted(
        f for f, b in zip(freqs, bandwidths) if 90 < f < 4000 and b < 400
    )
    return formants


def _hf_energy_ratio(frame: np.ndarray, sr: int, cutoff_hz=4000.0):
    spectrum = np.abs(np.fft.rfft(frame * np.hanning(len(frame))))
    freqs = np.fft.rfftfreq(len(frame), d=1.0 / sr)
    total = np.sum(spectrum**2) + 1e-12
    hf = np.sum(spectrum[freqs >= cutoff_hz] ** 2)
    return float(hf / total)


def extract_features(wav_bytes: bytes) -> FeatureSet:
    audio, sr = read_wav_mono(wav_bytes)
    duration_sec = len(audio) / sr

    frame_len = int(sr * FRAME_MS / 1000)
    hop_len = int(sr * HOP_MS / 1000)
    if frame_len <= 0 or len(audio) < frame_len:
        raise ValueError("clip too short to analyze")

    f0s, voicings, rms_values, hf_ratios, formant_frames = [], [], [], [], []

    for start in _frame_starts(len(audio), frame_len, hop_len):
        frame = audio[start:start + frame_len]
        f0, voicing = _autocorr_f0(frame, sr)
        rms = float(np.sqrt(np.mean(frame**2)))
        rms_values.append(rms)
        hf_ratios.append(_hf_energy_ratio(frame, sr))

        f0s.append(f0)
        voicings.append(voicing)
        if voicing >= VOICING_THRESHOLD and f0 > 0:
            formant_frames.append(_formants_from_frame(frame, sr))

    voicings = np.array(voicings)
    f0s = np.array(f0s)
    rms_values = np.array(rms_values)
    hf_ratios = np.array(hf_ratios)

    voiced_mask = voicings >= VOICING_THRESHOLD
    voiced_ratio = float(np.mean(voiced_mask)) if len(voiced_mask) else 0.0
    voiced_f0 = f0s[voiced_mask]

    mean_f0 = float(np.mean(voiced_f0)) if len(voiced_f0) else 0.0
    f0_std = float(np.std(voiced_f0)) if len(voiced_f0) else 0.0

    if len(voiced_f0) > 1:
        periods = 1.0 / voiced_f0
        jitter = float(np.mean(np.abs(np.diff(periods))) / (np.mean(periods) + 1e-9))
    else:
        jitter = 0.0

    voiced_rms = rms_values[voiced_mask]
    if len(voiced_rms) > 1:
        shimmer = float(np.mean(np.abs(np.diff(voiced_rms))) / (np.mean(voiced_rms) + 1e-9))
    else:
        shimmer = 0.0

    unvoiced_rms = rms_values[~voiced_mask]
    if len(unvoiced_rms) > 1:
        threshold = np.percentile(unvoiced_rms, SILENCE_ENERGY_PERCENTILE)
        quiet = unvoiced_rms[unvoiced_rms <= max(threshold, 1e-6)]
        silence_rms_std = float(np.std(quiet)) if len(quiet) > 1 else float(np.std(unvoiced_rms))
    else:
        silence_rms_std = 0.0

    hf_energy_ratio = float(np.mean(hf_ratios)) if len(hf_ratios) else 0.0
    hf_energy_std = float(np.std(hf_ratios)) if len(hf_ratios) else 0.0

    formant_positions = [[], [], []]
    for formants in formant_frames:
        for i in range(3):
            if i < len(formants):
                formant_positions[i].append(formants[i])
    mean_formants = tuple(
        float(np.mean(vals)) if vals else 0.0 for vals in formant_positions
    )

    return FeatureSet(
        duration_sec=duration_sec,
        voiced_ratio=voiced_ratio,
        mean_f0=mean_f0,
        f0_std=f0_std,
        jitter=jitter,
        shimmer=shimmer,
        mean_formants=mean_formants,
        hf_energy_ratio=hf_energy_ratio,
        hf_energy_std=hf_energy_std,
        silence_rms_std=silence_rms_std,
        voiced_frame_count=int(np.sum(voiced_mask)),
        sample_rate=sr,
    )
