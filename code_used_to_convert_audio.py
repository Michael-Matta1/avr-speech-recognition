"""
Dataset Audio Conversion Utility.

Copyright (c) 2026 Michael Matta
GitHub: https://github.com/Michael-Matta1

Licensed under the MIT License. See LICENSE for details.

Converts Speech Commands WAV files into the deployment-friendly format used by
this project:
- mono channel
- 8 kHz sample rate
- unsigned 8-bit PCM WAV

Output structure:
    ./converted/<word>/<filename>.wav
"""

import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
from math import gcd

# Configuration
CATEGORIES  = ["down", "go", "left", "off", "on", "right", "stop", "up"]
TARGET_RATE = 8000
OUTPUT_DIR  = "converted"


def convert_wav(src_path: str, dst_path: str) -> bool:
    try:
        src_rate, data = wavfile.read(src_path)

        # 1) Convert to float32 in [-1, 1] for a consistent processing path.
        if data.dtype == np.uint8:
            data = (data.astype(np.float32) - 128.0) / 128.0
        elif data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        else:
            data = data.astype(np.float32)

        # 2) Downmix multi-channel input to mono.
        if data.ndim > 1:
            data = data.mean(axis=1)

        # 3) Resample to the deployment sample rate (8 kHz).
        if src_rate != TARGET_RATE:
            common = gcd(TARGET_RATE, src_rate)
            up     = TARGET_RATE // common
            down   = src_rate    // common
            data   = resample_poly(data, up, down)

        # 4) Quantize to unsigned 8-bit PCM, as required by WAV format.
        data = np.clip(data, -1.0, 1.0)
        data = ((data + 1.0) * 127.5).astype(np.uint8)

        # 5) Write output file.
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        wavfile.write(dst_path, TARGET_RATE, data)
        return True

    except Exception as exc:
        print(f"  [ERROR] {src_path}: {exc}")
        return False


def main():
    root = os.getcwd()
    print(f"Working directory : {root}")
    print(f"Target format     : Mono / {TARGET_RATE} Hz / 8-bit unsigned PCM")
    print(f"Output directory  : {os.path.join(root, OUTPUT_DIR)}")
    print(f"Categories        : {CATEGORIES}\n")

    total_ok = total_err = 0

    for category in CATEGORIES:
        src_dir = os.path.join(root, category)
        dst_dir = os.path.join(root, OUTPUT_DIR, category)

        if not os.path.isdir(src_dir):
            print(f"[SKIP] '{category}' folder not found at {src_dir}")
            continue

        wav_files = [f for f in os.listdir(src_dir) if f.lower().endswith(".wav")]
        ok = err = 0

        print(f"[{category}]  {len(wav_files)} files found ...")

        for fname in wav_files:
            src_path = os.path.join(src_dir, fname)
            dst_path = os.path.join(dst_dir, fname)
            if convert_wav(src_path, dst_path):
                ok  += 1
            else:
                err += 1

        print(f"  converted: {ok}   errors: {err}")
        total_ok  += ok
        total_err += err

    print("\n" + "-" * 50)
    print(f"Done!  Total converted: {total_ok}  |  Total errors: {total_err}")
    print(f"Output saved to: {os.path.join(root, OUTPUT_DIR)}/")


if __name__ == "__main__":
    main()
