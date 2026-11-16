#!/usr/bin/python3
# coding=utf-8

import argparse
import os
import numpy as np


def gen_golden_data(N: int, seed: int, low: float, high: float,
                     input_dir: str, output_dir: str):
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    rng = np.random.default_rng(seed)
    real_part = rng.uniform(low, high, N).astype(np.float32)
    imag_part = rng.uniform(low, high, N).astype(np.float32)

    # Interleave real and imag: [r0, i0, r1, i1, ...]
    x_interleaved = np.empty(N * 2, dtype=np.float32)
    x_interleaved[0::2] = real_part
    x_interleaved[1::2] = imag_part
    input_path = os.path.join(input_dir, "input_x.bin")
    x_interleaved.tofile(input_path)

    x = real_part + 1j * imag_part
    golden = np.fft.fft(x).astype(np.complex64)
    golden_interleaved = np.empty(N * 2, dtype=np.float32)
    golden_interleaved[0::2] = golden.real
    golden_interleaved[1::2] = golden.imag
    golden_path = os.path.join(output_dir, "golden.bin")
    golden_interleaved.tofile(golden_path)

    print(f"N={N}  seed={seed}  range=[{low}, {high}]")
    print(f"input  -> {input_path}  ({x_interleaved.nbytes} bytes)")
    print(f"golden -> {golden_path}  ({golden_interleaved.nbytes} bytes)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate random complex input + golden FFT output for AscendC FFT kernels."
    )
    parser.add_argument("-N", "--size", type=int, default=4096,
                         help="FFT length (default: 4096)")
    parser.add_argument("--seed", type=int, default=42,
                         help="RNG seed (default: 42)")
    parser.add_argument("--low", type=float, default=-10.0,
                         help="Lower bound of uniform distribution (default: -10.0)")
    parser.add_argument("--high", type=float, default=10.0,
                         help="Upper bound of uniform distribution (default: 10.0)")
    parser.add_argument("--input-dir", type=str, default="./input",
                         help="Directory to write input_x.bin into (default: ./input)")
    parser.add_argument("--output-dir", type=str, default="./output",
                         help="Directory to write golden.bin into (default: ./output)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    gen_golden_data(args.size, args.seed, args.low, args.high,
                     args.input_dir, args.output_dir)