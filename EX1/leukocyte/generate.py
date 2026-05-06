#!/usr/bin/env python3
"""
Leukocyte Tracking Input Data Generator

Generates AVI video files in raw/uncompressed format for leukocyte tracking benchmark.
The videos contain synthetic grayscale frames simulating microscopy cell images.
"""

import argparse
import os
import struct
import sys
from pathlib import Path
from typing import Tuple
import numpy as np


# Preset configurations: (width, height, frames, fps, description)
# NOTE: All presets use 640x480 resolution because the leukocyte code has
# hardcoded crop region (TOP=110, BOTTOM=328 in find_ellipse.h)
PRESETS = {
    "mini": (640, 480, 50, 30, "Mini video for quick testing (~1s)"),
    "small": (640, 480, 150, 30, "Small video (~5s)"),
    "medium": (640, 480, 600, 30, "Medium video (reference dataset, ~20s)"),
    "large": (640, 480, 1000, 30, "Large video (~60s)"),
    "extra-large": (640, 480, 1500, 30, "Extra large video (~120s)"),
}


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def safe_join(directory: str, filename: str, overwrite: bool) -> str:
    """Safely construct file path, checking for overwrites."""
    filepath = os.path.join(directory, filename)
    if os.path.exists(filepath) and not overwrite:
        raise FileExistsError(f"File already exists: {filepath}. Use --overwrite to replace.")
    return filepath


def write_fourcc(f, fourcc: str):
    """Write a FOURCC code (4-character code)."""
    f.write(fourcc.encode('ascii'))


def write_u32(f, value: int):
    """Write unsigned 32-bit integer (little-endian)."""
    f.write(struct.pack('<I', value))


def write_u16(f, value: int):
    """Write unsigned 16-bit integer (little-endian)."""
    f.write(struct.pack('<H', value))


def generate_synthetic_frame(width: int, height: int, frame_idx: int, seed: int) -> np.ndarray:
    """
    Generate a synthetic grayscale frame simulating microscopy cell images.
    
    Returns:
        8-bit grayscale image as numpy array
    """
    np.random.seed(seed + frame_idx)
    
    # Start with background noise
    frame = np.random.randint(20, 60, (height, width), dtype=np.uint8)
    
    # Add 3-8 circular "cells" with darker/lighter regions
    num_cells = np.random.randint(3, 9)
    for _ in range(num_cells):
        cx = np.random.randint(width // 4, 3 * width // 4)
        cy = np.random.randint(height // 4, 3 * height // 4)
        radius = np.random.randint(15, 40)
        intensity = np.random.randint(80, 200)
        
        # Create coordinate grids
        y, x = np.ogrid[:height, :width]
        mask = (x - cx)**2 + (y - cy)**2 <= radius**2
        frame[mask] = intensity
        
        # Add gradient/texture
        if np.random.rand() > 0.5:
            gradient = np.linspace(0, 30, radius*2)
            for r in range(radius):
                mask_ring = ((x - cx)**2 + (y - cy)**2 >= r**2) & ((x - cx)**2 + (y - cy)**2 < (r+1)**2)
                if r < len(gradient):
                    frame[mask_ring] = np.clip(frame[mask_ring] + gradient[r], 0, 255).astype(np.uint8)
    
    # Add slight motion blur effect (simulate cell movement)
    if frame_idx > 0:
        offset = np.random.randint(-2, 3, size=2)
        frame = np.roll(frame, offset, axis=(0, 1))
    
    return frame


def write_avi_file(filepath: str, width: int, height: int, fps: int, frames: np.ndarray):
    """
    Write raw/uncompressed AVI file with 8-bit grayscale frames.
    
    Args:
        filepath: Output AVI file path
        width: Frame width in pixels
        height: Frame height in pixels
        fps: Frames per second
        frames: Array of shape (num_frames, height, width) with dtype uint8
    """
    num_frames = len(frames)
    frame_size = width * height  # 8-bit grayscale
    
    # Pad width to multiple of 4 for alignment
    padded_width = (width + 3) & ~3
    padded_frame_size = padded_width * height
    
    # Build palette (256 grayscale entries)
    palette = bytearray()
    for i in range(256):
        palette.extend([i, i, i, 0])  # B, G, R, reserved
    
    # Calculate sizes for RIFF structure
    # idx1 chunk: 'idx1' (4) + size (4) + data
    idx1_chunk_size = 8 + 16 * num_frames
    
    # movi LIST content size (value in the size field, includes 'movi' fourcc + data)
    movi_content_size = 4 + num_frames * (8 + padded_frame_size)  # 'movi' fourcc + frames
    
    # hdrl LIST content
    strf_size = 40 + len(palette)  # BITMAPINFOHEADER + palette
    avih_chunk = 8 + 56  # 'avih' + size + data
    strh_chunk = 8 + 56  # 'strh' + size + data
    strf_chunk = 8 + strf_size  # 'strf' + size + data
    strl_content = 4 + strh_chunk + strf_chunk  # 'strl' fourcc + chunks
    hdrl_content_size = 4 + avih_chunk + 8 + strl_content  # 'hdrl' + avih + LIST + strl
    
    # RIFF size = everything after 'RIFF' + size field
    # = 'AVI ' (4) + LIST hdrl (8 + hdrl_content) + LIST movi (8 + movi_content) + idx1 chunk
    avi_size = 4 + (8 + hdrl_content_size) + (8 + movi_content_size) + idx1_chunk_size
    
    with open(filepath, 'wb') as f:
        # RIFF header
        write_fourcc(f, 'RIFF')
        write_u32(f, avi_size)
        write_fourcc(f, 'AVI ')
        
        # hdrl LIST
        write_fourcc(f, 'LIST')
        write_u32(f, hdrl_content_size)
        write_fourcc(f, 'hdrl')
        
        # avih (Main AVI Header)
        write_fourcc(f, 'avih')
        write_u32(f, 56)  # avih size
        write_u32(f, int(1000000 / fps))  # dwMicroSecPerFrame
        write_u32(f, frame_size * fps)    # dwMaxBytesPerSec
        write_u32(f, 0)                   # dwPaddingGranularity
        write_u32(f, 0x10)                # dwFlags (AVIF_HASINDEX)
        write_u32(f, num_frames)          # dwTotalFrames
        write_u32(f, 0)                   # dwInitialFrames
        write_u32(f, 1)                   # dwStreams
        write_u32(f, 0)                   # dwSuggestedBufferSize
        write_u32(f, width)               # dwWidth
        write_u32(f, height)              # dwHeight
        write_u32(f, 0)                   # dwReserved[4]
        write_u32(f, 0)
        write_u32(f, 0)
        write_u32(f, 0)
        
        # strl LIST (Stream list)
        write_fourcc(f, 'LIST')
        write_u32(f, strl_content)
        write_fourcc(f, 'strl')
        
        # strh (Stream header)
        write_fourcc(f, 'strh')
        write_u32(f, 56)  # strh size
        write_fourcc(f, 'vids')  # fccType (video stream)
        write_fourcc(f, '\x00\x00\x00\x00')  # fccHandler (uncompressed)
        write_u32(f, 0)       # dwFlags
        write_u16(f, 0)       # wPriority
        write_u16(f, 0)       # wLanguage
        write_u32(f, 0)       # dwInitialFrames
        write_u32(f, 1)       # dwScale
        write_u32(f, fps)     # dwRate (fps = dwRate/dwScale)
        write_u32(f, 0)       # dwStart
        write_u32(f, num_frames)  # dwLength
        write_u32(f, padded_frame_size)  # dwSuggestedBufferSize
        write_u32(f, 0)       # dwQuality (-1 = default)
        write_u32(f, 0)       # dwSampleSize
        write_u16(f, 0)       # rcFrame.left
        write_u16(f, 0)       # rcFrame.top
        write_u16(f, width)   # rcFrame.right
        write_u16(f, height)  # rcFrame.bottom
        
        # strf (Stream format - BITMAPINFOHEADER)
        write_fourcc(f, 'strf')
        write_u32(f, strf_size)
        write_u32(f, 40)      # biSize (BITMAPINFOHEADER size)
        write_u32(f, width)   # biWidth
        write_u32(f, height)  # biHeight
        write_u16(f, 1)       # biPlanes
        write_u16(f, 8)       # biBitCount (8-bit)
        write_u32(f, 0)       # biCompression (BI_RGB = uncompressed)
        write_u32(f, padded_frame_size)  # biSizeImage
        write_u32(f, 0)       # biXPelsPerMeter
        write_u32(f, 0)       # biYPelsPerMeter
        write_u32(f, 256)     # biClrUsed (palette entries)
        write_u32(f, 0)       # biClrImportant
        f.write(palette)      # Color palette
        
        # movi LIST (Movie data)
        write_fourcc(f, 'LIST')
        write_u32(f, movi_content_size)
        write_fourcc(f, 'movi')
        
        # Write frames and collect index entries
        frame_offsets = []
        # According to AVI spec (used by avilib), index offsets are relative to 
        # the position of the 'movi' fourcc itself (LIST + 8), not after it.
        # At this point we've written: 'LIST' (4) + size (4) + 'movi' (4)
        # Current position is right after 'movi', so we subtract 4 to get movi fourcc pos
        movi_base = f.tell() - 4
        
        for frame in frames:
            write_fourcc(f, '00db')  # Uncompressed video frame
            write_u32(f, padded_frame_size)
            
            # Record offset pointing to frame DATA (after chunk ID and size)
            # For idx_type=2, avilib calculates: pos + movi_start + 4
            # We want this to equal current position (start of data)
            # So: pos + movi_base + 8 = f.tell()
            # Therefore: pos = f.tell() - movi_base - 8
            frame_offset = f.tell() - movi_base - 8
            frame_offsets.append(frame_offset)
            
            # Write frame data with padding
            # AVI format stores bitmaps bottom-up (last row first)
            for row in frame[::-1]:  # Reverse row order
                f.write(row.tobytes())
                if padded_width > width:
                    f.write(b'\x00' * (padded_width - width))
        
        # Write index chunk (idx1)
        idx1_data_size = 16 * num_frames  # Each index entry is 16 bytes
        write_fourcc(f, 'idx1')
        write_u32(f, idx1_data_size)
        
        for offset in frame_offsets:
            write_fourcc(f, '00db')  # Chunk ID
            write_u32(f, 0x10)       # Flags (AVIIF_KEYFRAME)
            write_u32(f, offset)     # Offset from movi
            write_u32(f, padded_frame_size)  # Size
    
    print(f"  Created AVI: {filepath} ({width}x{height}, {num_frames} frames, {fps} fps)")


def generate_synthetic(args):
    """Generate synthetic AVI video files."""
    width, height, num_frames, fps, desc = PRESETS[args.preset]
    
    # Override with explicit parameters if provided
    if args.width:
        width = args.width
    if args.height:
        height = args.height
    if args.frames:
        num_frames = args.frames
    if args.fps:
        fps = args.fps
    
    # Validate parameters
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid dimensions: {width}x{height}")
    if num_frames <= 0:
        raise ValueError(f"Invalid frame count: {num_frames}")
    if fps <= 0:
        raise ValueError(f"Invalid fps: {fps}")
    
    print(f"Generating synthetic AVI video ({args.preset}): {width}x{height}, {num_frames} frames, {fps} fps")
    
    # Generate frames
    frames = []
    for i in range(num_frames):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  Generating frame {i+1}/{num_frames}...")
        frame = generate_synthetic_frame(width, height, i, args.seed)
        frames.append(frame)
    
    frames = np.array(frames, dtype=np.uint8)
    
    # Write AVI file
    ensure_dir(args.out_dir)
    output_path = safe_join(args.out_dir, "testfile.avi", args.overwrite)
    write_avi_file(output_path, width, height, fps, frames)
    
    print(f"✓ Synthetic AVI generated successfully")
    print(f"  Size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Generate input AVI video files for leukocyte tracking benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate mini dataset
  python generate.py synthetic --preset mini --out-dir input_data/mini --seed 42
  
  # Generate medium dataset (reference)
  python generate.py synthetic --preset medium --out-dir input_data/medium
  
  # Custom size
  python generate.py synthetic --preset small --width 512 --height 384 --frames 200 --out-dir input_data/custom
"""
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Synthetic generation
    synthetic_parser = subparsers.add_parser('synthetic', help='Generate synthetic AVI video')
    synthetic_parser.add_argument('--out-dir', type=str, required=True,
                                   help='Output directory (will be created if needed)')
    synthetic_parser.add_argument('--preset', type=str, default='medium',
                                   choices=list(PRESETS.keys()),
                                   help='Preset configuration')
    synthetic_parser.add_argument('--width', type=int, help='Override frame width')
    synthetic_parser.add_argument('--height', type=int, help='Override frame height')
    synthetic_parser.add_argument('--frames', type=int, help='Override number of frames')
    synthetic_parser.add_argument('--fps', type=int, help='Override frames per second')
    synthetic_parser.add_argument('--seed', type=int, default=42,
                                   help='Random seed for reproducibility')
    synthetic_parser.add_argument('--overwrite', action='store_true',
                                   help='Overwrite existing files')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == 'synthetic':
            generate_synthetic(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

