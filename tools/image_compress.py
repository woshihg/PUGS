#!/usr/bin/env python3
"""
Compress and resize images in a folder by a given scale and save to another folder.
Usage examples:
  python tools/image_compress.py --src images/ --dst images_small/ --scale 0.5 --quality 85 --recursive

Features:
- Scales images by a multiplier (scale). If scale is <= 0 raises error.
- Preserves folder structure when --recursive is used.
- Keeps image format by default. For JPEG you can set --quality (1-95).
- Optionally overwrite existing files.
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Iterable

try:
    from PIL import Image, ExifTags
except Exception:
    raise ImportError("Pillow is required. Install with: pip install pillow")


SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}


# Determine LANCZOS resampling constant in a way compatible with multiple Pillow versions
if hasattr(Image, 'Resampling'):
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
elif hasattr(Image, 'LANCZOS'):
    RESAMPLE_LANCZOS = Image.LANCZOS
elif hasattr(Image, 'BICUBIC'):
    RESAMPLE_LANCZOS = Image.BICUBIC
else:
    # Fallback numeric constant (most common enums: 1=NEAREST, 2=BILINEAR, 3=BICUBIC, 4=LANCZOS)
    RESAMPLE_LANCZOS = 4


def iter_image_files(src_dir: Path, recursive: bool = True) -> Iterable[Path]:
    if recursive:
        for p in src_dir.rglob('*'):
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
                yield p
    else:
        for p in src_dir.iterdir():
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
                yield p


def compress_image(in_path: Path, out_path: Path, scale: float, quality: int = 85) -> None:
    if scale <= 0:
        raise ValueError('scale must be > 0')
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(in_path) as im:
        # Preserve mode for PNG transparency
        orig_format = im.format
        w, h = im.size
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        if (new_w, new_h) != (w, h):
            im = im.resize((new_w, new_h), resample=RESAMPLE_LANCZOS)

        save_kwargs = {}
        suffix = out_path.suffix.lower()
        if orig_format and orig_format.lower() in ('jpeg', 'jpg') or suffix in ('.jpg', '.jpeg'):
            # For JPEG use quality and optimize
            save_kwargs['quality'] = max(1, min(95, int(quality)))
            save_kwargs['optimize'] = True
            # Try to preserve exif
            try:
                exif = im.info.get('exif')
                if exif:
                    save_kwargs['exif'] = exif
            except Exception:
                pass
            if im.mode in ('RGBA', 'LA'):
                # JPEG doesn't support alpha, convert to RGB using white background
                background = Image.new('RGB', im.size, (255, 255, 255))
                background.paste(im, mask=im.split()[-1])
                im = background
            else:
                if im.mode != 'RGB':
                    im = im.convert('RGB')
        elif suffix == '.png' or (orig_format and orig_format.lower() == 'png'):
            # For PNG use optimize and let Pillow choose compression
            save_kwargs['optimize'] = True
        else:
            # Other formats: let Pillow handle defaults
            pass

        # Save
        im.save(str(out_path), **save_kwargs)


def process_directory(src: str, dst: str, scale: float, recursive: bool = True, quality: int = 85, overwrite: bool = False) -> int:
    src_path = Path(src)
    dst_path = Path(dst)
    if not src_path.exists() or not src_path.is_dir():
        raise FileNotFoundError(f'source directory not found: {src}')
    dst_path.mkdir(parents=True, exist_ok=True)

    files = list(iter_image_files(src_path, recursive))
    total = len(files)
    if total == 0:
        print('No supported image files found.')
        return 0

    print(f'Found {total} images. Processing...')
    processed = 0
    for i, f in enumerate(files, 1):
        # compute relative path to preserve structure
        rel = f.relative_to(src_path)
        out_file = dst_path / rel
        # ensure extension remains same
        if not overwrite and out_file.exists():
            print(f'[{i}/{total}] Skipping existing: {rel}')
            continue
        try:
            compress_image(f, out_file, scale=scale, quality=quality)
            processed += 1
            print(f'[{i}/{total}] -> {rel} ({f.suffix})')
        except Exception as e:
            print(f'[{i}/{total}] Failed {rel}: {e}')
    print(f'Done. Processed {processed}/{total} images.')
    return processed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Compress/resize all images in a folder and save to another folder')
    p.add_argument('--src', '-s', required=True, help='Source directory containing images')
    p.add_argument('--dst', '-d', required=True, help='Destination directory to save compressed images')
    p.add_argument('--scale', type=float, required=True, help='Scale multiplier (e.g. 0.5 to reduce to half size)')
    p.add_argument('--quality', type=int, default=85, help='JPEG quality (1-95). Only used for JPEG output')
    p.add_argument('--no-recursive', dest='recursive', action='store_false', help='Do not recurse into subdirectories')
    p.add_argument('--overwrite', action='store_true', help='Overwrite existing files in destination')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        processed = process_directory(args.src, args.dst, args.scale, recursive=args.recursive, quality=args.quality, overwrite=args.overwrite)
    except Exception as e:
        print('Error:', e)


if __name__ == '__main__':
    main()
