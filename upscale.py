#!/usr/bin/env python3
"""Batch-upscale images with Real-ESRGAN, with optional GFPGAN face restoration.

Usage:
    python upscale.py -i photos/ -o output/
    python upscale.py -i photo.jpg -o output/ --face-enhance
    python upscale.py -i photos.zip -o output/ --model RealESRGAN_x4plus_anime_6B --zip
"""

import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

# Model registry: RRDBNet architecture params + official weight URLs (xinntao/Real-ESRGAN releases).
MODELS = {
    "RealESRGAN_x4plus": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "scale": 4,
        "num_block": 23,
    },
    "RealESRNet_x4plus": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth",
        "scale": 4,
        "num_block": 23,
    },
    "RealESRGAN_x4plus_anime_6B": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "scale": 4,
        "num_block": 6,
    },
    "RealESRGAN_x2plus": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "scale": 2,
        "num_block": 23,
    },
}

GFPGAN_URL = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"


def _patch_torchvision_functional_tensor():
    """basicsr (unmaintained since 2022) imports a module torchvision>=0.17 removed.

    This registers a shim module before basicsr is imported anywhere, aliasing
    the missing `functional_tensor` submodule to the current `functional` one.
    """
    import sys as _sys
    import types

    if "torchvision.transforms.functional_tensor" in _sys.modules:
        return
    import torchvision.transforms.functional as F

    shim = types.ModuleType("torchvision.transforms.functional_tensor")
    shim.rgb_to_grayscale = F.rgb_to_grayscale
    _sys.modules["torchvision.transforms.functional_tensor"] = shim


def detect_device(requested):
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    print("No GPU detected -- running on CPU. This will be significantly slower.", file=sys.stderr)
    return "cpu"


def build_upsampler(model_name, device, tile, tile_pad, pre_pad, half):
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from basicsr.utils.download_util import load_file_from_url
    from realesrgan import RealESRGANer

    spec = MODELS[model_name]
    WEIGHTS_DIR.mkdir(exist_ok=True)
    model_path = WEIGHTS_DIR / f"{model_name}.pth"
    if not model_path.exists():
        print(f"Downloading {model_name} weights (first run only)...")
        load_file_from_url(spec["url"], model_dir=str(WEIGHTS_DIR), file_name=model_path.name)

    arch = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=spec["num_block"],
        num_grow_ch=32,
        scale=spec["scale"],
    )
    upsampler = RealESRGANer(
        scale=spec["scale"],
        model_path=str(model_path),
        model=arch,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=half,
        device=device,
    )
    return upsampler, spec["scale"]


def build_face_enhancer(upsampler, outscale, device):
    from basicsr.utils.download_util import load_file_from_url
    from gfpgan import GFPGANer

    WEIGHTS_DIR.mkdir(exist_ok=True)
    model_path = WEIGHTS_DIR / "GFPGANv1.4.pth"
    if not model_path.exists():
        print("Downloading GFPGAN face-restoration weights (first run only)...")
        load_file_from_url(GFPGAN_URL, model_dir=str(WEIGHTS_DIR), file_name=model_path.name)

    return GFPGANer(
        model_path=str(model_path),
        upscale=outscale,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=upsampler,
        device=device,
    )


def collect_input_files(input_path, work_dir):
    """Stage input images (from a folder, .zip, or single file) into work_dir.

    Always copies -- never reads or modifies the user's original files in place.
    """
    input_path = Path(input_path)
    staged = work_dir / "staged"
    staged.mkdir(parents=True, exist_ok=True)

    if input_path.is_dir():
        result = []
        for src in sorted(input_path.rglob("*")):
            if src.suffix.lower() in IMAGE_EXTS:
                dst = staged / src.name
                shutil.copy2(src, dst)
                result.append(dst)
        return result

    if input_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(input_path) as zf:
            for member in zf.namelist():
                if member.endswith("/"):
                    continue
                name = Path(member).name
                if Path(name).suffix.lower() not in IMAGE_EXTS or not name:
                    continue
                with zf.open(member) as src, open(staged / name, "wb") as dst:
                    shutil.copyfileobj(src, dst)
        return sorted(staged.iterdir())

    if input_path.suffix.lower() in IMAGE_EXTS:
        dst = staged / input_path.name
        shutil.copy2(input_path, dst)
        return [dst]

    raise ValueError(f"Unsupported input: {input_path} (expected an image file, a folder, or a .zip archive)")


def ensure_rgb(path):
    """Convert grayscale/palette/RGBA images to RGB; GFPGAN crashes on non-3-channel input."""
    from PIL import Image

    with Image.open(path) as img:
        if img.mode != "RGB":
            img.convert("RGB").save(path)


def process_images(files, upsampler, face_enhancer, outscale, output_dir, ext_override):
    import cv2
    from tqdm import tqdm

    output_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    for path in tqdm(files, desc="Upscaling", unit="img"):
        try:
            ensure_rgb(path)
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError("could not be read (corrupt or unsupported format)")

            if face_enhancer is not None:
                _, _, output = face_enhancer.enhance(
                    img, has_aligned=False, only_center_face=False, paste_back=True
                )
            else:
                output, _ = upsampler.enhance(img, outscale=outscale)

            out_ext = (ext_override or path.suffix.lstrip(".")).lower()
            out_path = output_dir / f"{path.stem}.{out_ext}"
            cv2.imwrite(str(out_path), output)
        except Exception as exc:
            failures.append((path.name, str(exc)))
    return failures


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Batch-upscale images with Real-ESRGAN, with optional GFPGAN face restoration."
    )
    parser.add_argument("-i", "--input", required=True, help="Image file, folder, or .zip archive")
    parser.add_argument("-o", "--output", default="output", help="Output folder (default: ./output)")
    parser.add_argument(
        "-n", "--model", default="RealESRGAN_x4plus", choices=sorted(MODELS), help="Model to use"
    )
    parser.add_argument(
        "--outscale", type=float, default=None, help="Final upscale factor (default: model's native scale)"
    )
    parser.add_argument("--face-enhance", action="store_true", help="Enable GFPGAN face restoration")
    parser.add_argument(
        "--tile", type=int, default=512, help="Tile size to limit GPU memory use, 0 disables tiling (default: 512)"
    )
    parser.add_argument("--tile-pad", type=int, default=10, help="Tile padding in pixels (default: 10)")
    parser.add_argument("--pre-pad", type=int, default=0, help="Pre-padding in pixels (default: 0)")
    parser.add_argument(
        "--half", action="store_true", help="Use fp16 inference (faster on most CUDA GPUs; ignored elsewhere)"
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu", "mps"],
        help="Compute device (default: auto-detect; mps is experimental)",
    )
    parser.add_argument("--ext", default=None, help="Force output file extension (default: keep original per file)")
    parser.add_argument("--zip", action="store_true", help="Also package results into <output>.zip")
    return parser


def main():
    args = build_arg_parser().parse_args()

    _patch_torchvision_functional_tensor()
    device = detect_device(args.device)

    half = args.half and device == "cuda"
    if args.half and not half:
        print("--half ignored: fp16 inference requires a CUDA GPU.", file=sys.stderr)

    upsampler, native_scale = build_upsampler(
        args.model, device, args.tile, args.tile_pad, args.pre_pad, half
    )
    outscale = args.outscale or native_scale
    face_enhancer = build_face_enhancer(upsampler, outscale, device) if args.face_enhance else None

    output_dir = Path(args.output)
    with tempfile.TemporaryDirectory(prefix="upscale_") as tmp:
        files = collect_input_files(args.input, Path(tmp))
        if not files:
            print("No supported image files found in input.", file=sys.stderr)
            sys.exit(1)

        print(f"Found {len(files)} image(s). Using {args.model} on {device}.")
        failures = process_images(files, upsampler, face_enhancer, outscale, output_dir, args.ext)

    succeeded = len(files) - len(failures)
    print(f"\nDone: {succeeded}/{len(files)} succeeded. Output in {output_dir}/")
    if failures:
        print("Failed files:")
        for name, err in failures:
            print(f"  - {name}: {err}")

    if args.zip:
        archive_path = shutil.make_archive(str(output_dir), "zip", str(output_dir))
        print(f"Packaged results: {archive_path}")

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
