# Real-ESRGAN Batch Upscaler

A practical, self-contained Python CLI to **upscale images** using AI (Real-ESRGAN + optional GFPGAN face restoration), running **locally on your own machine** (GPU or CPU) or in **Docker**.

> This project previously ran as a Google Colab notebook. It has been rewritten from scratch as a local tool and hasn't been battle-tested across every environment yet. If you need the older, tested Colab-based version, see [Release v1.0.0](https://github.com/hcristosm/image_batch_upscale/releases/tag/v1.0.0).

---

## Key Features

- **Batch processing** — point it at a single image, a folder (including subfolders), or a `.zip` archive.
- **Face restoration** — optional GFPGAN integration to enhance and restore facial details (`--face-enhance`).
- **Multiple models** — choose between general-purpose, anime, and 2x/4x Real-ESRGAN variants (`--model`).
- **Automatic GPU/CPU detection** — uses CUDA or Apple Silicon (MPS) when available, falls back to CPU with a warning otherwise.
- **Memory-safe** — tile-based processing (`--tile`) avoids GPU out-of-memory errors on large images.
- **Never touches your originals** — inputs are always copied to a temporary workspace before processing.
- **Docker support** — run without installing PyTorch/CUDA on your host at all.

---

## Requirements

- Python 3.9–3.12 (very new Python releases may not yet have PyTorch wheels available)
- ~4 GB of disk space for model weights and dependencies
- A CUDA-capable GPU is recommended but not required (CPU inference works, just slower)

---

## Installation (local, with a virtual environment)

```bash
git clone https://github.com/hcristosm/image_batch_upscale.git
cd image_batch_upscale

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 1. Install PyTorch first, matching your hardware:
#    https://pytorch.org/get-started/locally/
#    Examples:
pip install torch torchvision                                             # CPU-only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # NVIDIA GPU (CUDA 12.8)

# 2. Install the rest of the dependencies
pip install -r requirements.txt
```

Model weights are downloaded automatically into `weights/` the first time you use them, and reused afterward.

---

## Installation (Docker)

No local Python/CUDA setup required — the image bundles PyTorch with CUDA support.

```bash
docker build -t image-upscaler .

docker run --rm --gpus all \
  -v "$(pwd)/photos:/app/input" \
  -v "$(pwd)/output:/app/output" \
  -v "$(pwd)/weights:/app/weights" \
  image-upscaler -i /app/input -o /app/output --face-enhance
```

- Drop `--gpus all` to run on CPU.
- Mounting `weights/` as a volume keeps downloaded models cached between runs.

---

## Usage

```bash
# Basic: upscale every image in a folder
python upscale.py -i photos/ -o output/

# Single photo with face restoration
python upscale.py -i photo.jpg -o output/ --face-enhance

# A .zip archive, anime model, also zip the results
python upscale.py -i photos.zip -o output/ --model RealESRGAN_x4plus_anime_6B --zip
```

### Options

| Flag | Default | Description |
|---|---|---|
| `-i, --input` | *(required)* | Image file, folder, or `.zip` archive |
| `-o, --output` | `output` | Output folder |
| `-n, --model` | `RealESRGAN_x4plus` | Model to use (see table below) |
| `--outscale` | model's native scale | Final upscale factor |
| `--face-enhance` | off | Enable GFPGAN face restoration |
| `--tile` | `512` | Tile size in pixels to limit GPU memory use; `0` disables tiling |
| `--tile-pad` | `10` | Tile padding in pixels |
| `--pre-pad` | `0` | Pre-padding in pixels |
| `--half` | off | fp16 inference (CUDA only, ignored elsewhere) |
| `--device` | `auto` | `auto`, `cuda`, `cpu`, or `mps` (Apple Silicon, experimental) |
| `--ext` | keep original | Force output file extension |
| `--zip` | off | Also package the output folder into `<output>.zip` |

### Supported models

| Model | Scale | Best for |
|---|---|---|
| `RealESRGAN_x4plus` | 4x | General photos (default) |
| `RealESRNet_x4plus` | 4x | General photos, milder/less sharpened output |
| `RealESRGAN_x4plus_anime_6B` | 4x | Anime / illustrations, smaller and faster network |
| `RealESRGAN_x2plus` | 2x | General photos, 2x scale |

---

## Troubleshooting

- **`CUDA out of memory`** — lower `--tile` (e.g. `256` or `128`).
- **`ImportError: torchvision.transforms.functional_tensor`** — this is a known incompatibility between the (unmaintained) `basicsr` package and newer `torchvision` releases. `upscale.py` patches around it automatically at startup; you shouldn't need to do anything.
- **`libGL.so.1: cannot open shared object file`** (Docker only) — already handled by the provided `Dockerfile`; rebuild the image if you hit this.
- **Duplicate filenames** — when processing a folder with subfolders or a `.zip`, files are flattened into a single output folder by filename; two inputs with the same basename will overwrite each other.
- **Slow processing** — expected on CPU; a CUDA GPU (or Apple Silicon via `--device mps`, experimental) is much faster.

---

## Built With

- **[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN):** Core super-resolution algorithm.
- **[GFPGAN](https://github.com/TencentARC/GFPGAN):** Facial feature restoration model.
- **PyTorch & Torchvision:** Deep learning framework.
- **OpenCV & Pillow:** Image I/O and color mode handling.

---

## License

This project is licensed under the [MIT License](LICENSE). Feel free to use, modify, and distribute it.
