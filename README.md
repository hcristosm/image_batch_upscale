# 📸 Real-ESRGAN Batch Upscaler (Google Colab)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hcristosm/image_batch_upscale/blob/main/batch_upscale.ipynb)

A practical and fully automated Python pipeline to **upscale images by 4x** using Artificial Intelligence (Real-ESRGAN + GFPGAN), running directly in Google Colab.

---
> ⚠️ **Stability Notice:** The `main` branch contains active development code and may be subject to unannounced updates or bugs. For the tested and official stable version, please use **[Release v1.0.0](https://github.com/hcristosm/image_batch_upscale/releases/tag/v1.0.0)**.
## ⚡ Key Features

- 📦 **Batch Processing:** Supports individual photos or `.zip` archives (automatically extracts files from subdirectories).
- 👤 **Face Restoration:** Integrated with GFPGAN to enhance and restore facial details with high clarity.
- 🛡️ **Anti-Crash & Error Handling:**
  - Automatically converts Grayscale (black & white) images to RGB to prevent processing crashes.
  - Tile-based processing (`--tile 512`) prevents GPU Out-Of-Memory (OOM) errors on large images.
- 📥 **Automatic Download:** Packages and downloads `fotos_upscaled.zip` as soon as processing finishes.

---

## 📖 Step-by-Step Usage Guide

### Step 1: Open in Google Colab
Click the **Open in Colab** badge at the top of this repository or open the `.ipynb` file directly in Colab.

### Step 2: Enable GPU Acceleration
Before running the script, ensure your Colab environment is using a GPU:
1. In the top menu, go to **Runtime** > **Change runtime type**.
2. Under *Hardware accelerator*, select **T4 GPU**.
3. Click **Save**.

### Step 3: Run the Code
1. Click the **Play (▶)** button on the main code cell (or press `Ctrl + Enter`).
2. Wait a few seconds while the script installs dependencies and applies environment fixes.

### Step 4: Upload Your Photos
1. When the **"Choose Files"** button appears, select:
   - One or multiple individual images (`.jpg`, `.png`, etc.); **OR**
   - A `.zip` file containing your images.
2. Upload will begin automatically.

### Step 5: Download Results
Once processing is complete, all upscaled images will be zipped, and the download for `fotos_upscaled.zip` will start **automatically in your browser**.

---

## 🛠️ Built With

- **[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN):** Core super-resolution algorithm.
- **[GFPGAN](https://github.com/TencentARC/GFPGAN):** Facial feature restoration model.
- **PyTorch & Torchvision:** Deep learning framework.
- **PIL (Pillow):** Image handling and color mode conversion.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE). Feel free to use, modify, and distribute it.
