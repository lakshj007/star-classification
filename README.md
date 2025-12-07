# Star Classification

This repository now includes a full training pipeline for a convolutional neural network that detects stars directly on the supplied FITS frames.

## Setup

1. Create a virtual environment (recommended) and install the dependencies:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Ensure your FITS files live underneath `Practice Image Sets/` (subdirectories are scanned recursively).

## Training the detector

Run the training script to automatically mine positive/negative patches, perform heavy augmentation, and fit the CNN:

```bash
python train_star_detector.py \
  --data-dir "Practice Image Sets" \
  --output-dir artifacts \
  --epochs 30 \
  --patch-size 32
```

Useful flags:

- `--pos-per-image` / `--neg-per-image`: how many star/background patches to mine per frame.
- `--candidate-method`: choose `opencv` (default) to run the denoising + contour-based star finder, or `localmax` for the older local-maxima heuristic.
- `--cv-blur-ksize`, `--cv-thresh-factor`, `--cv-thresh-offset`: control the smoothing and adaptive/global thresholds used by the OpenCV pipeline (lower offsets/factors => more detections).
- `--cv-min-area` / `--cv-max-area`: acceptable contour areas in pixels.
- `--cv-min-distance`: minimum separation between detections in pixels.
- `--cv-tile-size` / `--cv-max-per-tile`: cap how many detections each tile can contribute so a single bright cluster can't dominate positives.
- `--threshold`: only used when `--candidate-method localmax` is selected (sigma threshold for local maxima).
- `--device`: force `cpu` if CUDA is unavailable.
- `--predict path/to/file.fits`: after training, slide the model over this frame and save a FITS probability map.
- `--detect-threshold`: probability cut-off when reporting predicted stars.
- `--save-overlay`: also export a PNG overlay with red circles showing detected stars.
- `--show-overlay`: pop up an interactive matplotlib window of the detections (close to continue).

Training artifacts:

- `artifacts/star_detector.pt`: best model checkpoint.
- `artifacts/training_metrics.json`: loss/accuracy history you can plot for debugging.
- `artifacts/<frame>_probability_map.fits`: probability heatmap for a predicted frame.
- `artifacts/<frame>_detections.json`: pixel coordinates and probabilities for each reported detection.
- `artifacts/<frame>_overlay.png`: optional visualization when `--save-overlay` is specified.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
