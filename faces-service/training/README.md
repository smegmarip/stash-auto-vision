# faces-service/training

Reference tooling for converting a trained occlusion-classifier PyTorch
checkpoint into the ONNX file that the faces-service runtime loads.

## What this directory is

The published face occlusion classifier used by the faces-service at runtime
(`faces-service/app/occlusion_detector.py`) is a ResNet18 binary classifier
trained on MAFA + ImageNet-crop augmentation. The weights live at
[smegmarip/face-recognition](https://huggingface.co/smegmarip/face-recognition/tree/main/models)
as `occlusion_classifier.onnx` and are auto-downloaded into the
`faces_models_cache` Docker volume by `faces-service/app/bootstrap.py` on
first container start. End users don't need anything in this directory —
the runtime pulls the published weights automatically.

**This directory exists for two narrow use cases:**

1. **Traceability** — a minimal, human-readable record of how the published
   ONNX file was built, so you can verify or reproduce the export path.
2. **Custom retraining** — if you retrain the classifier on your own
   dataset and want to ship a replacement ONNX, this is the tooling for
   the `.pth → .onnx` conversion step.

## What this directory is NOT

This is not the full training pipeline. Training the classifier requires:

- The MAFA (MAsked FAces) dataset and/or FaceOcc
- ImageNet crops for random-object occlusion augmentation
- ~50 epochs on a ≥16 GB CUDA GPU with thermal management
- The `train_mafa_augmented.py` / `dataset_loader.py` / `augmentation.py`
  modules from the broader training experiment repo (not shipped here)

See [`docs/FACE_OCCLUSION.md`](../../docs/FACE_OCCLUSION.md) at the repo
root for the full training write-up — datasets, hyperparameters, MAFA
loader bug fix, results table, etc.

## Files

| File                 | Purpose                                                                     |
| -------------------- | --------------------------------------------------------------------------- |
| `export_onnx.py`     | CLI tool: loads a .pth checkpoint, exports to .onnx (opset 11, dynamic batch) |
| `models.py`          | torchvision-backed ResNet18 / ConvNeXt-Small factories with 2-class heads   |
| `export_config.yaml` | Minimal 1-field config naming the backbone architecture                     |
| `requirements.txt`   | torch + torchvision + pyyaml (export-time only; not in the runtime image)   |

## Usage

```bash
cd faces-service/training
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python export_onnx.py \
    --checkpoint /path/to/your/best_model.pth \
    --output     /path/to/occlusion_classifier.onnx \
    --config     export_config.yaml
```

The script accepts either a full training checkpoint dict (with
`model_state_dict`) or a raw state dict. ONNX opset defaults to 11,
which is what the faces-service runtime expects.

## Deploying a re-exported model

Once you have a new `occlusion_classifier.onnx`, you can wire it into the
faces-service two ways:

### Option 1: Replace the published HF weights

Upload your ONNX to a fork of `smegmarip/face-recognition` (or your own
public / private repo) and set in `.env`:

```bash
FACES_HF_REPO=your-org/your-repo
FACES_HF_OCCLUSION_MODEL=models/occlusion_classifier.onnx   # adjust path as needed
FACES_HF_TOKEN=hf_xxx   # only if the repo is private
```

On next `docker compose up faces-service`, `bootstrap.py` will download
the replacement from your repo instead of `smegmarip/face-recognition`.

### Option 2: Local-path override (air-gapped / offline)

Drop the ONNX file somewhere the container can see it and point the local
override env var at it:

```bash
# in .env
FACES_LOCAL_OCCLUSION_PATH=/app/models/occlusion_classifier.onnx
```

Then mount the file into the container via a bind mount or volume, and
restart. `bootstrap.py` will detect the local file and skip the HF
download entirely.

See the **Offline / air-gapped install** section in
[`docs/FACE_OCCLUSION.md`](../../docs/FACE_OCCLUSION.md) for a full walkthrough.

## Reference: current published checkpoint

- **Architecture:** ResNet18 (torchvision, 2-class head)
- **Input:** 224×224 RGB, ImageNet mean/std normalized
- **Output:** raw logits, shape `(batch, 2)`, classes `[clean, occluded]`
- **File size:** ~42.6 MB
- **ONNX opset:** 11
- **Training accuracy:** ~99.56% validation (MAFA + FaceOcc + ImageNet-crop augmentation)
- **Hand-occlusion recall:** ~100% on the local hand-occlusion test set
  (previous ConvNeXt-Small model was ~50%)

Full training write-up: [`docs/FACE_OCCLUSION.md`](../../docs/FACE_OCCLUSION.md).
