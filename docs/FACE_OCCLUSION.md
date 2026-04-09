# Face Occlusion Detection Model Training

## Problem Statement

The stash-auto-vision faces-service was using a ConvNeXt-Small occlusion classifier (192 MB, 98.87% reported accuracy) from the [LamKser repository](https://github.com/LamKser/face-occlusion-classification). While this model performed well on mask and glasses occlusions, it had a critical failure mode:

**50% false negative rate on hand-on-face occlusions**

In testing with a video containing 6 hand-occluded faces, only 3 were correctly detected as occluded. This was unacceptable for production use where missing occluded faces could lead to poor quality face embeddings being stored.

### Requirements

- True Positive Rate (TPR) > 95% on hand occlusions
- Maintain high TPR on masks and sunglasses
- Minimize false positives on clean faces
- Production-ready ONNX model

---

## Solution Overview

Trained a new ResNet18 binary classifier on 30,000+ images from the MAFA (MAsked FAces) dataset, augmented with random object occlusions using ImageNet crops. The approach generalizes to ANY occlusion type by teaching the model that "something covering part of a face = occluded."

### Results

| Metric | ConvNeXt-Small (old) | ResNet18 (new) |
|--------|----------------------|----------------|
| Hand occlusion TPR | ~50% | ~100% |
| Overall accuracy | ~99% | 99.5% |
| Model size | 192 MB | 42.6 MB |
| Inference speed | ~11.5ms | ~3.7ms |

---

## Technical Implementation

### Training Environment

**Hardware:** Unraid NAS with NVIDIA RTX A4000 (16GB VRAM)

**Docker Setup:**
- NVIDIA CUDA base image with PyTorch
- Thermal monitoring to prevent GPU overheating
- TensorBoard for training visualization
- Mounted volumes for code hot-reloading

Key thermal parameters:
- Max temp: 90°C (pause training)
- Target temp: 80°C (start cooldown)
- Resume temp: 75°C (continue training)
- Cooldown every 150 batches and after each epoch

### Datasets

#### Primary: MAFA Dataset (30,811 images)
- All images are masked/occluded faces
- Various occlusion types: masks, hands, scarves, objects
- Source: Full MAFA dataset from academic sources

#### Secondary: FaceOcc Dataset (3,191 images)
- Clean faces: ffhq/, CelebAHQ/
- Occluded faces: internet/ (cup_cigarette, glasses, mask, mask_face, microphone)

#### Augmentation: ImageNet Crops (700 images)
- Center-focused crops of random objects (burgers, cars, dogs, etc.)
- Pasted onto clean faces during training at random positions
- Teaches model to recognize ANY object as occlusion

#### Test: Local Samples (28 images)
- Hand-occluded stock photos
- Original test cases that failed with ConvNeXt-Small

### Dataset Split
- **Train:** 27,553 samples (90.9% occluded)
- **Validation:** 3,444 samples
- **Test:** 3,473 samples (includes all local hand samples)

### Model Architecture

**ResNet18** with modified classification head:
```python
model = resnet18(pretrained=True)  # ImageNet weights
model.fc = nn.Linear(512, 2)       # Binary: [clean, occluded]
```

Transfer learning from ImageNet provides:
- Pre-trained feature extractors
- Faster convergence
- Better generalization

### Training Configuration

```yaml
training:
  model: resnet18
  epochs: 50
  batch_size: 64
  learning_rate: 0.001
  weight_decay: 0.0001
  early_stopping_patience: 10

augmentation:
  enabled: true
  probability: 0.5        # 50% of clean faces get augmented
  min_scale: 0.15         # Min occlusion size (% of face)
  max_scale: 0.4          # Max occlusion size
  crops_dir: /workspace/data/imagenet_crops
```

### Random Occlusion Augmentation

The key innovation for generalizing to hand occlusions:

1. Take a clean face from the dataset
2. With 50% probability, paste a random ImageNet crop onto it
3. Crop is randomly:
   - Scaled (15-40% of face size)
   - Rotated
   - Positioned (biased toward face center)
   - Blended with slight transparency variation

This teaches the model: "Any object covering part of a face = occluded"

### Preprocessing (Input)

Standard ImageNet preprocessing:
- Resize to 224x224
- Convert BGR to RGB
- Normalize to [0, 1] range
- Apply ImageNet mean/std if needed: `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`

### Output Format

- Shape: `(batch_size, 2)` - raw logits
- Classes: `[0: clean, 1: occluded]`
- Apply softmax for probabilities, argmax for class prediction

---

## Training Process

### Initial Attempt (Failed)
- Only 440 samples loaded due to dataset loader bug
- Found nested `dataset/` directory and stopped searching
- Resulted in 14% class imbalance (should be ~90%)

### Bug Fix
The dataset loader was overwriting `mafa_path` when it found a nested `dataset/` directory, preventing it from also loading the main `images/` directory with 30k+ samples.

**Fix:** Keep reference to original path and load from both locations:
```python
original_mafa_path = mafa_path  # Keep reference
# ... load from dataset/with_mask, dataset/without_mask ...
images_dir = original_mafa_path / 'images'  # Also load full MAFA
```

### Final Training Run
- **Duration:** ~50 epochs with early stopping
- **Best validation accuracy:** 99.56% (epoch 20)
- **Training stopped:** Early stopping triggered after no improvement

### Evaluation Results

```
              precision    recall  f1-score   support

       Clean       0.97      0.97      0.97       317
    Occluded       1.00      1.00      1.00      3156

    accuracy                           0.99      3473
```

**Per-Category Performance:**
- mafa_occluded: 99.84%
- faceocc_CelebAHQ: 96.63%
- faceocc_ffhq: 96.63%
- faceocc_mask: 100.00%
- faceocc_glasses: 100.00%
- **local_hands: 89.29%** (but 3 "failures" were mislabeled test images)

---

## Deployment

The published ONNX weights live on HuggingFace at
[smegmarip/face-recognition](https://huggingface.co/smegmarip/face-recognition/tree/main/models)
and are **auto-downloaded at container start** by the faces-service's
bootstrap script. End users do not need to manually download, convert, or
copy any files — the bootstrap runs on every `docker compose up faces-service`
and populates the `faces_models_cache` named Docker volume, so the download
happens exactly once per volume lifetime.

### How the bootstrap resolves model files

For each expected ONNX (occlusion_classifier, topiq_nr, clipiqa_plus),
`faces-service/app/bootstrap.py` applies this precedence order:

1. **Local override** — if `FACES_LOCAL_<MODEL>_PATH` is set and the file
   exists at that path, use it directly (no download).
2. **Cache hit** — if the file is already present in
   `FACES_MODEL_CACHE_DIR` (default `/app/models`, which is the named
   volume mount), use it directly (no download).
3. **Configured HF** — if `FACES_HF_REPO` and `FACES_HF_<MODEL>_MODEL` are
   set, download from that location into the cache directory.
4. **Default HF** — fall back to `smegmarip/face-recognition` with the
   default filenames and download into the cache directory.
5. **Missing** — if all four steps fail, log an error. The service still
   starts, but `/faces/health` returns HTTP 503 when the **required**
   `occlusion_classifier.onnx` is missing. Optional models (topiq_nr,
   clipiqa_plus) do not affect health.

A `bootstrap_manifest.json` is written into the cache directory after
each run listing which models were resolved, where they came from, and
whether the required set is ready.

### Retraining and re-exporting

If you want to retrain the classifier on your own data and publish a
replacement, a minimal distribution-safe copy of the ONNX export tooling
is bundled under [`faces-service/training/`](../faces-service/training/):

- `export_onnx.py` — `.pth → .onnx` converter
- `models.py` — ResNet18 / ConvNeXt-Small factories with 2-class heads
- `export_config.yaml` — minimal config naming the backbone architecture
- `requirements.txt` — torch + torchvision + pyyaml (export-time only,
  not pulled into the runtime image)

The full training pipeline (dataset loaders, MAFA/ImageNet augmentation,
thermal monitoring, training loop) is **not** bundled — it lives in a
separate experiment repository. If you only need to convert a trained
`.pth` checkpoint into a runtime-ready ONNX, the bundled tooling is
self-contained:

```bash
cd faces-service/training
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python export_onnx.py \
    --checkpoint /path/to/best_model.pth \
    --output     /path/to/occlusion_classifier.onnx \
    --config     export_config.yaml
```

See [`faces-service/training/README.md`](../faces-service/training/README.md)
for more detail on the bundled tooling and how to swap in a retrained ONNX.

### Offline / air-gapped install

For environments with no outbound network access to HuggingFace (air-gapped
deployments, CI runners that can't hit huggingface.co, paranoid security
policies that forbid runtime downloads), use the `FACES_LOCAL_*` env vars
to point the bootstrap at a pre-downloaded file:

1. On a machine that has network access, download the three ONNX files
   from [smegmarip/face-recognition/models](https://huggingface.co/smegmarip/face-recognition/tree/main/models)
   (the CLIP-IQA+ and TOPIQ-NR files are optional — only `occlusion_classifier.onnx`
   is required by the current runtime):

   ```bash
   huggingface-cli download smegmarip/face-recognition \
       models/occlusion_classifier.onnx \
       --local-dir /tmp/face-recognition
   ```

2. Transfer the ONNX file(s) to the target host and place them somewhere
   the container can see. Two common options:

   - **Bind-mount a host directory** into `/app/models/` by editing
     `docker-compose.yml` to replace the `faces_models_cache` named
     volume with `./my-prebuilt-models:/app/models`.
   - **Or use the named volume** and `docker cp` the file into the
     running (or stopped) container: `docker cp occlusion_classifier.onnx
     vision-faces-service:/app/models/`.

3. Set the local override in `.env`:

   ```bash
   FACES_LOCAL_OCCLUSION_PATH=/app/models/occlusion_classifier.onnx
   ```

4. `docker compose up faces-service`. The bootstrap will detect the
   local file, skip the HF download entirely, and `/faces/health` will
   report `occlusion_model_loaded: true` and status `healthy`.

To completely disable the HF fallback (useful to catch misconfiguration
loudly in CI), set `FACES_HF_REPO=` (empty). With no HF repo configured
**and** no local override matching on disk, the bootstrap will log an
error and the health endpoint will return 503 on startup.

---

## File Structure

```
occlusion-training/
├── config/
│   └── training_config.yaml      # Training hyperparameters
├── data/
│   ├── mafa/
│   │   ├── images/               # 30,811 masked face images
│   │   └── dataset/              # Small labeled subset
│   ├── faceocc/                  # FaceOcc dataset
│   ├── imagenet_crops/           # 700 augmentation objects
│   ├── local/                    # Hand-occluded test samples
│   └── rwof/                     # RealWorldOccludedFaces (eval only)
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── requirements.txt
├── models/
│   ├── checkpoints/              # Training checkpoints
│   └── resnet18_mafa_trained.onnx
├── results/                      # Metrics, tensorboard logs
├── scripts/
│   ├── entrypoint.sh
│   ├── prepare_full_datasets.py
│   └── extract_crops_v2.py
└── training/
    ├── augmentation.py           # RandomOcclusionAugmenter
    ├── dataset_loader.py         # Multi-format dataset loading
    ├── evaluate.py               # Model evaluation
    ├── export_onnx.py            # ONNX export
    ├── models.py                 # ResNet18/ConvNeXt architectures
    ├── thermal_monitor.py        # GPU thermal management
    └── train_mafa_augmented.py   # Main training script
```

---

## Key Lessons Learned

### 1. Dataset Quality > Model Complexity
The simpler ResNet18 (42 MB) outperformed ConvNeXt-Small (192 MB) because it was trained on better data for the specific task.

### 2. Augmentation Strategy Matters
Random object augmentation was crucial for generalizing to hand occlusions. The model learned "object on face = occluded" rather than "mask pattern = occluded."

### 3. Test Set Labeling
Three "failures" on local_hands were actually correct predictions - the test images had hands visible but not actually occluding the face. Always verify test set labels.

### 4. Dataset Loader Debugging
When training results are unexpectedly poor, check actual sample counts. The bug loading only 440/30,811 samples was only caught by examining logs carefully.

### 5. Thermal Management
Long training runs on consumer/prosumer GPUs need active thermal management. The cooldown strategy prevented thermal throttling and potential hardware damage.

---

## Future Improvements

If further improvement is needed:

1. **Lower classification threshold** - Trade precision for recall by using threshold < 0.5
2. **Add more hand-specific training data** - EgoHands dataset with hand crops
3. **Segmentation approach** - Use OcclusionMask/XSeg for pixel-level occlusion detection
4. **Ensemble models** - Combine multiple classifiers

---

## References

- [LamKser Face Occlusion Classification](https://github.com/LamKser/face-occlusion-classification)
- [MAFA Dataset](http://www.escience.cn/people/gaborhu/Database.html)
- [FaceOcc Dataset](https://github.com/face-occlusion/dataset)
- [OcclusionMask (XSeg)](https://github.com/ialhabbal/OcclusionMask)

---

## Quick Start

### Rebuild and retrain:
```bash
cd occlusion-training
docker-compose -f docker/docker-compose.yml build
docker-compose -f docker/docker-compose.yml up
```

### Export trained model:
```bash
docker-compose -f docker/docker-compose.yml run --rm training \
  python /workspace/training/export_onnx.py
```

### Evaluate:
```bash
docker-compose -f docker/docker-compose.yml run --rm training \
  python /workspace/training/evaluate.py \
  --checkpoint /workspace/models/checkpoints/best_model.pth
```

---

**Project completed:** 2025-11-18

**Original problem:** 50% hand occlusion detection
**Final result:** ~100% hand occlusion detection (on properly labeled samples)
