# Evaluation Pipeline Scripts

This folder now supports a two-stage unlabeled-video workflow before finetuning:

1. Prepare processed videos from the raw CSV of source videos
2. Post-pretrain on the processed unlabeled clips
3. Finetune on labeled split files
4. Evaluate on test split

Each pipeline run is stored in one run folder under `RUNS_ROOT/RUN_NAME`, including:

- Processed videos and per-video face/head metadata
- Stage logs and checkpoints
- Exported stage checkpoints
- Exact command history
- Resolved config values
- Git head and working tree status
- Snapshot copy of `train.csv`, `val.csv`, `test.csv` and checksums

## Scripts

- `run_full_pipeline.sh`: orchestrates selected stages
- `00_prepare_clips.sh`: preprocess the raw unlabeled CSV into cropped videos
- `01_post_pretrain.sh`: stage 1 only
- `02_finetune.sh`: stage 2 only
- `03_evaluate.sh`: stage 3 only
- `preprocess_face_centered_videos.py`: uses ModelScope DAMO-YOLO head detection and keeps the face/head closest to frame center
- `convert_hf_safetensors_to_pth.py`: converts Hugging Face `model.safetensors` into a torch checkpoint for stage 1
- `pipeline_common.sh`: shared helper functions

## Preprocessing Behavior

`00_prepare_clips.sh` reads the source CSV, opens each input video from the `实际路径` column, runs the ModelScope detector `iic/cv_tinynas_head-detection_damoyolo` on every frame, and:

- keeps all detected boxes in per-video JSON metadata
- selects the box closest to the image center when multiple heads are detected
- smooths the selected box across time
- crops a square region around the selected box with configurable expansion
- resizes the processed video to `224x224` by default

The output of stage 0 is a directory of processed MP4 files in `Data/dataset_post_pretrain`, plus `preprocess_summary.csv` and compact per-video JSON metadata.

## Quick Start

1. Create config file:

```bash
cp Evaluation/pipeline_config.example.env Evaluation/pipeline_config.env
```

2. Edit `Evaluation/pipeline_config.env` if needed.

3. Run the requested workflow:

```bash
bash Evaluation/run_full_pipeline.sh Evaluation/pipeline_config.env
```

By default this now runs `prepare_clips,post_pretrain`, with processed videos exported to `Data/dataset_post_pretrain`.

## Checkpoint Handling

`01_post_pretrain.sh` accepts `POST_INIT_CKPT` as either:

- `.safetensors`
- `.pth`, `.pt`, or `.bin`

If a `.safetensors` path is provided, the script auto-converts it into a torch checkpoint inside the run folder before starting `run_mae_pretraining.py`.

## Recommended Settings For This Dataset

For the current infant-video setup:

- detector preprocess on GPU: `PREP_DEVICE=cuda`
- processed frame size: `PREP_OUTPUT_SIZE=224`
- post-pretrain model: `POST_MODEL=pretrain_videomae_huge_patch16_224`
- post-pretrain clip setting: `POST_NUM_FRAMES=16`
- post-pretrain temporal stride: `POST_SAMPLING_RATE=4`
- post-pretrain crop size: `POST_INPUT_SIZE=224`
- initial weights: `POST_INIT_CKPT=/root/Desktop/vit_h_model.safetensors`

This gives VideoMAEv2 a standard 16x4 training window on face-centered infant clips.

## Notes

- The detector model from ModelScope is a head detector rather than a strict face detector. In this pipeline it is used as the infant-face proxy because it is robust to partial face views and off-angle recordings.
- The preprocessing stage will now target CUDA by default once the PyTorch build supports the local V100 GPUs.
- Detector preprocessing is intended to run on the local Tesla V100 GPUs. If CUDA validation fails, the PyTorch build in the venv is still wrong and must be replaced before running stage 0.
- Stage 2/3 still use `torchrun` for compatibility with distributed barriers in `run_class_finetuning.py`.
