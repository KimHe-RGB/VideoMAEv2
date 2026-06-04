# Evaluation Pipeline Scripts

This folder contains scripts to run a full experiment pipeline:

1. Post-pretraining on unlabeled clips
2. Finetuning on labeled split files
3. Evaluation on test split

Each pipeline run is stored in one run folder under `RUNS_ROOT/RUN_NAME`, including:

- Stage logs and checkpoints
- Exported stage checkpoints
- Exact command history
- Resolved config values
- Git head and working tree status
- Snapshot copy of `train.csv`, `val.csv`, `test.csv` and checksums

## Scripts

- `run_full_pipeline.sh`: orchestrates selected stages
- `01_post_pretrain.sh`: stage 1 only
- `02_finetune.sh`: stage 2 only
- `03_evaluate.sh`: stage 3 only
- `pipeline_common.sh`: shared helper functions

## Quick Start

1. Create config file:

```bash
cp Evaluation/pipeline_config.example.env Evaluation/pipeline_config.env
```

2. Edit `Evaluation/pipeline_config.env` for your dataset paths and parameters.

3. Run full pipeline:

```bash
bash Evaluation/run_full_pipeline.sh Evaluation/pipeline_config.env
```

## Notes

- `UNLABELED_CLIPS_DIR` can contain filenames with spaces. Stage 1 auto-creates safe symlinks and manifest.
- `FINETUNE_DATA_PATH` must contain `train.csv`, `val.csv`, and `test.csv`.
- Stage 1 uses `run_mae_pretraining.py` and requires torch checkpoint format for `POST_INIT_CKPT`.
- Stage 2/3 use `torchrun` for compatibility with distributed barriers in `run_class_finetuning.py`.
