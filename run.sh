source .venv/bin/activate

pip install -r requirements.txt

cd ~/Datasets
conda run -n videomae python ./make_dataset.py \
    --source_dir ./深圳盐田区筛查婴幼儿视频 \
    --output_dir ./ChildVideo \
    --csv_output_dir ./csv_output \
    --mode pretrain \
    --dry_run
