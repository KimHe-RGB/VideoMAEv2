#!/usr/bin/env python3
"""
Script to prepare video dataset for VideoMAEv2 training.
- Finds all .mp4 files in a directory tree
- Renames videos by removing Chinese characters (keeping only 6-digit IDs)
- Copies them to a single output folder
- Generates train.csv, val.csv, and test.csv files

For pre-training: video_path,0,-1
For fine-tuning: video_path,label
"""

import os
import re
import shutil
import argparse
from pathlib import Path
from collections import defaultdict
import random


def extract_video_id(filename):
    """
    Extract video ID from filename.
    Format: 2 capital letters + 4 digits (e.g., AB1234)
    Or: 4 capital letters + 2 digits (e.g., ABCD12)
    Example: "AB1234中文.mp4" -> "AB1234"
    """
    match = re.search(r'([A-Z]{2}\d{4}|[A-Z]{4}\d{2})', filename)
    if match:
        return match.group(1)
    return None


def find_all_videos(source_dir):
    """
    Recursively find all .mp4 files in directory tree.
    Returns list of tuples: (original_path, video_id)
    """
    videos = []
    videos_without_id = []
    total_mp4_files = 0
    source_path = Path(source_dir)
    
    print(f"Scanning directory: {source_dir}")
    for video_file in source_path.rglob("*.mp4"):
        total_mp4_files += 1
        video_id = extract_video_id(video_file.name)
        if video_id:
            videos.append((str(video_file), video_id))
        else:
            videos_without_id.append(str(video_file))
            print(f"Warning: Could not extract ID from {video_file.name}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Video File Summary:")
    print(f"{'='*60}")
    print(f"Total .mp4 files found:        {total_mp4_files}")
    print(f"Files with valid ID:           {len(videos)}")
    print(f"Files without valid ID:        {len(videos_without_id)}")
    print(f"{'='*60}\n")
    
    if videos_without_id:
        print(f"Files without valid ID pattern ([A-Z]{{2}}\\d{{4}}):")
        for i, filepath in enumerate(videos_without_id[:10], 1):
            print(f"  {i}. {Path(filepath).name}")
        if len(videos_without_id) > 10:
            print(f"  ... and {len(videos_without_id) - 10} more")
        print()
    
    return videos


def copy_and_rename_videos(videos, output_dir):
    """
    Copy videos to output directory with cleaned names.
    Returns dict mapping: video_id -> new_path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    video_mapping = {}
    duplicates = defaultdict(list)
    
    print(f"\nCopying videos to: {output_dir}")
    for original_path, video_id in videos:
        new_filename = f"{video_id}.mp4"
        new_path = output_path / new_filename
        
        # Check for duplicates
        if video_id in video_mapping:
            duplicates[video_id].append(original_path)
            print(f"Warning: Duplicate ID {video_id} found:")
            print(f"  Kept: {video_mapping[video_id]}")
            print(f"  Skipped: {original_path}")
            continue
        
        # Copy file
        try:
            shutil.copy2(original_path, new_path)
            video_mapping[video_id] = str(new_path)
            print(f"Copied: {original_path} -> {new_filename}")
        except Exception as e:
            print(f"Error copying {original_path}: {e}")
    
    if duplicates:
        print(f"\nFound {len(duplicates)} duplicate video IDs")
    
    return video_mapping


def split_dataset(video_list, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Split video list into train/val/test sets.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    random.seed(seed)
    shuffled = video_list.copy()
    random.shuffle(shuffled)
    
    total = len(shuffled)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_set = shuffled[:train_end]
    val_set = shuffled[train_end:val_end]
    test_set = shuffled[val_end:]
    
    return train_set, val_set, test_set


def write_pretrain_csv(video_paths, output_file, data_root=None):
    """
    Write pre-training CSV file.
    Format: video_path,0,-1
    If data_root is provided, write relative paths.
    """
    with open(output_file, 'w') as f:
        for video_path in sorted(video_paths):
            if data_root:
                # Write relative path
                rel_path = os.path.relpath(video_path, data_root)
                f.write(f"{rel_path},0,-1\n")
            else:
                # Write absolute path
                f.write(f"{video_path},0,-1\n")
    
    print(f"Wrote {len(video_paths)} entries to {output_file}")


def write_finetune_csv(video_paths, output_file, label=0, data_root=None):
    """
    Write fine-tuning CSV file.
    Format: video_path,label
    If data_root is provided, write relative paths.
    """
    with open(output_file, 'w') as f:
        for video_path in sorted(video_paths):
            if data_root:
                # Write relative path
                rel_path = os.path.relpath(video_path, data_root)
                f.write(f"{rel_path},{label}\n")
            else:
                # Write absolute path
                f.write(f"{video_path},{label}\n")
    
    print(f"Wrote {len(video_paths)} entries to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare video dataset for VideoMAEv2"
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="Source directory containing videos in tree structure"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for renamed videos"
    )
    parser.add_argument(
        "--csv_output_dir",
        type=str,
        required=True,
        help="Directory to save CSV files (default: same as output_dir)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["pretrain", "finetune"],
        required=True,
        help="Dataset mode: pretrain or finetune"
    )
    parser.add_argument(
        "--label",
        type=int,
        default=0,
        help="Label for fine-tuning (default: 0)"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Training set ratio (default: 0.8)"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation set ratio (default: 0.1)"
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Test set ratio (default: 0.1)"
    )
    parser.add_argument(
        "--relative_paths",
        action="store_true",
        help="Write relative paths in CSV (relative to output_dir)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for dataset split"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Don't copy files, just show what would be done"
    )
    
    args = parser.parse_args()
    
    # Set CSV output directory
    csv_dir = args.csv_output_dir if args.csv_output_dir else args.output_dir
    Path(csv_dir).mkdir(parents=True, exist_ok=True)
    
    # Step 1: Find all videos
    print("=" * 60)
    print("Step 1: Finding videos...")
    print("=" * 60)
    videos = find_all_videos(args.source_dir)
    
    if len(videos) == 0:
        print("No videos found with valid IDs! Please check your source directory.")
        return
    
    # Step 2: Copy and rename videos
    print("\n" + "=" * 60)
    print("Step 2: Copying and renaming videos...")
    print("=" * 60)
    
    if args.dry_run:
        print("DRY RUN MODE - No files will be copied")
        video_mapping = {vid: os.path.join(args.output_dir, f"{vid}.mp4") 
                        for _, vid in videos}
    else:
        video_mapping = copy_and_rename_videos(videos, args.output_dir)
    
    print(f"\nProcessed {len(video_mapping)} unique videos")
    
    # Step 3: Split dataset
    print("\n" + "=" * 60)
    print("Step 3: Splitting dataset...")
    print("=" * 60)
    
    video_paths = list(video_mapping.values())
    train_set, val_set, test_set = split_dataset(
        video_paths,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed
    )
    
    print(f"Train set: {len(train_set)} videos ({args.train_ratio*100:.1f}%)")
    print(f"Val set:   {len(val_set)} videos ({args.val_ratio*100:.1f}%)")
    print(f"Test set:  {len(test_set)} videos ({args.test_ratio*100:.1f}%)")
    
    # Step 4: Write CSV files
    print("\n" + "=" * 60)
    print("Step 4: Writing CSV files...")
    print("=" * 60)
    
    data_root = args.output_dir if args.relative_paths else None
    
    if args.mode == "pretrain":
        write_pretrain_csv(
            train_set,
            os.path.join(csv_dir, "train.csv"),
            data_root
        )
        write_pretrain_csv(
            val_set,
            os.path.join(csv_dir, "val.csv"),
            data_root
        )
        write_pretrain_csv(
            test_set,
            os.path.join(csv_dir, "test.csv"),
            data_root
        )
    else:  # finetune
        write_finetune_csv(
            train_set,
            os.path.join(csv_dir, "train.csv"),
            args.label,
            data_root
        )
        write_finetune_csv(
            val_set,
            os.path.join(csv_dir, "val.csv"),
            args.label,
            data_root
        )
        write_finetune_csv(
            test_set,
            os.path.join(csv_dir, "test.csv"),
            args.label,
            data_root
        )
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    print(f"\nVideos location: {args.output_dir}")
    print(f"CSV files location: {csv_dir}")
    print(f"\nGenerated files:")
    print(f"  - train.csv ({len(train_set)} videos)")
    print(f"  - val.csv ({len(val_set)} videos)")
    print(f"  - test.csv ({len(test_set)} videos)")


if __name__ == "__main__":
    main()
