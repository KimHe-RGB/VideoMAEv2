#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
except ImportError as exc:
    raise SystemExit(
        'modelscope is required for face/head preprocessing. Install it with `pip install modelscope`.'
    ) from exc

MODEL_ID = 'iic/cv_tinynas_head-detection_damoyolo'
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.mpeg', '.mpg'}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Detect the center-most infant head/face proxy in each frame and write cropped videos.'
    )
    parser.add_argument('--input_csv', required=True)
    parser.add_argument('--videos_dir', required=True)
    parser.add_argument('--metadata_dir', required=True)
    parser.add_argument('--manifest_path', required=True)
    parser.add_argument('--model_id', default=MODEL_ID)
    parser.add_argument('--path_column', default='实际路径')
    parser.add_argument('--id_column', default='编号')
    parser.add_argument('--score_threshold', type=float, default=0.35)
    parser.add_argument('--crop_scale', type=float, default=1.15)
    parser.add_argument('--output_size', type=int, default=224)
    parser.add_argument('--smooth_alpha', type=float, default=0.8)
    parser.add_argument('--max_videos', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max_frames', type=int, default=0)
    parser.add_argument('--skip_existing', action='store_true')
    parser.add_argument('--draw_boxes', action='store_true')
    return parser.parse_args()


def sanitize_name(raw: str, fallback: str) -> str:
    value = ''.join(ch if ch.isalnum() or ch in '._-' else '_' for ch in str(raw).strip())
    value = value.strip('._')
    return value or fallback


def clamp_box(box: np.ndarray, width: int, height: int) -> np.ndarray:
    x1, y1, x2, y2 = box.astype(np.float32)
    x1 = max(0.0, min(x1, width - 1.0))
    y1 = max(0.0, min(y1, height - 1.0))
    x2 = max(x1 + 1.0, min(x2, width * 1.0))
    y2 = max(y1 + 1.0, min(y2, height * 1.0))
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def center_distance(box: np.ndarray, width: int, height: int) -> float:
    cx = float(box[0] + box[2]) / 2.0
    cy = float(box[1] + box[3]) / 2.0
    return (cx - width / 2.0) ** 2 + (cy - height / 2.0) ** 2


def choose_center_box(boxes: list[np.ndarray], width: int, height: int) -> np.ndarray | None:
    if not boxes:
        return None
    return min(boxes, key=lambda item: center_distance(item, width, height))


def smooth_box(previous_box: np.ndarray | None, current_box: np.ndarray, alpha: float) -> np.ndarray:
    if previous_box is None:
        return current_box
    return alpha * previous_box + (1.0 - alpha) * current_box


def crop_side_from_box(box: np.ndarray, crop_scale: float) -> float:
    x1, y1, x2, y2 = [float(v) for v in box]
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    return max(box_w, box_h) * crop_scale


def square_crop(
    frame: np.ndarray,
    box: np.ndarray,
    crop_scale: float,
    output_size: int,
    fixed_side: float | None = None,
) -> np.ndarray:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    side = fixed_side if fixed_side is not None else crop_side_from_box(box, crop_scale)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    half = side / 2.0

    crop_x1 = int(math.floor(cx - half))
    crop_y1 = int(math.floor(cy - half))
    crop_x2 = int(math.ceil(cx + half))
    crop_y2 = int(math.ceil(cy + half))

    pad_left = max(0, -crop_x1)
    pad_top = max(0, -crop_y1)
    pad_right = max(0, crop_x2 - width)
    pad_bottom = max(0, crop_y2 - height)

    if any(v > 0 for v in (pad_left, pad_top, pad_right, pad_bottom)):
        frame = cv2.copyMakeBorder(
            frame,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_REPLICATE,
        )
        crop_x1 += pad_left
        crop_x2 += pad_left
        crop_y1 += pad_top
        crop_y2 += pad_top

    cropped = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    if cropped.size == 0:
        cropped = cv2.resize(frame, (output_size, output_size), interpolation=cv2.INTER_AREA)
        return cropped
    return cv2.resize(cropped, (output_size, output_size), interpolation=cv2.INTER_AREA)


def parse_detection_result(result: Any, width: int, height: int, threshold: float) -> tuple[list[np.ndarray], list[float]]:
    if result is None:
        return [], []

    boxes = result.get('boxes')
    if boxes is None:
        boxes = result.get('bboxes')
    if boxes is None:
        boxes = []

    scores = result.get('scores')
    if scores is None:
        scores = []

    if isinstance(boxes, np.ndarray):
        boxes = boxes.tolist()
    if isinstance(scores, np.ndarray):
        scores = scores.tolist()

    parsed_boxes: list[np.ndarray] = []
    parsed_scores: list[float] = []

    if boxes and isinstance(boxes[0], dict):
        for item in boxes:
            score = float(item.get('score', item.get('confidence', 1.0)))
            if score < threshold:
                continue
            coords = item.get('box') or item.get('bbox') or item.get('boxes')
            if not coords or len(coords) < 4:
                continue
            box = clamp_box(np.array(coords[:4], dtype=np.float32), width, height)
            parsed_boxes.append(box)
            parsed_scores.append(score)
        return parsed_boxes, parsed_scores

    for idx, coords in enumerate(boxes):
        if coords is None or len(coords) < 4:
            continue
        score = float(scores[idx]) if idx < len(scores) else 1.0
        if score < threshold:
            continue
        box = clamp_box(np.array(coords[:4], dtype=np.float32), width, height)
        parsed_boxes.append(box)
        parsed_scores.append(score)
    return parsed_boxes, parsed_scores


def detect_boxes(detector: Any, frame_bgr: np.ndarray, threshold: float) -> tuple[list[np.ndarray], list[float]]:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    attempts = [frame_rgb]
    try:
        from PIL import Image
        attempts.append(Image.fromarray(frame_rgb))
    except Exception:
        pass

    last_error: Exception | None = None
    for payload in attempts:
        try:
            result = detector(payload)
            return parse_detection_result(result, frame_bgr.shape[1], frame_bgr.shape[0], threshold)
        except Exception as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    return [], []


def open_writer(path: Path, fps: float, size: int) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(path), fourcc, fps, (size, size))
    if not writer.isOpened():
        raise RuntimeError(f'Failed to open output video writer: {path}')
    return writer


def process_video(
    detector: Any,
    input_path: Path,
    output_path: Path,
    metadata_path: Path,
    output_size: int,
    crop_scale: float,
    threshold: float,
    smooth_alpha: float,
    max_frames: int,
    draw_boxes: bool,
) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f'Failed to open video: {input_path}')

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-6:
        fps = 25.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    writer = open_writer(output_path, fps, output_size)

    previous_box: np.ndarray | None = None
    fixed_crop_side: float | None = None
    frame_count = 0
    detected_frames = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames and frame_count >= max_frames:
                break
            frame_count += 1

            boxes, scores = detect_boxes(detector, frame, threshold)
            selected_box = choose_center_box(boxes, frame.shape[1], frame.shape[0])
            if selected_box is not None:
                detected_frames += 1
                if fixed_crop_side is None:
                    fixed_crop_side = crop_side_from_box(selected_box, crop_scale)
                selected_box = smooth_box(previous_box, selected_box, smooth_alpha)
                previous_box = selected_box
            elif previous_box is None:
                previous_box = np.array([0.0, 0.0, frame.shape[1], frame.shape[0]], dtype=np.float32)

            if draw_boxes and boxes:
                debug_frame = frame.copy()
                for box in boxes:
                    x1, y1, x2, y2 = [int(v) for v in box]
                    cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                if previous_box is not None:
                    x1, y1, x2, y2 = [int(v) for v in previous_box]
                    cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                processed = square_crop(debug_frame, previous_box, crop_scale, output_size, fixed_side=fixed_crop_side)
            else:
                processed = square_crop(frame, previous_box, crop_scale, output_size, fixed_side=fixed_crop_side)

            writer.write(processed)
    except Exception:
        writer.release()
        cap.release()
        if output_path.exists():
            output_path.unlink()
        if metadata_path.exists():
            metadata_path.unlink()
        raise
    finally:
        try:
            writer.release()
        except Exception:
            pass
        try:
            cap.release()
        except Exception:
            pass

    stats = {
        'input_path': str(input_path),
        'output_path': str(output_path),
        'metadata_path': str(metadata_path),
        'frames_total': frame_count,
        'frames_with_detection': detected_frames,
        'fps': fps,
        'output_size': output_size,
        'crop_scale': crop_scale,
        'fixed_crop_side': fixed_crop_side,
    }
    metadata_path.write_text(json.dumps({'stats': stats}, ensure_ascii=False, indent=2), encoding='utf-8')
    return stats


def iter_rows(path: Path) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()

    input_csv = Path(args.input_csv).expanduser().resolve()
    videos_dir = Path(args.videos_dir).expanduser().resolve()
    metadata_dir = Path(args.metadata_dir).expanduser().resolve()
    manifest_path = Path(args.manifest_path).expanduser().resolve()

    videos_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    rows = iter_rows(input_csv)
    if args.max_videos > 0:
        rows = rows[:args.max_videos]

    detector = pipeline(
        Tasks.domain_specific_object_detection,
        model=args.model_id,
        trust_remote_code=True,
        device=args.device,
    )

    manifest_lines: list[str] = []
    summary_rows: list[dict[str, Any]] = []

    for index, row in enumerate(rows, start=1):
        raw_path = row.get(args.path_column, '').strip()
        if not raw_path:
            summary_rows.append({'row_index': index, 'status': 'missing_path'})
            continue

        input_path = Path(raw_path)
        if input_path.suffix.lower() not in VIDEO_EXTENSIONS:
            summary_rows.append({'row_index': index, 'input_path': raw_path, 'status': 'unsupported_extension'})
            continue
        if not input_path.is_file():
            summary_rows.append({'row_index': index, 'input_path': raw_path, 'status': 'not_found'})
            continue

        sample_name = sanitize_name(row.get(args.id_column, input_path.stem), f'video_{index:04d}')
        output_path = videos_dir / f'{index:04d}_{sample_name}.mp4'
        metadata_path = metadata_dir / f'{index:04d}_{sample_name}.json'

        if args.skip_existing and output_path.is_file() and metadata_path.is_file():
            manifest_lines.append(f'{output_path} 0 -1')
            summary_rows.append({'row_index': index, 'input_path': raw_path, 'output_path': str(output_path), 'status': 'skipped_existing'})
            continue

        stats = process_video(
            detector=detector,
            input_path=input_path,
            output_path=output_path,
            metadata_path=metadata_path,
            output_size=args.output_size,
            crop_scale=args.crop_scale,
            threshold=args.score_threshold,
            smooth_alpha=args.smooth_alpha,
            max_frames=args.max_frames,
            draw_boxes=args.draw_boxes,
        )
        manifest_lines.append(f'{output_path} 0 -1')
        summary_rows.append({'row_index': index, **stats, 'status': 'processed'})
        print(f'[INFO] processed {index}/{len(rows)}: {input_path} -> {output_path}')

    manifest_path.write_text('\n'.join(manifest_lines) + ('\n' if manifest_lines else ''), encoding='utf-8')

    summary_path = metadata_dir / 'preprocess_summary.csv'
    if summary_rows:
        fieldnames = sorted({key for row in summary_rows for key in row.keys()})
        with summary_path.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
    print(f'[INFO] wrote manifest: {manifest_path}')
    print(f'[INFO] wrote summary: {summary_path}')


if __name__ == '__main__':
    main()
