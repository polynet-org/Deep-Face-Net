import sys
import cv2
import time
import threading
from pathlib import Path
from queue import Queue


def process_video(source_face, face_analyser, args):
    """Process a video file.

    Args:
        mode (via args.mode or args.enhance):
            'swap'         – face swap only (default)
            'enhance'      – GFPGAN on original faces, no swap
            'swap_enhance' – face swap then GFPGAN enhancement
    """
    from core.engine.face_swapper import detect_and_swap

    mode = getattr(args, "mode", None)
    if mode is None:
        # backward-compat: if old --enhance flag is used
        mode = "swap_enhance" if getattr(args, "enhance", False) else "swap"

    needs_enhance = mode in ("enhance", "swap_enhance")
    needs_swap    = mode in ("swap",    "swap_enhance")

    enhance_faces = None
    if needs_enhance:
        try:
            from core.engine.face_enhancer import enhance_faces as _ef
            enhance_faces = _ef
        except FileNotFoundError as e:
            print(f"Warning: Enhancement disabled — {e}")
            needs_enhance = False
            if mode == "enhance":
                print("Error: Cannot run Enhance-Only mode without GFPGAN model.")
                sys.exit(1)
            mode = "swap"

    mode_label = {"swap": "Swap", "enhance": "Enhance Only", "swap_enhance": "Swap + Enhance"}[mode]
    print(f"Processing video: {args.target}  [mode: {mode_label}]")

    # ── Open video ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.target)
    if not cap.isOpened():
        print(f"Error: Failed to open video: {args.target}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video info: {width}x{height} @ {fps:.2f} FPS, {total_frames} frames")

    # ── Output path ───────────────────────────────────────────────────────────
    if args.output:
        output_path = args.output
    else:
        target_path = Path(args.target)
        suffix_map  = {"swap": "_swapped", "enhance": "_enhanced", "swap_enhance": "_swapped_enhanced"}
        output_path = f"{target_path.stem}{suffix_map[mode]}{target_path.suffix}"

    # ── Video writer ──────────────────────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # ── Threaded I/O queues ───────────────────────────────────────────────────
    read_queue  = Queue(maxsize=32)
    write_queue = Queue(maxsize=32)

    def _read_frames():
        while True:
            ret, frame = cap.read()
            if not ret:
                read_queue.put(None)
                break
            read_queue.put(frame)

    def _write_frames():
        while True:
            frame = write_queue.get()
            if frame is None:
                break
            out.write(frame)

    read_thread  = threading.Thread(target=_read_frames, daemon=True)
    write_thread = threading.Thread(target=_write_frames, daemon=False)
    read_thread.start()
    write_thread.start()

    # ── Process frames ────────────────────────────────────────────────────────
    print("Processing frames...")
    start_time  = time.time()
    frame_count = 0

    try:
        while True:
            frame = read_queue.get()
            if frame is None:
                break

            if needs_swap:
                frame, _ = detect_and_swap(source_face, frame, face_analyser)
                
            if needs_enhance:
                # Need to detect faces for enhancement
                from core.face_analyser import get_many_faces
                detect_faces = get_many_faces(frame)
                if detect_faces:
                    frame = enhance_faces(frame, detect_faces, (width, height))

            write_queue.put(frame)
            frame_count += 1

            if frame_count % 30 == 0:
                elapsed    = time.time() - start_time
                progress   = (frame_count / total_frames) * 100
                fps_actual = frame_count / elapsed if elapsed > 0 else 0
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames}) - {fps_actual:.1f} FPS")

    finally:
        write_queue.put(None)
        write_thread.join()   # flush all frames before releasing
        cap.release()
        out.release()

        elapsed = time.time() - start_time
        print(f"\n✓ Done!  {frame_count} frames in {elapsed:.1f}s  ({frame_count/elapsed:.1f} FPS avg)")
        print(f"  Output: {output_path}")
