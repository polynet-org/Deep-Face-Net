import sys
import cv2
from pathlib import Path


def process_image(source_face, face_analyser, args):
    """Process a single image.

    Args:
        mode (via args.mode or args.enhance):
            'swap'         – face swap only (default)
            'enhance'      – GFPGAN on original faces, no swap
            'swap_enhance' – face swap then GFPGAN enhancement
    """
    from core.engine.face_swapper import swap_face

    mode = getattr(args, "mode", None)
    if mode is None:
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
    print(f"Processing image: {args.target}  [mode: {mode_label}]")

    # Load target image
    target_img = cv2.imread(args.target)
    if target_img is None:
        print(f"Error: Failed to load target image: {args.target}")
        sys.exit(1)

    h, w = target_img.shape[:2]

    # Detect faces
    print("Detecting faces...")
    faces = face_analyser.get(target_img)
    print(f"Found {len(faces)} face(s)")
    if not faces:
        print("Warning: No faces detected in target image")

    result = target_img.copy()

    # Step 1: Swap (if needed)
    if needs_swap and faces:
        for face in faces:
            result = swap_face(source_face, face, result)

    # Step 2: Enhance — re-detect after swap for fresh landmarks
    if needs_enhance and faces:
        print("Enhancing faces...")
        detect_faces = face_analyser.get(result) if needs_swap else faces
        if detect_faces:
            result = enhance_faces(result, detect_faces, (w, h))

    # Save output
    if args.output:
        output_path = args.output
    else:
        target_path = Path(args.target)
        suffix_map  = {"swap": "_swapped", "enhance": "_enhanced", "swap_enhance": "_swapped_enhanced"}
        output_path = f"{target_path.stem}{suffix_map[mode]}{target_path.suffix}"

    cv2.imwrite(output_path, result)
    print(f"✓ Output saved: {output_path}")
