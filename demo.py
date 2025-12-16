import argparse
import os
import sys
import tempfile
from typing import List, Tuple
import glob

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# import inference code
sys.path.append("notebook")
from inference import (  # noqa: E402
    Inference,
    load_image,
    load_single_mask,
    ready_gaussian_for_video_rendering,
    render_video,
)


def discover_image_paths(root_dir: str) -> List[str]:
    """
    Recursively discover images using glob (**).
    Supports: .png, .jpg, .jpeg, .webp (case-insensitive).
    """
    exts = ["png", "jpg", "jpeg", "webp"]
    paths: List[str] = []
    for ext in exts:
        # recursive glob; matches case-sensitive, so add uppercase too
        pattern = os.path.join(root_dir, "**", f"*.{ext}")
        pattern_uc = os.path.join(root_dir, "**", f"*.{ext.upper()}")
        paths.extend(glob.glob(pattern, recursive=True))
        paths.extend(glob.glob(pattern_uc, recursive=True))
    # De-duplicate while preserving order
    seen = set()
    unique_paths: List[str] = []
    for p in paths:
        if p not in seen and os.path.isfile(p):
            seen.add(p)
            unique_paths.append(p)
    return unique_paths


def ensure_mask_dir_for_image(image_path: str) -> Tuple[str, int]:
    """
    Create a temporary mask directory for the given image.
    The mask will be derived from the alpha channel if present;
    otherwise, a full-image mask is generated.
    Returns (mask_dir, mask_index).
    """
    pil_img: Image.Image = Image.open(image_path).convert("RGBA")
    alpha = pil_img.getchannel("A")
    # Build RGBA mask image with alpha channel
    mask_rgba = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    mask_rgba.putalpha(alpha)

    tmp_dir = tempfile.mkdtemp(prefix="mask_")
    mask_path = os.path.join(tmp_dir, "0.png")
    mask_rgba.save(mask_path)
    return tmp_dir, 0


def save_visualization(gs, out_video_path: str, fov: int = 60, resolution: int = 512, radius: float = 1.0, fps: int = 30) -> None:
    scene = ready_gaussian_for_video_rendering(gs)
    frames = render_video(
        scene,
        r=radius,
        fov=fov,
        resolution=resolution,
    )["color"]
    imageio.mimsave(out_video_path, frames, fps=fps, format="FFMPEG")


def preview_image_from_gaussian(gs, fov: int = 60, resolution: int = 512, radius: float = 1.0) -> Image.Image:
    scene = ready_gaussian_for_video_rendering(gs)
    frames = render_video(
        scene,
        r=radius,
        fov=fov,
        resolution=resolution,
    )["color"]
    frame0 = frames[0]  # HxWx3 uint8
    return Image.fromarray(np.asarray(frame0))


def make_labelled_tile(img: Image.Image, label: str, size: Tuple[int, int] = (384, 384)) -> Image.Image:
    tile_w, tile_h = size
    bar_h = 36
    # fit image to tile_w x (tile_h - bar_h)
    max_h = tile_h - bar_h
    im = img.convert("RGB")
    im.thumbnail((tile_w, max_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (tile_w, tile_h), (255, 255, 255))
    # center image
    x = (tile_w - im.width) // 2
    y = (max_h - im.height) // 2
    canvas.paste(im, (x, y))
    # draw label
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    text = label
    tw, th = draw.textbbox((0, 0), text, font=font)[2:]
    tx = max(8, (tile_w - tw) // 2)
    ty = max_h + (bar_h - th) // 2
    draw.text((tx, ty), text, fill=(0, 0, 0), font=font)
    return canvas


def save_summary_image(pairs: List[Tuple[str, Image.Image]], out_path: str) -> None:
    """
    pairs: list of (input_image_path, preview_PIL_image)
    Produces a 2-column grid: [input | preview] per row.
    """
    if not pairs:
        return
    tile_sz = (384, 384)
    rows = []
    for in_path, preview in pairs:
        # input thumb
        input_img = Image.open(in_path).convert("RGB")
        in_tile = make_labelled_tile(input_img, os.path.basename(in_path), tile_sz)
        # preview thumb
        pv_tile = make_labelled_tile(preview, "preview", tile_sz)
        # concat horizontally
        row = Image.new("RGB", (tile_sz[0] * 2, tile_sz[1]), (255, 255, 255))
        row.paste(in_tile, (0, 0))
        row.paste(pv_tile, (tile_sz[0], 0))
        rows.append(row)
    # stack vertically
    sheet_h = tile_sz[1] * len(rows)
    sheet_w = tile_sz[0] * 2
    sheet = Image.new("RGB", (sheet_w, sheet_h), (255, 255, 255))
    for i, r in enumerate(rows):
        sheet.paste(r, (0, i * tile_sz[1]))
    sheet.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch 3D reconstruction for images in a folder.")
    parser.add_argument("input_dir", type=str, help="Input folder containing images.")
    parser.add_argument("--visualize", action="store_true", help="Enable preview video export (MP4).")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"[ERROR] Input directory does not exist: {args.input_dir}")
        sys.exit(1)

    image_paths = discover_image_paths(args.input_dir)
    if not image_paths:
        print(f"[ERROR] No supported images found in: {args.input_dir}")
        sys.exit(1)

    # Load model once
    tag = "hf"
    config_path = os.path.join("checkpoints", tag, "pipeline.yaml")
    inference = Inference(config_path, compile=False)

    # Prepare outputs dir
    outputs_dir = os.path.join(args.input_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    summary_pairs: List[Tuple[str, Image.Image]] = []
    for img_path in image_paths:
        try:
            print(f"[INFO] Processing: {img_path}")
            base = os.path.splitext(os.path.basename(img_path))[0]

            # Load image
            image = load_image(img_path)

            # Prepare mask directory (from alpha or full)
            mask_dir, mask_index = ensure_mask_dir_for_image(img_path)
            mask = load_single_mask(mask_dir, index=mask_index)

            # Run inference
            output = inference(image, mask, seed=42)

            # --- Object pose (translation / rotation / scale) ---
            # Inference pipeline already decodes object pose using
            # `pose_decoder_name` from `pipeline.yaml` (e.g. ScaleShiftInvariant),
            # so we can read it directly from `output`.
            translation = output.get("translation", None)
            rotation = output.get("rotation", None)
            scale = output.get("scale", None)
            if translation is not None and rotation is not None and scale is not None:
                # Move to CPU and convert to numpy for cleaner printing, if tensors
                try:
                    translation_np = translation.detach().cpu().numpy()
                    rotation_np = rotation.detach().cpu().numpy()
                    scale_np = scale.detach().cpu().numpy()
                except AttributeError:
                    # Fallback if these are already numpy arrays or lists
                    translation_np = translation
                    rotation_np = rotation
                    scale_np = scale

                print(f"[POSE] translation (scene coords): {translation_np}")
                print(f"[POSE] rotation (quaternion wxyz): {rotation_np}")
                print(f"[POSE] scale (xyz): {scale_np}")

            gs = output["gs"]

            # Export PLY
            ply_path = os.path.join(outputs_dir, f"splat_{base}.ply")
            gs.save_ply(ply_path)
            print(f"[OK] Saved PLY -> {ply_path}")

            # Export GLB
            if "glb" in output and output["glb"] is not None:
                glb_path = os.path.join(outputs_dir, f"mesh_{base}.glb")
                output["glb"].export(glb_path)
                print(f"[OK] Saved GLB -> {glb_path}")

            # Optional visualization
            if args.visualize:
                mp4_path = os.path.join(outputs_dir, f"preview_{base}.mp4")
                save_visualization(gs, mp4_path)
                print(f"[OK] Saved preview MP4 -> {mp4_path}")

            # Collect preview still for summary sheet
            try:
                preview_img = preview_image_from_gaussian(gs)
                summary_pairs.append((img_path, preview_img))
            except Exception as _:
                pass
        except Exception as e:
            print(f"[WARN] Failed on '{img_path}': {e}")

    # Save one summary image showing input ↔ output mapping
    try:
        summary_path = os.path.join(outputs_dir, "summary.png")
        save_summary_image(summary_pairs, summary_path)
        print(f"[OK] Saved summary image -> {summary_path}")
    except Exception as e:
        print(f"[WARN] Failed to save summary image: {e}")

if __name__ == "__main__":
    main()
