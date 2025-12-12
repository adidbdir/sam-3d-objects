import argparse
import os
import sys
import tempfile
import shutil
from typing import List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

# notebook/inference.py へのパスを通す
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "notebook"))

from inference import (  # noqa: E402
    Inference,
    load_image,
    load_single_mask,
)


def discover_image_paths(root_dir: str) -> List[str]:
    """
    再帰的に画像ファイルを探索するユーティリティ.
    対応拡張子: .png, .jpg, .jpeg, .webp
    """
    import glob

    exts = ["png", "jpg", "jpeg", "webp"]
    paths: List[str] = []
    for ext in exts:
        pattern = os.path.join(root_dir, "**", f"*.{ext}")
        pattern_uc = os.path.join(root_dir, "**", f"*.{ext.upper()}")
        paths.extend(glob.glob(pattern, recursive=True))
        paths.extend(glob.glob(pattern_uc, recursive=True))

    # 重複を削除しつつ順序保持
    seen = set()
    unique_paths: List[str] = []
    for p in paths:
        if p not in seen and os.path.isfile(p):
            seen.add(p)
            unique_paths.append(p)
    return unique_paths


def ensure_mask_dir_for_image(image_path: str) -> Tuple[str, int]:
    """
    画像の alpha チャンネルからマスク画像 (0.png) を作成し,
    一時ディレクトリのパスとマスク index (常に 0) を返す.
    alpha がない画像は全画素マスク扱い (alpha=255) とする.
    """
    pil_img: Image.Image = Image.open(image_path).convert("RGBA")
    alpha = pil_img.getchannel("A")

    mask_rgba = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    mask_rgba.putalpha(alpha)

    tmp_dir = tempfile.mkdtemp(prefix="mask_")
    mask_path = os.path.join(tmp_dir, "0.png")
    mask_rgba.save(mask_path)
    return tmp_dir, 0


def compute_voxel_and_rotation(
    inference: Inference,
    img_path: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    1枚の画像から voxel, rotation(quaternion), polygon mesh(vertices, faces)
    を推論して numpy 配列で返す.
    - voxel: (N, 3)
    - rotation: (1, 4) など (元の tensor 形状をそのまま numpy 化)
    - mesh_vertices: (V, 3)
    - mesh_faces: (F, 3) int
    """
    # 画像読み込み
    image = load_image(img_path)

    # alpha からマスク生成
    mask_dir, mask_index = ensure_mask_dir_for_image(img_path)
    try:
        mask = load_single_mask(mask_dir, index=mask_index)

        # stage1_only=False で stage2 まで実行し,
        # - coords から voxel を算出
        # - rotation を取得
        # - mesh decoder からポリゴンメッシュ(vertices, faces) を取得
        output = inference._pipeline.run(  # type: ignore[attr-defined]
            image=image,
            mask=mask,
            seed=42,
            stage1_only=False,
            with_mesh_postprocess=False,
            with_texture_baking=False,
            with_layout_postprocess=False,
            use_vertex_color=False,
            stage1_inference_steps=None,
            stage2_inference_steps=None,
            use_stage1_distillation=False,
            use_stage2_distillation=False,
            pointmap=None,
            # ポリゴンメッシュを取得したいので "mesh" をデコード
            # かつ postprocess_slat_output 内で gaussian も参照されるため "gaussian" も必要
            decode_formats=["mesh", "gaussian"],
            estimate_plane=False,
        )
    finally:
        # 一時マスクディレクトリは都度削除
        shutil.rmtree(mask_dir, ignore_errors=True)

    # voxel: stage1_only=False の場合は自前で coords から計算
    coords = output.get("coords", None)
    if coords is None:
        raise RuntimeError(
            f"'coords' が出力に存在しませんでした: {img_path}"
        )
    # InferencePipelinePointMap.run(stage1_only=True) の実装に合わせる
    # ss_return_dict["coords"] は (N, 4) [batch, x, y, z] なので 1: 以降を使用
    coords_np = None

    def to_numpy(x) -> np.ndarray:
        try:
            # torch.Tensor 想定
            return x.detach().cpu().numpy()
        except AttributeError:
            return np.asarray(x)

    coords_np = to_numpy(coords)
    voxel_np = coords_np[:, 1:] / 64.0 - 0.5

    # rotation
    rotation = output.get("rotation", None)
    if rotation is None:
        raise RuntimeError(f"'rotation' が出力に存在しませんでした: {img_path}")

    rotation_np = to_numpy(rotation)

    # polygon mesh: MeshExtractResult (vertices, faces) を numpy に変換
    mesh_list = output.get("mesh", None)
    if mesh_list is None or len(mesh_list) == 0:
        raise RuntimeError(f"'mesh' が出力に存在しませんでした: {img_path}")

    mesh0 = mesh_list[0]
    verts = mesh0.vertices
    faces = mesh0.faces

    mesh_vertices_np = to_numpy(verts)
    mesh_faces_np = to_numpy(faces)

    return voxel_np, rotation_np, mesh_vertices_np, mesh_faces_np


def save_npz_for_image(
    img_path: str,
    voxel: np.ndarray,
    rotation: np.ndarray,
    mesh_vertices: np.ndarray,
    mesh_faces: np.ndarray,
    input_root: str,
    output_root: str,
) -> str:
    """
    1枚の画像に対応する .npz を保存する。
    - input_root からの相対パスを使って, output_root 以下に同じフォルダ構造を作る.
    - ファイル名は拡張子だけ .npz に差し替え.
    """
    rel_path = os.path.relpath(img_path, input_root)
    base_no_ext, _ = os.path.splitext(rel_path)
    out_path = os.path.join(output_root, base_no_ext + ".npz")

    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)

    # npz 内のキー:
    # - img_path: 画像のフルパス (dtype=object の 0次元配列)
    # - voxel: (N, 3) など
    # - rotate: (4,) or (1, 4) など
    # - mesh_vertices: (V, 3) ポリゴンメッシュ頂点
    # - mesh_faces: (F, 3) ポリゴンメッシュ面インデックス
    np.savez_compressed(
        out_path,
        img_path=np.array(img_path, dtype=object),
        voxel=voxel,
        rotate=rotation,
        mesh_vertices=mesh_vertices,
        mesh_faces=mesh_faces,
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "画像から voxel と姿勢(rotation) を推論し, "
            "フォルダ構造を保った .npz を出力するスクリプト"
        )
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="入力パス (画像ファイル or 画像を含むディレクトリ)",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help=(
            "出力ルートディレクトリ. "
            "未指定の場合, ディレクトリ入力なら <input_dir>/outputs_voxel, "
            "ファイル入力なら <その親ディレクトリ>/outputs_voxel"
        ),
    )
    parser.add_argument(
        "--config-tag",
        type=str,
        default="hf",
        help="使用するチェックポイントタグ (既定: hf)",
    )

    args = parser.parse_args()

    input_path = os.path.abspath(args.input_path)

    if os.path.isdir(input_path):
        input_root = input_path
        image_paths = discover_image_paths(input_root)
        if not image_paths:
            print(f"[ERROR] 画像が見つかりませんでした: {input_root}")
            sys.exit(1)
    elif os.path.isfile(input_path):
        input_root = os.path.dirname(input_path)
        image_paths = [input_path]
    else:
        print(f"[ERROR] 入力パスが存在しません: {input_path}")
        sys.exit(1)

    # 出力ルート
    if args.output_root is not None:
        output_root = os.path.abspath(args.output_root)
    else:
        output_root = os.path.join(input_root, "outputs_voxel")
    os.makedirs(output_root, exist_ok=True)

    # モデルは一度だけロード
    tag = args.config_tag
    config_path = os.path.join(BASE_DIR, "checkpoints", tag, "pipeline.yaml")
    if not os.path.exists(config_path):
        print(f"[ERROR] config が見つかりません: {config_path}")
        sys.exit(1)

    print(f"[INFO] Loading Inference model from: {config_path}")
    inference = Inference(config_path, compile=False)

    num_images = len(image_paths)
    print(f"[INFO] Found {num_images} images.")

    # まとめ書き用に全サンプルを保持
    all_img_paths: List[str] = []
    all_voxels: List[np.ndarray] = []
    all_rotations: List[np.ndarray] = []
    all_mesh_vertices: List[np.ndarray] = []
    all_mesh_faces: List[np.ndarray] = []

    for idx, img_path in enumerate(
        tqdm(image_paths, total=num_images, desc="Exporting voxel/pose", ncols=80),
        start=1,
    ):
        try:
            print(f"[INFO] ({idx}/{len(image_paths)}) Processing: {img_path}")
            (
                voxel_np,
                rotation_np,
                mesh_vertices_np,
                mesh_faces_np,
            ) = compute_voxel_and_rotation(inference, img_path)

            # 個別ファイル保存
            out_path = save_npz_for_image(
                img_path=img_path,
                voxel=voxel_np,
                rotation=rotation_np,
                mesh_vertices=mesh_vertices_np,
                mesh_faces=mesh_faces_np,
                input_root=input_root,
                output_root=output_root,
            )
            print(f"[OK] Saved NPZ -> {out_path}")

            # まとめ用リストに追加
            all_img_paths.append(img_path)
            all_voxels.append(voxel_np)
            all_rotations.append(rotation_np)
            all_mesh_vertices.append(mesh_vertices_np)
            all_mesh_faces.append(mesh_faces_np)
        except Exception as e:
            print(f"[WARN] Failed on '{img_path}': {e}")

    # === 最後に全データを 1 つの npz にまとめて保存 ===
    if all_img_paths:
        if os.path.isdir(input_path):
            combined_path = os.path.join(output_root, "all_voxels.npz")
        else:
            base = os.path.splitext(os.path.basename(input_path))[0]
            combined_path = os.path.join(output_root, f"{base}_all.npz")

        np.savez_compressed(
            combined_path,
            img_path=np.array(all_img_paths, dtype=object),
            voxel=np.array(all_voxels, dtype=object),
            rotate=np.array(all_rotations, dtype=object),
            mesh_vertices=np.array(all_mesh_vertices, dtype=object),
            mesh_faces=np.array(all_mesh_faces, dtype=object),
        )
        print(f"[OK] Saved combined NPZ -> {combined_path}")


if __name__ == "__main__":
    main()


