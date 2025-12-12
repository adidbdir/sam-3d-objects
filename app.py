# https://note.com/tori29umai/n/n9adb3215b183
import os
import sys
import shutil
import tempfile
import uuid
import zipfile

import numpy as np
from PIL import Image
import torch
import imageio
import gradio as gr

# ==========================================
# Gaussian Splatting inference 関連
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# inference.py を読み込むためのパス追加
sys.path.append(os.path.join(BASE_DIR, "notebook"))

from inference import (
    Inference,
    ready_gaussian_for_video_rendering,
    load_image,
    load_masks,
    load_single_mask,
    make_scene,
    render_video,
)

# Gaussian Splatting モデル（tag = 固定 "hf"）
TAG_DEFAULT = "hf"
CONFIG_PATH_DEFAULT = os.path.join(BASE_DIR, "checkpoints", TAG_DEFAULT, "pipeline.yaml")

print("[GS] Loading Gaussian Splatting inference model...")
inference_default = Inference(CONFIG_PATH_DEFAULT, compile=False)
print("[GS] Ready.")

# ==========================================
# FFmpeg チェック
# ==========================================
def ensure_ffmpeg():
    try:
        import imageio_ffmpeg  # noqa
    except ImportError:
        raise gr.Error(
            "❌ MP4 生成に FFmpeg が必要です。\n"
            "以下を実行してください：\n"
            "`pip install imageio[ffmpeg]`"
        )

# ==========================================
# ZIP 解凍 → image.png + 連番マスク (0.png,1.png,...) を取得
# ==========================================
def _resolve_file_path(file_obj):
    """
    Gradio の File コンポーネントはバージョンによって
    str, dict, tempfile など形式が異なり得るので、
    それらをうまく吸収してパスを返すヘルパー。
    """
    if file_obj is None:
        return None

    # すでに文字列パス
    if isinstance(file_obj, str):
        return file_obj

    # dict 形式 { 'name': '/tmp/xxx.zip', ... }
    if isinstance(file_obj, dict) and "name" in file_obj:
        return file_obj["name"]

    # file-like オブジェクト
    if hasattr(file_obj, "name"):
        return file_obj.name

    raise gr.Error("❌ ZIP ファイルのパスを解決できませんでした。Gradio のバージョンを確認してください。")


def unzip_image_and_masks(zip_file, work_prefix: str):
    """
    ZIP ファイルから image.png と 連番 PNG マスク (0.png,1.png,...) を取り出す。
    構成前提:
      - ZIP 直下に image.png
      - 同じ階層に 0.png,1.png,2.png,... (RGBA, alpha がマスク)
      - サブフォルダなし
    """
    zip_path = _resolve_file_path(zip_file)
    if zip_path is None:
        raise gr.Error("❌ ZIP ファイルが指定されていません。")

    if not zipfile.is_zipfile(zip_path):
        raise gr.Error("❌ ZIP ファイルではありません。")

    work_dir = tempfile.mkdtemp(prefix=work_prefix)
    print(f"[ZIP] work_dir: {work_dir}")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(work_dir)

    # image.png を探す（必須）
    image_path = os.path.join(work_dir, "image.png")
    if not os.path.exists(image_path):
        raise gr.Error("❌ ZIP 内に image.png が見つかりませんでした。")

    # マスク格納ディレクトリ
    mask_dir = os.path.join(work_dir, "masks")
    os.makedirs(mask_dir, exist_ok=True)

    # work_dir 直下の PNG のうち image.png 以外をマスクとして mask_dir に移動
    for fname in os.listdir(work_dir):
        if not fname.lower().endswith(".png"):
            continue
        if fname == "image.png":
            continue

        src = os.path.join(work_dir, fname)

        # 数値ファイル名のみ採用 (例: "0.png", "1.png" ...)
        stem, ext = os.path.splitext(fname)
        if not stem.isdigit():
            print(f"[ZIP] 非連番 PNG をスキップ: {fname}")
            continue

        dst = os.path.join(mask_dir, fname)
        shutil.move(src, dst)
        print(f"[ZIP] Move mask: {src} -> {dst}")

    # mask_dir 内の連番 PNG を確認
    pngs = [
        os.path.join(mask_dir, f)
        for f in os.listdir(mask_dir)
        if f.lower().endswith(".png")
    ]

    if len(pngs) == 0:
        raise gr.Error(
            "❌ ZIP 内に 連番 PNG マスク (0.png,1.png,...) が見つかりませんでした。\n"
            "image.png と同じ階層に 0.png,1.png,... を配置してください。"
        )

    # 連番順にソート
    def key_sort(p):
        return int(os.path.splitext(os.path.basename(p))[0])

    pngs_sorted = sorted(pngs, key=key_sort)

    print(f"[ZIP] total masks: {len(pngs_sorted)}")
    return image_path, mask_dir, pngs_sorted, work_dir

# =========================================================
# タブ1：マルチ – ZIP → 全マスクで 3D 再構築
# =========================================================
def preprocess_multi_from_zip(zip_file):
    """
    マルチ用前処理：
      - ZIP から image.png と 連番マスクを抽出
      - マスク一覧を Gallery に表示
      - state に image_path, mask_dir, work_dir を保存
    """
    if zip_file is None:
        raise gr.Error("❌ マスク ZIP が指定されていません。")

    image_path, mask_dir, mask_paths, work_dir = unzip_image_and_masks(
        zip_file, work_prefix="multi3d_"
    )

    options = [str(i) for i in range(len(mask_paths))]

    state = {
        "image_path": image_path,
        "mask_dir": mask_dir,
        "work_dir": work_dir,
    }

    return mask_paths, gr.update(choices=options, value=[]), state


def remove_masks_multi(selected_indices, state):
    """
    マルチ用：複数選択されたマスクインデックスを削除し、
    残ったファイルを 0.png,1.png,... に再連番して Gallery と Dropdown を更新。
    """
    if state is None:
        raise gr.Error("❌ まず『前処理（ZIP展開）』を実行してください。")

    mask_dir = state["mask_dir"]

    # 選択なしなら現在の一覧を返すだけ
    if not selected_indices:
        pngs = [
            os.path.join(mask_dir, f)
            for f in os.listdir(mask_dir)
            if f.lower().endswith(".png")
        ]
        if not pngs:
            raise gr.Error("❌ マスクが存在しません。再度前処理を実行してください。")

        def key_sort(p):
            return int(os.path.splitext(os.path.basename(p))[0])

        pngs_sorted = sorted(pngs, key=key_sort)
        options = [str(i) for i in range(len(pngs_sorted))]
        return pngs_sorted, gr.update(choices=options, value=[])

    # 削除対象インデックスを整数へ
    indices_to_remove = [int(i) for i in selected_indices]

    # 対象ファイルを削除
    for idx in indices_to_remove:
        path = os.path.join(mask_dir, f"{idx}.png")
        if os.path.exists(path):
            os.remove(path)
            print(f"[MULTI] Removed mask → {path}")

    # 残った PNG を取得
    pngs = [f for f in os.listdir(mask_dir) if f.lower().endswith(".png")]
    if not pngs:
        raise gr.Error("❌ 全てのマスクが削除されました。再度前処理を実行してください。")

    # 現在の番号順にソート
    def key_sort_name(name):
        return int(os.path.splitext(name)[0])

    pngs_sorted_names = sorted(pngs, key=key_sort_name)

    # 0,1,2,... にリネーム
    for new_idx, old_name in enumerate(pngs_sorted_names):
        old_path = os.path.join(mask_dir, old_name)
        new_path = os.path.join(mask_dir, f"{new_idx}.png")
        if old_path != new_path:
            os.rename(old_path, new_path)
            print(f"[MULTI] Renamed {old_path} → {new_path}")

    # 再取得（フルパス）
    new_pngs = [os.path.join(mask_dir, f"{i}.png") for i in range(len(pngs_sorted_names))]
    options = [str(i) for i in range(len(new_pngs))]

    return new_pngs, gr.update(choices=options, value=[])


def auto_multi3d_pipeline(state, fov, resolution, radius, fps):
    """
    マルチ用 3D 再構築：
      - state 内の image_path & mask_dir を使って、
        残っているマスクだけで multi-object 3D (PLY + MP4) を生成する。
    """
    if state is None:
        raise gr.Error("❌ まず『前処理（ZIP展開）』を実行してください。")

    ensure_ffmpeg()

    image_path = state["image_path"]
    mask_dir = state["mask_dir"]
    work_dir = state["work_dir"]

    # マスクが存在するか確認
    pngs = [f for f in os.listdir(mask_dir) if f.lower().endswith(".png")]
    if not pngs:
        raise gr.Error("❌ 有効なマスクがありません。再度前処理を実行してください。")

    inference = inference_default  # tag 固定

    image = load_image(image_path)
    masks = load_masks(mask_dir, extension=".png")

    if len(masks) == 0:
        raise gr.Error("❌ マスクが 0 枚です。再度前処理を実行してください。")

    outputs = []
    for i, mask in enumerate(masks):
        print(f"[MULTI] Inference {i+1}/{len(masks)}")
        outputs.append(inference(image, mask, seed=42))

    scene_gs = make_scene(*outputs)
    scene_gs = ready_gaussian_for_video_rendering(scene_gs)

    ply_path = os.path.join(work_dir, "scene_multi.ply")
    mp4_path = os.path.join(work_dir, "scene_multi.mp4")

    frames = render_video(
        scene_gs,
        r=radius,
        fov=fov,
        resolution=resolution,
    )["color"]

    imageio.mimsave(mp4_path, frames, fps=fps, format="FFMPEG")
    print(f"[MULTI] MP4 saved → {mp4_path}")


    # === 上下反転 ===
    xyz = scene_gs._xyz.data
    xyz[:, 1] *= -1
    # xyz[:, 2] *= -1   # ← 必要なら Z も反転
    scene_gs._xyz.data = xyz

    ply_path = os.path.join(work_dir, "scene_multi.ply")
    scene_gs.save_ply(ply_path)

    # Viewer 用（Model3D, Video）を返す
    return ply_path, mp4_path

# =========================================================
# タブ2：シングル – ZIP → 1マスク選択で 3D
# =========================================================
def preprocess_single_from_zip(zip_file):
    """
    シングル用前処理：
      - ZIP から image.png と 連番マスクを抽出
      - マスク一覧を Gallery に表示
      - Dropdown にインデックス（0,1,2,...) を設定
    """
    if zip_file is None:
        raise gr.Error("❌ マスク ZIP が指定されていません。")

    image_path, mask_dir, mask_paths, work_dir = unzip_image_and_masks(
        zip_file, work_prefix="single3d_"
    )

    def key_sort(p):
        return int(os.path.splitext(os.path.basename(p))[0])

    mask_paths_sorted = sorted(mask_paths, key=key_sort)
    options = [str(i) for i in range(len(mask_paths_sorted))]

    state = {
        "image_path": image_path,
        "mask_dir": mask_dir,
        "work_dir": work_dir,
    }

    return mask_paths_sorted, gr.update(choices=options, value=None), state


def single_object_3d(mask_index_str, state):
    """
    シングル用 3D 再構築：
      - state 内の image_path & mask_dir から
        指定インデックスのマスクを用いて 1 オブジェクトの PLY を生成。
    """
    if state is None:
        raise gr.Error("❌ まず『前処理（ZIP展開）』を実行してください。")

    if mask_index_str is None or mask_index_str == "":
        raise gr.Error("❌ マスクが選択されていません。")

    idx = int(mask_index_str)

    image_path = state["image_path"]
    mask_dir = state["mask_dir"]
    work_dir = state["work_dir"]

    inference = inference_default  # 固定

    image = load_image(image_path)
    mask = load_single_mask(mask_dir, index=idx)

    print(f"[SINGLE] Running inference for mask index {idx}")
    output = inference(image, mask, seed=42)

    gs = output["gs"]

    # === 上下反転 ===
    xyz = gs._xyz.data
    xyz[:, 1] *= -1     # Y 軸反転
    # xyz[:, 2] *= -1   # ← 必要なら Z も反転
    gs._xyz.data = xyz

    ply_path = os.path.join(work_dir, f"splat_{idx}.ply")
    gs.save_ply(ply_path)
    print(f"[SINGLE] PLY saved → {ply_path}")

    # マスク一式はここで消してもOK（要件次第）
    try:
        shutil.rmtree(mask_dir, ignore_errors=True)
    except Exception as e:
        print("[SINGLE] mask_dir cleanup failed:", e)

    # viewer 用 + DL 用
    return ply_path, ply_path

# =========================================================
# Gradio UI
# =========================================================
with gr.Blocks(title="Gaussian Splatting – ZIP Masks & 3D") as demo:
    gr.Markdown("## 🧱 Gaussian Splatting – ZIP マスク & 3D 再構築")

    gr.Markdown(
        "### 入力 ZIP の構成\n"
        "- `image.png` （元画像 / RGBA or RGB）\n"
        "- `0.png, 1.png, 2.png, ...` （RGBA, alpha がマスク）\n"
        "- サブフォルダなし\n"
    )

    # ---------------------------------------------------------
    # ① マルチ：ZIP → 全マスクで 3D
    # ---------------------------------------------------------
    with gr.Tab("① マルチ：ZIP → 全マスクで 3D（PLY + MP4）"):
        gr.Markdown(
            "1. マスク ZIP をアップロード\n"
            "2. 『🧩 前処理（ZIP展開）』でマスク一覧を表示\n"
            "3. 不要なマスクを選択して『🗑 選択マスクを除去』\n"
            "4. 『▶ 残りのマスクで 3D 再構築』で PLY + MP4 を生成"
        )

        mask_zip_input_multi = gr.File(
            label="mask ZIP（image.png と 0.png,1.png,... を含む）",
            file_types=[".zip"],
        )

        preprocess_multi_btn = gr.Button("🧩 前処理（ZIP展開）")

        multi_gallery = gr.Gallery(
            label="生成されたマスク一覧（クリックして中身確認可）",
            columns=4,
            rows=2,
            height=300,
        )

        multi_selector = gr.Dropdown(
            label="除外したいマスクのインデックス（複数選択可）",
            choices=[],
            multiselect=True,
        )

        state_multi = gr.State()

        preprocess_multi_btn.click(
            fn=preprocess_multi_from_zip,
            inputs=[mask_zip_input_multi],
            outputs=[multi_gallery, multi_selector, state_multi],
        )

        remove_multi_btn = gr.Button("🗑 選択マスクを除去してリスト更新")

        remove_multi_btn.click(
            fn=remove_masks_multi,
            inputs=[multi_selector, state_multi],
            outputs=[multi_gallery, multi_selector],
        )

        with gr.Row():
            auto_fov = gr.Slider(20, 120, value=60, step=1, label="FOV")
            auto_resolution = gr.Slider(256, 1024, value=512, step=64, label="Resolution")
            auto_radius = gr.Slider(0.5, 3.0, value=1.0, step=0.1, label="Camera Radius")
            auto_fps = gr.Slider(10, 60, value=30, step=1, label="FPS")

        auto_run_button = gr.Button("▶ 残りのマスクで 3D 再構築（multi-object）")

        auto_ply_viewer = gr.Model3D(label="PLY プレビュー（multi-object scene）")
        auto_mp4_player = gr.Video(label="MP4 プレビュー（multi-object scene）")

        auto_run_button.click(
            fn=auto_multi3d_pipeline,
            inputs=[state_multi, auto_fov, auto_resolution, auto_radius, auto_fps],
            outputs=[auto_ply_viewer, auto_mp4_player],
        )

    # ---------------------------------------------------------
    # ② シングル：ZIP → 1 つ選んで 3D
    # ---------------------------------------------------------
    with gr.Tab("② シングル：ZIP → 1つ選んで 3D（単一オブジェクト）"):
        gr.Markdown(
            "1. マスク ZIP をアップロード\n"
            "2. 『🧩 前処理（ZIP展開）』でマスク一覧を表示\n"
            "3. 3D化したいマスクを1つ選択\n"
            "4. 『▶ 選択マスクで 3D 再構築』で PLY を生成"
        )

        mask_zip_input_single = gr.File(
            label="mask ZIP（image.png と 0.png,1.png,... を含む）",
            file_types=[".zip"],
        )

        preprocess_single_button = gr.Button("🧩 前処理（ZIP展開）")

        single_gallery = gr.Gallery(
            label="生成されたマスク一覧",
            columns=4,
            rows=2,
            height=300,
        )
        single_selector = gr.Dropdown(
            label="3D化するマスク番号を選択（0,1,2,...）",
            choices=[],
        )

        state_single = gr.State()

        preprocess_single_button.click(
            fn=preprocess_single_from_zip,
            inputs=[mask_zip_input_single],
            outputs=[single_gallery, single_selector, state_single],
        )

        run_single_button = gr.Button("▶ 選択マスクで 3D 再構築（single object）")

        single_ply_viewer = gr.Model3D(label="PLY プレビュー（single object）")
        single_ply_file = gr.File(label="Download PLY")

        run_single_button.click(
            fn=single_object_3d,
            inputs=[single_selector, state_single],
            outputs=[single_ply_viewer, single_ply_file],
        )

# スクリプトとして実行された場合のみ起動
if __name__ == "__main__":
    demo.launch(server_port=7861)
