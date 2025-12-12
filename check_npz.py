import numpy as np
import matplotlib
matplotlib.use("Agg")  # 画面なし環境用
import matplotlib.pyplot as plt

path = "outputs/small_data/train/rgb_0000431.npz"

data = np.load(path, allow_pickle=True)

img_path = data["img_path"].item() if data["img_path"].shape == () else data["img_path"]
voxel = data["voxel"]        # (N, 3)
rotate = data["rotate"]      # (1, 4) 想定

print("keys:", data.files)
print("img_path:", img_path)
print("voxel shape / dtype:", voxel.shape, voxel.dtype)
print("rotate shape / dtype:", rotate.shape, rotate.dtype)

# 統計情報
print("voxel min:", voxel.min(axis=0))
print("voxel max:", voxel.max(axis=0))
print("voxel mean:", voxel.mean(axis=0))

print("rotate (quaternion wxyz):", rotate)

# 3D散布図で保存（点が多いので一部サンプリング）
N = voxel.shape[0]
idx = np.random.choice(N, size=min(5000, N), replace=False)
v = voxel[idx]

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(v[:, 0], v[:, 1], v[:, 2], s=1, alpha=0.3)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("Voxel point cloud")

# 軸スケールをだいたい等しく
xyz_min = v.min(axis=0)
xyz_max = v.max(axis=0)
center = (xyz_min + xyz_max) / 2
radius = (xyz_max - xyz_min).max() / 2
for i, lab in enumerate(["x", "y", "z"]):
    getattr(ax, f"set_{lab}lim")(center[i] - radius, center[i] + radius)

plt.tight_layout()
plt.savefig("voxel_cloud.png", dpi=300)
print("saved: voxel_cloud.png")