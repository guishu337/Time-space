import os
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import itertools

#穷举排列
def estimate_rigid_transform(src_pts, tgt_pts):
    """直接给出R,t矩阵"""
    src_center = src_pts.mean(axis=0)
    tgt_center = tgt_pts.mean(axis=0)
    src_centered = src_pts - src_center
    tgt_centered = tgt_pts - tgt_center

    H = src_centered.T @ tgt_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = tgt_center - R @ src_center
    return R, t

def find_best_marker_matching(src_pts, tgt_pts):
    """
    穷举所有排列，找最佳匹配（最小均方误差）
    """
    min_error = np.inf
    best_src = None
    best_tgt = None

    for perm in itertools.permutations(range(len(src_pts))):
        permuted_src = src_pts[list(perm)]
        R, t = estimate_rigid_transform(permuted_src, tgt_pts)
        transformed = (R @ permuted_src.T).T + t

        error = np.mean(np.linalg.norm(transformed - tgt_pts, axis=1))
        if error < min_error:
            min_error = error
            best_src = permuted_src
            best_tgt = tgt_pts

    print(f"最佳匹配的均方误差: {min_error:.6f}")
    return best_src, best_tgt

#定位球检测策略
def extract_white_spheres_centroids(pcd, color_tolerance=0.01, min_white_points=30, cluster_eps=0.1, cluster_min_points=30):
    print(" 根据颜色提取所有白球（多球支持）...")

    colors = np.asarray(pcd.colors)
    points = np.asarray(pcd.points)

    if len(colors) == 0:
        raise RuntimeError(" 点云没有颜色信息，无法提取白球。")

    white = np.array([1.0, 1.0, 1.0])
    color_dist = np.linalg.norm(colors - white, axis=1)

    mask = color_dist < color_tolerance
    white_points = points[mask]

    print(f" 提取到的白色点数：{len(white_points)}")

    if len(white_points) < min_white_points:
        raise RuntimeError(" 白色点太少，可能白球未被正确识别。请检查颜色或调整 color_tolerance。")

    white_pcd = o3d.geometry.PointCloud()
    white_pcd.points = o3d.utility.Vector3dVector(white_points)

    labels = np.array(white_pcd.cluster_dbscan(eps=cluster_eps, min_points=cluster_min_points, print_progress=False))
    unique_labels = np.unique(labels[labels >= 0])
    print(f" 初步检测到 {len(unique_labels)} 个白球簇。")

    # 尺寸过滤
    valid_clusters = []
    for label in unique_labels:
        cluster_pts = white_points[labels == label]
        bbox = np.ptp(cluster_pts, axis=0)  # 最大-最小，轴范围
        size = np.linalg.norm(bbox)
        if size > 0.03:  # 过滤掉尺寸小于5cm的杂点簇
            centroid = np.mean(cluster_pts, axis=0)
            valid_clusters.append(centroid)
        else:
            print(f" 忽略一个小白点簇，尺寸过小: {size:.4f}m")

    centroids = np.array(valid_clusters)
    print(f"最终保留 {len(centroids)} 个白球。")
    return centroids

#定位球中心点配准
def compute_transform_from_markers(src_pts, tgt_pts):
    print("使用定位球中心点进行配准...")
    src_pts = np.asarray(src_pts)
    tgt_pts = np.asarray(tgt_pts)
    assert src_pts.shape == tgt_pts.shape

    src_center = src_pts.mean(axis=0)
    tgt_center = tgt_pts.mean(axis=0)

    src_centered = src_pts - src_center
    tgt_centered = tgt_pts - tgt_center

    H = src_centered.T @ tgt_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = tgt_center - R @ src_center

    transform = np.eye(4)
    transform[:3, :3] = R
    transform[:3, 3] = t

    print("定位球变换矩阵：\n", transform)
    return transform

def evaluate_registration_error(T_est, T_gt, src_points=None):
    """
    评估配准误差（旋转、平移、点云误差）
    Args:
        T_est: 估计的变换矩阵 (4x4)
        T_gt: 真实变换矩阵 (4x4)
        src_points: 源点云坐标 (Nx3)，可选（用于计算点云误差）
    Returns:
        dict: 包含旋转误差（度）、平移误差（米）、点云误差（米）
    """
    # 提取旋转和平移
    R_est = T_est[:3, :3]
    t_est = T_est[:3, 3]
    R_gt = T_gt[:3, :3]
    t_gt = T_gt[:3, 3]
    # 1. 旋转误差（角度差）
    trace = np.clip((np.trace(R_gt.T @ R_est) - 1) / 2, -1, 1)
    rotation_error_deg = np.arccos(trace) * 180 / np.pi

    # 2. 平移误差（欧氏距离）
    translation_error = np.linalg.norm(t_gt - t_est)

    # 3. 点云误差（若提供源点云）
    point_error = None
    if src_points is not None:
        src_hom = np.column_stack([src_points, np.ones(len(src_points))])
        pts_est = (T_est @ src_hom.T).T[:, :3]
        pts_gt = (T_gt @ src_hom.T).T[:, :3]
        point_error = np.mean(np.linalg.norm(pts_gt - pts_est, axis=1))

    return {
        "rotation_error_deg": rotation_error_deg,
        "translation_error_m": translation_error,
        "point_error_m": point_error
    }

#多层点云集成
def load_multilayer_pointclouds(input_dir, file_prefix="layer_", file_ext=".ply"):
    """
    加载目录下的多层点云文件
    """
    files = sorted([
        f for f in os.listdir(input_dir)
        if f.startswith(file_prefix) and f.endswith(file_ext)
    ], key=lambda x: int(x[len(file_prefix):-len(file_ext)]))

    if not files:
        raise RuntimeError(f"目录中未找到匹配 {file_prefix}*{file_ext} 的文件")

    pointclouds = []
    for f in files:
        path = os.path.join(input_dir, f)
        pcd = o3d.io.read_point_cloud(path)
        if not pcd.has_points():
            raise RuntimeError(f"点云为空: {f}")
        pointclouds.append(pcd)
    return pointclouds, files

def hierarchical_marker_registration(layers_pcd, base_layer_idx=-1, output_dir=None, T_gt=None):
    """
    多层点云基于定位球配准到基准层
    """
    base_pcd = layers_pcd[base_layer_idx]
    registered_pcds = []

    print(f"\n提取基准层 layer_{base_layer_idx} 定位球...")
    base_markers = extract_white_spheres_centroids(base_pcd)

    for i, src_pcd in enumerate(layers_pcd):
        if i == base_layer_idx:
            registered_pcds.append(base_pcd)
            continue

        print(f"\n>>> 正在处理 layer_{i} 到 layer_{base_layer_idx}")
        src_markers = extract_white_spheres_centroids(src_pcd)
        matched_src, matched_tgt = find_best_marker_matching(src_markers, base_markers)
        transform = compute_transform_from_markers(matched_src, matched_tgt)

        # 评估误差
        if T_gt is not None:
            error = evaluate_registration_error(transform, T_gt, np.asarray(src_pcd.points))
            print(
                f"配准误差: 旋转={error['rotation_error_deg']:.3f}°, 平移={error['translation_error_m']:.6f}m, 点云误差={error['point_error_m']:.6f}m")

        src_pcd.transform(transform)

        registered_pcds.append(src_pcd)

    combined = o3d.geometry.PointCloud()
    for pcd in registered_pcds:
        combined += pcd

    final_pcd = combined.voxel_down_sample(0.02)
    if output_dir:
        o3d.io.write_point_cloud(os.path.join(output_dir, "final_merged.ply"), final_pcd)

    return registered_pcds, final_pcd

#主函数
def hierarchical_marker_registration_with_io(input_dir, output_dir, base_layer_idx=-1, T_gt=None):
    try:
        layers_pcd, filenames = load_multilayer_pointclouds(input_dir)
    except Exception as e:
        print(f"加载失败: {e}")
        return

    registered_pcds, final_pcd = hierarchical_marker_registration(
        layers_pcd=layers_pcd,
        base_layer_idx=base_layer_idx,
        output_dir=output_dir,
        T_gt=T_gt
    )
    print(f"多层定位球配准完成，结果保存在：{output_dir}")

if __name__ == "__main__":
    input_dir = r"D:\ply\conference\plypoint\pointcheck\4point"  # 存放 layer_0.ply ~ layer_N.ply
    output_dir = r"D:\ply\conference\plypoint\pointcheck\20250917"
    base_layer = 1  # 最后一层为参考层

#评估误差的真实变换矩阵
    T_gt = np.array([
        [0.00545031, -0.78594042, 0.62566942, -0.21043692],
        [0.79895919, 0.38266882, 0.47373312, -0.01801463],
        [-0.60895713, 0.49503181, 0.62714352, 0.01951526],
        [0, 0, 0, 1]
    ])

    hierarchical_marker_registration_with_io(
        input_dir=input_dir,
        output_dir=output_dir,
        base_layer_idx=base_layer,
        T_gt=T_gt
    )

###########################################################成功版本2###############################################################################