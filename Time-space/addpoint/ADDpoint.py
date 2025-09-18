import open3d as o3d
import numpy as np

# 1. 加载原始点云
pcd = o3d.io.read_point_cloud(r"D:\ply\conference\plypoint\pointcheck\4point\pointcloud_with_marker12.ply")

# 2. 创建球体 mesh
sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
sphere.compute_vertex_normals()

# 3. 平移球体
# sphere.translate([-2.218621, 2.271904, 2.104797]) ##11
# sphere.translate([-0.065681, -1.254818, 4.995250]) ##12
# sphere.translate([-2.209028, 4.750197, 1.238394]) ##13
# sphere.translate([-0.688605, -0.447605, 4.287989])##21
# sphere.translate([4.020420, 1.878616, 2.730907])##22
# sphere.translate([-3.151752, 0.590375, 4.454330]) ##23
# sphere.translate([2.235898, 2.309355, 2.061301]) ##24
# sphere.translate([2.250554, 4.099477, 0.676315]) ##25
# sphere.translate([-3.207488, 0.530573, 4.399609]) ##41
# sphere.translate([-0.731607, 0.150104, 3.905314]) ##42
# sphere.translate([4.185523, 1.398218, 3.469508]) ##43
# sphere.translate([-0.799959, 3.634074, 1.079901]) ##44
sphere.translate([4.195292, 1.433058, 3.664387]) ##25
# sphere.translate([-2.890787, -0.133198, 4.782284]) ##25
# sphere.translate([-0.515527, -0.436994, 4.318385]) ##25
# sphere.translate([-0.417412, 3.439524, 2.137572]) ##25

# 4. 采样为点云
sphere_pcd = sphere.sample_points_poisson_disk(number_of_points=500)

# 给球体点云上色为白色
sphere_pcd.paint_uniform_color([1.0, 1.0, 1.0])  # 白色


# 5. 合并点云
combined_pcd = pcd + sphere_pcd
print("是否有颜色：", combined_pcd.has_colors())
# 6. 保存点云
o3d.io.write_point_cloud("pointcloud_with_marker12.ply", combined_pcd)

print("保存完成")
