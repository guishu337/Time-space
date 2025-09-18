import gradio as gr
import open3d as o3d
import numpy as np
from scipy.integrate import odeint
import tempfile
import os
import time
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


class EnhancedWasteModel:
    def __init__(self, initial_height, initial_mass, season="夏季", material='organic'):
        """时空演化模型，结合温湿度、材质和微生物因素"""
        self.h0 = initial_height
        self.m0 = initial_mass
        self.material = material
        self.season = season

        # 基础参数
        self.params = {
            'T_opt': 290,  # 最优温度 17°C (290K)
            'H_opt': 0.6,  # 最优湿度 (60%)
            'Ea_over_R': 4000,  # 活化能/气体常数 (K)
            'Ks': 200,  # 半饱和常数 (ton)
            'mu_max': 1.0,  # 最大比生长速率 (day^-1)
            'B0': 1.0,  # 初始微生物量
            'eta': 0.7,  # 压缩效率
            'alpha': 0.5  # 质量-体积响应系数
        }

        # 材质参数
        self.material_params = {
            'organic': {'w_org': 0.62, 'k_org': 0.5, 'w_inert': 0.38, 'k_inert': 0.001, 'beta': -0.05},
            'mixed': {'w_org': 0.3, 'k_org': 0.05, 'w_inert': 0.7, 'k_inert': 0.001, 'beta': -0.02},
            'metal': {'w_org': 0.0, 'k_org': 0.0, 'w_inert': 1.0, 'k_inert': 0.0001, 'beta': 0.0}
        }

        # 季节参数
        self.season_params = {
            '夏季': {'T': 293, 'H': 0.7},  # 30°C, 70%
            '冬季': {'T': 283, 'H': 0.5}  # 10°C, 50%
        }

    def dynamic_degradation_rate(self, m, B, T, H):
        """动态降解系数计算"""
        f_temp = np.exp(-self.params['Ea_over_R'] * (1 / T - 1 / self.params['T_opt']) ** 2)
        f_hum = (H ** 2) / (self.params['H_opt'] ** 2 + H ** 2)

        mat = self.material_params[self.material]
        f_material = (mat['w_org'] * mat['k_org'] +
                      mat['w_inert'] * mat['k_inert'] +
                      mat['beta'] * mat['w_org'] * mat['w_inert'])

        f_microbe = B / (self.params['Ks'] + B)

        return self.params['mu_max'] * f_temp * f_hum * f_material * f_microbe

    def microbe_growth(self, m, B):
        """微生物生长模型"""
        return self.params['mu_max'] * (m / self.m0) * B

    def compression_factor(self, T, H):
        """温湿度对压缩的影响"""
        T_min = 283  # 10°C
        g = ((T - T_min) / (self.params['T_opt'] - T_min)) * (H / self.params['H_opt'])
        return np.clip(g, 0, 1)

    def solve(self, days):
        """求解耦合系统"""
        t = np.linspace(0, days, days + 1)
        y0 = [self.m0, self.params['B0']]
        solution = odeint(self._system_equations, y0, t)

        m = solution[:, 0]
        B = solution[:, 1]
        season = self.season_params[self.season]
        T, H = season['T'], season['H']

        gamma = self.params['alpha'] * self.dynamic_degradation_rate(m, B, T, H)
        g = self.compression_factor(T, H)
        h = self.h0 * (1 - self.params['eta'] * (1 - np.exp(-gamma * t)) * g)

        return t, m, h

    def _system_equations(self, y, t):
        """微分方程系统"""
        m, B = y
        season = self.season_params[self.season]
        T, H = season['T'], season['H']

        k = self.dynamic_degradation_rate(m, B, T, H)
        dmdt = -k * m
        dBdt = self.microbe_growth(m, B)

        return [dmdt, dBdt]

    def get_values_at_day(self, day):
        """获取指定天数的预测值"""
        t, m, h = self.solve(day + 1)  # +1确保包含目标天数
        return {
            'mass': m[day],
            'height': h[day],
            'density': m[day] / (self.h0 * 1e-3) if h[day] > 0 else 0
        }


class WasteLayer:
    def __init__(self, point_cloud, model, material='organic'):
        self.point_cloud = point_cloud
        self.model = model
        self.material = material
        self.initial_height = model.h0
        self.initial_points = np.asarray(point_cloud.points)  # 保存初始坐标

        # 颜色设置
        self.base_colors = np.asarray(point_cloud.colors)
        if len(self.base_colors) == 0:
            self.base_colors = np.ones((len(point_cloud.points), 3))

        self.color_end = {
            'organic': np.array([0.3, 0.15, 0]),
            'metal': np.array([0.5, 0.5, 0.5]),
            'mixed': np.array([0.4, 0.2, 0.1])
        }.get(material, np.array([0.3, 0.15, 0]))

    def update(self, day):
        """更新点云状态"""
        # 获取模型预测值
        state = self.model.get_values_at_day(day)
        current_height = state['height']

        # 计算压缩比例因子（非线性）
        compression_ratio = current_height / self.initial_height

        # 更新高度
        updated_points = np.copy(self.initial_points)
        mask = updated_points[:, 2] > 2.7  # 地面阈值

        # 非线性压缩
        z_normalized = (updated_points[mask, 2] - 2.7) / (self.initial_height - 2.7)  # 归一化高度
        updated_points[mask, 2] = 2.7 + (self.initial_height - 2.7) * z_normalized * compression_ratio

        # 更新颜色
        color_shift_days = 3
        alpha = min(1.0, day / color_shift_days)
        new_colors = (1 - alpha) * self.base_colors + alpha * self.color_end

        # 创建新点云
        updated_pc = o3d.geometry.PointCloud()
        updated_pc.points = o3d.utility.Vector3dVector(updated_points)
        updated_pc.colors = o3d.utility.Vector3dVector(new_colors)

        self.point_cloud = updated_pc
        return updated_pc


def animate_multi_layers(layers, days_range, save_day=None, save_dir=None):
    """可视化动画"""
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1000, height=800)

    for day in range(days_range + 1):
        vis.clear_geometries()
        for layer in layers:
            updated_pc = layer.update(day)
            vis.add_geometry(updated_pc)

        vis.poll_events()
        vis.update_renderer()

        # 保存指定天数点云
        if save_day is not None and day == save_day and save_dir:
            os.makedirs(save_dir, exist_ok=True)
            for i, layer in enumerate(layers):
                save_path = os.path.join(save_dir, f"layer_{i}_day_{day}.ply")
                o3d.io.write_point_cloud(save_path, layer.point_cloud)

        time.sleep(0.3)

    vis.destroy_window()



uploaded_layers = []


def add_layer(ply_files, height, mass, material, season):
    global uploaded_layers
    if not ply_files:
        return "请上传点云文件"

    for ply_file in ply_files:
        model = EnhancedWasteModel(height, mass, season, material)
        pcd = o3d.io.read_point_cloud(ply_file.name)
        layer = WasteLayer(pcd, model, material)
        uploaded_layers.append(layer)

    return f"✅ 已添加 {len(ply_files)} 层（{season}）"


def run_simulation(season, save_day):
    global uploaded_layers
    if not uploaded_layers:
        return "请先添加点云层"

    # 根据季节决定模拟天数
    days = 5 if season == "夏季" else 9
    save_day = min(save_day, days)

    # 运行模拟
    save_dir = os.path.join(tempfile.gettempdir(), "waste_sim_output")
    animate_multi_layers(uploaded_layers, days, save_day, save_dir)

    # 生成结果图表
    fig = plt.figure(figsize=(10, 8))
    for i, layer in enumerate(uploaded_layers):
        t = np.arange(days + 1)
        m = [layer.model.get_values_at_day(d)['mass'] for d in t]
        h = [layer.model.get_values_at_day(d)['height'] for d in t]

        plt.subplot(2, 1, 1)
        plt.plot(t, m, label=f'Layer {i} Mass')
        plt.subplot(2, 1, 2)
        plt.plot(t, h, label=f'Layer {i} Height')

    plt.subplot(2, 1, 1)
    plt.title("Mass Change")
    plt.ylabel("Mass (ton)")
    plt.legend()
    plt.grid(False)

    plt.subplot(2, 1, 2)
    plt.title("Height Change")
    plt.xlabel("Days")
    plt.ylabel("Height (m)")
    plt.legend()
    plt.grid(False)

    plt.tight_layout()
    chart_path = os.path.join(save_dir, "results.png")
    plt.savefig(chart_path)
    plt.close()

    return f"模拟完成！结果已保存至: {save_dir}", chart_path


with gr.Blocks() as demo:
    gr.Markdown("""
    # 🗑️ 垃圾时空演化模拟系统
    ### 基于温湿度-材质-微生物耦合模型
    """)

    with gr.Row():
        with gr.Column():
            ply_input = gr.Files(label="上传点云文件(.ply)", file_types=[".ply"])
            height_input = gr.Number(label="初始高度 (m)", value=5.0)
            mass_input = gr.Number(label="初始质量 (ton)", value=1000.0)
            material_input = gr.Dropdown(
                choices=["organic", "metal", "mixed"],
                value="organic",
                label="垃圾类型"
            )
            season_input = gr.Dropdown(
                choices=["夏季", "冬季"],
                value="夏季",
                label="季节"
            )
            add_btn = gr.Button("添加层")

        with gr.Column():
            layer_status = gr.Textbox(label="层状态")
            save_day = gr.Slider(
                minimum=0, maximum=10, step=1,
                label="保存第几天的点云", value=5
            )
            run_btn = gr.Button("开始模拟", variant="primary")
            result_out = gr.Textbox(label="模拟结果")
            chart_output = gr.Image(label="质量与高度变化曲线")

    add_btn.click(
        add_layer,
        inputs=[ply_input, height_input, mass_input, material_input, season_input],
        outputs=layer_status
    )

    run_btn.click(
        run_simulation,
        inputs=[season_input, save_day],
        outputs=[result_out, chart_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861)