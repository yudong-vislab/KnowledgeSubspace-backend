import json
import numpy as np
import plotly.graph_objects as go

def pixel_to_axial(x, y, size):
    """
    将二维平面坐标转换为六边形网格的轴坐标 (q, r)。
    """
    q = (np.sqrt(3)/3 * x - 1.0/3 * y) / size
    r = (2.0/3 * y) / size
    return cube_round(q, -q-r, r)

def cube_round(x, y, z):
    """
    将笛卡尔坐标 (x, y, z) 四舍五入到最近的六边形坐标 (q, r)。
    """
    rx, ry, rz = round(x), round(y), round(z)
    dx, dy, dz = abs(rx-x), abs(ry-y), abs(rz-z)
    if dx > dy and dx > dz:
        rx = -ry - rz
    elif dy > dz:
        ry = -rx - rz
    else:
        rz = -rx - ry
    return int(rx), int(rz)
def generate_hexagon_boundary(q, r, size):
    """根据六边形的q, r坐标生成六边形的顶点坐标"""
    angle = np.pi / 3  # 60度
    hexagon = []
    for i in range(6):
        angle_rad = angle * i
        x = q + size * np.cos(angle_rad)
        y = r + size * np.sin(angle_rad)
        hexagon.append([x, y])
    return np.array(hexagon)

def group_points_into_hexagons(data, hex_size=1):
    """
    将句子根据 2D 坐标分组到六边形网格中。
    每个六边形会包含一个 (q, r) 坐标和该六边形内所有属于该六边形的句子的 MSU_id，
    同时根据 paper_id 对句子进行分组。
    每个六边形格子由 (q, r, paper_id) 组合唯一标识。
    """
    hexagons = {}  # 存储六边形格子，键是 (q, r, paper_id)，值是该六边形内的句子 MSU_id 列表
    
    x_coords = [item["2d_coord"][0] for item in data]
    y_coords = [item["2d_coord"][1] for item in data]
    print(f"原始 X 坐标范围: [{min(x_coords):.4f}, {max(x_coords):.4f}]")
    print(f"原始 Y 坐标范围: [{min(y_coords):.4f}, {max(y_coords):.4f}]")
    
    for item in data:
        # 获取句子的 2D 坐标和 paper_id
        x, y = item["2d_coord"]
        paper_id = item["paper_id"]
        
        # 将 2D 坐标转换为六边形坐标 (q, r)
        q, r = pixel_to_axial(x, y, hex_size)
        
        # 组合成 (q, r, paper_id) 作为六边形格子的唯一标识
        hex_coord = (q, r, paper_id)
        
        # 如果该六边形 (q, r, paper_id) 不存在，初始化一个新的列表
        if hex_coord not in hexagons:
            hexagons[hex_coord] = []
        
        # 将句子的 MSU_id 加入到对应的六边形 (q, r, paper_id) 中
        hexagons[hex_coord].append(item["MSU_id"])
    
    # 打印转换后的坐标范围
    q_coords = [hex_coord[0] for hex_coord in hexagons.keys()]
    r_coords = [hex_coord[1] for hex_coord in hexagons.keys()]
    print(f"转换后 Q 坐标范围: [{min(q_coords):.4f}, {max(q_coords):.4f}]")
    print(f"转换后 R 坐标范围: [{min(r_coords):.4f}, {max(r_coords):.4f}]")
    
    return hexagons



def visualize_hexagons(data, hex_size=1, save_html="hexagon_visualization.html"):
    hexagons = group_points_into_hexagons(data, hex_size)

    fig = go.Figure()

    # 添加六边形网格
    for (q, r), msu_ids in hexagons.items():
        hexagon_boundary = generate_hexagon_boundary(q, r, hex_size)
        fig.add_trace(go.Scatter(
            x=hexagon_boundary[:, 0],
            y=hexagon_boundary[:, 1],
            mode='lines',
            fill='toself',
            line=dict(color='rgba(0,0,0,0.2)', width=1),
            name=f"Hex ({q},{r})",
            showlegend=False
        ))

    # 添加句子点
    for item in data:
        x, y = item["2d_coord"]
        q, r = pixel_to_axial(x, y, hex_size)
        fig.add_trace(go.Scatter(
            x=[q],
            y=[r],
            mode='markers',
            marker=dict(
                size=8,
                color='rgba(255,0,0,0.7)',
                opacity=0.7
            ),
            text=f"MSU_id: {item['MSU_id']}",
            hoverinfo='text'
        ))

    # 更新图表布局
    fig.update_layout(
        title="Hexagonal Grid with Sentences",
        xaxis_title="Hex Q Coordinate",
        yaxis_title="Hex R Coordinate",
        hovermode='closest',
        width=1000,
        height=700,
        showlegend=False
    )

    # 保存为HTML
    fig.write_html(save_html)
    print(f"图表已保存为 {save_html}")
def save_hexagon_info(hexagons, output_file="hexagon_info.json"):
    hexagon_info = []
    
    # 保存六边形的 (q, r) 和对应的 MSU_ids
    for hex_coord, msu_ids in hexagons.items():
        hexagon_info.append({
            "hex_coord": [hex_coord[0], hex_coord[1]],  # (q, r, paper_id)
            "country": hex_coord[2],  # paper_id
            "MSU_ids": msu_ids
        })
    
    # 将信息保存到 JSON 文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(hexagon_info, f, ensure_ascii=False, indent=4)
    
    print(f"六边形信息已保存到 {output_file}")

# 读取 formdatabase_v2.0.json 文件
with open('pollution_result/formdatabase_v2.0.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 将句子根据 2D 坐标分组到六边形中
hexagons = group_points_into_hexagons(data, hex_size=0.15)
# visualize_hexagons(data, hex_size=0.2, save_html="hexagon_visualization.html")
# 输出每个六边形格子的信息
save_hexagon_info(hexagons, output_file="hexagon_info_0.15.json")

