import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 36

# 特别设置数学公式字体（覆盖全局设置）
rcParams['mathtext.fontset'] = 'cm'  # Computer Modern字体
rcParams['mathtext.rm'] = 'serif'    # 常规数学字体
rcParams['mathtext.it'] = 'serif:italic'  # 斜体数学字体
rcParams['mathtext.bf'] = 'serif:bold'    # 粗体数学字体


groups = 7
bars_per_group = 3

# 示例数据（保持原始数据格式）
data = np.array([
    [0.112693866666666, 0.430543933333332, 0.456762199999999],   # 组0
    [0.0818591999999999, 0.490213066666666, 0.427927733333333],   # 组1
    [0.0558218000000001, 0.570234, 0.373944199999999],   # 组2
    [0.0348277333333333, 0.698026200000001, 0.267146066666666],   # 组3
    [0.229822066666666, 0.0202441333333332, 0.7499338],   # 组4
    [0.0207607333333332, 0.0000113999999999999, 0.979227866666669],   # 组5
    [0.000989066666666623, 0.00988033333333333, 0.989130599999997],   # 组6
])

group_positions = np.arange(groups)
bar_height = 0.20  # 增大条形高度
offsets = np.linspace(-0.22, 0.22, bars_per_group)  # 减小偏移量
colors = ['#325276',  '#AE3733', '#D6AD38']
labels = ['$C$', '$D$', '$Q$']
hatches = ['/','x', '\\' ]

plt.figure(figsize=(7, 16))  # 调整画布比例

for i in range(groups):
    for j in range(bars_per_group):
        y_pos = group_positions[i] + offsets[j]
        plt.barh(
            y_pos,
            data[i, j],
            height=bar_height,
            color=colors[j],
            edgecolor='black',
            linewidth=1.5,
            hatch=hatches[j],
            label=labels[j] if i == 0 else None
        )

# 坐标轴设置
y_labels = [r'$1/k_i^4$',
            r'$1/k_i^3$',
            r'$1/k_i^2$',
            r'$1/k_i$',
            r'$1/4$',
            r'$k_i$',
            r'$k_i^2$',
            ]

ax = plt.gca()
ax.set_yticks(group_positions)
ax.set_yticklabels(y_labels, rotation=90, va='center')  # 设置90度旋转
ax.tick_params(axis='y', which='both', length=0, pad=10)  # 调整标签间距

plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
plt.xlabel(r'$f_S$', labelpad=15)
plt.ylabel(r'$\rho$', labelpad=15, rotation=0, va='center', ha='right')

# 优化图例位置
plt.legend(frameon=False)

# 调整边距
plt.subplots_adjust(left=0.28, right=0.95, bottom=0.15, top=0.95)

# 保存为 PDF 并显示
plt.savefig('TEMP.pdf', format='pdf', bbox_inches='tight', dpi=300, pad_inches=0.03)

plt.show()