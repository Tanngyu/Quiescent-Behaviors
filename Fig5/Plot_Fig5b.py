import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import FormatStrFormatter

# 设置字体和大小
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 42
# 特别设置数学公式字体（覆盖全局设置）
rcParams['mathtext.fontset'] = 'cm'  # Computer Modern字体
rcParams['mathtext.rm'] = 'serif'    # 常规数学字体
rcParams['mathtext.it'] = 'serif:italic'  # 斜体数学字体
rcParams['mathtext.bf'] = 'serif:bold'    # 粗体数学字体

# 读取Excel文件
df = pd.read_excel('..\..\data\Personalised_Ho.xlsx', header=None)

# 取出前3行数据，每行作为一条曲线
row1 = df.iloc[7, 2:]  # 第一行数据
row2 = df.iloc[8, 2:]  # 第二行数据
row3 = df.iloc[9, 2:]  # 第三行数据

# 创建一个图形，大小为8x7
plt.figure(figsize=(8, 7))

# 生成 x 坐标：假设一共 row1.size 个点，均匀分布在 [0.05, 0.95]
x = np.linspace(1, 20, row1.size)

plt.plot(x, row1, marker='o', label=r'$C$', color='#036EB8', linewidth=5, markersize=13)
plt.plot(x, row2, marker='^', label=r'$D$', color='#E60012', linewidth=5, markersize=13)
plt.plot(x, row3, marker='s', label=r'$Q$', color='#FFBC32', linewidth=5, markersize=13)

# 用数学模式显示
plt.xlabel(r'E($\rho$)')
plt.ylabel(r'$f_s$')

# plt.legend(fontsize=32)

# 1) 固定横坐标范围为 [0.05, 0.95]
plt.xlim(0.8, 20.2)

# 2) 手动设置 x 轴刻度位置与显示标签
x_ticks = [1, 5, 10, 15, 20]
x_labels = ['0.1', '0.3', '0.5', '0.7', '0.9']
plt.xticks(x_ticks, x_labels)
plt.yticks([0.00, 0.25, 0.50, 0.75, 1.00])

# 设置y轴刻度格式，保留两位小数
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

# 保存为 PDF 并显示
plt.savefig('Temp.pdf', format='pdf', bbox_inches='tight', dpi=300, pad_inches=0.03)
plt.show()
