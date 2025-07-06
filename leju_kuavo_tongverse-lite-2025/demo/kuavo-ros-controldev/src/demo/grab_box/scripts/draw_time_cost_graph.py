import pandas as pd
import matplotlib.pyplot as plt
import sys

# 读取CSV文件
file_path = sys.argv[1]
data = pd.read_csv(file_path)

# 过滤掉 "Total" 项
total_value = data[data['NodeName'].str.replace('"', '') == 'Total']['ExecutionTime_ms'].values[0]
print("Total execution time: ", total_value)
data = data[data['NodeName'].str.replace('"', '') != 'Total']

# 创建柱状图
plt.figure(figsize=(10, 6))
bars = plt.bar(data['NodeName'], data['ExecutionTime_ms'], color='skyblue')
plt.title(f'Execution Time of Behavior Tree Nodes, Total time cost: {total_value/1000.0:.1f} s')
plt.xlabel('Node Name')
plt.ylabel('Execution Time (ms)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()  # 调整图形以适应标签

# 在每个柱上添加具体数字
for bar in bars:
    yval = bar.get_height()  # 获取每个柱子的高度
    plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom')

# 展示图形
plt.show()

# 过滤掉 "Total" 项
data = data[data['NodeName'].str.replace('"', '') != 'Total']

# 创建饼图
plt.figure(figsize=(8, 8))
plt.pie(data['ExecutionTime_ms'], labels=data['NodeName'], autopct='%1.1f%%', startangle=140)
plt.title('Execution Time Distribution of Behavior Tree Nodes')
plt.axis('equal')  # 确保饼图是圆形

# 展示饼图
plt.show()