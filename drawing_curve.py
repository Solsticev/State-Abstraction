import numpy as np
import matplotlib.pyplot as plt


data_wood = np.load("./log/subtask_wood/evaluations.npz")
data_wood_pickaxe = np.load("./log/subtask_wood_pickaxe/evaluations.npz")
data_stone = np.load("./log/subtask_stone/evaluations.npz")
data_stone_pickaxe = np.load("./log/subtask_stone_pickaxe/evaluations.npz")
data_iron = np.load("./log/subtask_iron/evaluations.npz")

rl_only_stone_pickaxe = np.load("./log/RL_only_stone_pickaxe/evaluations.npz")

rl_only_iron = np.load("./log/RL_only_iron/evaluations.npz")
print(np.mean(rl_only_iron["results"], axis=1))

stage1_data = np.mean(data_wood["results"], axis=1)
stage1_data = np.insert(stage1_data, 0, 0)
stage2_data = np.mean(data_wood_pickaxe["results"], axis=1)
stage3_data = np.mean(data_stone["results"], axis=1)
stage4_data = np.mean(data_stone_pickaxe["results"], axis=1)
stage5_data = np.mean(data_iron["results"], axis=1)
stage5_data = np.delete(stage5_data, -1)

baseline_data = np.mean(rl_only_iron["results"], axis=1)
baseline_data = np.insert(baseline_data, 0, 0)
baseline_data = np.delete(baseline_data, -1)


# 2. 累加计算（保持不变）
all_data = [stage1_data]
prev_last_value = stage1_data[-1]
for stage_data in [stage2_data, stage3_data, stage4_data, stage5_data]:
    current_data = stage_data + prev_last_value
    all_data.append(current_data)
    prev_last_value = current_data[-1]
final_data = np.concatenate(all_data)

# 3. 创建并修改x轴数据：将每个值乘以10000
x_values = np.arange(len(final_data)) * 10000

# 4. 绘制折线图
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(10, 6))

plt.plot(x_values, final_data, marker='o', linestyle='-', color='b', label='Our Method')

# 5. 添加图表元素，使横坐标标签更清晰
plt.title('Training Reward Curve', fontsize=16)
plt.xlabel('Training Steps', fontsize=12) # 这里的单位是"步骤"，但数值放大了10000倍
plt.ylabel('Cumulative Reward', fontsize=12)

# 在每个阶段的连接点上添加垂直线
stage1_end = len(stage1_data) * 10000
stage2_end = stage1_end + len(stage2_data) * 10000
stage3_end = stage2_end + len(stage3_data) * 10000
stage4_end = stage3_end + len(stage4_data) * 10000

plt.axvline(x=(stage1_end/10000-1)*10000, color='gray', linestyle='--', linewidth=1)
plt.axvline(x=(stage2_end/10000-1)*10000, color='gray', linestyle='--', linewidth=1)
plt.axvline(x=(stage3_end/10000-1)*10000, color='gray', linestyle='--', linewidth=1)
plt.axvline(x=(stage4_end/10000-1)*10000, color='gray', linestyle='--', linewidth=1)

# 添加文本注释来标记每个阶段
plt.text(stage1_end/2-30000, 4, 'Wood', horizontalalignment='center', verticalalignment='bottom', fontsize=8)
plt.text((stage1_end + stage2_end)/2-20000, 4, 'Wood Pickaxe', horizontalalignment='center', verticalalignment='bottom', fontsize=8)
plt.text((stage2_end + stage3_end)/2-20000, 4, 'Stone', horizontalalignment='center', verticalalignment='bottom', fontsize=8)
plt.text((stage3_end + stage4_end)/2-20000, 4, 'Stone Pickaxe', horizontalalignment='center', verticalalignment='bottom', fontsize=8)
plt.text((stage4_end + len(final_data)*10000)/2-20000, 4, 'Iron', horizontalalignment='center', verticalalignment='bottom', fontsize=8)

total_len = sum(len(d) for d in all_data)
baseline_x_values = np.arange(total_len) * 10000
plt.plot(baseline_x_values[:len(baseline_data)], baseline_data, color='red', linestyle='-', marker='^', linewidth=2, label='RL Only')

plt.legend(loc='lower right', fontsize=10)

total_steps = len(final_data) * 10000
plt.xticks(np.linspace(0, total_steps -1, 30))

plt.tight_layout()
plt.savefig('cumulative_training_curve.png')

print("Plot saved to 'cumulative_training_curve.png'")

