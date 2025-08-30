# Stable-Retro Integration for DreamerV3

这个集成允许DreamerV3在stable-retro环境中进行训练，支持多种经典游戏机游戏。

## 安装

1. 确保你已经安装了DreamerV3的基础环境：
```bash
conda activate dreamerv3-torch
```

2. 安装stable-retro：
```bash
pip install stable-retro
```

或者运行提供的安装脚本：
```bash
bash envs/setup_scripts/stable_retro.sh
```

### 重要限制

⚠️ **单模拟器限制**: stable-retro有一个重要限制 - 每个进程只能创建一个模拟器实例。这意味着：

- `envs` 配置必须设置为 1
- 训练和评估环境将共享同一个模拟器实例
- 不支持并行环境

这是stable-retro库本身的限制，不是DreamerV3的问题。

### 基本训练

使用以下命令在stable-retro环境中训练DreamerV3：

```bash
python dreamer.py --configs retro --task retro_<GAME_NAME> --logdir ./logdir/retro_<game_name>
```

例如：
```bash
python dreamer.py --configs retro --task retro_Airstriker-Genesis --logdir ./logdir/retro_airstriker
```

### 支持的游戏

stable-retro支持多种游戏，但需要相应的ROM文件。默认情况下，`Airstriker-Genesis`包含免费的ROM。

查看可用游戏：
```python
import retro
games = retro.data.list_games()
print(games)
```

### 配置参数

retro配置继承了基础配置并添加了以下默认设置：

```yaml
retro:
  steps: 4e5                    # 训练步数（400K）
  envs: 1                       # 并行环境数量
  action_repeat: 4              # 动作重复次数
  train_ratio: 1024             # 训练比率
  video_pred_log: true          # 视频预测日志
  eval_episode_num: 100         # 评估回合数
  actor: {dist: 'onehot', std: 'none'}  # 演员网络配置（离散动作）
  imag_gradient: 'reinforce'    # 想象梯度方法
  grayscale: False              # 是否使用灰度图像
  time_limit: 108000            # 时间限制
  size: [84, 84]                # 观察图像大小
```

### 自定义配置

你可以创建自定义配置来调整特定游戏的参数：

```bash
python dreamer.py --configs retro --task retro_SonicTheHedgehog-Genesis \
  --action_repeat 4 --size [64,64] --grayscale True --logdir ./logdir/sonic
```

## 测试集成

运行测试脚本验证集成是否正常工作：

```bash
# 测试基本环境功能
python test_stable_retro.py

# 测试特定游戏
python test_stable_retro.py --game SonicTheHedgehog-Genesis --steps 200

# 测试DreamerV3集成
python test_stable_retro.py --test-dreamer
```

## 环境特性

### 观察空间
- **图像观察**：默认为84x84 RGB图像
- **RAM观察**：可选择使用内存状态作为观察
- **变量观察**：游戏特定变量可在info字典中获取

### 动作空间
- **离散动作**：自动转换为适合DreamerV3的one-hot编码
- **多键组合**：支持多个按键的组合操作

### 预处理
- **图像调整**：自动调整到指定尺寸
- **灰度转换**：可选的RGB到灰度转换
- **动作重复**：支持动作重复以提高训练效率

## 游戏ROM管理

### 导入ROM
如果你有合法的ROM文件，可以使用以下方式导入：

```bash
python -m retro.import /path/to/your/roms/
```

### 查看已导入的游戏
```python
import retro
print("Available games:")
for game in retro.data.list_games():
    print(f"  {game}")
```

## 监控训练

使用TensorBoard监控训练进度：

```bash
tensorboard --logdir ./logdir
```

## 故障排除

### 常见问题

1. **ImportError: No module named 'retro'**
   - 解决方案：`pip install stable-retro`

2. **ROM not found**
   - 确保游戏ROM已正确导入
   - 使用`retro.data.list_games()`检查可用游戏

3. **CUDA out of memory**
   - 减少批量大小或环境数量
   - 使用更小的图像尺寸

4. **训练不稳定**
   - 调整学习率
   - 尝试不同的动作重复值
   - 考虑使用灰度图像以减少复杂性

### 性能优化

1. **图像尺寸**：较小的图像（如64x64）训练更快
2. **灰度模式**：减少内存使用和计算复杂度
3. **动作重复**：适当的动作重复可以提高学习效率
4. **并行环境**：在有足够资源时增加并行环境数量

## 示例结果

训练成功后，你应该能看到：
- 智能体学会游戏的基本操作
- 奖励逐渐增加
- 视频预测质量改善

## 贡献

如果你在使用过程中发现问题或有改进建议，欢迎提交issue或pull request。

## 许可证

此集成遵循原DreamerV3项目的许可证条款。请确保你拥有所使用游戏ROM的合法权利。
