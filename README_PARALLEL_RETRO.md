# Dreamer并行Retro环境训练

本修改为Dreamer添加了对多个retro游戏并行训练的支持，参考了PPO中的多任务训练方法。

## 主要特性

### 🎮 多游戏并行训练
- 支持在多个不同的retro游戏上同时训练
- 每个游戏运行在独立的进程中
- 自动处理不同游戏的动作空间差异

### 🔧 兼容性保持
- 完全兼容原有的单游戏训练模式
- 不影响其他环境（dmc、atari等）的使用
- 保持原有的API和配置结构

### ⚙️ 配置灵活
- 可通过配置文件或命令行参数控制
- 支持自定义游戏列表
- 支持视觉奖励集成

## 使用方法

### 方法1: 使用配置文件

#### 单游戏训练（原始方式）
```bash
python dreamer.py --configs retro --task retro_SuperMarioBros-Nes
```

#### 多游戏并行训练
```bash
python dreamer.py --configs retro_parallel --task retro_multitask
```

### 方法2: 使用便捷脚本

```bash
# 使用默认游戏列表
python run_parallel_retro.py

# 自定义游戏列表
python run_parallel_retro.py --games SuperMarioBros-Nes MegaMan-Nes DonkeyKong-Nes

# 启用视觉奖励
python run_parallel_retro.py --visual-reward --visual-weight 0.1

# 自定义训练步数和设备
python run_parallel_retro.py --steps 5e6 --device cuda:1
```

### 方法3: 命令行自定义配置

```bash
python dreamer.py \\
  --configs retro \\
  --task retro_multitask \\
  --retro_parallel_games=True \\
  --retro_games_list="['SuperMarioBros-Nes','MegaMan-Nes','DonkeyKong-Nes']" \\
  --steps=2e6 \\
  --logdir=./logs_parallel
```

## 配置参数

### 并行训练相关参数
- `retro_parallel_games`: 是否启用多游戏并行训练 (默认: False)
- `retro_games_list`: 游戏列表，例如 `['SuperMarioBros-Nes', 'MegaMan-Nes']`
- `envs`: 环境数量（在并行模式下会自动设置为游戏数量）

### 推荐的游戏列表

#### 经典平台游戏
```yaml
retro_games_list: [
  'SuperMarioBros-Nes',
  'SuperMarioWorld-Snes',
  'DonkeyKong-Nes',
  'DonkeyKongCountry-Snes'
]
```

#### 动作游戏
```yaml
retro_games_list: [
  'MegaMan-Nes',
  'MegaMan2-Nes',
  'ContraForce-Nes',
  'GunNac-Nes'
]
```

#### 混合类型
```yaml
retro_games_list: [
  'SuperMarioBros-Nes',     # 平台游戏
  'MegaMan-Nes',           # 动作游戏
  'BubbleBobble-Nes',      # 休闲游戏
  'Gradius-Nes',           # 射击游戏
  'Jackal-Nes'             # 射击游戏
]
```

## 技术细节

### 动作空间处理
- 自动检测所有游戏的动作空间
- 如果动作空间不同，使用最大动作空间
- 显示警告信息以便调试

### 环境创建流程
1. 解析配置，检查是否启用并行模式
2. 为每个游戏创建独立的环境实例
3. 使用Dreamer的Parallel类进行进程管理
4. 自动设置环境数量匹配游戏数量

### 与PPO的差异
- PPO使用SubprocVecEnv，Dreamer使用自定义Parallel类
- PPO支持向量化环境，Dreamer为每个游戏创建独立环境
- 保持了Dreamer原有的环境包装器链

## 文件结构

```
dreamerv3-torch/
├── dreamer.py              # 主训练脚本（已修改）
├── configs.yaml            # 配置文件（已添加并行配置）
├── run_parallel_retro.py   # 便捷启动脚本（新增）
├── test_parallel_setup.py  # 测试脚本（新增）
└── envs/
    └── stable_retro.py     # Retro环境包装器
```

## 测试

运行测试脚本验证设置：

```bash
# 完整测试（需要ROM文件）
python test_parallel_setup.py

# 跳过环境创建测试
python test_parallel_setup.py --skip-env
```

## 故障排除

### 常见问题

1. **ImportError: stable-retro模块未找到**
   ```bash
   pip install stable-retro
   ```

2. **ROM文件未找到**
   - 确保ROMs在正确的目录下
   - 参考stable-retro文档配置ROM路径

3. **动作空间不匹配警告**
   - 这是正常的，系统会自动使用最大动作空间
   - 可以选择动作空间相似的游戏来避免

4. **内存不足**
   - 减少游戏数量
   - 调整batch_size参数
   - 使用更少的并行环境

5. **GPU内存不足**
   - 设置较小的batch_size
   - 禁用visual_reward
   - 使用CPU训练

### 性能优化建议

1. **游戏选择**
   - 选择动作空间相似的游戏
   - 避免混合差异很大的游戏类型

2. **硬件配置**
   - 使用SSD存储以加快环境加载
   - 确保足够的CPU核心数支持多进程
   - 使用高内存GPU支持大批次训练

3. **训练参数**
   - 适当增加训练步数（建议2e6以上）
   - 调整学习率以适应多任务学习
   - 考虑使用curriculum learning

## 贡献

基于PPO多任务训练的经验，欢迎提出改进建议：
- 更好的动作空间统一方法
- 自适应学习率调整
- 游戏难度平衡策略
- 更高效的多进程通信
