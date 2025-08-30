
# StableRetroScorer 参数一致性参考

## 核心参数 (必须在所有评估中保持一致)

### 1. 视觉编码器
```python
visual_encoder = "CLIP"  # 默认值，所有评估必须使用
```

### 2. DenStream聚类参数
```python
denstream_params = {
    'lambda_': 0.001,  # 衰减因子
    'eps': 2.0,        # 空间半径  
    'beta': 0.3,       # 异常点阈值
    'mu': 2           # 核心微簇阈值
}
```

### 3. 帧尺寸
```python
frame_shape = (84, 84, 3)  # Height, Width, Channels
```

### 4. 评分逻辑
- L1: survival_steps (存活步数)
- L2: cluster_width (聚类宽度) 
- L3: state_judger prediction (win/loss/else)

### 5. Episode逻辑
- reset_l2_clusterer: 仅第一个episode为True
- terminated: 基于episode自然结束
- verbose: 每25步打印一次

## 使用示例
```python
scorer = StableRetroScorer(
    visual_encoder="CLIP",
    state_judger_model_path=None,
    denstream_params=denstream_params,
    device="auto"
)
```

## 注意事项
- 所有模型评估必须使用相同的参数
- 更改任何参数都会影响横向比较的有效性
- 建议在评估开始前运行verify_scoring_consistency.py进行验证
