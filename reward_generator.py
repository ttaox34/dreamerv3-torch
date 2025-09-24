import numpy as np
from cluster import DenStreamClusterer  # 导入DenStream聚类器

class RewardGenerator:
    def __init__(self, denstream_params=None):
        """初始化奖励生成器"""
        # 初始化DenStream聚类器
        if denstream_params is None:
            denstream_params = {'lambda_': 0.01, 'eps': 5, 'mu': 5}
        self.clusterer = DenStreamClusterer(** denstream_params)
        
        # 缓存当前簇心
        self.cached_centers = []
        # 存储历史最小距离，用于计算均值
        self.history_min_distances = []

    def generate_reward(self, vector):
        """
        处理输入向量，返回奖励值
        奖励计算逻辑：当前最小距离 - 历史最小距离的均值
        """
        vector = np.array(vector)
        current_min_distance = 0.0
        
        # 1. 计算与历史簇心的最小欧式距离
        if self.cached_centers:
            distances = [np.linalg.norm(vector - center) for center in self.cached_centers]
            current_min_distance = min(distances)

        # 2. 更新历史最小距离记录
        self.history_min_distances.append(current_min_distance)
        
        # 3. 计算历史最小距离的均值
        if len(self.history_min_distances) > 1:
            # 排除当前值计算历史均值（避免自身影响）
            history_mean = np.mean(self.history_min_distances[:-1])
            history_std =np.std(self.history_min_distances[:-1])

        else:
            # 若只有一个历史值，均值为0
            history_mean = 0.0
            history_std =1.0
        if history_std == 0.0:
            history_std =1.0
        # 4. 计算奖励：当前最小距离 - 历史最小距离的均值
        reward =((current_min_distance - history_mean)/history_std)+1
        if reward<0.75:
            reward=0.75
        reward = np.log(reward)
        
        # 5. 更新聚类和簇心
        _, new_centers = self.clusterer.process_vector(vector)
        self.cached_centers = new_centers

        return reward


# 示例用法
if __name__ == "__main__":
    # 生成示例数据（3个聚类的5维向量）
    np.random.seed(42)
    dim = 5
    n_samples = 100
    
    cluster1 = np.random.normal(loc=[1]*dim, scale=0.3, size=(n_samples, dim))
    cluster2 = np.random.normal(loc=[5]*dim, scale=0.3, size=(n_samples, dim))
    cluster3 = np.random.normal(loc=[9]*dim, scale=0.3, size=(n_samples, dim))
    data_stream = np.vstack([cluster1, cluster2, cluster3])
    
    vector_zero = np.array([0.0,0.0,0.0,0.0,0.0])
    
    # 初始化奖励生成器
    reward_generator = RewardGenerator()
    
    # 处理数据流并输出结果
    for i, vec in enumerate(data_stream):
        reward = reward_generator.generate_reward(vec)
        
        # 每20个向量输出一次状态
        if (i + 1) % 20 == 0:
            print(f"处理第{i+1}个向量:")
            print(f"  当前最小距离: {reward_generator.history_min_distances[-1]:.4f}")
            print(f"  历史最小距离均值: {np.mean(reward_generator.history_min_distances[:-1]):.4f}")
            print(f"  奖励值: {reward:.4f}")
            print(f"  当前簇心数量: {len(reward_generator.cached_centers)}")
            print("-" * 60)
    
    # 测试零向量
    zero_reward = reward_generator.generate_reward(vector_zero)
    print(f"零向量的奖励为: {zero_reward:.4f}")
    print(f"零向量的最小距离: {reward_generator.history_min_distances[-1]:.4f}")
    print(f"历史最小距离均值: {np.mean(reward_generator.history_min_distances[:-1]):.4f}")
    