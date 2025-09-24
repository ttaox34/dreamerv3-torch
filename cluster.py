import numpy as np
import math
from collections import defaultdict

"""
Use like this:
    clusterer = DenStreamClusterer(
        lambda_=0.01,  # 减小衰减因子，减缓权重衰减
        eps=2.0,       # 增大空间半径，允许同一聚类的向量合并
        mu=5           # 降低核心微簇的权重阈值
    )
    
    cluster_count(返回簇心的个数), centers(返回簇心的平均向量) = clusterer.process_vector(vec)
    
"""
class MicroCluster:
    """微簇类，存储聚类的核心统计信息"""
    def __init__(self, lambda_, creation_time):
        self.lambda_ = lambda_  # 时间衰减因子
        self.creation_time = creation_time  # 创建时间
        self.last_updated = creation_time  # 最后更新时间
        self.n = 0  # 加权点数
        self.linear_sum = None  # 向量加权和
        self.squared_sum = 0.0  # 向量平方和（用于计算半径）
        
    def update(self, point, current_time):
        """用新向量更新微簇"""
        point = np.array(point)
        if self.linear_sum is None:
            self.linear_sum = np.zeros_like(point)
            
        # 计算时间衰减因子
        delta_t = current_time - self.last_updated
        weight = math.exp(-self.lambda_ * delta_t)
        
        # 更新统计信息
        self.n = self.n * weight + 1
        self.linear_sum = self.linear_sum * weight + point
        self.squared_sum = self.squared_sum * weight + np.dot(point, point)
        self.last_updated = current_time
        
    def get_center(self):
        """获取聚类中心（平均向量）"""
        if self.n == 0 or self.linear_sum is None:
            return None
        return self.linear_sum / self.n
    
    def get_weight(self, current_time):
        """计算当前时间的微簇权重"""
        delta_t = current_time - self.last_updated
        return self.n * math.exp(-self.lambda_ * delta_t)


class DenStreamClusterer:
    """Denstream聚类器，提供简单接口处理单个向量并返回聚类结果"""
    def __init__(self, lambda_=0.1, eps=0.5, beta=0.5, mu=10):
        """
        初始化参数
        
        lambda_: 时间衰减因子，控制旧数据的权重衰减速度
        eps: 空间半径阈值，决定向量间的相似度阈值
        beta: 异常点微簇的阈值因子
        mu: 形成核心微簇所需的最小权重
        """
        self.lambda_ = lambda_
        self.eps = eps
        self.beta = beta
        self.mu = mu
        self.current_time = 0  # 当前时间步
        
        # 阈值计算
        self.theta_online = beta * mu  # 异常点微簇升级为核心微簇的阈值
        self.theta_offline = mu  # 离线聚类的最小权重阈值
        
        # 存储核心微簇和异常点微簇
        self.core_micro_clusters = []
        self.outlier_micro_clusters = []
    
    def process_vector(self, vector):
        """
        处理单个输入向量，更新聚类状态并返回结果
        
        参数:
            vector: 输入的高维向量（list或numpy数组）
            
        返回:
            tuple: (类别数量, 每个类别的平均向量列表)
        """
        self.current_time += 1  # 时间步递增
        vector = np.array(vector)

        # 1. 尝试加入核心微簇 (向量化优化)
        if self.core_micro_clusters:
            core_centers = np.array([cmc.get_center() for cmc in self.core_micro_clusters if cmc.get_center() is not None])
            if core_centers.size > 0:
                distances = np.linalg.norm(core_centers - vector, axis=1)
                min_dist_idx = np.argmin(distances)
                if distances[min_dist_idx] <= self.eps:
                    self.core_micro_clusters[min_dist_idx].update(vector, self.current_time)
                    return self._get_cluster_results()

        # 2. 尝试加入异常点微簇 (向量化优化)
        if self.outlier_micro_clusters:
            outlier_centers = np.array([omc.get_center() for omc in self.outlier_micro_clusters if omc.get_center() is not None])
            if outlier_centers.size > 0:
                distances = np.linalg.norm(outlier_centers - vector, axis=1)
                min_dist_idx = np.argmin(distances)
                if distances[min_dist_idx] <= self.eps:
                    closest_outlier = self.outlier_micro_clusters[min_dist_idx]
                    closest_outlier.update(vector, self.current_time)
                    # 检查是否可以升级为核心微簇
                    if closest_outlier.get_weight(self.current_time) >= self.theta_online:
                        self.outlier_micro_clusters.pop(min_dist_idx)
                        self.core_micro_clusters.append(closest_outlier)
                    return self._get_cluster_results()

        # 3. 创建新的异常点微簇
        new_omc = MicroCluster(self.lambda_, self.current_time)
        new_omc.update(vector, self.current_time)
        self.outlier_micro_clusters.append(new_omc)
        
        # 4. 清理权重过低的异常点微簇
        self._cleanup_outliers()
        
        return self._get_cluster_results()
    
    def _cleanup_outliers(self):
        """移除权重过低的异常点微簇"""
        self.outlier_micro_clusters = [
            omc for omc in self.outlier_micro_clusters
            if omc.get_weight(self.current_time) >= self.theta_online * 0.1
        ]
    
    def _get_cluster_results(self):
        """计算并返回当前的聚类结果：类别数量和平均向量"""
        # 过滤有效的核心微簇
        valid_cores = [
            cmc for cmc in self.core_micro_clusters
            if cmc.get_weight(self.current_time) >= self.theta_offline
        ]
        
        # 合并距离过近的核心微簇
        merged = []
        while valid_cores:
            cluster = valid_cores.pop(0)
            to_merge = []
            
            for i, other in enumerate(valid_cores):
                dist = np.linalg.norm(cluster.get_center() - other.get_center())
                if dist <= 2 * self.eps:  # 中心距离小于2*eps则合并
                    to_merge.append(i)
            
            # 合并选中的微簇
            for i in reversed(to_merge):
                other = valid_cores.pop(i)
                cluster = self._merge_micro_clusters(cluster, other)
            
            merged.append(cluster)
        
        # 提取平均向量（聚类中心）
        cluster_centers = [mc.get_center() for mc in merged if mc.get_center() is not None]
        return len(cluster_centers), cluster_centers
    
    def _merge_micro_clusters(self, mc1, mc2):
        """合并两个微簇"""
        merged = MicroCluster(self.lambda_, min(mc1.creation_time, mc2.creation_time))
        w1 = mc1.get_weight(self.current_time)
        w2 = mc2.get_weight(self.current_time)
        
        merged.n = w1 + w2
        merged.linear_sum = (mc1.linear_sum * math.exp(-self.lambda_ * (self.current_time - mc1.last_updated)) +
                            mc2.linear_sum * math.exp(-self.lambda_ * (self.current_time - mc2.last_updated)))
        merged.last_updated = self.current_time
        return merged
    
    def get_cluster_centers(self):
        """获取当前的聚类中心列表"""
        # 过滤有效的核心微簇
        valid_cores = [
            cmc for cmc in self.core_micro_clusters
            if cmc.get_weight(self.current_time) >= self.theta_offline
        ]
        
        # 合并距离过近的核心微簇
        merged = []
        while valid_cores:
            cluster = valid_cores.pop(0)
            to_merge = []
            
            for i, other in enumerate(valid_cores):
                dist = np.linalg.norm(cluster.get_center() - other.get_center())
                if dist <= 2 * self.eps:  # 中心距离小于2*eps则合并
                    to_merge.append(i)
            
            # 合并选中的微簇
            for i in reversed(to_merge):
                other = valid_cores.pop(i)
                cluster = self._merge_micro_clusters(cluster, other)
            
            merged.append(cluster)
        
        # 提取平均向量（聚类中心）
        cluster_centers = [mc.get_center() for mc in merged if mc.get_center() is not None]
        return cluster_centers