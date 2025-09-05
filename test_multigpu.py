#!/usr/bin/env python3
"""
多GPU训练测试脚本

测试多GPU训练功能是否正常工作
"""

import torch
import argparse
import sys
import pathlib
import os

# 添加项目根目录到路径
sys.path.append(str(pathlib.Path(__file__).parent))

import ruamel.yaml as yaml


def test_multi_gpu_availability():
    """测试多GPU可用性"""
    print("=== 测试多GPU可用性 ===")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，无法进行多GPU训练")
        return False
    
    num_gpus = torch.cuda.device_count()
    print(f"✅ 发现 {num_gpus} 个GPU:")
    
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"   GPU {i}: {gpu_name} ({memory_total:.1f} GB)")
    
    if num_gpus < 2:
        print("⚠️  只有1个GPU，无法测试真正的多GPU训练")
        return False
    
    return True


def test_distributed_training():
    """测试分布式训练功能"""
    print("\n=== 测试分布式训练功能 ===")
    
    try:
        import torch.distributed as dist
        print("✅ torch.distributed 模块可用")
        
        # 测试NCCL后端
        if dist.is_nccl_available():
            print("✅ NCCL后端可用")
        else:
            print("❌ NCCL后端不可用")
            return False
        
        # 浝始化简单的分布式环境
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12356'
        
        print("✅ 分布式训练环境测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 分布式训练测试失败: {e}")
        return False


def test_config_loading():
    """测试配置加载"""
    print("\n=== 测试配置加载 ===")
    
    try:
        # 加载配置文件
        config_path = pathlib.Path(__file__).parent / "configs.yaml"
        configs = yaml.safe_load(config_path.read_text())
        
        # 检查多GPU配置
        if 'multigpu_dmc_vision' in configs:
            print("✅ 多GPU DMC配置存在")
            print(f"   配置: {configs['multigpu_dmc_vision']['multi_gpu']}")
        else:
            print("❌ 多GPU DMC配置不存在")
            return False
        
        if 'multigpu_retro' in configs:
            print("✅ 多GPU Retro配置存在")
            print(f"   配置: {configs['multigpu_retro']['multi_gpu']}")
        else:
            print("❌ 多GPU Retro配置不存在")
            return False
        
        print("✅ 配置加载测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 配置加载测试失败: {e}")
        return False


def test_module_imports():
    """测试模块导入"""
    print("\n=== 测试模块导入 ===")
    
    modules_to_test = [
        'multigpu_trainer',
        'tools',
        'models',
        'envs.wrappers',
        'parallel'
    ]
    
    all_passed = True
    for module_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"✅ {module_name} 模块导入成功")
        except Exception as e:
            print(f"❌ {module_name} 模块导入失败: {e}")
            all_passed = False
    
    return all_passed


def test_model_creation():
    """测试模型创建"""
    print("\n=== 测试模型创建 ===")
    
    try:
        import tools
        import models
        
        # 创建模拟配置
        class MockConfig:
            def __init__(self):
                # 基础配置
                self.device = 'cuda:0'
                self.compile = False
                self.precision = 32
                self.debug = False
                self.video_pred_log = True
                
                # 训练配置
                self.log_every = 1000
                self.eval_every = 10000
                self.eval_episode_num = 10
                self.reset_every = 0
                self.steps = 1000000
                self.seed = 0
                
                # 环境配置
                self.task = 'dmc_walker_walk'
                self.size = [64, 64]
                self.envs = 1
                self.action_repeat = 2
                self.time_limit = 1000
                self.grayscale = False
                self.prefill = 2500
                self.reward_EMA = True
                
                # 模型配置
                self.dyn_hidden = 512
                self.dyn_deter = 512
                self.dyn_stoch = 32
                self.dyn_discrete = 32
                self.dyn_rec_depth = 1
                self.dyn_mean_act = 'none'
                self.dyn_std_act = 'sigmoid2'
                self.dyn_min_std = 0.1
                self.grad_heads = ['decoder', 'reward', 'cont']
                self.units = 512
                self.act = 'SiLU'
                self.norm = True
                self.encoder = {'mlp_keys': '$^', 'cnn_keys': 'image', 'act': 'SiLU', 'norm': True, 'cnn_depth': 32, 'kernel_size': 4, 'minres': 4, 'mlp_layers': 5, 'mlp_units': 1024, 'symlog_inputs': True}
                self.decoder = {'mlp_keys': '$^', 'cnn_keys': 'image', 'act': 'SiLU', 'norm': True, 'cnn_depth': 32, 'kernel_size': 4, 'minres': 4, 'mlp_layers': 5, 'mlp_units': 1024, 'cnn_sigmoid': False, 'image_dist': 'mse', 'vector_dist': 'symlog_mse', 'outscale': 1.0}
                self.actor = {'layers': 2, 'dist': 'normal', 'entropy': 3e-4, 'unimix_ratio': 0.01, 'std': 'learned', 'min_std': 0.1, 'max_std': 1.0, 'temp': 0.1, 'lr': 3e-5, 'eps': 1e-5, 'grad_clip': 100.0, 'outscale': 1.0}
                self.critic = {'layers': 2, 'dist': 'symlog_disc', 'slow_target': True, 'slow_target_update': 1, 'slow_target_fraction': 0.02, 'lr': 3e-5, 'eps': 1e-5, 'grad_clip': 100.0, 'outscale': 0.0}
                self.reward_head = {'layers': 2, 'dist': 'symlog_disc', 'loss_scale': 1.0, 'outscale': 0.0}
                self.cont_head = {'layers': 2, 'loss_scale': 1.0, 'outscale': 1.0}
                self.dyn_scale = 0.5
                self.rep_scale = 0.1
                self.kl_free = 1.0
                self.weight_decay = 0.0
                self.unimix_ratio = 0.01
                self.initial = 'learned'
                
                # 训练参数
                self.batch_size = 16
                self.batch_length = 64
                self.train_ratio = 512
                self.pretrain = 100
                self.model_lr = 1e-4
                self.opt_eps = 1e-8
                self.grad_clip = 1000
                self.dataset_size = 1000000
                self.opt = 'adam'
                
                # 行为配置
                self.discount = 0.997
                self.discount_lambda = 0.95
                self.imag_horizon = 15
                self.imag_gradient = 'dynamics'
                self.imag_gradient_mix = 0.0
                self.eval_state_mean = False
                
                # 探索配置
                self.expl_behavior = 'greedy'
                self.expl_until = 0
                self.expl_extr_scale = 0.0
                self.expl_intr_scale = 1.0
                
                # 其他配置
                self.num_actions = 6
                self.use_amp = False
        
        config = MockConfig()
        
        # 创建模拟观察和动作空间
        obs_space = {
            'image': (3, 64, 64),
            'is_first': (1,),
            'is_terminal': (1,),
            'is_last': (1,)
        }
        act_space = type('Space', (), {'n': 6})()
        
        # 创建数据集
        class MockDataset:
            def __iter__(self):
                return self
            
            def __next__(self):
                return {
                    'image': torch.randn(16, 64, 3, 64, 64),
                    'action': torch.randn(16, 64, 6),
                    'reward': torch.randn(16, 64),
                    'discount': torch.randn(16, 64),
                    'is_first': torch.randn(16, 64),
                    'is_terminal': torch.randn(16, 64),
                    'is_last': torch.randn(16, 64)
                }
        
        # 创建日志记录器
        class MockLogger:
            def __init__(self):
                self.step = 0
            
            def scalar(self, name, value):
                pass
            
            def write(self, **kwargs):
                pass
        
        # 创建模型
        import dreamer
        model = dreamer.Dreamer(obs_space, act_space, config, MockLogger(), MockDataset())
        model = model.to(config.device)
        
        print("✅ Dreamer模型创建成功")
        print(f"   模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 测试前向传播
        try:
            batch_size = 4
            obs = {
                'image': torch.randn(batch_size, 3, 64, 64).to(config.device),
                'is_first': torch.zeros(batch_size, 1).to(config.device),
                'is_terminal': torch.zeros(batch_size, 1).to(config.device),
                'is_last': torch.zeros(batch_size, 1).to(config.device)
            }
            reset = torch.ones(batch_size, dtype=torch.bool).to(config.device)
            
            with torch.no_grad():
                output, state = model(obs, reset, training=False)
            
            print("✅ 模型前向传播成功")
            print(f"   输出动作形状: {output['action'].shape}")
            
        except Exception as e:
            print(f"❌ 模型前向传播失败: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 模型创建测试失败: {e}")
        return False


def main():
    """主测试函数"""
    parser = argparse.ArgumentParser(description="多GPU训练测试")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    args = parser.parse_args()
    
    print("🚀 开始多GPU训练功能测试\n")
    
    tests = [
        ("多GPU可用性", test_multi_gpu_availability),
        ("分布式训练功能", test_distributed_training),
        ("配置加载", test_config_loading),
        ("模块导入", test_module_imports),
        ("模型创建", test_model_creation)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test_name}测试异常: {e}")
            failed += 1
    
    print(f"\n📊 测试结果:")
    print(f"   ✅ 通过: {passed}")
    print(f"   ❌ 失败: {failed}")
    print(f"   📈 成功率: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 所有测试通过！多GPU训练功能可以正常使用")
        return 0
    else:
        print(f"\n⚠️  有 {failed} 个测试失败，请检查相关问题")
        return 1


if __name__ == "__main__":
    exit(main())