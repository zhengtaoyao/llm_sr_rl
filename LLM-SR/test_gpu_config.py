#!/usr/bin/env python3
"""
GPU配置测试脚本
验证是否能正确使用指定的GPU卡
"""

import os
import torch

def test_gpu_config():
    """测试GPU配置"""
    print("🔍 GPU配置测试")
    print("=" * 50)
    
    # 检查CUDA_VISIBLE_DEVICES环境变量
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    print(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")
    
    # 检查PyTorch是否可用CUDA
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        
        # 列出所有可用的GPU
        for i in range(torch.cuda.device_count()):
            device = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {device.name} ({device.total_memory / 1024**3:.1f} GB)")
        
        # 测试GPU内存
        print("\n🧪 测试GPU内存使用:")
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            cached = torch.cuda.memory_reserved(i) / 1024**3
            print(f"GPU {i}: {allocated:.1f} GB allocated, {cached:.1f} GB cached")
    
    print("✅ GPU配置测试完成")

if __name__ == "__main__":
    test_gpu_config() 