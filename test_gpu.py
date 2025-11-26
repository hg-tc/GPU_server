#!/usr/bin/env python3
"""测试GPU使用情况的脚本"""
import requests
import time
import subprocess
import sys

def check_gpu_before():
    """检查调用前的GPU状态"""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", 
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        mem_used, gpu_util = result.stdout.strip().split(", ")
        return int(mem_used), int(gpu_util)
    return None, None

def test_embedding():
    """测试嵌入API"""
    print("=" * 60)
    print("测试嵌入API...")
    url = "http://localhost:18001/embed"
    
    mem_before, util_before = check_gpu_before()
    print(f"调用前 - GPU显存使用: {mem_before} MB, GPU利用率: {util_before}%")
    
    data = {
        "texts": [
            "这是一个测试文本",
            "This is a test text",
            "测试GPU加速是否正常工作"
        ]
    }
    
    start_time = time.time()
    try:
        response = requests.post(url, json=data, timeout=30)
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 嵌入成功 | 耗时: {elapsed:.3f}s | 返回向量数: {len(result.get('embeddings', []))}")
        else:
            print(f"❌ 嵌入失败 | 状态码: {response.status_code}")
    except Exception as e:
        print(f"❌ 请求失败: {e}")
        return
    
    time.sleep(2)  # 等待GPU操作完成
    mem_after, util_after = check_gpu_before()
    print(f"调用后 - GPU显存使用: {mem_after} MB, GPU利用率: {util_after}%")
    
    if mem_after and mem_before:
        mem_diff = mem_after - mem_before
        print(f"显存变化: {mem_diff:+d} MB")
        if mem_diff > 0:
            print("✅ 模型正在使用GPU显存")
        else:
            print("⚠️  未检测到显存增加，可能在使用CPU")

def test_rerank():
    """测试重排序API"""
    print("\n" + "=" * 60)
    print("测试重排序API...")
    url = "http://localhost:18001/rerank"
    
    mem_before, util_before = check_gpu_before()
    print(f"调用前 - GPU显存使用: {mem_before} MB, GPU利用率: {util_before}%")
    
    data = {
        "query": "什么是人工智能？",
        "documents": [
            "人工智能是计算机科学的一个分支",
            "机器学习是人工智能的核心技术",
            "深度学习是机器学习的一个子领域",
            "自然语言处理是AI的重要应用",
            "计算机视觉也是AI的应用领域"
        ]
    }
    
    start_time = time.time()
    try:
        response = requests.post(url, json=data, timeout=30)
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            scores = result.get('scores', [])
            print(f"✅ 重排序成功 | 耗时: {elapsed:.3f}s | 返回分数数: {len(scores)}")
            if scores:
                print(f"   分数范围: [{min(scores):.4f}, {max(scores):.4f}]")
        else:
            print(f"❌ 重排序失败 | 状态码: {response.status_code}")
    except Exception as e:
        print(f"❌ 请求失败: {e}")
        return
    
    time.sleep(2)  # 等待GPU操作完成
    mem_after, util_after = check_gpu_before()
    print(f"调用后 - GPU显存使用: {mem_after} MB, GPU利用率: {util_after}%")
    
    if mem_after and mem_before:
        mem_diff = mem_after - mem_before
        print(f"显存变化: {mem_diff:+d} MB")
        if mem_diff > 0:
            print("✅ 模型正在使用GPU显存")
        else:
            print("⚠️  未检测到显存增加，可能在使用CPU")

if __name__ == "__main__":
    print("GPU使用情况测试")
    print("=" * 60)
    
    # 检查服务是否运行
    try:
        response = requests.get("http://localhost:18001/health", timeout=5)
        if response.status_code != 200:
            print("❌ 服务未正常运行")
            sys.exit(1)
    except Exception as e:
        print(f"❌ 无法连接到服务: {e}")
        sys.exit(1)
    
    print("✅ 服务运行正常\n")
    
    # 测试嵌入
    test_embedding()
    
    # 测试重排序
    test_rerank()
    
    print("\n" + "=" * 60)
    print("测试完成！请查看日志文件了解详细信息：")
    print("  tail -f /root/GPU_server/logs/gpu_server.log | grep -E 'device|GPU|使用GPU|使用CPU'")


