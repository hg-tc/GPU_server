#!/usr/bin/env python3
"""验证离线模式是否生效的脚本"""
import os
import sys
import subprocess
import re

print("=" * 60)
print("离线模式验证")
print("=" * 60)

# 检查服务进程的环境变量
def check_service_env():
    """检查运行中的服务进程的环境变量"""
    print("\n1. 服务进程环境变量检查:")
    try:
        # 查找服务进程
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True
        )
        
        # 查找uvicorn进程
        uvicorn_pids = []
        for line in result.stdout.split('\n'):
            if 'uvicorn main:app' in line and 'grep' not in line:
                parts = line.split()
                if len(parts) > 1:
                    uvicorn_pids.append(parts[1])
        
        if not uvicorn_pids:
            print("  ⚠️  未找到运行中的服务进程")
            return False
        
        # 检查第一个进程的环境变量
        pid = uvicorn_pids[0]
        try:
            env_result = subprocess.run(
                ["cat", f"/proc/{pid}/environ"],
                capture_output=True,
                text=True
            )
            
            if env_result.returncode == 0:
                env_vars = {}
                for item in env_result.stdout.split('\0'):
                    if '=' in item:
                        key, value = item.split('=', 1)
                        env_vars[key] = value
                
                # 检查关键环境变量
                key_vars = [
                    "HF_HUB_OFFLINE",
                    "TRANSFORMERS_OFFLINE",
                    "HF_DATASETS_OFFLINE",
                ]
                
                all_set = True
                for var in key_vars:
                    value = env_vars.get(var, "未设置")
                    status = "✅" if value == "1" else "❌"
                    print(f"  {status} {var}: {value}")
                    if value != "1":
                        all_set = False
                
                return all_set
            else:
                print(f"  ⚠️  无法读取进程 {pid} 的环境变量（可能需要root权限）")
                return False
        except Exception as e:
            print(f"  ⚠️  检查进程环境变量失败: {e}")
            return False
    except Exception as e:
        print(f"  ⚠️  查找服务进程失败: {e}")
        return False

# 检查当前shell的环境变量（仅作参考）
def check_shell_env():
    """检查当前shell的环境变量（仅作参考）"""
    print("\n2. 当前Shell环境变量检查（仅作参考）:")
    print("   ℹ️  这些变量只在服务进程中设置，当前shell中未设置是正常的")
    
    env_vars = [
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_DATASETS_OFFLINE",
    ]
    
    for var in env_vars:
        value = os.getenv(var, "未设置")
        status = "✅" if value in ("1", "true", "True") else "ℹ️"
        print(f"  {status} {var}: {value}")

# 检查服务日志
def check_service_logs():
    """检查服务日志中的离线模式信息"""
    print("\n3. 服务日志检查:")
    log_file = "/root/GPU_server/logs/gpu_server.log"
    
    if not os.path.exists(log_file):
        print("  ⚠️  日志文件不存在")
        return False
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            log_content = f.read()
        
        # 检查离线模式相关日志
        offline_patterns = [
            r"离线模式.*启用",
            r"OFFLINE.*enabled",
            r"离线模式.*尝试从本地缓存",
        ]
        
        found_offline = False
        for pattern in offline_patterns:
            if re.search(pattern, log_content, re.IGNORECASE):
                found_offline = True
                matches = re.findall(pattern, log_content, re.IGNORECASE)
                print(f"  ✅ 找到离线模式日志: {matches[0] if matches else '已启用'}")
                break
        
        if not found_offline:
            print("  ⚠️  未在日志中找到离线模式相关信息")
        
        # 检查是否有retry日志
        retry_count = len(re.findall(r'retry|Retry|RETRY', log_content, re.IGNORECASE))
        if retry_count == 0:
            print("  ✅ 无retry日志（离线模式正常工作）")
        else:
            print(f"  ⚠️  发现 {retry_count} 条retry相关日志")
        
        # 检查网络请求日志
        network_patterns = [
            r'urllib3.*hf-mirror',
            r'HEAD.*hf-mirror',
            r'GET.*hf-mirror',
        ]
        
        network_requests = 0
        for pattern in network_patterns:
            network_requests += len(re.findall(pattern, log_content, re.IGNORECASE))
        
        if network_requests == 0:
            print("  ✅ 无网络请求日志（完全离线）")
        else:
            print(f"  ⚠️  发现 {network_requests} 条网络请求日志")
        
        return found_offline and retry_count == 0
    except Exception as e:
        print(f"  ❌ 读取日志文件失败: {e}")
        return False

# 检查服务配置
def check_service_config():
    """检查服务配置文件"""
    print("\n4. 服务配置文件检查:")
    config_file = "/root/GPU_server/gpu-server.service"
    
    if not os.path.exists(config_file):
        print("  ⚠️  服务配置文件不存在")
        return False
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config_content = f.read()
        
        key_vars = [
            "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE",
            "HF_DATASETS_OFFLINE",
        ]
        
        all_configured = True
        for var in key_vars:
            pattern = rf'Environment="{var}=(\w+)"'
            match = re.search(pattern, config_content)
            if match:
                value = match.group(1)
                status = "✅" if value == "1" else "⚠️"
                print(f"  {status} {var}={value}")
                if value != "1":
                    all_configured = False
            else:
                print(f"  ❌ {var}: 未配置")
                all_configured = False
        
        return all_configured
    except Exception as e:
        print(f"  ❌ 读取配置文件失败: {e}")
        return False

# 检查模型文件
def check_model_files():
    """检查模型文件是否存在"""
    print("\n5. 模型文件检查:")
    models = [
        "BAAI/bge-large-zh-v1.5",
        "BAAI/bge-reranker-v2-m3",
    ]
    
    cache_base = os.path.expanduser("~/.cache/huggingface/hub")
    all_exist = True
    
    for model in models:
        model_dir = cache_base + "/models--" + model.replace("/", "--")
        if os.path.exists(model_dir):
            # 检查是否有模型文件
            result = subprocess.run(
                ["find", model_dir, "-name", "*.safetensors", "-o", "-name", "*.bin"],
                capture_output=True,
                text=True
            )
            file_count = len(result.stdout.strip().split("\n")) if result.stdout.strip() else 0
            status = "✅" if file_count > 0 else "⚠️"
            print(f"  {status} {model}: 找到 {file_count} 个模型文件")
            if file_count == 0:
                all_exist = False
        else:
            print(f"  ❌ {model}: 缓存目录不存在")
            all_exist = False
    
    return all_exist

# 主验证流程
def main():
    results = {
        "service_env": check_service_env(),
        "service_config": check_service_config(),
        "service_logs": check_service_logs(),
        "model_files": check_model_files(),
    }
    
    check_shell_env()  # 仅作参考，不参与结果判断
    
    # 总结
    print("\n" + "=" * 60)
    print("验证总结")
    print("=" * 60)
    
    all_pass = all(results.values())
    
    if all_pass:
        print("\n✅ 离线模式配置正确且已生效！")
        print("\n验证结果:")
        print("  ✅ 服务进程环境变量: 已设置")
        print("  ✅ 服务配置文件: 已配置")
        print("  ✅ 服务日志: 离线模式已启用，无retry")
        print("  ✅ 模型文件: 完整存在")
    else:
        print("\n⚠️  部分验证未通过，请检查以下项目:")
        for key, value in results.items():
            status = "✅" if value else "❌"
            name_map = {
                "service_env": "服务进程环境变量",
                "service_config": "服务配置文件",
                "service_logs": "服务日志",
                "model_files": "模型文件",
            }
            print(f"  {status} {name_map.get(key, key)}")
    
    print("\n说明:")
    print("  - 当前Shell中的环境变量未设置是正常的")
    print("  - 环境变量只在服务进程中生效")
    print("  - 查看服务日志可以确认离线模式是否真正生效")
    print("\n如果看到 retry 日志，可能是:")
    print("  1. 某些库的内部重试机制（无法完全禁用）")
    print("  2. 模型文件不完整，库尝试从网络获取")
    print("  3. 首次加载时的缓存验证")
    print("\n建议:")
    print("  - 确保模型文件完整下载")
    print("  - 检查日志中的错误信息")
    print("  - 如果模型加载成功，retry 不影响功能")

if __name__ == "__main__":
    main()

