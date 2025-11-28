#!/usr/bin/env python3
"""
GPU Server 测试脚本
测试所有 API 端点
"""

import sys
import time
import json
import argparse
from pathlib import Path
from typing import Optional, Tuple

try:
    import httpx
except ImportError:
    print("请安装 httpx: pip install httpx")
    sys.exit(1)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def get_gpu_memory() -> Optional[Tuple[float, float]]:
    """获取 GPU 显存使用情况（GB）"""
    if not TORCH_AVAILABLE:
        return None
    try:
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            return (allocated, reserved)
    except:
        pass
    return None


def check_gpu_usage(before: Optional[Tuple[float, float]], after: Optional[Tuple[float, float]], operation: str):
    """检查 GPU 使用情况变化"""
    if before is None or after is None:
        if TORCH_AVAILABLE and torch.cuda.is_available():
            current = get_gpu_memory()
            if current:
                allocated, reserved = current
                print(f"  GPU 显存当前状态 ({operation}):")
                print(f"    已分配: {allocated:.2f} GB")
                print(f"    已保留: {reserved:.2f} GB")
                if allocated > 0.1:
                    print(f"    ✅ GPU 正在使用中")
        return
    
    allocated_before, reserved_before = before
    allocated_after, reserved_after = after
    
    allocated_diff = allocated_after - allocated_before
    reserved_diff = reserved_after - reserved_before
    
    if abs(allocated_diff) > 0.01 or abs(reserved_diff) > 0.01:
        print(f"  GPU 显存变化 ({operation}):")
        print(f"    已分配: {allocated_before:.2f} GB → {allocated_after:.2f} GB (变化: {allocated_diff:+.2f} GB)")
        print(f"    已保留: {reserved_before:.2f} GB → {reserved_after:.2f} GB (变化: {reserved_diff:+.2f} GB)")
        if allocated_diff > 0.1:
            print(f"    ✅ GPU 正在被使用")
        elif allocated_after > 0.1:
            print(f"    ✅ GPU 已在使用中（模型已加载）")
    else:
        if allocated_after > 0.1:
            print(f"  GPU 显存: 无明显变化，但显存已在使用 ({allocated_after:.2f} GB) - 模型可能已加载")
        else:
            print(f"  GPU 显存: 无明显变化 (可能使用 CPU 或模型未加载)")


def find_test_files(test_dir: Path = None) -> Tuple[Optional[Path], Optional[Path]]:
    """在 test 目录中查找测试文件"""
    if test_dir is None:
        test_dir = Path(__file__).parent / "test"
    
    if not test_dir.exists():
        return None, None
    
    # 查找图片文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']
    image_file = None
    for ext in image_extensions:
        images = list(test_dir.glob(f"*{ext}")) + list(test_dir.glob(f"*{ext.upper()}"))
        if images:
            image_file = images[0]
            break
    
    # 查找 PDF 文件
    pdf_files = list(test_dir.glob("*.pdf")) + list(test_dir.glob("*.PDF"))
    pdf_file = pdf_files[0] if pdf_files else None
    
    return image_file, pdf_file


def test_health(base_url: str) -> bool:
    """测试健康检查"""
    print("\n" + "=" * 50)
    print("🔍 测试健康检查 /health")
    print("=" * 50)
    
    try:
        resp = httpx.get(f"{base_url}/health", timeout=10)
        data = resp.json()
        
        print(f"状态码: {resp.status_code}")
        print(f"版本: {data.get('version', 'N/A')}")
        print(f"设备: {data.get('device', 'N/A')}")
        
        gpu = data.get('gpu', {})
        if gpu.get('available'):
            print(f"GPU: ✅ {gpu.get('device_name', 'N/A')}")
            mem = gpu.get('memory', {})
            allocated = mem.get('allocated_gb', 0)
            total = mem.get('total_gb', 0)
            print(f"  显存: {allocated:.2f} / {total:.2f} GB")
            if allocated > 0.1:
                print(f"  ✅ GPU 正在使用中")
        else:
            print("GPU: ❌ 不可用")
        
        models = data.get('models', {})
        print("模型状态:")
        for name, state in models.items():
            if state == "loaded":
                status = "✅ 已加载"
            elif state == "lazy":
                status = "⏳ 懒加载（首次使用时自动加载）"
            else:
                status = f"❓ {state}"
            print(f"  - {name}: {status}")
        
        return resp.status_code == 200
    except Exception as e:
        print(f"❌ 失败: {e}")
        return False


def test_embed(base_url: str) -> bool:
    """测试文本嵌入"""
    print("\n" + "=" * 50)
    print("🔍 测试文本嵌入 /embed")
    print("=" * 50)
    
    texts = ["这是一个测试文本", "Hello world", "人工智能"]
    
    try:
        gpu_before = get_gpu_memory()
        start = time.time()
        resp = httpx.post(
            f"{base_url}/embed",
            json={"texts": texts},
            timeout=300  # 增加超时时间，因为首次加载模型需要时间
        )
        elapsed = time.time() - start
        gpu_after = get_gpu_memory()
        
        print(f"状态码: {resp.status_code}")
        print(f"耗时: {elapsed:.3f}s")
        
        if resp.status_code == 200:
            data = resp.json()
            embeddings = data.get("embeddings", [])
            print(f"向量数量: {len(embeddings)}")
            if embeddings:
                print(f"向量维度: {len(embeddings[0])}")
            check_gpu_usage(gpu_before, gpu_after, "文本嵌入")
            return True
        else:
            print(f"❌ 错误: {resp.text}")
            return False
    except Exception as e:
        print(f"❌ 失败: {e}")
        return False


def test_rerank(base_url: str) -> bool:
    """测试文档重排序"""
    print("\n" + "=" * 50)
    print("🔍 测试文档重排序 /rerank")
    print("=" * 50)
    
    query = "什么是人工智能？"
    documents = [
        "人工智能是计算机科学的一个分支",
        "今天天气很好",
        "机器学习是人工智能的核心技术",
        "我喜欢吃苹果"
    ]
    
    try:
        gpu_before = get_gpu_memory()
        start = time.time()
        resp = httpx.post(
            f"{base_url}/rerank",
            json={"query": query, "documents": documents},
            timeout=300  # 增加超时时间，因为首次加载模型需要时间
        )
        elapsed = time.time() - start
        gpu_after = get_gpu_memory()
        
        print(f"状态码: {resp.status_code}")
        print(f"耗时: {elapsed:.3f}s")
        
        if resp.status_code == 200:
            data = resp.json()
            scores = data.get("scores", [])
            print(f"查询: {query}")
            print("排序结果:")
            
            # 按分数排序
            ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
            for i, (doc, score) in enumerate(ranked, 1):
                print(f"  {i}. [{score:.4f}] {doc[:40]}...")
            check_gpu_usage(gpu_before, gpu_after, "文档重排序")
            return True
        else:
            print(f"❌ 错误: {resp.text}")
            return False
    except Exception as e:
        print(f"❌ 失败: {e}")
        return False


def test_ocr(base_url: str, image_path: str = None) -> bool:
    """测试图片 OCR"""
    print("\n" + "=" * 50)
    print("🔍 测试图片 OCR /ocr_image")
    print("=" * 50)
    
    # 如果没有提供路径，尝试从 test 目录查找
    if not image_path:
        image_path, _ = find_test_files()
        if image_path:
            print(f"📁 自动使用测试文件: {image_path.name}")
        else:
            print("⏭️ 跳过（未找到测试图片，使用 --image 参数指定）")
            return True
    
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"❌ 图片不存在: {image_path}")
        return False
    
    try:
        gpu_before = get_gpu_memory()
        start = time.time()
        with open(image_path, 'rb') as f:
            files = {"file": (image_path.name, f)}
            resp = httpx.post(f"{base_url}/ocr_image", files=files, timeout=120)
        elapsed = time.time() - start
        gpu_after = get_gpu_memory()
        
        print(f"状态码: {resp.status_code}")
        print(f"耗时: {elapsed:.3f}s")
        print(f"文件: {image_path.name}")
        
        if resp.status_code == 200:
            data = resp.json()
            text = data.get("text", "")
            confidence = data.get("confidence", 0)
            lines = data.get("lines", [])
            
            print(f"置信度: {confidence:.4f}")
            print(f"识别行数: {len(lines)}")
            if text:
                print(f"文本预览: {text[:200]}..." if len(text) > 200 else f"文本: {text}")
            else:
                print("⚠️ 未识别到文本")
            check_gpu_usage(gpu_before, gpu_after, "图片 OCR")
            return True
        else:
            print(f"❌ 错误: {resp.text}")
            return False
    except Exception as e:
        print(f"❌ 失败: {e}")
        return False


def test_structure_image(base_url: str, image_path: str = None) -> bool:
    """测试图片版面分析"""
    print("\n" + "=" * 50)
    print("🔍 测试图片版面分析 /structure_image")
    print("=" * 50)
    
    # 如果没有提供路径，尝试从 test 目录查找
    if not image_path:
        image_path, _ = find_test_files()
        if image_path:
            print(f"📁 自动使用测试文件: {image_path.name}")
        else:
            print("⏭️ 跳过（未找到测试图片，使用 --image 参数指定）")
            return True
    
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"❌ 图片不存在: {image_path}")
        return False
    
    try:
        gpu_before = get_gpu_memory()
        start = time.time()
        with open(image_path, 'rb') as f:
            files = {"file": (image_path.name, f)}
            resp = httpx.post(f"{base_url}/structure_image", files=files, timeout=120)
        elapsed = time.time() - start
        gpu_after = get_gpu_memory()
        
        print(f"状态码: {resp.status_code}")
        print(f"耗时: {elapsed:.3f}s")
        print(f"文件: {image_path.name}")
        
        if resp.status_code == 200:
            data = resp.json()
            markdown = data.get("markdown", "")
            print(f"Markdown 长度: {len(markdown)} 字符")
            if markdown:
                print(f"内容预览:\n{markdown[:500]}..." if len(markdown) > 500 else f"内容:\n{markdown}")
            else:
                print("⚠️ 未生成 Markdown 内容")
            check_gpu_usage(gpu_before, gpu_after, "图片版面分析")
            return True
        else:
            print(f"❌ 错误: {resp.text}")
            return False
    except Exception as e:
        print(f"❌ 失败: {e}")
        return False


def test_pdf(base_url: str, pdf_path: str = None) -> bool:
    """测试 PDF 转 Markdown"""
    print("\n" + "=" * 50)
    print("🔍 测试 PDF 转 Markdown /pdf_to_markdown")
    print("=" * 50)
    
    # 如果没有提供路径，尝试从 test 目录查找
    if not pdf_path:
        _, pdf_path = find_test_files()
        if pdf_path:
            print(f"📁 自动使用测试文件: {pdf_path.name}")
        else:
            print("⏭️ 跳过（未找到测试 PDF，使用 --pdf 参数指定）")
            return True
    
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        print(f"❌ PDF 不存在: {pdf_path}")
        return False
    
    try:
        file_size = pdf_path.stat().st_size / 1024 / 1024  # MB
        print(f"文件大小: {file_size:.2f} MB")
        
        gpu_before = get_gpu_memory()
        start = time.time()
        with open(pdf_path, 'rb') as f:
            files = {"file": (pdf_path.name, f)}
            resp = httpx.post(f"{base_url}/pdf_to_markdown", files=files, timeout=300)
        elapsed = time.time() - start
        gpu_after = get_gpu_memory()
        
        print(f"状态码: {resp.status_code}")
        print(f"耗时: {elapsed:.3f}s")
        
        if resp.status_code == 200:
            data = resp.json()
            content = data.get("content", "")
            method = data.get("conversion_method", "N/A")
            
            print(f"转换方法: {method}")
            print(f"内容长度: {len(content)} 字符")
            if content:
                print(f"内容预览:\n{content[:500]}..." if len(content) > 500 else f"内容:\n{content}")
            else:
                print("⚠️ 未生成内容")
            check_gpu_usage(gpu_before, gpu_after, "PDF 转 Markdown")
            return True
        else:
            print(f"❌ 错误: {resp.text}")
            return False
    except Exception as e:
        print(f"❌ 失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="GPU Server 测试脚本")
    parser.add_argument("--url", default="http://localhost:8000", help="服务器地址")
    parser.add_argument("--image", help="测试图片路径")
    parser.add_argument("--pdf", help="测试 PDF 路径")
    parser.add_argument("--all", action="store_true", help="运行所有测试")
    parser.add_argument("--health", action="store_true", help="仅测试健康检查")
    parser.add_argument("--embed", action="store_true", help="仅测试嵌入")
    parser.add_argument("--rerank", action="store_true", help="仅测试重排序")
    parser.add_argument("--ocr", action="store_true", help="仅测试 OCR")
    parser.add_argument("--structure", action="store_true", help="仅测试版面分析")
    
    args = parser.parse_args()
    
    base_url = args.url.rstrip("/")
    print(f"🚀 GPU Server 测试")
    print(f"📍 服务器地址: {base_url}")
    
    # 检查测试文件
    found_test_image, found_test_pdf = find_test_files()
    if found_test_image or found_test_pdf:
        print(f"📁 测试文件目录: test/")
        if found_test_image:
            print(f"  图片: {found_test_image.name}")
        if found_test_pdf:
            print(f"  PDF: {found_test_pdf.name}")
    print()
    
    results = {}
    
    # 确定要运行的测试
    run_all = args.all or not any([args.health, args.embed, args.rerank, args.ocr, args.structure])
    
    if run_all or args.health:
        results["health"] = test_health(base_url)
    
    if run_all or args.embed:
        results["embed"] = test_embed(base_url)
    
    if run_all or args.rerank:
        results["rerank"] = test_rerank(base_url)
    
    if run_all or args.ocr:
        # 如果没有指定图片，使用自动找到的测试文件
        image_path = args.image if args.image else (str(found_test_image) if found_test_image else None)
        results["ocr"] = test_ocr(base_url, image_path)
    
    if run_all or args.structure:
        # 如果没有指定图片，使用自动找到的测试文件
        image_path = args.image if args.image else (str(found_test_image) if found_test_image else None)
        results["structure"] = test_structure_image(base_url, image_path)
    
    if args.pdf or run_all:
        # 如果没有指定 PDF，使用自动找到的测试文件
        pdf_path = args.pdf if args.pdf else (str(found_test_pdf) if found_test_pdf else None)
        results["pdf"] = test_pdf(base_url, pdf_path)
    
    # 汇总结果
    print("\n" + "=" * 50)
    print("📊 测试结果汇总")
    print("=" * 50)
    
    passed = 0
    failed = 0
    for name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\n总计: {passed} 通过, {failed} 失败")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
