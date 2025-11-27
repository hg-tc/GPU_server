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

try:
    import httpx
except ImportError:
    print("请安装 httpx: pip install httpx")
    sys.exit(1)


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
            print(f"  显存: {mem.get('allocated_gb', 0):.2f} / {mem.get('total_gb', 0):.2f} GB")
        else:
            print("GPU: ❌ 不可用")
        
        models = data.get('models', {})
        print("模型状态:")
        for name, loaded in models.items():
            status = "✅ 已加载" if loaded else "⏳ 未加载"
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
        start = time.time()
        resp = httpx.post(
            f"{base_url}/embed",
            json={"texts": texts},
            timeout=60
        )
        elapsed = time.time() - start
        
        print(f"状态码: {resp.status_code}")
        print(f"耗时: {elapsed:.3f}s")
        
        if resp.status_code == 200:
            data = resp.json()
            embeddings = data.get("embeddings", [])
            print(f"向量数量: {len(embeddings)}")
            if embeddings:
                print(f"向量维度: {len(embeddings[0])}")
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
        start = time.time()
        resp = httpx.post(
            f"{base_url}/rerank",
            json={"query": query, "documents": documents},
            timeout=60
        )
        elapsed = time.time() - start
        
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
    
    if not image_path:
        print("⏭️ 跳过（未提供图片路径，使用 --image 参数）")
        return True
    
    if not Path(image_path).exists():
        print(f"❌ 图片不存在: {image_path}")
        return False
    
    try:
        start = time.time()
        with open(image_path, 'rb') as f:
            files = {"file": (Path(image_path).name, f)}
            resp = httpx.post(f"{base_url}/ocr_image", files=files, timeout=120)
        elapsed = time.time() - start
        
        print(f"状态码: {resp.status_code}")
        print(f"耗时: {elapsed:.3f}s")
        
        if resp.status_code == 200:
            data = resp.json()
            text = data.get("text", "")
            confidence = data.get("confidence", 0)
            lines = data.get("lines", [])
            
            print(f"置信度: {confidence:.4f}")
            print(f"识别行数: {len(lines)}")
            print(f"文本预览: {text[:200]}..." if len(text) > 200 else f"文本: {text}")
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
    
    if not image_path:
        print("⏭️ 跳过（未提供图片路径，使用 --image 参数）")
        return True
    
    if not Path(image_path).exists():
        print(f"❌ 图片不存在: {image_path}")
        return False
    
    try:
        start = time.time()
        with open(image_path, 'rb') as f:
            files = {"file": (Path(image_path).name, f)}
            resp = httpx.post(f"{base_url}/structure_image", files=files, timeout=120)
        elapsed = time.time() - start
        
        print(f"状态码: {resp.status_code}")
        print(f"耗时: {elapsed:.3f}s")
        
        if resp.status_code == 200:
            data = resp.json()
            markdown = data.get("markdown", "")
            print(f"Markdown 长度: {len(markdown)}")
            print(f"内容预览:\n{markdown[:500]}..." if len(markdown) > 500 else f"内容:\n{markdown}")
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
    
    if not pdf_path:
        print("⏭️ 跳过（未提供 PDF 路径，使用 --pdf 参数）")
        return True
    
    if not Path(pdf_path).exists():
        print(f"❌ PDF 不存在: {pdf_path}")
        return False
    
    try:
        start = time.time()
        with open(pdf_path, 'rb') as f:
            files = {"file": (Path(pdf_path).name, f)}
            resp = httpx.post(f"{base_url}/pdf_to_markdown", files=files, timeout=300)
        elapsed = time.time() - start
        
        print(f"状态码: {resp.status_code}")
        print(f"耗时: {elapsed:.3f}s")
        
        if resp.status_code == 200:
            data = resp.json()
            content = data.get("content", "")
            method = data.get("conversion_method", "N/A")
            
            print(f"转换方法: {method}")
            print(f"内容长度: {len(content)}")
            print(f"内容预览:\n{content[:500]}..." if len(content) > 500 else f"内容:\n{content}")
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
        results["ocr"] = test_ocr(base_url, args.image)
    
    if run_all or args.structure:
        results["structure"] = test_structure_image(base_url, args.image)
    
    if args.pdf or run_all:
        results["pdf"] = test_pdf(base_url, args.pdf)
    
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
