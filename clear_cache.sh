#!/bin/bash

# HuggingFace 缓存清理脚本
# 用于清理损坏的模型缓存文件

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 获取 HuggingFace 缓存目录
HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
HF_CACHE_DIR="$HF_HOME/hub"

echo -e "${BLUE}=========================================="
echo "  HuggingFace 缓存清理工具"
echo "==========================================${NC}"
echo ""
echo "缓存目录: $HF_CACHE_DIR"
echo ""

# 检查缓存目录是否存在
if [ ! -d "$HF_CACHE_DIR" ]; then
    echo -e "${YELLOW}警告: 缓存目录不存在: $HF_CACHE_DIR${NC}"
    exit 0
fi

# 显示缓存大小
get_cache_size() {
    if [ -d "$1" ]; then
        du -sh "$1" 2>/dev/null | cut -f1
    else
        echo "0"
    fi
}

echo -e "${BLUE}当前缓存大小: $(get_cache_size "$HF_CACHE_DIR")${NC}"
echo ""

# 显示可用的模型
echo -e "${GREEN}可用的模型缓存:${NC}"
ls -1 "$HF_CACHE_DIR" 2>/dev/null | head -20
echo ""

# 函数：将模型名称转换为缓存目录名
convert_model_name() {
    local name=$1
    # 如果已经是缓存目录格式，直接返回
    if [[ "$name" == models--* ]]; then
        echo "$name"
    else
        # 将 / 替换为 --
        echo "models--${name//\//--}"
    fi
}

# 函数：查找模型缓存目录
find_model_cache() {
    local model_name=$1
    local cache_name=$(convert_model_name "$model_name")
    local model_dir="$HF_CACHE_DIR/$cache_name"
    
    # 如果直接找到，返回
    if [ -d "$model_dir" ]; then
        echo "$cache_name"
        return 0
    fi
    
    # 尝试模糊匹配
    local found=$(ls -1 "$HF_CACHE_DIR" 2>/dev/null | grep -i "${model_name//\//--}" | head -1)
    if [ -n "$found" ]; then
        echo "$found"
        return 0
    fi
    
    return 1
}

# 函数：清理特定模型
clear_model() {
    local model_name=$1
    local cache_name=$(find_model_cache "$model_name")
    
    if [ -z "$cache_name" ]; then
        echo -e "${YELLOW}模型缓存不存在: $model_name${NC}"
        echo -e "${YELLOW}尝试搜索类似的模型...${NC}"
        ls -1 "$HF_CACHE_DIR" 2>/dev/null | grep -i "${model_name//\//--}" || echo "未找到匹配的模型"
        return 1
    fi
    
    local model_dir="$HF_CACHE_DIR/$cache_name"
    local size=$(get_cache_size "$model_dir")
    echo -e "${YELLOW}正在清理: $cache_name (大小: $size)${NC}"
    rm -rf "$model_dir"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ 已清理: $cache_name${NC}"
        return 0
    else
        echo -e "${RED}❌ 清理失败: $cache_name${NC}"
        return 1
    fi
}

# 函数：清理所有缓存
clear_all() {
    local size=$(get_cache_size "$HF_CACHE_DIR")
    echo -e "${YELLOW}正在清理所有缓存 (大小: $size)${NC}"
    echo -e "${RED}警告: 这将删除所有 HuggingFace 模型缓存！${NC}"
    
    read -p "确认删除? (yes/no): " confirm
    if [ "$confirm" != "yes" ]; then
        echo "已取消"
        return 1
    fi
    
    rm -rf "$HF_CACHE_DIR"/*
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ 已清理所有缓存${NC}"
        return 0
    else
        echo -e "${RED}❌ 清理失败${NC}"
        return 1
    fi
}

# 主菜单
if [ $# -eq 0 ]; then
    echo "用法:"
    echo "  $0 <model_name>    清理特定模型"
    echo "  $0 --all           清理所有缓存"
    echo "  $0 --bge           清理 BGE 相关模型"
    echo "  $0 --list          列出所有模型"
    echo ""
    echo "示例:"
    echo "  $0 BAAI/bge-large-zh-v1.5"
    echo "  $0 models--BAAI--bge-large-zh-v1.5"
    echo "  $0 --bge"
    echo ""
    exit 0
fi

case "$1" in
    --all)
        clear_all
        ;;
    --bge)
        echo -e "${GREEN}清理 BGE 相关模型...${NC}"
        for model in "$HF_CACHE_DIR"/models--BAAI--bge-*; do
            if [ -d "$model" ]; then
                model_name=$(basename "$model")
                clear_model "$model_name"
            fi
        done
        ;;
    --list)
        echo -e "${GREEN}所有模型缓存:${NC}"
        ls -lh "$HF_CACHE_DIR" 2>/dev/null | tail -n +2 | awk '{printf "%-50s %s\n", $9, $5}'
        ;;
    *)
        clear_model "$1"
        ;;
esac

echo ""
echo -e "${BLUE}清理后缓存大小: $(get_cache_size "$HF_CACHE_DIR")${NC}"
echo ""
echo -e "${GREEN}提示: 清理后重新运行 ./setup.sh 重新下载模型${NC}"

