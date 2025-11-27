#!/usr/bin/env bash
# ================================================
# GPU Server 日志查看脚本
# ================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$ROOT_DIR/logs"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 检查日志目录
if [ ! -d "$LOG_DIR" ]; then
    echo -e "${RED}日志目录不存在: $LOG_DIR${NC}"
    exit 1
fi

case "${1:-help}" in
    all|main|f)
        echo -e "${GREEN}=== 实时查看主日志 (Ctrl+C 退出) ===${NC}"
        tail -f "$LOG_DIR/gpu_server.log"
        ;;
    
    error|err|e)
        echo -e "${RED}=== 实时查看错误日志 (Ctrl+C 退出) ===${NC}"
        tail -f "$LOG_DIR/gpu_server_error.log"
        ;;
    
    last|tail|l)
        lines="${2:-50}"
        echo -e "${BLUE}=== 最后 $lines 行日志 ===${NC}"
        tail -n "$lines" "$LOG_DIR/gpu_server.log"
        ;;
    
    grep|search|g)
        if [ -z "$2" ]; then
            echo "用法: $0 grep <关键词>"
            exit 1
        fi
        echo -e "${BLUE}=== 搜索: $2 ===${NC}"
        grep -i --color=auto "$2" "$LOG_DIR/gpu_server.log" | tail -100
        ;;
    
    pdf)
        echo -e "${BLUE}=== PDF 处理日志 ===${NC}"
        grep -E "\[PDF\]" "$LOG_DIR/gpu_server.log" | tail -50
        ;;
    
    ocr)
        echo -e "${BLUE}=== OCR 处理日志 ===${NC}"
        grep -E "\[OCR\]" "$LOG_DIR/gpu_server.log" | tail -50
        ;;
    
    structure)
        echo -e "${BLUE}=== 版面分析日志 ===${NC}"
        grep -E "\[Structure\]" "$LOG_DIR/gpu_server.log" | tail -50
        ;;
    
    embed)
        echo -e "${BLUE}=== 嵌入任务日志 ===${NC}"
        grep -E "\[Embed\]" "$LOG_DIR/gpu_server.log" | tail -50
        ;;
    
    rerank)
        echo -e "${BLUE}=== 重排序任务日志 ===${NC}"
        grep -E "\[Rerank\]" "$LOG_DIR/gpu_server.log" | tail -50
        ;;
    
    request|req)
        echo -e "${BLUE}=== 请求日志 ===${NC}"
        grep -E "\[请求" "$LOG_DIR/gpu_server.log" | tail -50
        ;;
    
    success|ok)
        echo -e "${GREEN}=== 成功日志 ===${NC}"
        grep -E "✅" "$LOG_DIR/gpu_server.log" | tail -50
        ;;
    
    fail|failed)
        echo -e "${RED}=== 失败日志 ===${NC}"
        grep -E "❌|失败|错误|ERROR" "$LOG_DIR/gpu_server.log" | tail -50
        ;;
    
    stats|stat|s)
        echo -e "${GREEN}=== 日志统计 ===${NC}"
        echo ""
        
        if [ -f "$LOG_DIR/gpu_server.log" ]; then
            echo "主日志: $(du -h "$LOG_DIR/gpu_server.log" | cut -f1)"
        else
            echo "主日志: 不存在"
        fi
        
        if [ -f "$LOG_DIR/gpu_server_error.log" ]; then
            echo "错误日志: $(du -h "$LOG_DIR/gpu_server_error.log" | cut -f1)"
        else
            echo "错误日志: 不存在"
        fi
        
        echo ""
        echo -e "${BLUE}请求统计:${NC}"
        echo "  总请求: $(grep -c "\[请求完成\]" "$LOG_DIR/gpu_server.log" 2>/dev/null || echo 0)"
        echo "  成功: $(grep -c "✅" "$LOG_DIR/gpu_server.log" 2>/dev/null || echo 0)"
        echo "  失败: $(grep -c "❌" "$LOG_DIR/gpu_server.log" 2>/dev/null || echo 0)"
        
        echo ""
        echo -e "${BLUE}任务统计:${NC}"
        echo "  PDF: $(grep -c "\[PDF\].*✅" "$LOG_DIR/gpu_server.log" 2>/dev/null || echo 0)"
        echo "  OCR: $(grep -c "\[OCR\].*✅" "$LOG_DIR/gpu_server.log" 2>/dev/null || echo 0)"
        echo "  Structure: $(grep -c "\[Structure\].*✅" "$LOG_DIR/gpu_server.log" 2>/dev/null || echo 0)"
        echo "  Embed: $(grep -c "\[Embed\].*✅" "$LOG_DIR/gpu_server.log" 2>/dev/null || echo 0)"
        echo "  Rerank: $(grep -c "\[Rerank\].*✅" "$LOG_DIR/gpu_server.log" 2>/dev/null || echo 0)"
        ;;
    
    clear|clean)
        echo -e "${YELLOW}清空日志文件...${NC}"
        > "$LOG_DIR/gpu_server.log"
        > "$LOG_DIR/gpu_server_error.log"
        echo -e "${GREEN}日志已清空${NC}"
        ;;
    
    *)
        echo -e "${GREEN}GPU Server 日志查看工具${NC}"
        echo ""
        echo "用法: $0 <命令> [参数]"
        echo ""
        echo -e "${BLUE}实时查看:${NC}"
        echo "  all, main, f     实时查看主日志"
        echo "  error, err, e    实时查看错误日志"
        echo ""
        echo -e "${BLUE}历史查看:${NC}"
        echo "  last, l [N]      查看最后 N 行（默认50）"
        echo "  grep, g <词>     搜索日志"
        echo ""
        echo -e "${BLUE}按类型查看:${NC}"
        echo "  pdf              PDF 处理日志"
        echo "  ocr              OCR 处理日志"
        echo "  structure        版面分析日志"
        echo "  embed            嵌入任务日志"
        echo "  rerank           重排序任务日志"
        echo "  request          请求日志"
        echo "  success          成功日志"
        echo "  fail             失败日志"
        echo ""
        echo -e "${BLUE}其他:${NC}"
        echo "  stats, s         统计信息"
        echo "  clear            清空日志"
        echo ""
        echo -e "${BLUE}示例:${NC}"
        echo "  $0 f              # 实时查看日志"
        echo "  $0 l 100          # 最后100行"
        echo "  $0 g 'error'      # 搜索 error"
        echo "  $0 pdf            # PDF 处理日志"
        echo "  $0 s              # 统计信息"
        ;;
esac
