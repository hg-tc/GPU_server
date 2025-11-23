#!/usr/bin/env bash
# GPU 服务器日志查看脚本

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$ROOT_DIR/logs"

case "${1:-all}" in
  all|main)
    echo "=== 查看主日志 (按 Ctrl+C 退出) ==="
    tail -f "$LOG_DIR/gpu_server.log"
    ;;
  error|errors)
    echo "=== 查看错误日志 (按 Ctrl+C 退出) ==="
    tail -f "$LOG_DIR/gpu_server_error.log"
    ;;
  last|tail)
    lines="${2:-50}"
    echo "=== 最后 $lines 行日志 ==="
    tail -n "$lines" "$LOG_DIR/gpu_server.log"
    ;;
  grep|search)
    if [ -z "$2" ]; then
      echo "用法: $0 grep <搜索关键词>"
      exit 1
    fi
    echo "=== 搜索日志: $2 ==="
    grep -i "$2" "$LOG_DIR/gpu_server.log" | tail -50
    ;;
  pdf)
    echo "=== PDF转换任务日志 ==="
    grep "\[PDF转换任务\]" "$LOG_DIR/gpu_server.log" | tail -50
    ;;
  embed)
    echo "=== 嵌入任务日志 ==="
    grep "\[嵌入任务\]" "$LOG_DIR/gpu_server.log" | tail -50
    ;;
  rerank)
    echo "=== 重排序任务日志 ==="
    grep "\[重排序任务\]" "$LOG_DIR/gpu_server.log" | tail -50
    ;;
  request|requests)
    echo "=== 请求日志 ==="
    grep "\[请求" "$LOG_DIR/gpu_server.log" | tail -50
    ;;
  stats)
    echo "=== 日志统计 ==="
    echo "主日志文件大小: $(du -h "$LOG_DIR/gpu_server.log" | cut -f1)"
    echo "错误日志文件大小: $(du -h "$LOG_DIR/gpu_server_error.log" | cut -f1)"
    echo ""
    echo "最近24小时统计:"
    echo "  - 总请求数: $(grep -c "\[请求完成\]" "$LOG_DIR/gpu_server.log" 2>/dev/null || echo 0)"
    echo "  - PDF转换任务: $(grep -c "\[PDF转换任务\].*任务完成" "$LOG_DIR/gpu_server.log" 2>/dev/null || echo 0)"
    echo "  - 嵌入任务: $(grep -c "\[嵌入任务\].*任务完成" "$LOG_DIR/gpu_server.log" 2>/dev/null || echo 0)"
    echo "  - 重排序任务: $(grep -c "\[重排序任务\].*任务完成" "$LOG_DIR/gpu_server.log" 2>/dev/null || echo 0)"
    echo "  - 错误数: $(grep -c "\[.*\].*❌" "$LOG_DIR/gpu_server.log" 2>/dev/null || echo 0)"
    ;;
  *)
    echo "GPU 服务器日志查看工具"
    echo ""
    echo "用法: $0 {命令} [参数]"
    echo ""
    echo "命令:"
    echo "  all, main        - 实时查看主日志（默认）"
    echo "  error, errors    - 实时查看错误日志"
    echo "  last, tail [N]   - 查看最后 N 行日志（默认50行）"
    echo "  grep <关键词>    - 搜索日志内容"
    echo "  pdf              - 查看PDF转换任务日志"
    echo "  embed            - 查看嵌入任务日志"
    echo "  rerank           - 查看重排序任务日志"
    echo "  request          - 查看请求日志"
    echo "  stats            - 查看日志统计信息"
    echo ""
    echo "示例:"
    echo "  $0                # 实时查看所有日志"
    echo "  $0 last 100        # 查看最后100行"
    echo "  $0 grep 'PDF转换'  # 搜索包含'PDF转换'的日志"
    echo "  $0 pdf             # 查看PDF转换任务"
    echo "  $0 stats           # 查看统计信息"
    exit 1
    ;;
esac

