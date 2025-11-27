#!/usr/bin/env bash
# GPU 模型服务启动脚本
# 支持 systemd 服务管理和手动启动两种方式

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT_DIR/venv"
SERVICE_NAME="gpu-server"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
NGINX_CONF_PATH="/etc/nginx/conf.d/gpu_server.conf"

# 启动 Nginx 的辅助函数
_start_nginx() {
  if [[ $EUID -ne 0 ]]; then
    return 0  # 非 root 用户跳过
  fi
  
  if ! command -v nginx >/dev/null 2>&1; then
    echo "[WARN] Nginx 未安装，跳过启动" >&2
    return 0
  fi
  
  # 检查 Nginx 配置
  if [[ -f "$NGINX_CONF_PATH" ]]; then
    if ! nginx -t >/dev/null 2>&1; then
      echo "[WARN] Nginx 配置检查失败，跳过启动" >&2
      return 0
    fi
  fi
  
  # 检查 Nginx 是否已运行
  if pgrep -x nginx >/dev/null 2>&1; then
    echo "[INFO] Nginx 已在运行中"
    # 重载配置以确保使用最新配置
    if command -v systemctl >/dev/null 2>&1; then
      systemctl reload nginx 2>/dev/null || nginx -s reload 2>/dev/null || true
    else
      nginx -s reload 2>/dev/null || true
    fi
    return 0
  fi
  
  # 启动 Nginx
  echo "[INFO] 启动 Nginx 反向代理..."
  
  # 方式1: 使用 systemctl（如果可用）
  if command -v systemctl >/dev/null 2>&1; then
    if systemctl start nginx 2>/dev/null; then
      echo "[SUCCESS] Nginx 已启动（systemctl）"
      return 0
    fi
  fi
  
  # 方式2: 使用 service 命令（SysV init）
  if command -v service >/dev/null 2>&1; then
    if service nginx start 2>/dev/null; then
      echo "[SUCCESS] Nginx 已启动（service）"
      return 0
    fi
  fi
  
  # 方式3: 直接启动 nginx 进程
  if command -v nginx >/dev/null 2>&1; then
    # 检查是否已有 nginx 进程
    if pgrep -x nginx >/dev/null 2>&1; then
      echo "[INFO] Nginx 进程已存在"
      return 0
    fi
    
    # 尝试启动
    if nginx 2>/dev/null; then
      echo "[SUCCESS] Nginx 已启动（直接启动）"
      return 0
    else
      # 检查错误日志
      if [[ -f /var/log/nginx/error.log ]]; then
        echo "[WARN] Nginx 启动失败，查看错误日志: /var/log/nginx/error.log" >&2
        tail -5 /var/log/nginx/error.log 2>/dev/null || true
      fi
      echo "[WARN] Nginx 启动失败，请手动检查配置: sudo nginx -t" >&2
      return 1
    fi
  else
    echo "[WARN] 未找到 nginx 命令" >&2
    return 1
  fi
}

# 停止 Nginx 的辅助函数（可选）
_stop_nginx() {
  if [[ $EUID -ne 0 ]]; then
    return 0
  fi
  
  if ! command -v nginx >/dev/null 2>&1; then
    return 0
  fi
  
  if ! pgrep -x nginx >/dev/null 2>&1; then
    return 0
  fi
  
  echo "[INFO] 停止 Nginx..."
  if command -v systemctl >/dev/null 2>&1; then
    systemctl stop nginx 2>/dev/null || true
  else
    nginx -s stop 2>/dev/null || pkill -x nginx 2>/dev/null || true
  fi
}

# 检查虚拟环境是否存在
if [[ ! -d "$VENV_DIR" ]]; then
  echo "[ERROR] 虚拟环境不存在: $VENV_DIR" >&2
  echo "请先运行安装脚本: sudo ./install_gpu_server.sh" >&2
  exit 1
fi

# 检查 uvicorn 是否安装
if [[ ! -f "$VENV_DIR/bin/uvicorn" ]]; then
  echo "[ERROR] uvicorn 未安装，请先运行安装脚本" >&2
  exit 1
fi

# 解析命令行参数
MODE="${1:-systemd}"
PORT="${GPU_SERVER_PORT_INTERNAL:-18001}"

case "$MODE" in
  systemd|service)
    # 使用 systemd 管理服务
    if [[ $EUID -ne 0 ]]; then
      echo "[ERROR] 使用 systemd 模式需要 root 权限，请使用: sudo $0 systemd" >&2
      exit 1
    fi

    # 检查 systemctl 是否可用
    if ! command -v systemctl >/dev/null 2>&1; then
      echo "[WARN] 系统中未找到 systemctl 命令，无法使用 systemd 模式" >&2
      echo "[INFO] 自动切换到后台启动模式..." >&2
      echo ""
      # 递归调用，使用 background 模式
      "$0" background
      exit $?
    fi

    echo "[INFO] 配置 systemd 服务..."

    # 创建临时服务文件并更新路径
    TEMP_SERVICE=$(mktemp)
    sed "s|WorkingDirectory=.*|WorkingDirectory=$ROOT_DIR|g" "$ROOT_DIR/gpu-server.service" | \
    sed "s|ExecStart=.*|ExecStart=$VENV_DIR/bin/uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1|g" | \
    sed "s|Environment=\"PATH=.*|Environment=\"PATH=$VENV_DIR/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\"|g" | \
    sed "s|ExecStartPost=.*|ExecStartPost=/bin/bash -c 'sleep 2 \&\& systemctl start nginx || true'|g" > "$TEMP_SERVICE"

    # 复制服务文件到 systemd 目录
    cp "$TEMP_SERVICE" "$SERVICE_FILE"
    rm -f "$TEMP_SERVICE"
    
    # 重载 systemd 配置
    if ! systemctl daemon-reload; then
      echo "[ERROR] systemd daemon-reload 失败" >&2
      exit 1
    fi
    
    # 启用服务（开机自启）
    systemctl enable "$SERVICE_NAME"
    
    # 启动服务
    echo "[INFO] 启动 systemd 服务: $SERVICE_NAME"
    if ! systemctl start "$SERVICE_NAME"; then
      echo "[ERROR] 服务启动失败，查看日志: journalctl -u $SERVICE_NAME -n 50" >&2
      exit 1
    fi
    
    # 等待一下，检查状态
    sleep 2
    if systemctl is-active --quiet "$SERVICE_NAME"; then
      echo "[SUCCESS] GPU 模型服务已启动"
      echo "[INFO] 查看服务状态: systemctl status $SERVICE_NAME"
      echo "[INFO] 查看服务日志: journalctl -u $SERVICE_NAME -f"
      echo "[INFO] 停止服务: systemctl stop $SERVICE_NAME"
      echo "[INFO] 重启服务: systemctl restart $SERVICE_NAME"
      
      # 启动 Nginx
      _start_nginx
    else
      echo "[ERROR] 服务启动失败，查看日志: journalctl -u $SERVICE_NAME -n 50" >&2
      exit 1
    fi
    ;;
    
  manual|foreground)
    # 手动前台启动（用于调试）
    echo "[INFO] 手动启动服务（前台模式）..."
    echo "[INFO] 工作目录: $ROOT_DIR"
    echo "[INFO] 端口: $PORT"
    echo "[INFO] 按 Ctrl+C 停止服务"
    echo ""
    
    cd "$ROOT_DIR"
    source "$VENV_DIR/bin/activate"
    
    # 设置环境变量
    # 离线模式配置（默认启用）
    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
    export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
    export HF_HUB_DISABLE_EXPERIMENTAL_WARNING="${HF_HUB_DISABLE_EXPERIMENTAL_WARNING:-1}"
    export HF_HUB_DISABLE_VERSION_CHECK="${HF_HUB_DISABLE_VERSION_CHECK:-1}"
    
    # 如果未启用离线模式，则设置镜像源
    if [[ "${HF_HUB_OFFLINE}" != "1" ]]; then
        export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
    fi
    
    exec uvicorn main:app --host 0.0.0.0 --port "$PORT" --workers 1
    ;;
    
  background|nohup)
    # 后台启动（使用 nohup）
    echo "[INFO] 后台启动服务..."
    echo "[INFO] 工作目录: $ROOT_DIR"
    echo "[INFO] 端口: $PORT"
    
    cd "$ROOT_DIR"
    source "$VENV_DIR/bin/activate"
    
    # 设置环境变量
    # 离线模式配置（默认启用）
    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
    export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
    export HF_HUB_DISABLE_EXPERIMENTAL_WARNING="${HF_HUB_DISABLE_EXPERIMENTAL_WARNING:-1}"
    export HF_HUB_DISABLE_VERSION_CHECK="${HF_HUB_DISABLE_VERSION_CHECK:-1}"
    
    # 如果未启用离线模式，则设置镜像源
    if [[ "${HF_HUB_OFFLINE}" != "1" ]]; then
        export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
    fi
    
    # 创建日志目录
    LOG_DIR="$ROOT_DIR/logs"
    mkdir -p "$LOG_DIR"
    
    # 使用 nohup 后台启动（添加超时参数以防止连接断开）
    nohup uvicorn main:app --host 0.0.0.0 --port "$PORT" --workers 1 \
      --timeout-keep-alive 300 --timeout-graceful-shutdown 30 \
      > "$LOG_DIR/gpu_server.log" 2>&1 &
    
    PID=$!
    echo "[SUCCESS] GPU 模型服务已在后台启动，PID: $PID"
    echo "[INFO] 日志文件: $LOG_DIR/gpu_server.log"
    echo "[INFO] 查看日志: tail -f $LOG_DIR/gpu_server.log"
    echo "[INFO] 停止服务: kill $PID"
    echo "$PID" > "$ROOT_DIR/gpu_server.pid"
    
    # 等待服务启动
    sleep 2
    
    # 启动 Nginx
    _start_nginx
    
    # 提示可以配置开机自启（如果没有 systemd）
    if [[ $EUID -eq 0 ]] && ! command -v systemctl >/dev/null 2>&1; then
      echo ""
      echo "[TIP] 要配置开机自启，请运行: sudo $0 enable-autostart"
    fi
    ;;
    
  enable-autostart)
    # 配置开机自启（无 systemd 环境）
    if [[ $EUID -ne 0 ]]; then
      echo "[ERROR] 配置开机自启需要 root 权限，请使用: sudo $0 enable-autostart" >&2
      exit 1
    fi
    
    echo "[INFO] 配置开机自启..."
    
    # 方案1: 使用 cron @reboot
    if command -v crontab >/dev/null 2>&1; then
      CRON_CMD="@reboot sleep 30 && $ROOT_DIR/start_gpu_server.sh background >/dev/null 2>&1"
      
      # 检查是否已存在
      EXISTING_CRON=$(crontab -l 2>/dev/null || echo "")
      if echo "$EXISTING_CRON" | grep -q "start_gpu_server.sh"; then
        echo "[INFO] cron 开机自启任务已存在，跳过添加"
      else
        # 确保正确添加（处理空 crontab 的情况）
        if [[ -z "$EXISTING_CRON" ]]; then
          echo "$CRON_CMD" | crontab -
        else
          (echo "$EXISTING_CRON"; echo "$CRON_CMD") | crontab -
        fi
        echo "[SUCCESS] 已添加 cron 开机自启任务"
        echo "[INFO] 当前 cron 任务:"
        crontab -l 2>/dev/null | grep "start_gpu_server.sh" || true
      fi
    fi
    
    # 方案2: 创建 init.d 脚本
    INIT_SCRIPT="/etc/init.d/gpu-server"
    if [[ -d /etc/init.d ]]; then
      cat > "$INIT_SCRIPT" <<EOF
#!/bin/bash
# GPU Model Server init.d script
# chkconfig: 2345 90 10
# description: GPU Model Server (PDF + Embedding + Rerank)

### BEGIN INIT INFO
# Provides:          gpu-server
# Required-Start:    \$network \$remote_fs
# Required-Stop:     \$network \$remote_fs
# Default-Start:     2 3 4 5
# Default-Stop:      0 1 6
# Short-Description: GPU Model Server
# Description:       GPU Model Server for PDF conversion, embedding and reranking
### END INIT INFO

ROOT_DIR="$ROOT_DIR"
START_SCRIPT="\$ROOT_DIR/start_gpu_server.sh"
PID_FILE="\$ROOT_DIR/gpu_server.pid"

case "\$1" in
  start)
    if [ -f "\$PID_FILE" ] && kill -0 "\$(cat \$PID_FILE)" 2>/dev/null; then
      echo "服务已在运行中"
      exit 0
    fi
    echo "启动 GPU 模型服务..."
    \$START_SCRIPT background
    ;;
  stop)
    echo "停止 GPU 模型服务..."
    \$START_SCRIPT stop
    ;;
  restart)
    \$0 stop
    sleep 2
    \$0 start
    ;;
  status)
    \$START_SCRIPT status
    ;;
  *)
    echo "用法: \$0 {start|stop|restart|status}"
    exit 1
    ;;
esac

exit 0
EOF
      chmod +x "$INIT_SCRIPT"
      echo "[SUCCESS] 已创建 init.d 脚本: $INIT_SCRIPT"
      
      # 尝试使用 chkconfig 或 update-rc.d 启用
      if command -v chkconfig >/dev/null 2>&1; then
        chkconfig --add gpu-server 2>/dev/null && echo "[SUCCESS] 已使用 chkconfig 启用开机自启" || echo "[INFO] chkconfig 配置可能需要手动检查"
      elif command -v update-rc.d >/dev/null 2>&1; then
        update-rc.d gpu-server defaults 2>/dev/null && echo "[SUCCESS] 已使用 update-rc.d 启用开机自启" || echo "[INFO] update-rc.d 配置可能需要手动检查"
      else
        echo "[INFO] 请手动创建符号链接以启用开机自启："
        echo "       ln -s $INIT_SCRIPT /etc/rc2.d/S90gpu-server"
        echo "       ln -s $INIT_SCRIPT /etc/rc3.d/S90gpu-server"
        echo "       ln -s $INIT_SCRIPT /etc/rc4.d/S90gpu-server"
        echo "       ln -s $INIT_SCRIPT /etc/rc5.d/S90gpu-server"
      fi
    fi
    
    echo ""
    echo "[完成] 开机自启配置完成"
    echo "[INFO] 可以使用以下命令管理服务："
    echo "       启动:   /etc/init.d/gpu-server start"
    echo "       停止:   /etc/init.d/gpu-server stop"
    echo "       重启:   /etc/init.d/gpu-server restart"
    echo "       状态:   /etc/init.d/gpu-server status"
    ;;
    
  disable-autostart)
    # 禁用开机自启
    if [[ $EUID -ne 0 ]]; then
      echo "[ERROR] 禁用开机自启需要 root 权限，请使用: sudo $0 disable-autostart" >&2
      exit 1
    fi
    
    echo "[INFO] 禁用开机自启..."
    
    # 移除 cron 任务
    if command -v crontab >/dev/null 2>&1; then
      crontab -l 2>/dev/null | grep -v "start_gpu_server.sh" | crontab - 2>/dev/null
      echo "[SUCCESS] 已移除 cron 开机自启任务"
    fi
    
    # 禁用 init.d 脚本
    INIT_SCRIPT="/etc/init.d/gpu-server"
    if [[ -f "$INIT_SCRIPT" ]]; then
      if command -v chkconfig >/dev/null 2>&1; then
        chkconfig --del gpu-server 2>/dev/null && echo "[SUCCESS] 已使用 chkconfig 禁用开机自启"
      elif command -v update-rc.d >/dev/null 2>&1; then
        update-rc.d -f gpu-server remove 2>/dev/null && echo "[SUCCESS] 已使用 update-rc.d 禁用开机自启"
      else
        echo "[INFO] 请手动删除符号链接以禁用开机自启"
      fi
    fi
    
    echo "[完成] 开机自启已禁用"
    ;;
    
  stop)
    # 停止服务
    if [[ $EUID -eq 0 ]] && command -v systemctl >/dev/null 2>&1 && systemctl is-active --quiet "$SERVICE_NAME" 2>/dev/null; then
      # 停止 systemd 服务
      systemctl stop "$SERVICE_NAME"
      echo "[SUCCESS] systemd 服务已停止"
    else
      # 尝试停止后台进程
      if [[ -f "$ROOT_DIR/gpu_server.pid" ]]; then
        PID=$(cat "$ROOT_DIR/gpu_server.pid")
        if kill -0 "$PID" 2>/dev/null; then
          kill "$PID"
          rm -f "$ROOT_DIR/gpu_server.pid"
          echo "[SUCCESS] 后台服务已停止"
        else
          echo "[WARN] PID 文件存在但进程不存在，清理 PID 文件"
          rm -f "$ROOT_DIR/gpu_server.pid"
        fi
      else
        echo "[INFO] 未找到 PID 文件，尝试查找进程..."
        if pkill -f "uvicorn main:app" 2>/dev/null; then
          echo "[SUCCESS] 已停止相关进程"
        else
          echo "[WARN] 未找到运行中的进程"
        fi
      fi
    fi
    
    # 注意：不自动停止 Nginx，因为它可能被其他服务使用
    # 如果需要停止 Nginx，请手动执行: sudo systemctl stop nginx
    ;;
    
  status)
    # 查看服务状态
    if [[ $EUID -eq 0 ]] && command -v systemctl >/dev/null 2>&1 && systemctl list-unit-files 2>/dev/null | grep -q "$SERVICE_NAME.service"; then
      echo "[INFO] Systemd 服务状态:"
      systemctl status "$SERVICE_NAME" --no-pager -l || true
    else
      if [[ -f "$ROOT_DIR/gpu_server.pid" ]]; then
        PID=$(cat "$ROOT_DIR/gpu_server.pid")
        if kill -0 "$PID" 2>/dev/null; then
          echo "[INFO] 后台服务运行中，PID: $PID"
          ps aux | grep "$PID" | grep -v grep || true
        else
          echo "[WARN] PID 文件存在但进程不存在"
          rm -f "$ROOT_DIR/gpu_server.pid"
        fi
      else
        echo "[INFO] 检查运行中的进程..."
        if pgrep -f "uvicorn main:app" > /dev/null; then
          echo "[INFO] 找到运行中的进程:"
          ps aux | grep "uvicorn main:app" | grep -v grep || true
        else
          echo "[WARN] 未找到运行中的服务"
        fi
      fi
    fi
    ;;
    
  restart)
    # 重启服务
    "$0" stop
    sleep 2
    "$0" "${2:-systemd}"
    ;;
    
  *)
    echo "用法: $0 {systemd|manual|background|stop|status|restart|enable-autostart|disable-autostart}" >&2
    echo ""
    echo "模式说明:"
    echo "  systemd          - 使用 systemd 管理服务（推荐，需要 root 权限，支持 systemd 的系统）"
    echo "  manual           - 前台启动（用于调试）"
    echo "  background       - 后台启动（使用 nohup）"
    echo "  stop             - 停止服务"
    echo "  status           - 查看服务状态"
    echo "  restart          - 重启服务"
    echo "  enable-autostart - 配置开机自启（无 systemd 环境，需要 root 权限）"
    echo "  disable-autostart - 禁用开机自启（需要 root 权限）"
    echo ""
    echo "环境变量:"
    echo "  GPU_SERVER_PORT_INTERNAL - 服务端口（默认: 18001）"
    echo "  HF_ENDPOINT              - Hugging Face 镜像源（默认: https://hf-mirror.com）"
    echo ""
    echo "示例:"
    echo "  # 启动服务（自动选择最佳方式）"
    echo "  sudo $0 systemd"
    echo ""
    echo "  # 配置开机自启（无 systemd 环境）"
    echo "  sudo $0 enable-autostart"
    exit 1
    ;;
esac

