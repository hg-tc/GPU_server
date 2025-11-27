#!/bin/bash
# ================================================
# Nginx 安装和配置脚本
# ================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 检测是否需要使用 sudo
if [ "$EUID" -eq 0 ]; then
    SUDO_CMD=""
    echo -e "${GREEN}检测到 root 用户，不使用 sudo${NC}"
else
    if command -v sudo &> /dev/null; then
        SUDO_CMD="sudo"
    else
        echo -e "${RED}错误: 需要 root 权限或 sudo 命令${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}=========================================="
echo "  Nginx 安装和配置"
echo "==========================================${NC}"

# 检查是否已安装nginx
if command -v nginx &> /dev/null; then
    echo -e "${GREEN}Nginx 已安装${NC}"
    nginx -v
else
    echo -e "${YELLOW}安装 Nginx...${NC}"
    
    # 检测系统类型
    if [ -f /etc/debian_version ]; then
        # Debian/Ubuntu
        $SUDO_CMD apt-get update
        $SUDO_CMD apt-get install -y nginx
    elif [ -f /etc/redhat-release ]; then
        # CentOS/RHEL
        $SUDO_CMD yum install -y nginx 2>/dev/null || $SUDO_CMD dnf install -y nginx
    else
        echo -e "${RED}无法自动检测系统类型，请手动安装 nginx${NC}"
        exit 1
    fi
fi

# 查找nginx配置目录
NGINX_CONF_DIR=""
if [ -d "/etc/nginx/conf.d" ]; then
    NGINX_CONF_DIR="/etc/nginx/conf.d"
elif [ -d "/etc/nginx/sites-available" ]; then
    NGINX_CONF_DIR="/etc/nginx/sites-available"
else
    echo -e "${YELLOW}未找到标准nginx配置目录，尝试查找...${NC}"
    NGINX_CONF_DIR=$(nginx -t 2>&1 | grep -oP 'file \K[^ ]+' | head -1 | xargs dirname 2>/dev/null || echo "")
fi

if [ -z "$NGINX_CONF_DIR" ]; then
    echo -e "${RED}无法找到nginx配置目录${NC}"
    exit 1
fi

echo -e "${GREEN}Nginx 配置目录: $NGINX_CONF_DIR${NC}"

# 复制配置文件
CONF_FILE="$NGINX_CONF_DIR/gpu_server.conf"
echo -e "${GREEN}创建配置文件: $CONF_FILE${NC}"

$SUDO_CMD cp "$SCRIPT_DIR/nginx.conf" "$CONF_FILE"

# 如果使用sites-available，创建符号链接
if [ -d "/etc/nginx/sites-enabled" ]; then
    if [ ! -L "/etc/nginx/sites-enabled/gpu_server.conf" ]; then
        echo -e "${GREEN}创建符号链接...${NC}"
        $SUDO_CMD ln -s "$CONF_FILE" "/etc/nginx/sites-enabled/gpu_server.conf"
    fi
fi

# 测试nginx配置
echo -e "${GREEN}测试 Nginx 配置...${NC}"
if $SUDO_CMD nginx -t; then
    echo -e "${GREEN}Nginx 配置测试通过${NC}"
    
    # 启动或重载nginx
    # 检查是否有 systemctl
    if command -v systemctl &> /dev/null; then
        if $SUDO_CMD systemctl is-active --quiet nginx 2>/dev/null; then
            echo -e "${GREEN}重载 Nginx...${NC}"
            $SUDO_CMD systemctl reload nginx 2>/dev/null || $SUDO_CMD nginx -s reload
        else
            echo -e "${GREEN}启动 Nginx...${NC}"
            $SUDO_CMD systemctl start nginx 2>/dev/null || $SUDO_CMD nginx
        fi
    else
        # 没有 systemctl，直接使用 nginx 命令
        if pgrep -x nginx > /dev/null; then
            echo -e "${GREEN}重载 Nginx...${NC}"
            $SUDO_CMD nginx -s reload
        else
            echo -e "${GREEN}启动 Nginx...${NC}"
            $SUDO_CMD nginx
        fi
    fi
    
    echo ""
    echo -e "${GREEN}=========================================="
    echo "  Nginx 配置完成！"
    echo "==========================================${NC}"
    echo ""
    echo "配置信息:"
    echo "  - 监听端口: 16000"
    echo "  - 后端服务: http://127.0.0.1:8000"
    echo "  - 配置文件: $CONF_FILE"
    echo ""
    echo "测试访问:"
    echo "  curl http://localhost:16000/health"
    echo ""
else
    echo -e "${RED}Nginx 配置测试失败，请检查配置文件${NC}"
    exit 1
fi

