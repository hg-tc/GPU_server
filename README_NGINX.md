# Nginx 反向代理和服务守护进程配置

## 概述

本项目配置了 Nginx 反向代理，将外部访问的 16000 端口代理到内部服务端口（默认 8000）。同时提供了守护进程脚本，在没有 systemctl 的情况下保持服务持续运行。

## 配置说明

### 端口配置
- **外部访问端口**: 16000（通过 Nginx 代理）
- **内部服务端口**: 8000（FastAPI 服务实际运行端口）

### 文件说明

1. **nginx.conf**: Nginx 反向代理配置文件
   - 监听 16000 端口
   - 代理到 http://127.0.0.1:8000
   - 支持大文件上传（500MB）
   - 超时时间设置为 600 秒

2. **daemon.sh**: 服务守护进程脚本
   - 自动监控服务状态
   - 服务异常退出时自动重启
   - 支持 start/stop/restart/status 命令

3. **install_nginx.sh**: Nginx 安装和配置脚本
   - 自动检测系统类型并安装 Nginx
   - 复制配置文件到系统目录
   - 测试并启动 Nginx

## 安装和配置步骤

### 1. 安装和配置 Nginx

```bash
# 运行安装脚本
./install_nginx.sh
```

脚本会自动：
- 检测并安装 Nginx（如果未安装）
- 复制配置文件到系统目录
- 测试配置并启动/重载 Nginx

### 2. 手动配置 Nginx（如果自动安装失败）

#### Debian/Ubuntu 系统：
```bash
# 安装 Nginx
sudo apt-get update
sudo apt-get install -y nginx

# 复制配置文件
sudo cp nginx.conf /etc/nginx/conf.d/gpu_server.conf

# 测试配置
sudo nginx -t

# 启动/重载 Nginx
sudo systemctl start nginx
# 或
sudo nginx -s reload
```

#### CentOS/RHEL 系统：
```bash
# 安装 Nginx
sudo yum install -y nginx
# 或
sudo dnf install -y nginx

# 复制配置文件
sudo cp nginx.conf /etc/nginx/conf.d/gpu_server.conf

# 测试配置
sudo nginx -t

# 启动 Nginx
sudo systemctl start nginx
```

### 3. 启动服务守护进程

```bash
# 启动服务（后台运行，自动监控和重启）
nohup ./daemon.sh start > logs/daemon.log 2>&1 &

# 或者使用 screen（推荐）
screen -S gpu_server
./daemon.sh start
# 按 Ctrl+A 然后按 D 退出 screen，服务继续运行

# 查看服务状态
./daemon.sh status

# 停止服务
./daemon.sh stop

# 重启服务
./daemon.sh restart
```

### 4. 使用 screen 保持服务运行（推荐）

```bash
# 创建新的 screen 会话
screen -S gpu_server

# 在 screen 中启动守护进程
./daemon.sh start

# 退出 screen（服务继续运行）
# 按 Ctrl+A，然后按 D

# 重新连接到 screen
screen -r gpu_server

# 查看所有 screen 会话
screen -ls
```

### 5. 使用 tmux 保持服务运行

```bash
# 创建新的 tmux 会话
tmux new -s gpu_server

# 在 tmux 中启动守护进程
./daemon.sh start

# 退出 tmux（服务继续运行）
# 按 Ctrl+B，然后按 D

# 重新连接到 tmux
tmux attach -t gpu_server

# 查看所有 tmux 会话
tmux ls
```

## 验证配置

### 1. 检查 Nginx 状态

```bash
# 检查 Nginx 是否运行
sudo systemctl status nginx
# 或
ps aux | grep nginx

# 检查端口监听
sudo netstat -tlnp | grep 16000
# 或
sudo ss -tlnp | grep 16000
```

### 2. 测试服务访问

```bash
# 测试内部服务（端口 8000）
curl http://127.0.0.1:8000/health

# 测试外部访问（端口 16000，通过 Nginx）
curl http://127.0.0.1:16000/health
curl http://your-server-ip:16000/health
```

### 3. 查看日志

```bash
# 查看守护进程日志
tail -f logs/daemon.log

# 查看服务日志
tail -f logs/server.log

# 查看 Nginx 访问日志
sudo tail -f /var/log/nginx/access.log

# 查看 Nginx 错误日志
sudo tail -f /var/log/nginx/error.log
```

## 环境变量

可以通过环境变量自定义配置：

```bash
# 设置服务端口（默认 8000）
export PORT=8000

# 设置主机（默认 0.0.0.0）
export HOST=0.0.0.0

# 设置工作进程数（默认 1）
export WORKERS=1

# 设置重启延迟（默认 10 秒）
export RESTART_DELAY=10
```

## 故障排除

### 1. Nginx 无法启动

```bash
# 检查配置文件语法
sudo nginx -t

# 查看错误日志
sudo tail -f /var/log/nginx/error.log

# 检查端口占用
sudo lsof -i :16000
```

### 2. 服务无法访问

```bash
# 检查服务是否运行
./daemon.sh status

# 检查端口占用
sudo lsof -i :8000

# 检查防火墙
sudo ufw status
# 或
sudo iptables -L
```

### 3. 服务频繁重启

```bash
# 查看服务日志
tail -f logs/server.log

# 查看守护进程日志
tail -f logs/daemon.log

# 检查系统资源
free -h
df -h
```

## 安全建议

1. **防火墙配置**：只开放必要的端口（16000）
   ```bash
   sudo ufw allow 16000/tcp
   sudo ufw deny 8000/tcp  # 禁止外部直接访问内部端口
   ```

2. **Nginx 安全**：考虑添加访问限制、SSL 证书等

3. **服务用户**：建议使用非 root 用户运行服务（需要相应权限调整）

## 更新配置

修改 `nginx.conf` 后：

```bash
# 复制新配置
sudo cp nginx.conf /etc/nginx/conf.d/gpu_server.conf

# 测试配置
sudo nginx -t

# 重载 Nginx
sudo systemctl reload nginx
# 或
sudo nginx -s reload
```


