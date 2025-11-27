# 修复 "Server disconnected without sending a response" 错误

## 问题描述

主服务器报错：`Server disconnected without sending a response`，这通常发生在GPU服务器处理请求时连接意外断开。

## 已实施的修复

### 1. 增强异常处理
- ✅ 在 `/embed` 端点添加了更细粒度的异常处理
- ✅ 添加了全局异常处理器，确保所有未捕获的异常都能返回HTTP响应
- ✅ 添加了HTTP异常处理器，确保HTTP异常正确返回

### 2. 添加超时配置
- ✅ 为 uvicorn 添加了 `--timeout-keep-alive 300` 参数（保持连接300秒）
- ✅ 为 uvicorn 添加了 `--timeout-graceful-shutdown 30` 参数（优雅关闭30秒）
- ✅ 更新了 `gpu-server.service` 和 `start_gpu_server.sh` 启动脚本

### 3. 超时检测和日志
- ✅ 在请求中间件中添加了超时检测
- ✅ 超过60秒发出警告
- ✅ 超过300秒发出严重警告

## 应用修复

### 方式1：如果使用 systemd 服务

```bash
# 1. 停止当前服务
sudo systemctl stop gpu-server

# 2. 重新加载配置（如果使用systemd）
sudo systemctl daemon-reload

# 3. 启动服务
sudo systemctl start gpu-server

# 4. 检查状态
sudo systemctl status gpu-server
```

### 方式2：如果使用后台启动脚本

```bash
# 1. 停止当前服务
cd /root/GPU_server
kill $(cat gpu_server.pid) 2>/dev/null || pkill -f "uvicorn main:app"

# 2. 重新启动服务
./start_gpu_server.sh background

# 3. 检查服务状态
ps aux | grep uvicorn | grep -v grep
```

### 方式3：手动重启

```bash
# 1. 找到并停止当前进程
ps aux | grep "uvicorn main:app" | grep -v grep
kill <PID>

# 2. 启动服务（使用新的超时参数）
cd /root/GPU_server
source venv/bin/activate
nohup uvicorn main:app --host 0.0.0.0 --port 18001 --workers 1 \
  --timeout-keep-alive 300 --timeout-graceful-shutdown 30 \
  > logs/gpu_server.log 2>&1 &
```

## 验证修复

### 1. 检查服务是否正常运行
```bash
curl http://localhost:18001/health
# 应该返回: {"status":"ok"}
```

### 2. 检查日志
```bash
tail -f /root/GPU_server/logs/gpu_server.log
```

### 3. 测试嵌入接口
```bash
curl -X POST http://localhost:18001/embed \
  -H "Content-Type: application/json" \
  -d '{"texts": ["测试文本"]}'
```

## 其他建议

### 1. 检查Nginx配置
确保Nginx的超时设置足够长：
- `proxy_read_timeout 1200s` (20分钟)
- `proxy_connect_timeout 75s`
- `proxy_send_timeout 300s`

### 2. 监控资源使用
```bash
# 检查内存使用
free -h

# 检查GPU使用
nvidia-smi

# 检查进程状态
ps aux | grep uvicorn
```

### 3. 如果问题仍然存在
- 检查是否有内存泄漏
- 检查是否有长时间运行的请求
- 查看错误日志：`tail -f /root/GPU_server/logs/gpu_server_error.log`
- 查看Nginx错误日志：`tail -f /var/log/nginx/gpu_server_error.log`

## 技术细节

### 超时参数说明
- `--timeout-keep-alive 300`: 保持空闲连接300秒，防止连接过早关闭
- `--timeout-graceful-shutdown 30`: 优雅关闭超时30秒，确保正在处理的请求能完成

### 异常处理改进
- 全局异常处理器确保所有异常都能返回JSON响应
- 细粒度的异常处理确保每个步骤的错误都能被捕获和记录
- HTTP异常处理器确保HTTP异常正确返回状态码

## 联系支持

如果问题持续存在，请提供：
1. 完整的错误日志
2. 服务运行时间
3. 最近的请求模式
4. 系统资源使用情况

