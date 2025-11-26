# 日志系统使用说明

## 📋 日志文件位置

- **主日志文件**: `logs/gpu_server.log` - 记录所有详细日志
- **错误日志文件**: `logs/gpu_server_error.log` - 仅记录错误和异常

## 🔍 查看日志的方法

### 方法1: 使用日志查看脚本（推荐）

```bash
# 实时查看所有日志
./view_logs.sh

# 查看最后50行日志
./view_logs.sh last 50

# 查看PDF转换任务日志
./view_logs.sh pdf

# 查看嵌入任务日志
./view_logs.sh embed

# 查看重排序任务日志
./view_logs.sh rerank

# 查看请求日志
./view_logs.sh request

# 搜索日志内容
./view_logs.sh grep "PDF转换"

# 查看错误日志
./view_logs.sh error

# 查看统计信息
./view_logs.sh stats
```

### 方法2: 直接使用 tail 命令

```bash
# 实时查看日志
tail -f logs/gpu_server.log

# 查看最后100行
tail -n 100 logs/gpu_server.log

# 查看错误日志
tail -f logs/gpu_server_error.log
```

### 方法3: 使用 grep 搜索

```bash
# 搜索特定任务
grep "\[PDF转换任务\]" logs/gpu_server.log

# 搜索错误
grep "❌" logs/gpu_server.log

# 搜索特定客户端
grep "客户端: 192.168.1.100" logs/gpu_server.log
```

## 📊 日志格式说明

### 请求日志

```
2025-11-23 21:37:33 [INFO    ] [gpu_pdf_server] [请求开始] GET /health | 客户端: 127.0.0.1
2025-11-23 21:37:33 [INFO    ] [gpu_pdf_server] [请求完成] ✅ GET /health | 状态码: 200 | 处理时间: 0.001s | 客户端: 127.0.0.1
```

- ✅ 表示成功（2xx状态码）
- ⚠️ 表示重定向（3xx状态码）
- ❌ 表示错误（4xx/5xx状态码）

### PDF转换任务日志

```
[PDF转换任务] 开始处理文件: example.pdf
[PDF转换任务] 文件已保存 | 文件名: example.pdf | 大小: 1,234,567 bytes (1.18 MB)
[PDF转换任务] 开始PDF转换: example.pdf
[PDF转换任务] PDF转换完成 | 转换时间: 2.345s
[PDF转换任务] ✅ 任务完成 | 文件名: example.pdf | 输入大小: 1,234,567 bytes | 输出大小: 5,678 chars | 总耗时: 2.456s
```

### 嵌入任务日志

```
[嵌入任务] 开始处理 | 文本数量: 10 | 总字符数: 5,000 | 平均长度: 500 chars
[嵌入任务] ✅ 任务完成 | 文本数量: 10 | 嵌入维度: 1024 | 编码时间: 0.123s | 总耗时: 0.125s | 速度: 81.3 texts/s
```

### 重排序任务日志

```
[重排序任务] 开始处理 | 查询长度: 50 chars | 文档数量: 20 | 总文档字符数: 10,000
[重排序任务] ✅ 任务完成 | 文档数量: 20 | 重排序时间: 0.234s | 总耗时: 0.245s | 分数范围: [0.1234, 0.9876] | 速度: 85.5 pairs/s
```

## 🎯 日志级别

- **DEBUG**: 详细的调试信息
- **INFO**: 一般信息（请求、任务处理等）
- **WARNING**: 警告信息
- **ERROR**: 错误信息（会同时写入错误日志文件）

## 📈 监控建议

1. **实时监控**: 使用 `./view_logs.sh` 实时查看日志
2. **错误监控**: 定期检查 `logs/gpu_server_error.log`
3. **性能监控**: 关注处理时间，如果超过预期需要优化
4. **统计信息**: 使用 `./view_logs.sh stats` 查看任务统计

## 🔧 日志轮转（可选）

如果日志文件过大，可以配置日志轮转：

```bash
# 安装 logrotate（如果未安装）
sudo apt-get install logrotate

# 创建 logrotate 配置
sudo nano /etc/logrotate.d/gpu-server
```

添加以下内容：

```
/root/GPU_server/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0644 root root
}
```

## 💡 常见问题

### Q: 日志文件太大怎么办？

A: 可以配置日志轮转，或者定期清理旧日志：
```bash
# 保留最近7天的日志
find logs/ -name "*.log" -mtime +7 -delete
```

### Q: 如何只查看错误日志？

A: 使用 `./view_logs.sh error` 或直接查看 `logs/gpu_server_error.log`

### Q: 如何搜索特定时间的日志？

A: 使用 grep 结合时间戳：
```bash
grep "2025-11-23 21:" logs/gpu_server.log
```

### Q: 日志没有输出怎么办？

A: 检查：
1. 服务是否正在运行：`./start_gpu_server.sh status`
2. 日志目录权限：`ls -la logs/`
3. 磁盘空间：`df -h`



