# 日志文件说明

## 日志文件列表

### 1. `logs/gpu_server.log` ⭐ **主要日志文件**
- **用途**: 应用主日志，记录所有 INFO/DEBUG 级别的日志
- **内容**: 
  - 服务启动信息
  - API 请求日志
  - 模型加载日志
  - OCR/PDF/Embedding 等任务处理日志
  - 成功/失败状态
- **更新频率**: 实时（已启用自动刷新）
- **查看方式**: 
  ```bash
  ./logs.sh f          # 实时查看
  ./logs.sh l 100      # 最后100行
  tail -f logs/gpu_server.log
  ```

### 2. `logs/gpu_server_error.log` ⚠️ **错误日志**
- **用途**: 仅记录 ERROR 级别的错误日志
- **内容**: 应用运行时的错误和异常
- **更新频率**: 仅在有错误时更新
- **查看方式**:
  ```bash
  ./logs.sh error      # 实时查看错误日志
  tail -f logs/gpu_server_error.log
  ```

### 3. `logs/server.log` 📋 **Uvicorn 服务器日志**
- **用途**: Uvicorn 服务器的标准输出和标准错误
- **内容**: 
  - Uvicorn 启动信息
  - HTTP 请求日志（来自 Uvicorn）
  - 服务器进程信息
- **更新频率**: 实时
- **说明**: 由 `daemon.sh` 重定向 uvicorn 输出生成
- **查看方式**:
  ```bash
  tail -f logs/server.log
  ```

### 4. `logs/daemon.log` 🔄 **守护进程日志**
- **用途**: 守护进程脚本的操作日志
- **内容**:
  - 服务启动/停止记录
  - 服务重启记录
  - 健康检查结果
  - 监控循环状态
- **更新频率**: 守护进程操作时更新
- **查看方式**:
  ```bash
  tail -f logs/daemon.log
  cat logs/daemon.log
  ```

## 日志文件有效性

### ✅ 有效的日志文件（推荐查看）

1. **`gpu_server.log`** - **最重要**，包含所有应用日志
2. **`gpu_server_error.log`** - 查看错误时使用
3. **`server.log`** - 查看 Uvicorn 服务器日志
4. **`daemon.log`** - 查看守护进程状态

### 📊 日志查看优先级

1. **日常查看**: `gpu_server.log` - 包含所有信息
2. **排查错误**: `gpu_server_error.log` - 只看错误
3. **服务器问题**: `server.log` - Uvicorn 相关
4. **进程管理**: `daemon.log` - 守护进程状态

## 日志刷新问题

### 问题
Python 的 `FileHandler` 默认有缓冲，日志可能不会立即写入文件。

### 解决方案
已添加自动刷新机制：
- 每次写入日志后自动调用 `flush()`
- 确保日志实时写入磁盘

### 验证日志是否刷新
```bash
# 方法1: 查看文件修改时间
stat logs/gpu_server.log

# 方法2: 实时监控
tail -f logs/gpu_server.log

# 方法3: 使用日志查看脚本
./logs.sh f
```

## 常用日志查看命令

### 使用日志脚本（推荐）
```bash
./logs.sh f              # 实时查看主日志
./logs.sh error          # 实时查看错误日志
./logs.sh l 100          # 最后100行
./logs.sh grep "OCR"     # 搜索 OCR 相关日志
./logs.sh ocr            # OCR 处理日志
./logs.sh pdf            # PDF 处理日志
./logs.sh stats          # 统计信息
```

### 直接查看文件
```bash
# 实时查看
tail -f logs/gpu_server.log

# 最后N行
tail -n 50 logs/gpu_server.log

# 搜索
grep "error" logs/gpu_server.log

# 查看所有日志
tail -f logs/*.log
```

## 日志文件大小管理

### 查看日志大小
```bash
du -h logs/*.log
```

### 清空日志（谨慎使用）
```bash
./logs.sh clear          # 清空主日志和错误日志
# 或手动清空
> logs/gpu_server.log
> logs/gpu_server_error.log
```

### 日志轮转（建议）
如果日志文件过大，可以考虑：
1. 使用 `logrotate` 工具
2. 定期备份和清理旧日志
3. 限制日志文件大小

## 日志级别说明

- **DEBUG**: 详细调试信息（仅在开发时有用）
- **INFO**: 一般信息（正常操作）
- **WARNING**: 警告信息（可能的问题）
- **ERROR**: 错误信息（需要关注）
- **CRITICAL**: 严重错误（必须处理）

## 重启服务后日志

重启服务后：
- ✅ `gpu_server.log` - 会继续追加新日志
- ✅ `gpu_server_error.log` - 会继续追加新错误
- ✅ `server.log` - 会记录新的服务器启动信息
- ✅ `daemon.log` - 会记录重启操作

**注意**: 日志文件不会自动清空，会一直追加。如果需要清空，使用 `./logs.sh clear`。

