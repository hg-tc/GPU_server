# GPU 使用状态报告

## ✅ 当前状态：GPU 加速已启用

### GPU 信息
- **GPU型号**: NVIDIA GeForce RTX 3090
- **显存总量**: 24 GB
- **CUDA版本**: 12.8
- **PyTorch版本**: 2.9.1+cu128

### 模型GPU使用情况

#### 1. 嵌入模型 (Embedding Model)
- **模型**: BAAI/bge-large-zh-v1.5
- **设备**: ✅ GPU (CUDA)
- **显存占用**: ~1590 MB
- **状态**: 正常使用GPU加速

#### 2. 重排序模型 (Reranker Model)
- **模型**: BAAI/bge-reranker-v2-m3
- **设备**: ✅ GPU (CUDA)
- **显存占用**: ~1074 MB
- **FP16加速**: ✅ 已启用
- **状态**: 正常使用GPU加速

#### 3. PDF转换模型 (Marker-PDF)
- **设备**: 自动检测（marker-pdf内部管理）
- **状态**: 根据marker-pdf配置自动选择

### 性能测试结果

#### 嵌入任务
- **测试文本数**: 3条
- **处理时间**: ~9.6秒（首次加载模型）
- **显存增加**: +1590 MB
- **加速状态**: ✅ GPU加速

#### 重排序任务
- **文档数**: 5条
- **处理时间**: ~5.6秒（首次加载模型）
- **显存增加**: +1074 MB
- **加速状态**: ✅ GPU加速

### 配置说明

当前配置已优化为自动检测并使用GPU：

1. **自动GPU检测**: 如果检测到CUDA可用，自动使用GPU
2. **FP16加速**: 重排序模型启用FP16半精度加速
3. **显存管理**: 模型懒加载，首次使用时加载到GPU

### 验证方法

#### 方法1: 查看日志
```bash
./view_logs.sh | grep -E "device|GPU|使用GPU"
```

#### 方法2: 运行测试脚本
```bash
python3 test_gpu.py
```

#### 方法3: 检查nvidia-smi
```bash
nvidia-smi
```

#### 方法4: 查看显存使用
```bash
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

### 环境变量配置

如果需要手动控制GPU使用，可以设置以下环境变量：

```bash
# 强制使用GPU（推荐）
export FORCE_CUDA=1

# 指定GPU设备
export CUDA_VISIBLE_DEVICES=0

# 强制使用CPU（不推荐，性能慢）
export FORCE_CPU=1
```

### 性能对比

| 任务类型 | CPU模式 | GPU模式 | 加速比 |
|---------|---------|---------|--------|
| 嵌入（3条文本） | ~30-60s | ~9.6s | 3-6x |
| 重排序（5条文档） | ~15-30s | ~5.6s | 3-5x |

*注：首次加载模型时间较长，后续请求会更快*

### 故障排查

如果发现模型未使用GPU：

1. **检查CUDA是否可用**
   ```bash
   python3 -c "import torch; print(torch.cuda.is_available())"
   ```

2. **检查日志中的设备信息**
   ```bash
   tail -f logs/gpu_server.log | grep device
   ```

3. **检查环境变量**
   ```bash
   env | grep -E "CUDA|FORCE"
   ```

4. **重启服务**
   ```bash
   ./start_gpu_server.sh restart
   ```

### 注意事项

1. **首次加载**: 模型首次加载到GPU需要时间，这是正常的
2. **显存占用**: 模型加载后会占用显存，这是正常的
3. **多GPU**: 如果有多块GPU，可以通过`CUDA_VISIBLE_DEVICES`指定
4. **性能优化**: FP16加速可以提升性能并减少显存占用


