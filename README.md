## 启动服务

```python
python ./funasr_wss_server.py
```

**注意事项：**
1. vLLM 需要安装：`pip install vllm>=0.13.0`
2. vLLM 仅支持 FunASRNano 类型的 ASR 模型
3. 如果 vLLM 初始化失败，系统会自动回退到传统推理方式
4. vLLM 主要用于离线/2pass ASR，流式ASR仍使用传统方式以保证实时性

## 客户端连接

```pytthon
python .\funasr_wss_client.py --audio_in .\example.wav
```


## WebSocket 接口文档

服务端提供基于 WebSocket 的实时语音识别服务，完全兼容 FunASR 客户端协议。

### 1. 连接地址

- **URL**: `ws://<SERVER_IP>:10095`
- **协议**: WebSocket (Binary Frames)

### 2. 通信流程

整个识别过程包含三个阶段：**握手配置 -> 音频流传输 -> 结果接收**。

#### a. 握手配置 (First Message)

建立连接后，客户端发送的**第一帧**必须是 JSON 格式的配置信息：

```json
{
  "mode": "2pass",                   // 推荐使用 2pass (流式+离线修正) 或 online
  "chunk_size": [5, 10, 5],          // 分块大小配置 [编码器历史, 当前块, 编码器未来]
  "chunk_interval": 10,              // 发送间隔 (ms)
  "encoder_chunk_look_back": 4,      // 编码器回溯步数
  "decoder_chunk_look_back": 1,      // 解码器回溯步数
  "audio_fs": 16000,                 // 音频采样率 (必须是 16000)
  "wav_name": "demo",                // 音频标识
  "is_speaking": true,               // 标记开始说话
  "hotwords": "{\"阿里巴巴\": 20, \"达摩院\": 30}", // 热词配置 (可选)
  "itn": true                        // 开启逆文本标准化 (数字转汉字等)
}
```


#### b. 音频流传输 (Streaming)

- 配置帧发送后，客户端持续发送**二进制音频数据 (Binary Frame)**。
- 格式：PCM, 16000Hz, 16bit, 单声道。
- 建议分块发送，每块大小约 60ms - 100ms 的数据。

#### c. 结束信号 (End of Stream)

- 当用户停止说话时，客户端发送一帧 JSON 结束信号：
  
  ```json
  {"is_speaking": false}
  ```

### 3. 服务端响应格式

服务端会通过 WebSocket 持续返回 JSON 格式的识别结果。

#### 流式中间结果 (Variable)

当 `mode="online"` 或 `mode="2pass"` 时，服务端会实时返回当前识别片段：

```json
{
  "mode": "2pass-online",
  "text": "正在识别的内容",
  "wav_name": "demo",
  "is_final": false // 通常为 false，但当检测到语音结束(is_speaking: false)时的最后一帧可能为 true
}
```

#### 最终结果 (Final)

当一句话结束 (VAD 检测到静音) 或收到 `is_speaking: false` 后，服务端会进行离线修正，并返回最终结果：

```json
{
  "mode": "2pass-offline",
  "text": "最终识别的修正结果。",
  "wav_name": "demo",
}
```
VAD在语音中间切句，返回的`is_final=False`, 只有当发送音频结束也就是服务端接收到`is_speaking: false` 才会返回`is_final=True`表示识别结束。

客户端判断是否是流式临时结果，还是二遍解码最终结果，根据mode字段判断。

is_final仅用于表示服务端是否已经完成全部输入语音的识别。客户端收到is_final=True，即可安全断开连接。

## 性能优化说明

### vLLM 异步推理

本版本集成了 vLLM 异步推理功能，可以显著提升高并发场景下的性能：

1. **离线 ASR 优化**：使用 vLLM 的批量处理能力，可以同时处理多个离线 ASR 请求
2. **并发控制**：通过 `--vllm_max_num_seqs` 参数控制最大并发数
3. **自动回退**：如果 vLLM 不可用或初始化失败，系统会自动回退到传统推理方式

### 并发性能提升

- **传统方式**：使用 ThreadPoolExecutor，每个请求独立处理，在高并发下可能遇到阻塞
- **vLLM 方式**：利用 vLLM 的批量处理和异步能力，可以更好地处理并发请求

### 使用建议

- **高并发场景**：推荐使用 vLLM，设置合适的 `--vllm_max_num_seqs` 和 `--vllm_gpu_memory_utilization`
- **低延迟场景**：可以适当降低批量处理大小和超时时间
- **资源受限**：如果 GPU 内存不足，可以降低 `--vllm_gpu_memory_utilization`


## Web 测试客户端 (New)

本项目提供了一个轻量级的 Web 页面，用于快速验证 ASR 服务及其 VAD 效果。

### 1. 启动 Web 服务

```bash
cd deploy/asr/web_client
python serve_client.py
```

服务默认监听 `8000` 端口。

### 2. 访问测试

- **推荐 (本地)**: 直接访问 `http://localhost:8000`。
  - 浏览器会自动允许麦克风权限。
  - 页面中 WebSocket 地址填入远程服务器 IP 即可 (例如 `ws://10.11.x.x:10095`)。
- **高级 (远程)**: 如果浏览器和 Web 服务不在同一台机器，需访问 `http://<Web_Server_IP>:8000`。
  - **注意**: Chrome 默认禁止非 HTTPS 网页使用麦克风。
  - **解决**: 需配置 `chrome://flags/#unsafely-treat-insecure-origin-as-secure` 才能使用麦克风。

