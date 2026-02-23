# vllm serve 命令参考手册

> 来源：https://docs.vllm.ai/en/stable/cli/serve/ (v0.15.1)

## 基本用法

```bash
vllm serve <model_name_or_path> [OPTIONS]
```

## JSON CLI 参数传递

支持两种等价的 JSON 参数传递方式：

```bash
# 方式一：完整 JSON
--json-arg '{"key1": "value1", "key2": {"key3": "value2"}}'

# 方式二：点号展开
--json-arg.key1 value1 --json-arg.key2.key3 value2

# 列表元素可用 + 传递
--json-arg '{"key4": ["value3", "value4", "value5"]}'
--json-arg.key4+ value3 --json-arg.key4+='value4,value5'
```

---

## 1. 通用参数 (Arguments)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--headless` | `False` | 无头模式运行，用于多节点数据并行 |
| `--api-server-count`, `-asc` | - | API 服务器进程数，默认等于 data_parallel_size |
| `--config` | - | 从 YAML 配置文件读取 CLI 选项 |
| `--disable-log-stats` | `False` | 禁用统计日志 |
| `--aggregate-engine-logging` | `False` | 数据并行时记录聚合统计而非逐引擎统计 |
| `--enable-log-requests` | `False` | 启用请求日志 |
| `--disable-log-requests` | `True` | [已弃用] 禁用请求日志 |

---

## 2. 前端参数 (Frontend)

OpenAI 兼容前端服务器参数。

### 2.1 网络与服务

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | - | 主机名 |
| `--port` | `8000` | 端口号 |
| `--uds` | - | Unix 域套接字路径，设置后忽略 host/port |
| `--uvicorn-log-level` | `info` | uvicorn 日志级别 (critical/debug/error/info/trace/warning) |
| `--disable-uvicorn-access-log` | `False` | 禁用 uvicorn 访问日志 |
| `--disable-access-log-for-endpoints` | - | 逗号分隔的排除访问日志的端点路径，如 `/health,/metrics,/ping` |
| `--root-path` | - | FastAPI root_path，用于反向代理 |
| `--disable-frontend-multiprocessing` | `False` | 在同一进程中运行前端服务器和模型引擎 |
| `--disable-fastapi-docs` | `False` | 禁用 FastAPI 的 OpenAPI/Swagger/ReDoc |
| `--enable-offline-docs` | `False` | 启用离线 FastAPI 文档（气隙环境） |

### 2.2 CORS 配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--allow-credentials` | `False` | 允许凭证 |
| `--allowed-origins` | `['*']` | 允许的来源 |
| `--allowed-methods` | `['*']` | 允许的方法 |
| `--allowed-headers` | `['*']` | 允许的头部 |

### 2.3 SSL/TLS 配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--ssl-keyfile` | - | SSL 密钥文件路径 |
| `--ssl-certfile` | - | SSL 证书文件路径 |
| `--ssl-ca-certs` | - | CA 证书文件 |
| `--enable-ssl-refresh` | `False` | SSL 证书文件变更时刷新 SSL 上下文 |
| `--ssl-cert-reqs` | `0` | 是否要求客户端证书 |
| `--ssl-ciphers` | - | HTTPS 加密套件（仅 TLS 1.2 及以下） |

### 2.4 聊天模板与响应

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--chat-template` | - | 聊天模板文件路径或单行模板 |
| `--chat-template-content-format` | `auto` | 消息内容渲染格式 (auto/openai/string) |
| `--trust-request-chat-template` | `False` | 是否信任请求中的聊天模板 |
| `--default-chat-template-kwargs` | - | 聊天模板默认关键字参数，如 `'{"enable_thinking": false}'` |
| `--response-role` | `assistant` | request.add_generation_prompt=true 时返回的角色名 |

### 2.5 工具调用

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--enable-auto-tool-choice` | `False` | 启用自动工具选择 |
| `--exclude-tools-when-tool-choice-none` | `False` | tool_choice='none' 时排除工具定义 |
| `--tool-call-parser` | - | 工具调用解析器（需配合 --enable-auto-tool-choice） |
| `--tool-parser-plugin` | `""` | 工具解析器插件路径 |
| `--tool-server` | - | 工具服务器地址列表，逗号分隔 (host:port) |

### 2.6 安全与认证

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--api-key` | - | API 密钥，设置后请求头中必须携带 |

### 2.7 日志与调试

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--log-config-file` | - | vllm 和 uvicorn 的日志配置 JSON 文件路径 |
| `--max-log-len` | `None` | 日志中打印的最大提示字符数 |
| `--enable-log-outputs` | `False` | 记录模型输出（需 --enable-log-requests） |
| `--enable-log-deltas` | `True` | 记录输出增量 |
| `--log-error-stack` | `False` | 记录错误响应的堆栈跟踪 |

### 2.8 其他前端参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--lora-modules` | - | LoRA 模块配置，支持 `name=path` 或 JSON 格式 |
| `--middleware` | `[]` | 附加 ASGI 中间件 |
| `--return-tokens-as-token-ids` | `False` | 以 `token_id:{id}` 格式返回 token |
| `--enable-request-id-headers` | `False` | 响应中添加 X-Request-Id 头 |
| `--enable-prompt-tokens-details` | `False` | 启用 usage 中的 prompt_tokens_details |
| `--enable-server-load-tracking` | `False` | 启用服务器负载指标跟踪 |
| `--enable-force-include-usage` | `False` | 每个请求都包含 usage |
| `--enable-tokenizer-info-endpoint` | `False` | 启用 /tokenizer_info 端点 |
| `--h11-max-incomplete-event-size` | `4194304` | h11 解析器最大不完整事件大小 (字节) |
| `--h11-max-header-count` | `256` | h11 解析器最大 HTTP 头数量 |
| `--tokens-only` | `False` | 仅启用 Tokens In/Out 端点（用于 Disaggregated 设置） |

---

## 3. 模型配置 (ModelConfig)

### 3.1 模型基础

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `Qwen/Qwen3-0.6B` | HuggingFace 模型名称或路径 |
| `--runner` | `auto` | 模型运行器类型 (auto/draft/generate/pooling) |
| `--convert` | `auto` | 模型转换适配器 (auto/classify/embed/none) |
| `--tokenizer` | - | HuggingFace tokenizer 名称或路径 |
| `--tokenizer-mode` | `auto` | tokenizer 模式 (auto/deepseek_v32/hf/mistral/slow) |
| `--trust-remote-code` | `False` | 信任远程代码 |
| `--served-model-name` | - | API 中使用的模型名称，支持多个 |
| `--config-format` | `auto` | 模型配置格式 (auto/hf/mistral) |
| `--hf-token` | - | HuggingFace 认证 token |
| `--hf-overrides` | `{}` | HuggingFace 配置覆盖参数 |
| `--hf-config-path` | - | HuggingFace 配置文件路径 |

### 3.2 数据类型与精度

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dtype` | `auto` | 模型权重数据类型 (auto/bfloat16/float/float16/float32/half) |
| `--quantization`, `-q` | - | 权重量化方法 |
| `--allow-deprecated-quantization` | `False` | 允许已弃用的量化方法 |
| `--override-attention-dtype` | - | 覆盖 attention 的数据类型 |

### 3.3 模型行为

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--seed` | `0` | 随机种子 |
| `--max-model-len` | - | 模型上下文长度，支持 `1k`/`1K`/`25.6k`/`-1`(auto) 格式 |
| `--max-logprobs` | `20` | 最大 logprobs 数量，-1 表示无上限 |
| `--logprobs-mode` | `raw_logprobs` | logprobs 内容模式 (raw_logprobs/processed_logprobs/raw_logits/processed_logits) |
| `--enforce-eager` | `False` | 强制使用 eager 模式（禁用 CUDA graph） |
| `--disable-sliding-window` | `False` | 禁用滑动窗口 |
| `--disable-cascade-attn` | `False` | 禁用级联注意力 |
| `--skip-tokenizer-init` | `False` | 跳过 tokenizer 初始化 |
| `--enable-prompt-embeds` | `False` | 允许通过 prompt_embeds 传递文本嵌入 |
| `--enable-return-routed-experts` | `False` | 返回路由的专家信息 |

### 3.4 版本控制

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--revision` | - | 模型版本（分支/标签/commit id） |
| `--code-revision` | - | 模型代码版本 |
| `--tokenizer-revision` | - | tokenizer 版本 |

### 3.5 生成配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--generation-config` | `auto` | 生成配置路径 (auto/vllm/文件夹路径) |
| `--override-generation-config` | `{}` | 覆盖生成配置，如 `{"temperature": 0.5}` |
| `--logits-processor-pattern` | - | 允许的 logits processor 正则模式 |
| `--logits-processors` | - | logits processor 类名或定义 |
| `--io-processor-plugin` | - | IOProcessor 插件名称 |

### 3.6 其他模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--allowed-local-media-path` | `""` | 允许读取本地媒体的路径（安全风险） |
| `--allowed-media-domains` | - | 允许的多模态媒体 URL 域名 |
| `--pooler-config` | - | 池化模型的 pooler 配置 (JSON) |
| `--enable-sleep-mode` | `False` | 启用引擎睡眠模式（仅 CUDA/HIP） |
| `--model-impl` | `auto` | 模型实现 (auto/terratorch/transformers/vllm) |

---

## 4. 加载配置 (LoadConfig)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--load-format` | `auto` | 权重加载格式 (auto/pt/safetensors/npcache/dummy/tensorizer/runai_streamer/runai_streamer_sharded/bitsandbytes/sharded_state/gguf/mistral) |
| `--download-dir` | - | 权重下载目录，默认 HuggingFace 缓存目录 |
| `--safetensors-load-strategy` | `lazy` | safetensors 加载策略 (lazy: 内存映射 / eager: 全量读入 / torchao: torchao 子类重建) |
| `--model-loader-extra-config` | `{}` | 模型加载器额外配置 |
| `--ignore-patterns` | `['original/**/*']` | 加载模型时忽略的文件模式 |
| `--use-tqdm-on-load` | `True` | 加载权重时显示进度条 |
| `--pt-load-map-location` | `cpu` | PyTorch checkpoint 的 map_location |
| `--max-parallel-loading-workers` | - | 最大并行加载 worker 数 |

---

## 5. 注意力配置 (AttentionConfig)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--attention-backend` | - | 注意力后端，None 时自动选择 |

---

## 6. 结构化输出配置 (StructuredOutputsConfig)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--reasoning-parser` | `""` | 推理内容解析器，用于解析为 OpenAI API 格式 |
| `--reasoning-parser-plugin` | `""` | 动态推理解析器插件路径 |

---

## 7. 并行配置 (ParallelConfig)

### 7.1 基础并行

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--tensor-parallel-size`, `-tp` | `1` | 张量并行组数 |
| `--pipeline-parallel-size`, `-pp` | `1` | 流水线并行组数 |
| `--data-parallel-size`, `-dp` | `1` | 数据并行组数 |
| `--distributed-executor-backend` | - | 分布式后端 (external_launcher/mp/ray/uni) |

### 7.2 多节点分布式

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--master-addr` | `127.0.0.1` | 多节点分布式推理的 master 地址 |
| `--master-port` | `29501` | 多节点分布式推理的 master 端口 |
| `--nnodes`, `-n` | `1` | 节点数 |
| `--node-rank`, `-r` | `0` | 节点排名 |

### 7.3 数据并行详细配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data-parallel-rank`, `-dpn` | - | 数据并行排名，启用外部负载均衡模式 |
| `--data-parallel-start-rank`, `-dpr` | - | 次要节点的起始数据并行排名 |
| `--data-parallel-size-local`, `-dpl` | - | 本节点运行的数据并行副本数 |
| `--data-parallel-address`, `-dpa` | - | 数据并行集群头节点地址 |
| `--data-parallel-rpc-port`, `-dpp` | - | 数据并行 RPC 通信端口 |
| `--data-parallel-backend`, `-dpb` | `mp` | 数据并行后端 (mp/ray) |
| `--data-parallel-hybrid-lb`, `-dph` | `False` | 混合 DP 负载均衡模式 |
| `--data-parallel-external-lb`, `-dpe` | `False` | 外部 DP 负载均衡模式 |

### 7.4 专家并行与 MoE

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--enable-expert-parallel`, `-ep` | `False` | MoE 层使用专家并行替代张量并行 |
| `--all2all-backend` | `allgather_reducescatter` | MoE 专家并行通信后端 (naive/allgather_reducescatter/pplx/deepep_high_throughput/deepep_low_latency/mori/flashinfer_all2allv) |
| `--enable-eplb` | `False` | 启用专家并行负载均衡 |
| `--eplb-config` | - | 专家并行负载均衡配置 (JSON) |
| `--expert-placement-strategy` | `linear` | 专家放置策略 (linear/round_robin) |

### 7.5 上下文并行

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--decode-context-parallel-size`, `-dcp` | `1` | 解码上下文并行组数 |
| `--prefill-context-parallel-size`, `-pcp` | `1` | 预填充上下文并行组数 |
| `--dcp-kv-cache-interleave-size` | `1` | DCP 的 KV 缓存交错大小（已弃用，用 cp-kv-cache-interleave-size） |
| `--cp-kv-cache-interleave-size` | `1` | DCP/PCP 的 KV 缓存交错大小 |

### 7.6 双批次重叠 (DBO)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--enable-dbo` | `False` | 启用双批次重叠 |
| `--ubatch-size` | `0` | 微批次大小 |
| `--dbo-decode-token-threshold` | `32` | 纯解码批次的 DBO 阈值 |
| `--dbo-prefill-token-threshold` | `512` | 含预填充批次的 DBO 阈值 |

### 7.7 其他并行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--ray-workers-use-nsight` | `False` | 使用 nsight 分析 Ray workers |
| `--disable-custom-all-reduce` | `False` | 禁用自定义 all-reduce，回退到 NCCL |
| `--disable-nccl-for-dp-synchronization` | - | DP 同步使用 Gloo 替代 NCCL |
| `--worker-cls` | `auto` | worker 类全名 |
| `--worker-extension-cls` | `""` | worker 扩展类全名 |

---

## 8. 缓存配置 (CacheConfig)

### 8.1 GPU 缓存

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--block-size` | 平台决定 | 缓存块大小 (1/8/16/32/64/128/256)，CUDA 最大 32 |
| `--gpu-memory-utilization` | `0.9` | GPU 显存利用率 (0~1) |
| `--kv-cache-memory-bytes` | - | 每 GPU 的 KV 缓存大小（字节），设置后忽略 gpu_memory_utilization |
| `--num-gpu-blocks-override` | - | 覆盖 GPU 块数量（测试用） |
| `--kv-cache-dtype` | `auto` | KV 缓存数据类型 (auto/bfloat16/fp8/fp8_e4m3/fp8_e5m2/fp8_inc/fp8_ds_mla) |
| `--calculate-kv-scales` | `False` | fp8 KV 缓存时动态计算 k/v scale |

### 8.2 前缀缓存

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--enable-prefix-caching` | - | 启用前缀缓存 |
| `--prefix-caching-hash-algo` | `sha256` | 前缀缓存哈希算法 (sha256/sha256_cbor/xxhash/xxhash_cbor) |

### 8.3 CPU 与卸载

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--swap-space` | `4` | 每 GPU 的 CPU 交换空间 (GiB) |
| `--cpu-offload-gb` | `0` | 每 GPU 卸载到 CPU 的空间 (GiB) |
| `--kv-offloading-size` | - | KV 缓存卸载缓冲区大小 (GiB) |
| `--kv-offloading-backend` | `native` | KV 缓存卸载后端 (native/lmcache) |

### 8.4 Mamba 缓存

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mamba-cache-dtype` | `auto` | Mamba 缓存数据类型 (auto/float16/float32) |
| `--mamba-ssm-cache-dtype` | `auto` | Mamba SSM 状态缓存数据类型 |
| `--mamba-block-size` | - | Mamba 缓存块大小（需启用前缀缓存，必须是 8 的倍数） |
| `--mamba-cache-mode` | `none` | Mamba 缓存策略 (none/all/align) |

### 8.5 其他缓存参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--kv-sharing-fast-prefill` | `False` | KV 共享快速预填充（实验性） |

---

## 9. 多模态配置 (MultiModalConfig)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--limit-mm-per-prompt` | `{}` | 每个 prompt 各模态的最大输入数量和选项 (JSON) |
| `--enable-mm-embeds` | `False` | 允许传递多模态嵌入 |
| `--media-io-kwargs` | `{}` | 媒体输入处理参数，如 `{"video": {"num_frames": 40}}` |
| `--mm-processor-kwargs` | - | 多模态处理器参数，如 `{"num_crops": 4}` |
| `--mm-processor-cache-gb` | `4` | 多模态处理器缓存大小 (GiB) |
| `--mm-processor-cache-type` | `lru` | 缓存类型 (lru/shm) |
| `--mm-shm-cache-max-object-size-mb` | `128` | 共享内存缓存单对象大小限制 (MiB) |
| `--mm-encoder-only` | `False` | 跳过语言模型组件（用于分离式编码器进程） |
| `--mm-encoder-tp-mode` | `weights` | 多模态编码器 TP 模式 (weights/data) |
| `--mm-encoder-attn-backend` | - | 多模态编码器注意力后端 |
| `--interleave-mm-strings` | `False` | 启用多模态提示的完全交错支持 |
| `--skip-mm-profiling` | `False` | 跳过多模态内存分析 |
| `--video-pruning-rate` | - | 视频剪枝率 [0, 1)，用于高效视频采样 |

---

## 10. LoRA 配置 (LoRAConfig)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--enable-lora` | - | 启用 LoRA 适配器 |
| `--max-loras` | `1` | 单批次最大 LoRA 数 |
| `--max-lora-rank` | `16` | 最大 LoRA 秩 (1/8/16/32/64/128/256/320/512) |
| `--lora-dtype` | `auto` | LoRA 数据类型 |
| `--enable-tower-connector-lora` | `False` | 多模态模型的视觉编码器/连接器 LoRA（实验性） |
| `--max-cpu-loras` | - | CPU 内存中最大 LoRA 数（≥ max_loras） |
| `--fully-sharded-loras` | `False` | 完全分片 LoRA 计算 |
| `--default-mm-loras` | - | 多模态默认 LoRA 映射 (JSON) |
| `--specialize-active-lora` | `False` | 按活跃 LoRA 数构建 kernel grid |

---

## 11. 可观测性配置 (ObservabilityConfig)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--show-hidden-metrics-for-version` | - | 启用指定版本以来隐藏的已弃用 Prometheus 指标 |
| `--otlp-traces-endpoint` | - | OpenTelemetry traces 目标 URL |
| `--collect-detailed-traces` | - | 收集详细 traces (all/model/worker 及组合) |
| `--kv-cache-metrics` | `False` | 启用 KV 缓存驻留指标 |
| `--kv-cache-metrics-sample` | `0.01` | KV 缓存指标采样率 |
| `--cudagraph-metrics` | `False` | 启用 CUDA graph 指标 |
| `--enable-layerwise-nvtx-tracing` | `False` | 启用逐层 NVTX 追踪 |
| `--enable-mfu-metrics` | `False` | 启用 MFU 指标 |
| `--enable-logging-iteration-details` | `False` | 启用迭代详情日志 |

---

## 12. 调度配置 (SchedulerConfig)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--max-num-batched-tokens` | - | 单次迭代最大处理 token 数，支持 `1k`/`1K` 格式 |
| `--max-num-seqs` | - | 单次迭代最大序列数 |
| `--max-num-partial-prefills` | `1` | 分块预填充时最大并发部分预填充序列数 |
| `--max-long-partial-prefills` | `1` | 长提示并发预填充数上限 |
| `--long-prefill-token-threshold` | `0` | 长请求 token 阈值 |
| `--scheduling-policy` | `fcfs` | 调度策略 (fcfs: 先到先服务 / priority: 优先级) |
| `--enable-chunked-prefill` | - | 启用分块预填充 |
| `--disable-chunked-mm-input` | `False` | 禁止部分调度多模态项 |
| `--scheduler-cls` | - | 自定义调度器类 |
| `--disable-hybrid-kv-cache-manager` | - | 禁用混合 KV 缓存管理器 |
| `--async-scheduling` | - | 异步调度（减少 GPU 空闲间隙） |
| `--stream-interval` | `1` | 流式输出间隔（token 数） |

---

## 13. 编译配置 (CompilationConfig)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--cudagraph-capture-sizes` | - | CUDA graph 捕获大小列表，None 时自动推断 |
| `--max-cudagraph-capture-size` | - | 最大 CUDA graph 捕获大小，默认 min(max_num_seqs*2, 512) |

---

## 14. Kernel 配置 (KernelConfig)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--enable-flashinfer-autotune` | - | 启用 FlashInfer 自动调优 |

---

## 15. 高级组合配置 (VllmConfig)

这些参数接受完整的 JSON 配置对象，用于高级场景。

| 参数 | 说明 |
|------|------|
| `--speculative-config` | 推测解码配置 (JSON) |
| `--kv-transfer-config` | 分布式 KV 缓存传输配置 (JSON) |
| `--kv-events-config` | 事件发布配置 (JSON) |
| `--ec-transfer-config` | 分布式 EC 缓存传输配置 (JSON) |
| `--compilation-config`, `-cc` | torch.compile 和 CUDA graph 捕获配置 (JSON)，支持 `-cc.mode=3` 简写 |
| `--attention-config`, `-ac` | 注意力配置 (JSON) |
| `--kernel-config` | Kernel 配置 (JSON) |
| `--additional-config` | 平台特定附加配置 (JSON) |
| `--structured-outputs-config` | 结构化输出配置 (JSON) |
| `--profiler-config` | 性能分析配置 (JSON) |
| `--weight-transfer-config` | RL 训练时的权重传输配置 (JSON) |
| `--optimization-level` | 优化级别，默认 `2`。-O0 启动快，-O3 性能最佳 |

---

## 常用示例

```bash
# 基础启动
vllm serve Qwen/Qwen2.5-7B-Instruct

# 指定端口和 GPU 显存利用率
vllm serve meta-llama/Llama-3-8B-Instruct --port 8080 --gpu-memory-utilization 0.8

# 多 GPU 张量并行
vllm serve meta-llama/Llama-3-70B-Instruct --tensor-parallel-size 4

# 启用量化
vllm serve TheBloke/Llama-2-7B-Chat-AWQ -q awq --dtype half

# 使用配置文件
vllm serve my-model --config config.yaml

# 启用工具调用
vllm serve model-name --enable-auto-tool-choice --tool-call-parser hermes

# 数据并行
vllm serve model-name --data-parallel-size 2 --tensor-parallel-size 2

# 设置上下文长度和分块预填充
vllm serve model-name --max-model-len 32K --enable-chunked-prefill
```
