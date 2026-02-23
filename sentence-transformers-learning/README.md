# Sentence Transformers 学习指南

官方文档：https://sbert.net/

## 一、入门指南 (Getting Started)

| 文档 | 链接 | 说明 |
|------|------|------|
| Installation | https://sbert.net/docs/installation.html | 安装配置 |
| Quickstart | https://sbert.net/docs/quickstart.html | 快速入门，包含三种模型的基本用法 |
| Migration Guide | https://sbert.net/docs/migration_guide.html | 从旧版本迁移 |

## 二、Sentence Transformer（嵌入模型）

用于将文本转换为向量表示，适用于语义搜索、文本相似度计算等场景。

| 文档 | 链接 | 说明 |
|------|------|------|
| Usage | https://sbert.net/docs/sentence_transformer/usage/usage.html | 使用方法 |
| Pretrained Models | https://sbert.net/docs/sentence_transformer/pretrained_models.html | 可用的预训练模型列表 |
| Training Overview | https://sbert.net/docs/sentence_transformer/training_overview.html | 如何训练/微调模型 |
| Dataset Overview | https://sbert.net/docs/sentence_transformer/dataset_overview.html | 训练数据集说明 |
| Loss Overview | https://sbert.net/docs/sentence_transformer/loss_overview.html | 损失函数选择指南 |
| Training Examples | https://sbert.net/docs/sentence_transformer/training/examples.html | 训练示例代码 |
| Speeding up Inference | https://sbert.net/docs/sentence_transformer/usage/efficiency.html | 推理加速技巧 |

## 三、Cross Encoder（重排序模型）

用于对文本对进行精确打分，常用于搜索结果重排序。

| 文档 | 链接 | 说明 |
|------|------|------|
| Usage | https://sbert.net/docs/cross_encoder/usage/usage.html | 使用方法 |
| Pretrained Models | https://sbert.net/docs/cross_encoder/pretrained_models.html | 可用的预训练模型列表 |
| Training Overview | https://sbert.net/docs/cross_encoder/training_overview.html | 如何训练/微调模型 |
| Loss Overview | https://sbert.net/docs/cross_encoder/loss_overview.html | 损失函数选择指南 |
| Training Examples | https://sbert.net/docs/cross_encoder/training/examples.html | 训练示例代码 |
| Speeding up Inference | https://sbert.net/docs/cross_encoder/usage/efficiency.html | 推理加速技巧 |

## 四、Sparse Encoder（稀疏编码模型）

用于生成稀疏向量表示，适合与传统搜索引擎结合使用。

| 文档 | 链接 | 说明 |
|------|------|------|
| Usage | https://sbert.net/docs/sparse_encoder/usage/usage.html | 使用方法 |
| Pretrained Models | https://sbert.net/docs/sparse_encoder/pretrained_models.html | 可用的预训练模型列表 |
| Training Overview | https://sbert.net/docs/sparse_encoder/training_overview.html | 如何训练/微调模型 |
| Loss Overview | https://sbert.net/docs/sparse_encoder/loss_overview.html | 损失函数选择指南 |
| Training Examples | https://sbert.net/docs/sparse_encoder/training/examples.html | 训练示例代码 |
| Speeding up Inference | https://sbert.net/docs/sparse_encoder/usage/efficiency.html | 推理加速技巧 |
| Vector Database Integration | https://sbert.net/examples/sparse_encoder/applications/semantic_search/README.html#vector-database-search | 向量数据库集成 |

## 五、API 参考 (Package Reference)

| 文档 | 链接 |
|------|------|
| Sentence Transformer API | https://sbert.net/docs/package_reference/sentence_transformer/index.html |
| Cross Encoder API | https://sbert.net/docs/package_reference/cross_encoder/index.html |
| Sparse Encoder API | https://sbert.net/docs/package_reference/sparse_encoder/index.html |
| util 工具函数 | https://sbert.net/docs/package_reference/util.html |

## 六、建议学习顺序

1. **Installation** → 安装 sentence-transformers
2. **Quickstart** → 了解三种模型的基本用法
3. **Sentence Transformer > Usage** → 深入学习嵌入模型
4. **Sentence Transformer > Pretrained Models** → 选择合适的模型
5. **Cross Encoder > Usage** → 学习重排序模型
6. **Training Overview** → 学习如何微调自己的模型
