# Transformers 学习目录

## 📁 目录结构

```
transformers-learning/
├── README.md                           # 本文件
├── TRANSFORMERS_LEARNING_GUIDE.md      # 完整学习指南
├── chapter01_quickstart/               # 第1章：快速入门
├── chapter02_pipeline/                 # 第2章：Pipeline 推理
├── chapter03_pretrained_models/        # 第3章：预训练模型加载
├── chapter04_trainer/                  # 第4章：Trainer 训练
├── chapter05_advanced/                 # 第5章：高级应用
├── projects/                           # 实战项目
└── examples/                           # 示例代码
```

## 🎯 学习路径

### 第1章：快速入门 (chapter01_quickstart)
- 环境配置
- 第一个 Pipeline 示例
- 基本概念理解

### 第2章：Pipeline 推理 (chapter02_pipeline)
- Pipeline 基本用法
- 支持的任务类型
- 性能优化技巧

### 第3章：预训练模型加载 (chapter03_pretrained_models)
- AutoClass API
- 模型加载参数
- 模型推理实践

### 第4章：Trainer 训练 (chapter04_trainer)
- Trainer 基本用法
- 训练参数配置
- 模型微调实战

### 第5章：高级应用 (chapter05_advanced)
- 量化技术
- 分布式训练
- 多模态任务

## 📚 开始学习

1. **阅读学习指南**
   ```bash
   # 打开学习指南
   code TRANSFORMERS_LEARNING_GUIDE.md
   ```

2. **按章节学习**
   - 从 chapter01 开始
   - 每章包含学习笔记和代码示例
   - 完成每章的练习

3. **实战项目**
   - 在 projects/ 目录下创建项目
   - 应用所学知识
   - 记录项目经验

## 🚀 快速开始

安装依赖：
```bash
pip install transformers datasets evaluate accelerate
```

运行第一个示例：
```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("I love learning Transformers!")
print(result)
```

## 📖 学习建议

- **循序渐进**：按章节顺序学习
- **动手实践**：每章都要写代码
- **记录笔记**：在各章节目录下记录学习心得
- **完成项目**：通过项目巩固知识

---

**祝学习顺利！🎓**
