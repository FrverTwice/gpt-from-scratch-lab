# gpt-from-scratch-lab

# 🧠 Build Your Own GPT — 从零实现一个可训练的 GPT 模型

本仓库记录了从 **0 到 1 构建 GPT 模型的完整过程**，包括：

- 实现 GPT 的基础组件（Token Embedding、Positional Embedding、Multi-Head Attention、FeedForward、Block 等）
- 自定义 Dataset（JSONL 格式、分块、token 积累、shift-label）
- 训练循环（optimizer、scheduler、loss 可视化）
- 自回归生成 generate() 实现
- 推理示例（输入 prompt → 输出模型续写）

整个项目旨在帮助你 **彻底理解 GPT2 的核心结构与训练方式**，并能在此基础上扩展更强大的模型。

项目主要参考 Chao Fa 老师的上课视频，在实现代码的同时加入了一些自己的理解
https://www.bilibili.com/video/BV1ZbFpeHEYr



***

## 📂 项目结构

如果需要复线这个代码，您可以参考下面这个目录：

.
├── gpt2_from_scratch.ipynb      # 主文件：从零开始构建 GPT 的完整 Notebook
├── data/                        # 存储训练数据
├── checkpoints/                 # 保存的模型 checkpoint（训练自动生成）
└── README.md                    # 本说明文档

***

## Nano GPT
参考GPT2的文章，实现了一个简单的GPT模型，具体实现在`gpt2_from_scratch.ipynb`中



### ✔ 1. 从零开始构建 GPT 的所有模块  
Notebook 中包含对每个模块的逐步构建：

- `nn.Embedding` 实现 Token Embedding & 位置编码
- 可训练 Position Embedding（非 sin/cos）
- Single-Head → Multi-Head Attention
- FeedForward（MLP）
- 残差连接 & LayerNorm
- Block 结构（Transformer Decoder Block）
- 最终的 GPT 模型封装

### ✔ 2. 自定义数据集（JSONL → Token → Blocks）  
实现了完整的数据处理流程：

- JSONL 文本文件逐行读取（节省内存）
- 使用 tiktoken(gpt2) tokenizer
- 拼接所有文本并加入 `<|endoftext|>` token
- 按 block_size 切片，每片 513（含 shift-label）
- 构建 (input, target) 对

### ✔ 3. 完整训练循环  
包括：

- AdamW 优化器
- 线性 warmup + decay Scheduler
- 训练 Loss
- 验证 Loss
- Step Loss 可视化（Matplotlib）

### ✔ 4. GPT 生成函数 generate()  
实现自回归生成逻辑：

- 限制 block_size（避免超过上下文窗口）
- 取最后一个 token 的 logits
- softmax 采样下一个 token
- 拼接输入继续生成

支持多条 prompt 批量生成。

### ✔ 5. 推理 Demo  
训练完后，你可以这样生成中文续写：

```python
prompt = "从前有座山，山里有座庙，庙里有个老和尚对小和尚说："
idx = torch.tensor([enc.encode(prompt)], device=device)

with torch.no_grad():
    out = model.generate(idx, max_new_tokens=50)

print(enc.decode(out[0].tolist()))
```
***

## 📜 许可证

MIT License，欢迎学习、使用、修改。

