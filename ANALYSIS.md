# Timestral-209M 模型分析

> 分析对象：`/home/feifei/develop/models/Timestral-209M`（来自 Mistral AI，
> [HF: mistralai/Timestral-209M](https://huggingface.co/mistralai/Timestral-209M)）

## 1. 模型定位

Timestral 是 Mistral AI 于 2025 年发布的**时间序列预测基础模型**（time series foundation
model），面向**零样本预测**（zero-shot forecasting）：无需在目标数据上训练，直接对任意
时间序列做概率预测。本仓库是家族中最大的 **209M** 参数版本（另有 5M / 28M / 123M）。

在三大公开基准上为当时的零样本 SOTA：

| 基准 | 指标 | Timestral-209M | 次优 |
|---|---|---|---|
| GIFT-Eval | MASE ↓ / CRPS ↓ | **0.658 / 0.446** | t0-gamma 0.661 / 0.452 |
| TIME | MASE ↓ / CRPS ↓ | **0.626 / 0.522** | t0-gamma 0.637 / 0.534 |
| fev-bench | SQL/MASE skill ↑ | **50.39 / 39.49** | TimesFM-3 48.99 / 37.99 |

## 2. 架构（config.json + 权重张量实证）

核心是 **patch-based transformer，同时使用时间注意力与变量注意力**（类似 iTransformer /
Moirai 的双轴注意力思路），权重张量结构如下：

| 组件 | 张量（模式） | 形状 | 说明 |
|---|---|---|---|
| Patch 嵌入 | `representation.patch_proj.{hidden,output,residual}_layer` | 64→3072→768 | 输入 64 = patch_size 32 × [值 + 缺失指示位]，两层 MLP + 残差 |
| 预测 query | `representation.forecast_query` | (768,) | 单个可学习 query 向量，用于生成未来 patch |
| Transformer | `transformer.layers.{0..23}` | d_model=768, heads=12 | 24 层，pre-norm (RMSNorm, eps=1e-6) |
| 注意力 | 每层 `attns.N.{wQKV,wO,_gate}` | 2304→768 / 768→768 | 融合 QKV 投影；`_gate` 为 sigmoid 门控（config `*_attn_gate: sdpa_sigmoid`），共 30 个注意力模块 |
| MLP | `mlp.w12` (4096,768) + `mlp.w3` (768,2048) | — | SwiGLU 门控 MLP（w12 = gate+up 2×2048） |
| 输出头 | `head.projection.{hidden,output,residual}` | 768→3072→672 | 输出 672 = 32 位置 × 21 分位数 |

- **block_schedule**：24 层的注意力布局为 18 层 `time` + 6 层 `time_variate`
  （每 4 层插入一个带变量注意力的层；18×1 + 6×2 = 30 与权重计数吻合）。
- **partial RoPE**：`rope_theta=10000`，`rope_partial_factor=0.5`（仅一半 head 维做旋转位置编码）。
- **参数量**：实测 `sum(p.numel())` = **209.3M**。

## 3. 能力边界（README + config）

| 维度 | 规格 |
|---|---|
| 上下文长度 | 最长 **32,768** 步（1,024 个 patch） |
| 单次预测 | 最长 **1,024** 步（32 个 patch）；更长则分块滚动，把每块中位数拼回上下文 |
| 输出 | **21 个分位数**（0.01, 0.05, 0.1, …, 0.9, 0.95, 0.99），`quantile_width_factor=3.0` |
| 输入 | 单变量或多变量目标；数值/类别协变量（过去型、已知未来型；类别做 target-encoding） |
| 缺失值 | 直接用 NaN 标记，或配 observation_mask；`attend_missing_patches=true` 表示缺失 patch 仍参与注意力 |
| 数值稳定 | `use_asinh=true` 逐 patch 缩放（`minimum_scale=1e-6`），对量纲不敏感 |
| 后处理 | `symmetric_averaging`（镜像对称平均，约 2 倍算力）、`clip_nonnegative`（非负序列截断） |

## 4. 推理 API（timestral 包实测）

```python
from timestral import TimestralForecaster
model = TimestralForecaster.from_pretrained(path, device="cuda")
fc = model.forecast(
    context,                # (T,) / (B,T) / (B,V,T)，NaN=缺失
    prediction_length=96,   # >=1，超 1024 自动分块
    observation_mask=...,   # 可选，与 context 同形，False=缺失
    past_covariates=...,    # (B,C,T)
    known_covariates=...,   # (B,C,T+H)
    categorical_past_covariates=..., categorical_known_covariates=...,
    symmetric_averaging=True, clip_nonnegative=True,
)
fc.quantiles  # (B,V,H,21)   fc.median -> (B,V,H)   fc.quantile(q) -> (B,V,H)
```

## 5. 实测性能（本机 RTX 5090 D 32GB，fp32 + TF32）

| 项目 | 数值 |
|---|---|
| 模型加载 | ~1.3 s |
| 显存占用 | ~850 MB |
| 单序列 (T=600, H=48, 含对称平均) | ~57 ms |
| 多变量 (V=2, T=300, H=24) | ~48 ms |
| 批量 3 条（分 2 组） | ~68 ms |
| 已知协变量小时周期 (T=240, H=24) | ~27 ms，且准确延续 0→23 的日内周期 |

## 6. 小结

- 这是一个**即用型零样本预测模型**：不针对具体业务微调，直接喂历史序列即可拿到带
  21 分位区间的概率预测；多变量是"联合建模"而非逐列独立预测。
- 对输入不需要归一化/填补缺失（模型内置 asinh 缩放与缺失 patch 注意力），工程接入成本极低。
- 服务化要点：单模型单卡实例需串行化推理；`symmetric_averaging` 关掉可近似减半延迟；
  预测长度 >1024 会触发分块滚动，长 horizon 的延迟近似线性增长。
