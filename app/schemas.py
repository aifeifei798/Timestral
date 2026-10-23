"""Pydantic request/response schemas for the Timestral forecasting API.

约定与 timestral 库一致：序列按行堆叠，即二维输入 shape 为 (变体数 V, 时间步 T)。
JSON 中的 null 表示缺失值（内部转为 NaN）。
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

FloatOrNone = Annotated[Union[float, None], Field(description="数值，null 表示缺失")]


class ForecastOptions(BaseModel):
    """模型推理选项（对整个请求生效）。"""

    prediction_length: int = Field(
        ge=1, description="预测步数（单次前向最多 1024 步，更长会分块滚动预测）"
    )
    quantiles: list[float] = Field(
        default=[0.1, 0.5, 0.9],
        description="要返回的分位数水平，必须是模型 21 个分位节点 (0.01..0.99) 的子集",
    )
    symmetric_averaging: bool = Field(
        default=True,
        description="用 -context 的镜像预测做对称平均（更稳但约 2 倍计算）",
    )
    clip_nonnegative: bool = Field(
        default=True, description="对历史上全部非负的序列，把预测截断到 >= 0"
    )


class Covariates(BaseModel):
    past_covariates: list[list[FloatOrNone]] | None = Field(
        default=None, description="仅在历史上可用的数值协变量，shape (C_past, T)"
    )
    known_covariates: list[list[FloatOrNone]] | None = Field(
        default=None,
        description="历史+未来都已知的数值协变量，shape (C_known, T + prediction_length)",
    )
    categorical_past_covariates: list[list[FloatOrNone]] | None = Field(
        default=None, description="历史类别协变量的类别编码，shape (C, T)"
    )
    categorical_known_covariates: list[list[FloatOrNone]] | None = Field(
        default=None, description="已知类别协变量的类别编码，shape (C, T + prediction_length)"
    )


class SeriesSpec(Covariates):
    """一条待预测序列：一维 list = 单变量；二维 list[list] = 多变量（行=变体）。"""

    series: list[FloatOrNone] | list[list[FloatOrNone]] = Field(
        description="历史观测，shape (T,) 或 (V, T)，最多 32768 步，缺失用 null",
    )
    observation_mask: list[bool] | list[list[bool]] | None = Field(
        default=None, description="与 series 同形；False 表示视为缺失，缺省全部有效"
    )


class ForecastRequest(ForecastOptions, SeriesSpec):
    """单序列预测请求 = 推理选项 + 序列数据。"""


class VariateForecast(BaseModel):
    variate_index: int = Field(description="变体在输入序列中的行索引")
    median: list[Union[float, None]] = Field(description="中位数预测，长度 = prediction_length")
    quantiles: dict[str, list[Union[float, None]]] = Field(
        description="各请求分位数的预测序列，key 为分位数水平的字符串"
    )


class SeriesForecast(BaseModel):
    context_length: int = Field(description="输入序列的时间步数")
    num_variates: int
    variates: list[VariateForecast]


class ForecastResponse(BaseModel):
    prediction_length: int
    quantile_levels: list[float]
    forecast: SeriesForecast
    inference_ms: float
    model: str


class BatchForecastRequest(ForecastOptions):
    """批量预测：输入形状相同的序列拼成 batch 一次前向，不同形状自动分组。"""

    requests: list[SeriesSpec] = Field(min_length=1, max_length=256)


class BatchForecastResponse(BaseModel):
    forecasts: list[SeriesForecast] = Field(description="与请求顺序一一对应")
    quantile_levels: list[float]
    inference_ms: float
    model: str


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model: str
    device: str
    params_millions: float
    max_context: int
    max_single_shot_horizon: int
    quantile_levels: list[float]
    supports_covariates: bool
    gpu_memory_allocated_mb: float | None = None
