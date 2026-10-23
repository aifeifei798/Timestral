"""模型加载与推理封装：单例 engine + 互斥锁（单卡单模型，串行推理）。"""

from __future__ import annotations

import math
import threading
import time
from pathlib import Path

import numpy as np
import torch

from .schemas import ForecastOptions, SeriesForecast, SeriesSpec, VariateForecast

DEFAULT_MODEL_PATH = "/home/feifei/develop/models/Timestral-209M"
MAX_CONTEXT = 32768
DEFAULT_MAX_HORIZON = 4096


def _to_nan_array(values, shape_desc: str) -> np.ndarray:
    """list[float|None]（或其嵌套）-> float32 ndarray，None 转 NaN。"""
    if not values:
        raise ValueError(f"序列为空，期望 {shape_desc}")
    if isinstance(values[0], list):
        if not values[0]:
            raise ValueError(f"序列为空，期望 {shape_desc}")
        arr = np.array(
            [[np.nan if v is None else v for v in row] for row in values],
            dtype=np.float32,
        )  # (V, T)
    else:
        arr = np.array(
            [np.nan if v is None else v for v in values], dtype=np.float32
        )  # (T,)
    if arr.ndim not in (1, 2):
        raise ValueError(f"期望 {shape_desc}，实际维度 {arr.ndim}")
    return arr


def _as_bool_mask(mask, ref_shape: tuple[int, ...]) -> np.ndarray:
    if mask is None:
        return np.ones(ref_shape, dtype=bool)
    arr = np.array(mask, dtype=bool)
    if arr.shape != ref_shape:
        raise ValueError(f"observation_mask 形状 {arr.shape} 与序列形状 {ref_shape} 不一致")
    return arr


def _finite(v: float) -> float | None:
    """NaN/Inf 在严格 JSON 里不合法，输出层统一转为 null。"""
    return float(v) if math.isfinite(v) else None


class TimestralEngine:
    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        device: str = "auto",
        max_horizon: int = DEFAULT_MAX_HORIZON,
    ) -> None:
        from timestral import TimestralForecaster  # 延迟导入，加快启动报错定位

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.set_float32_matmul_precision("high")

        self.model_path = str(model_path)
        self.device = str(device)
        self.max_horizon = max_horizon
        self.model = TimestralForecaster.from_pretrained(model_path, device=device)
        self.quantile_levels: tuple[float, ...] = tuple(self.model.quantile_levels)
        self._lock = threading.Lock()

        self.params_millions = round(
            sum(p.numel() for p in self.model.model.parameters()) / 1e6, 1
        )
        # 预热：触发 CUDA 内核编译/显存分配，避免第一个线上请求变慢
        t0 = time.perf_counter()
        warm = np.sin(np.arange(512, dtype=np.float32) / 10.0)
        self.model.forecast(warm, prediction_length=8, symmetric_averaging=False)
        self.warmup_ms = (time.perf_counter() - t0) * 1000

    # ------------------------------------------------------------------ 输入解析

    def _parse_spec(self, spec: SeriesSpec, options: ForecastOptions) -> dict:
        context = _to_nan_array(spec.series, "(T,) 或 (V, T)")
        t = context.shape[-1]
        if t < 1:
            raise ValueError("series 不能为空")
        if t > MAX_CONTEXT:
            raise ValueError(f"序列长度 {t} 超过模型上下文上限 {MAX_CONTEXT}")
        if options.prediction_length > self.max_horizon:
            raise ValueError(
                f"prediction_length {options.prediction_length} 超过上限 {self.max_horizon}"
            )

        bad = [q for q in options.quantiles if q not in self.quantile_levels]
        if bad:
            raise ValueError(f"不支持的分位数 {bad}，可选值：{list(self.quantile_levels)}")

        # 形状校验：库内部报错不友好，这里提前拦截
        for name, cov, expect_t in (
            ("past_covariates", spec.past_covariates, t),
            ("categorical_past_covariates", spec.categorical_past_covariates, t),
            ("known_covariates", spec.known_covariates, t + options.prediction_length),
            (
                "categorical_known_covariates",
                spec.categorical_known_covariates,
                t + options.prediction_length,
            ),
        ):
            if cov is not None:
                shape = np.array(cov, dtype=np.float32).shape
                if len(shape) != 2 or shape[-1] != expect_t:
                    raise ValueError(
                        f"{name} 应为 (C, {expect_t})（时间维含预测段），实际形状 {shape}"
                    )

        mask_ref = context.shape if context.ndim == 2 else (t,)
        return {
            "context": context,
            "observation_mask": _as_bool_mask(spec.observation_mask, mask_ref),
            "past_covariates": None
            if spec.past_covariates is None
            else _to_nan_array(spec.past_covariates, "(C, T)"),
            "known_covariates": None
            if spec.known_covariates is None
            else _to_nan_array(spec.known_covariates, "(C, T+H)"),
            "categorical_past_covariates": None
            if spec.categorical_past_covariates is None
            else _to_nan_array(spec.categorical_past_covariates, "(C, T)"),
            "categorical_known_covariates": None
            if spec.categorical_known_covariates is None
            else _to_nan_array(spec.categorical_known_covariates, "(C, T+H)"),
            "t": t,
        }

    @staticmethod
    def _group_key(parsed: dict, options: ForecastOptions) -> tuple:
        """可拼成同一 batch 的输入共享此 key。"""
        covs = tuple(
            parsed[k].shape if parsed[k] is not None else None
            for k in (
                "past_covariates",
                "known_covariates",
                "categorical_past_covariates",
                "categorical_known_covariates",
            )
        )
        return (parsed["context"].shape, covs)

    # ------------------------------------------------------------------ 推理

    def _call_model(
        self, batch: list[dict], options: ForecastOptions
    ) -> tuple[np.ndarray, np.ndarray]:
        """batch 内各元素形状一致；返回 (median, quantiles)，均含 batch 维。"""

        def stack(key: str):
            arrs = [b[key] for b in batch]
            return None if arrs[0] is None else np.stack(arrs)

        ctx = np.stack([b["context"] for b in batch])  # (B,T) 或 (B,V,T)
        with self._lock:  # 单 GPU 模型实例，串行化前向
            fc = self.model.forecast(
                ctx,
                prediction_length=options.prediction_length,
                observation_mask=stack("observation_mask"),
                past_covariates=stack("past_covariates"),
                known_covariates=stack("known_covariates"),
                categorical_past_covariates=stack("categorical_past_covariates"),
                categorical_known_covariates=stack("categorical_known_covariates"),
                symmetric_averaging=options.symmetric_averaging,
                clip_nonnegative=options.clip_nonnegative,
            )
            median = fc.median.cpu().numpy()  # (B,V,H) 或 (B,H)
            quantiles = fc.quantiles.cpu().numpy()  # (B,V,H,Q) 或 (B,H,Q)
        return median, quantiles

    def _split_forecast(
        self, median: np.ndarray, quantiles: np.ndarray,
        levels: list[float], context_length: int,
    ) -> list[SeriesForecast]:
        """median (V,H)（或 (H,) 单变量）、quantiles (V,H,Q)（或 (H,Q)）-> 按变体拆分。"""
        if median.ndim == 1:
            median, quantiles = median[None], quantiles[None]
        return [
            SeriesForecast(
                context_length=context_length,
                num_variates=int(median.shape[0]),
                variates=[
                    VariateForecast(
                        variate_index=v,
                        median=[_finite(x) for x in median[v]],
                        quantiles={
                            str(l): [_finite(x) for x in quantiles[v][:, i]]
                            for i, l in enumerate(levels)
                        },
                    )
                    for v in range(median.shape[0])
                ],
            )
        ]

    def run_single(
        self, spec: SeriesSpec, options: ForecastOptions
    ) -> tuple[SeriesForecast, float]:
        parsed = self._parse_spec(spec, options)
        levels = list(options.quantiles)
        idx = [self.quantile_levels.index(q) for q in levels]

        t0 = time.perf_counter()
        median, quantiles = self._call_model([parsed], options)
        elapsed = (time.perf_counter() - t0) * 1000
        # batch 维=1：median (1,V,H) -> (V,H)
        return self._split_forecast(
            median[0], quantiles[0][..., idx], levels, parsed["t"]
        )[0], elapsed

    def run_batch(
        self, specs: list[SeriesSpec], options: ForecastOptions
    ) -> tuple[list[SeriesForecast], float]:
        parsed_list = [self._parse_spec(s, options) for s in specs]
        levels = list(options.quantiles)
        idx = [self.quantile_levels.index(q) for q in levels]

        groups: dict[tuple, list[int]] = {}
        for i, parsed in enumerate(parsed_list):
            groups.setdefault(self._group_key(parsed, options), []).append(i)

        results: dict[int, SeriesForecast] = {}
        t0 = time.perf_counter()
        for members in groups.values():
            median, quantiles = self._call_model(
                [parsed_list[i] for i in members], options
            )
            for pos, i in enumerate(members):
                (results[i],) = self._split_forecast(
                    median[pos],
                    quantiles[pos][..., idx],
                    levels,
                    parsed_list[i]["t"],
                )
        elapsed = (time.perf_counter() - t0) * 1000
        return [results[i] for i in range(len(specs))], elapsed

    # ------------------------------------------------------------------ 健康信息

    def health(self) -> dict:
        gpu_mb = None
        if self.device.startswith("cuda") and torch.cuda.is_available():
            gpu_mb = round(torch.cuda.memory_allocated() / 1e6, 1)
        cfg = self.model.config
        return {
            "status": "ok",
            "model": Path(self.model_path).name,
            "device": self.device,
            "params_millions": self.params_millions,
            "max_context": int(cfg.context_length),
            "max_single_shot_horizon": int(cfg.block_size * cfg.patch_size),
            "quantile_levels": list(self.quantile_levels),
            "supports_covariates": bool(self.model.supports_covariates),
            "gpu_memory_allocated_mb": gpu_mb,
        }
