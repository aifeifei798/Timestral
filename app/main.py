"""Timestral-209M 时间序列预测 FastAPI 服务。

启动:
    .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000

环境变量:
    TIMESTRAL_MODEL_PATH    模型目录（默认 /home/feifei/develop/models/Timestral-209M）
    TIMESTRAL_DEVICE        auto / cuda / cpu（默认 auto）
    TIMESTRAL_MAX_HORIZON   单次请求最大预测步数（默认 4096）
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .engine import DEFAULT_MODEL_PATH, TimestralEngine
from .schemas import (
    BatchForecastRequest,
    BatchForecastResponse,
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
)

engine: TimestralEngine | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global engine
    engine = TimestralEngine(
        model_path=os.environ.get("TIMESTRAL_MODEL_PATH", DEFAULT_MODEL_PATH),
        device=os.environ.get("TIMESTRAL_DEVICE", "auto"),
        max_horizon=int(os.environ.get("TIMESTRAL_MAX_HORIZON", "4096")),
    )
    print(
        f"[timestral-api] model={engine.model_path} device={engine.device} "
        f"params={engine.params_millions}M warmup={engine.warmup_ms:.0f}ms"
    )
    yield
    engine = None


app = FastAPI(
    title="Timestral-209M Forecasting API",
    description=(
        "Mistral Timestral-209M 时间序列基础模型的零样本预测服务："
        "支持单/多变量、缺失值、过去/已知协变量，输出 21 分位数概率预测。"
    ),
    version="1.0.0",
    lifespan=lifespan,
)


STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", include_in_schema=False)
def index():
    """预测控制台（Web 客户端）。"""
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    """服务与模型状态。"""
    assert engine is not None
    return engine.health()


@app.post("/forecast", response_model=ForecastResponse, tags=["forecast"])
def forecast(req: ForecastRequest):
    """对一条序列做概率预测。

    - `series` 一维 list 为单变量，二维 list[list] 为多变量（每行一个变体，联合预测）
    - `null` 表示缺失观测；可选 `observation_mask` 进一步标记缺失
    - 返回每个变体的 median 及所请求分位数的整条预测序列
    """
    assert engine is not None
    try:
        sf, elapsed = engine.run_single(req, req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:  # noqa: BLE001 模型内部异常统一 500
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}") from e
    return ForecastResponse(
        prediction_length=req.prediction_length,
        quantile_levels=req.quantiles,
        forecast=sf,
        inference_ms=round(elapsed, 1),
        model=engine.model_path.rsplit("/", 1)[-1],
    )


@app.post("/forecast/batch", response_model=BatchForecastResponse, tags=["forecast"])
def forecast_batch(req: BatchForecastRequest):
    """对一个 list 中的多条序列分别预测（形状相同的序列会合并成 batch 一次前向）。"""
    assert engine is not None
    try:
        forecasts, elapsed = engine.run_batch(req.requests, req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}") from e
    return BatchForecastResponse(
        forecasts=forecasts,
        quantile_levels=req.quantiles,
        inference_ms=round(elapsed, 1),
        model=engine.model_path.rsplit("/", 1)[-1],
    )
