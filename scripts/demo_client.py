#!/usr/bin/env python
"""Timestral API 演示客户端（仅用标准库）。

用法: python scripts/demo_client.py [host] [port]
"""

import json
import math
import sys
import urllib.request

BASE = f"http://{sys.argv[1] if len(sys.argv) > 1 else '127.0.0.1'}:{sys.argv[2] if len(sys.argv) > 2 else '8000'}"


def post(path: str, payload: dict):
    req = urllib.request.Request(
        BASE + path, json.dumps(payload).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


print("== /health ==")
with urllib.request.urlopen(BASE + "/health") as r:
    h = json.loads(r.read())
print(f"model={h['model']} device={h['device']} params={h['params_millions']}M")

# 单变量：正弦波 + 周期 24 的已知协变量（未来小时数）
series = [math.sin(i / 12) * 3 + 20 for i in range(336)]  # 两周 hourly
hour = [[float(i % 24) for i in range(336 + 48)]]  # 已知未来 48 小时
print("\n== /forecast (univariate + known covariate) ==")
r = post(
    "/forecast",
    {
        "series": series,
        "known_covariates": hour,
        "prediction_length": 48,
        "quantiles": [0.1, 0.5, 0.9],
    },
)
v0 = r["forecast"]["variates"][0]
print(f"inference: {r['inference_ms']}ms")
print(f"median[0:6]      = {[round(x, 2) for x in v0['median'][:6]]}")
print(f"q0.1 [0:6]       = {[round(x, 2) for x in v0['quantiles']['0.1'][:6]]}")
print(f"q0.9 [0:6]       = {[round(x, 2) for x in v0['quantiles']['0.9'][:6]]}")

# 多变量 + 缺失值(null)
print("\n== /forecast/batch (含一条多变量、带缺失) ==")
r = post(
    "/forecast/batch",
    {
        "prediction_length": 24,
        "requests": [
            {"series": [i * 0.5 for i in range(200)]},
            {
                "series": [
                    [None if i % 40 == 5 else math.cos(i / 8) for i in range(200)],
                    [i % 10 for i in range(200)],
                ]
            },
        ],
    },
)
for f, tag in zip(r["forecasts"], ["linear", "multivariate(2)"]):
    print(f"{tag}: context={f['context_length']} V={f['num_variates']} median[:4]={[round(x, 2) for x in f['variates'][0]['median'][:4]]}")
