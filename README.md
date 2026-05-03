# Fine-Tuning Recommender 验证实验

验证小规模 pilot run 是否能预测 larger-budget run 的 LoRA rank ranking，从而支撑 fine-tuning recommender 的小规模试运行策略。

## 实验矩阵

- 3 任务: BANKING77（intent classification）、Bitext customer support（generation）、CUAD（legal clause classification）
- 4 PEFT 方法: LoRA / DoRA / QLoRA / LoRA+（统一 r=8, alpha=16；LoRA+ lr_ratio=16）
- 2 训练规模: pilot run / larger-budget run
- 单 seed: 42
- 共 24 runs，单卡 24GB GPU 上约 3-5 小时

base model: `Qwen/Qwen2.5-1.5B-Instruct`

**注意**: QLoRA 依赖 bitsandbytes 4-bit kernel,只能在 CUDA GPU 上跑;Mac MPS 上 smoke test 会自动 skip qlora。

## 本地准备（在 Mac 上跑一次）

```bash
pip install -r requirements.txt
python scripts/01_download_model.py
python scripts/02_prepare_all_data.py
python scripts/03_smoke_test.py
```

## 云端跑实验

```bash
# 同步项目目录到 GPU 实例后:
python -m venv .venv && .venv/bin/pip install -r requirements.txt
# 验证 bitsandbytes 装好(QLoRA 需要):
.venv/bin/python -c "import bitsandbytes; print(bitsandbytes.__version__)"

export ANTHROPIC_API_KEY=sk-ant-...
.venv/bin/python scripts/run_matrix.py
.venv/bin/python scripts/analyze_results.py
```

结果与图表落在 `outputs/aggregated/` 与 `outputs/figures/`。

跑某一个方法或任务做调试:

```bash
.venv/bin/python scripts/run_matrix.py --only-task banking77 --only-method lora
```

## 目录速查

- `configs/matrix.yaml` — 24 个 run 的笛卡尔积定义
- `configs/tasks/*.yaml` — 三个任务各自的 prompt / metric / 训练超参
- `src/training/run_one.py` — 单 run 入口（被 run_matrix.py subprocess 调用）
- `src/analysis/` — composite score、ranking 一致性、绘图
- `outputs/aggregated/all_runs.jsonl` — 每行一个 run 的完整指标记录
