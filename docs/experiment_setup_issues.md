# 实验搭建过程中遇到的问题(写报告 / 总结用)

本文档按"问题域"分类记录从项目搭建到云端开跑过程中遇到的全部技术问题、根因、修复办法、对应 commit。**用于复现性、踩坑警示、技术报告写作**。每条都按 `症状 → 根因 → 修复 → 经验教训` 的结构组织。

> **范围**:`ft-recommender-experiment/` 项目从 Initial Commit (`0142b17`) 到 QLoRA inference 修复 (`b4057aa`) 期间所有需要工程介入的问题,共 **14 项**。

---

## 一、数据准备阶段(3 个问题)

### 1. HuggingFace `datasets` 4.x 不支持 dataset scripts

**症状**:
```
RuntimeError: Dataset scripts are no longer supported, but found banking77.py
```

`scripts/02_prepare_all_data.py` 第一次运行就崩在 BANKING77 加载。

**根因**:
HuggingFace `datasets` 在 4.0 大版本里**移除了对 Python script-based datasets 的支持**(只接受纯 parquet / json / csv 在 hub 上的数据)。`PolyAI/banking77` / `theatticusproject/cuad-qa` 都依赖一个 `.py` loader script,新版 datasets 拒绝执行。

**修复**(commit 见数据 prep 相关初始 push):

`requirements.txt` 把 datasets 锁定到 3.x:
```
datasets>=3.0.0,<4.0.0
```

**经验教训**:
- HF 生态的 dataset 自带 loader script 是历史包袱,大半社区数据集都依赖。在 datasets 4.x 出来后,**始终在 requirements 里锁 `<4.0.0`** 直到上游集体迁移到纯 parquet。
- 报告里写 setup 时,**显式说明使用 `datasets==3.6.0`**,审稿人复现失败时第一反应就是版本问题。

---

### 2. `trust_remote_code=True` 是必需参数,不再有默认

**症状**:
```
ValueError: The repository for PolyAI/banking77 contains custom code which must
be executed to correctly load the dataset. Please pass the argument
`trust_remote_code=True`.
```

**根因**:
HF datasets 安全策略升级 — 任何依赖远程 .py 代码的 dataset 加载都强制要求显式同意,**不再静默允许**。

**修复**:
在三个 `prepare_*.py` 的 `load_dataset` 调用里加 `trust_remote_code=True`:
```python
ds = load_dataset(DATASET_ID, cache_dir="data/raw", trust_remote_code=True)
```

**经验教训**:
- `trust_remote_code` 不是可选的安全卫生 — 是 datasets 团队保护用户的硬卡口
- 在论文 / 报告里要**列出所有 `trust_remote_code=True` 的依赖**,这是合规审查会问到的
- 如果未来要做"零 remote code"的复现,需要把这三个数据集的 jsonl 直接打包进 repo(违反 size 限制)或用 raw GitHub 的下载链(脆弱)

---

### 3. CUAD 数据集需要 contract-level split 防泄露

**症状**:
没有出现错误,但**容易被忽略**的设计陷阱。如果按 clause-level random split,**同一份合同的不同 clause 同时出现在 train + test 中**,模型在测试集上看似性能极高,实际是数据泄漏。

**根因**:
CUAD 是 510 份合同的 ~13K clause 标注,同一合同内不同 clause 文风高度相似(同一律所、同一行业语言)。Random split 等于让模型在见过一部分 clause 后猜同合同剩下的 clause,严重过拟合。

**修复**:
`src/data/prepare_cuad.py` 实现 contract-level 70/15/15 split,记录 `split_manifest.json` 里每个 split 含哪些 contract_id,自动校验三个 split 互不相交:
```python
assert not (train_contracts & val_contracts)
assert not (train_contracts & test_contracts)
assert not (val_contracts & test_contracts)
```

**经验教训**:
- **CUAD 论文(Hendrycks et al. 2021)** 明确警告这个陷阱,但社区里**至少一半的 fine-tuning 实验照样 random split 拿到虚高分数**。如果你的 macro-F1 在 CUAD 上突然 > 0.85,**先怀疑 split**
- 任何 "实体—属性—值" 三元组的数据集都要查实体级泄漏(对话语料 → speaker level / 合同 → contract level / 推荐 → user level)
- 报告里 contract-level split 的细节是必写部分

---

## 二、训练框架兼容性(2 个问题)

### 4. trl 0.24 移除了 `dataset_text_field`,改用 `prompt+completion` / `messages`

**症状**(smoke 第一次跑):
```
KeyError: 'completion'
File "trl/trainer/sft_trainer.py", line 985, in tokenize_fn
    prompt_completion_ids = processing_class(
        text=example["prompt"] + example["completion"]
    )[
```

**根因**:
trl 0.24 重写了 `SFTTrainer` 的数据格式自动检测:
- 有 `prompt` + `completion` 列 → prompt-completion 模式(loss 只算 completion 部分)
- 有 `messages` 列 → conversational 模式(走 chat_template)
- 有 `text` 列 → text-only(loss 算全文,不再是默认)

我之前的 formatter 返回 `{"text": ..., "prompt": ..., "label": ...}`,trl 检测到 `prompt` 就走 prompt-completion 路径,但 "completion" key 不存在(我用了 `label`)→ KeyError。

**修复**:
- `src/data/formatters.py`:
  - classification 任务返回 `{"prompt": prompt + "\n", "completion": label}`
  - generation 任务返回 `{"messages": [...]}`
- `SFTConfig` 移除 `dataset_text_field="text"`(让 trl 自动检测)

**经验教训**:
- 大版本升级跨多 minor(0.10 → 0.24)文档变化不一定醒目;**永远在锁版本前先 smoke 一遍**
- 报告 / 复现说明里**写清 trl version + 用的是 prompt-completion 模式**,别人对着源码读起来会快得多

---

### 5. classification prompt-completion 边界让 trl 警告 tokenization mismatch

**症状**:
```
[RANK 0] Mismatch between tokenized prompt and the start of tokenized
prompt+completion. This may be due to unexpected tokenizer behavior,
whitespace issues, or special token handling.
```

trl 在每个样本上打这条警告。

**根因**:
trl 把 `prompt` 单独 tokenize 一次,把 `prompt + completion` 再 tokenize 一次,然后用第一次的 tokens 长度作 mask,要求第二次的前 N 个 token 跟第一次完全一致。**boundary 在合并后 tokenize 时偶尔被合并成新 token**,触发警告。

我原本的 formatter:
```python
return {"prompt": prompt, "completion": "\n" + label}
```
boundary 在 `prompt` 末尾的标点 + `\n`,某些 tokenizer 会合并 `<punctuation>\n` 为新 token。

**修复**:
把 `\n` 移到 prompt 末尾,completion 直接是 label:
```python
return {"prompt": prompt + "\n", "completion": label}
```
这样 prompt 单 tokenize 和 prompt+completion 合并 tokenize 在 prompt 末尾位置一致。

**经验教训**:
- prompt-completion mode 对 boundary 字符敏感,**永远把分隔字符放在 prompt 末尾**
- 这条警告是 yellow flag 不是 red flag(trl 仍能训练),但**没修会让 log 噪声大,也可能在某些 tokenizer 上变红 flag**

---

## 三、推理时的微妙 bug(2 个问题)

### 6. `predict_batch` 在 left-padded batch 里把 prompt 当作 completion 返回

**症状**:
classification eval 的 macro-F1 异常低 / 异常一致,与训练 loss 趋势脱节。

**根因(深坑)**:

原 `predict_batch` 用字符串相对位置提取 completion:
```python
input_len = enc["input_ids"][j].ne(pad_token_id).sum().item()
decoded_input = tokenizer.decode(enc[j][:input_len], skip_special=True)
full_decoded = tokenizer.decode(gen[j], skip_special=True)
if full_decoded.startswith(decoded_input):
    completion = full_decoded[len(decoded_input):]
```

**问题**:`tokenizer.padding_side = "left"`,real tokens 在序列**末尾**而非开头。`enc[j][:input_len]` 切的是**前** input_len 个 token,在 left padding 下:
- 短样本(N=180,L=360):`enc[j][:180]` = 180 个 PAD,decode → 空字符串
- 中样本(N=340,L=360):`enc[j][:340]` = 20 PAD + 320 real,decode → 部分 prompt

`full_decoded.startswith("")` 永远 True → completion = full_decoded(包含完整 prompt)。**label parser 在 prompt 的 label list 里乱匹配**,所有样本预测同一个长 label → macro-F1 接近 0。

**修复**(commit `b8516c0`):
改用 token-level slicing:
```python
prompt_len = enc["input_ids"].shape[1]  # padded length, same for all rows
completion_ids = gen[j][prompt_len:]
completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
```

**经验教训**:
- **left padding + batch generation 的标准做法是 token-level slice**,不要用 string-level startswith
- 这是个**沉默 bug**:测试不会失败(代码没崩),但实验结果整个失真。我自己的 smoke test 没抓到(因为 smoke 测的是单样本,不是 batch),需要看到具体 macro-F1 数字才能反推
- 写实验代码时,**每个推理函数加 unit test 要包括"3 个长度差很大的 prompt 在同一 batch"** 的 case

---

### 7. QLoRA 推理 dtype 不匹配(`bf16 hidden vs fp32 lm_head`)

**症状**:
```
RuntimeError: expected scalar type Float but found BFloat16
File "torch/nn/modules/linear.py", line 125, in forward
    return F.linear(input, self.weight, self.bias)
```

仅在 method=qlora 的 eval 路径触发,training 路径正常。

**根因**:
QLoRA 标准初始化:
```python
model = AutoModelForCausalLM.from_pretrained(..., quantization_config=bnb_4bit_nf4)
model = prepare_model_for_kbit_training(model)  # 关键
```

`prepare_model_for_kbit_training` **把所有非量化层 upcast 到 fp32**(为了梯度稳定):lm_head / norm / embed_tokens 都变 fp32。但 bnb 4-bit 模块在 forward 时输出 bf16 hidden(因为 `bnb_4bit_compute_dtype=bf16`)。

- **训练**:trl SFTTrainer 内部 `torch.autocast` 自动把 bf16 hidden cast 成 fp32 进 lm_head → 不崩
- **推理**:我的 `predict_batch` 没用 autocast → bf16 hidden 直接喂 fp32 lm_head → 崩

**修复**(commit `b4057aa`):
classification + generation 的 `predict_batch` 都包 autocast:
```python
with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                       enabled=use_autocast):
    gen = model.generate(...)
```

`enabled=use_autocast` 让非 CUDA 路径(MPS smoke)无副作用。LoRA / DoRA / LoRA+ 权重一直是 bf16,autocast 是 no-op,**修复对它们零影响**。

**经验教训**:
- QLoRA 推理需要显式 autocast,这是 Dettmers 论文实现里的隐含细节,大部分 tutorial 不写
- 训练通过 + 推理崩溃是 **PEFT bug 最常见的 signature** —— 训练框架自带 autocast,推理代码自己写就忘了
- **任何 QLoRA 的 README 必写**:"inference must be wrapped in `torch.autocast(dtype=bfloat16)`"

---

## 四、配置参数错误(1 个问题)

### 8. 三个 task 的 `max_length` 都设小了

**症状**:
没有报错,但 banking77 的 training prompt 实测 **404 tokens**,task config 设的是 256 → trl 训练时**截断 prompt 末尾**,模型在训练阶段从未看到 "Answer with only the label." 这条指令。

CUAD 长 clause 实测 533 tokens > 配置 512;Bitext response 实测 max ~500 tokens > 配置 512(prompt+response combined 经常溢出)。

**根因**:
我估算 prompt 长度时低估了 BANKING77 77-label list 的开销(实测 ~250 tokens 仅 label list)+ instruction(~50 tokens)+ user input(~50 tokens)= 350-400 tokens。CUAD 长 clause 输入可达 2000 字符(~500 tokens),也低估。

**修复**(commit `b8516c0`):
基于实测重新设定:
- banking77: 256 → **512**(prompt 404 + completion + buffer)
- cuad: 512 → **768**(max prompt 533 + buffer)
- bitext_support: 512 → **1024**(response p95 ~430,max ~580)

注释里写明实测数值。

**经验教训**:
- **永远跑实测 prompt token 长度**(在 prepare_data 之后、训练之前),用脚本一行算 p50/p95/max。我用 `tokenizer.encode(format_classification_prompt(...))` 的实际数字才发现问题
- max_length 设太小是**沉默 bug**:训练 loss 看不出明显异常(因为 completion 很短不被影响),但模型行为质量被悄悄拖低
- 报告里**写清每个 task 的 prompt 长度统计 + 选择的 max_length**,审稿人一眼就能验证合理性

---

## 五、工程基建(3 个问题)

### 9. `all_runs.jsonl` 重跑会 append 重复行,污染 composite score

**症状**:
分析 24 runs 的 ranking 时,如果用户重跑了某个 run(比如调参后),`all_runs.jsonl` 会有同一 (task, method, scale, seed) 的两行。`compute_composite` min-max normalize 时把这两行都算进去,等于一个 run 投了两票。

**根因**:
原始 `run_one.py` 用 `append_jsonl` 写记录,无去重。

**修复**(commit `b8516c0`):
两层防御:
1. `run_matrix.py` 启动前 warning 提醒已有 N 行
2. `analyze_results.py` 加 `(task, method, run_type, seed)` last-write-wins 去重

**经验教训**:
- jsonl append 是个**非常诱人但容易踩坑的设计模式**:看似日志友好,实际隐藏 dedup 责任。
- 修复架构上更干净的做法是 SQLite + UPSERT,但对 24 行数据用力过猛
- 报告里如果写 "we collected 24 measurements" 必须说明去重策略

---

### 10. 24 runs 没有醒目 summary,云端 stream log 难判断结果

**症状**:
跑 24 runs 时只能看到 trl 的训练 loss 进度条,每个 run 完成时 logger 只 print 一行 `[task/scale/method] DONE.`,关键指标(macro_f1 / memory / time)藏在 jsonl 里,**实时 debug 几乎不可能**。

**根因**:
原始 logging 设计偏向"机器可读 jsonl",忽略"人在 stream 看 log"的需求。

**修复**(commit `ecf3c92`):
- 每个 run 完成后 print 一段 fenced `=== RUN SUMMARY ===` block,8 维指标 + sanity 标记 (✓ / ⚠ / ✗) 基于经验阈值
- `run_matrix.py` 末尾打印总表(每行一个 run,task/scale/method × eval_q/loss/time/memory)
- 阈值在 `_sanity_quality` / `_sanity_loss` / `_sanity_memory` / `_sanity_overfit` 里集中,易调

**经验教训**:
- **任何会跑 > 30 分钟的脚本必须 print 醒目 summary**,不要只 dump jsonl
- 阈值要写"经验区间"而非硬截止,允许 ⚠ vs ✗ 分级
- 报告写法:把 sanity 表格直接 copy 到论文 appendix,自带可解释性

---

### 11. `run_matrix.py` 没有 resume / skip-completed

**症状**:
Colab 断线后用户被迫从 run 1 重跑全部 24 runs,**已完成的 12 runs 白浪费 1 小时**。

**根因**:
原始 `run_matrix.py` 是裸笛卡尔积循环,subprocess 跑每个 run 不检查 `all_runs.jsonl` 已有什么。

**修复**(commit `47c08e7`):
启动时读 `all_runs.jsonl`,build set of `(task, method, run_type, seed)` 已完成 keys,跳过这些 triple。新增 flags:
- `--force` / `--no-skip-completed`:强制全跑
- `--dry-run`:打印 will-skip 和 will-run 列表

**经验教训**:
- 任何"跑很多个独立 run"的脚本**必须自带 resume**,不能靠用户手工 `--only-X` 拼凑
- last-write-wins 是足够好的去重策略,不要过度设计 hash check
- 报告里"我们跑了 24 runs"背后的"跑了 24 runs 一遍"前提需要明示 — resume 机制让结果复现性提高

---

## 六、Colab 环境(4 个问题)

### 12. `outputs/` 在 Colab 临时盘,Runtime 重启后丢失

**症状**:
Colab session 闲置 90 分钟自动 disconnect,`/content/` 临时盘清空,所有跑过的 run 数据(jsonl + run dirs + metrics + figures)消失,**用户必须从头再跑**。

**根因**:
Colab `/content/` 是 ephemeral storage。`models/` 和 `data/processed/` 也在 `/content`,但相对小,重新下载只要 1-2 分钟;`outputs/` 装载的是几十 MB 的 metric jsonl 和图,但**每个 run 消耗 5-15 分钟训练**,丢失成本极高。

**修复**(commit `1c82387`):
Colab notebook 加 Cell 2.5,把 `outputs/` 软链到 Google Drive:
```python
drive_outputs = Path('/content/drive/MyDrive/ft-recommender-experiment-outputs')
project_outputs = Path('/content/ft-recommender-experiment/outputs')
project_outputs.symlink_to(drive_outputs, target_is_directory=True)
```
之后所有 jsonl / metrics 写入 直接落 Drive,断线重启 → 重新挂 Drive → 软链恢复 → resume 机制接续。

**经验教训**:
- Colab 上**任何长时间产出物必须挂 Drive**,这是 Colab 工程的硬铁律
- 软链优于"完成后 cp"模式 — 后者有"完成但还没 cp"的 race window
- 报告里说 "实验在 Colab L4 上跑" 时,**必须说明持久化方案**,否则审稿人合理质疑可复现性

---

### 13. Colab `torchvision` / `torch` ABI 冲突

**症状**:
```
RuntimeError: operator torchvision::nms does not exist
ModuleNotFoundError: Could not import module 'PreTrainedModel'
```

`pip install -q -r requirements.txt` 装完后 smoke test 立刻爆,看似简单的 import 都失败。

**根因**:
Colab 默认装 torchvision 2.x + torchaudio 2.x,**绑定它内置的 torch 2.6 版本**。我们 requirements 装 torch==2.8,**pip 不会自动升级 torchvision**(因为 requirements 里没列),留下 ABI 不匹配的 torchvision。

但 `transformers.image_utils` 触链 import torchvision → torchvision 内部 `torch.library.register_fake("torchvision::nms")` 调用与 torch 2.8 不兼容 → import 全链崩溃。peft 间接 import transformers,也崩。

**修复**(commit `76aa6d9`):
Cell 3 在 `pip install` 前**先卸载** torchvision + torchaudio:
```bash
pip uninstall -y torchvision torchaudio
pip install -q -r requirements.txt
```
本项目纯文本,根本不需要 vision 包。

**经验教训**:
- Colab 预装的 ML 栈是个"互相绑死"的版本组合,**要升级任何一个就要打破整个组合**
- "pip install 不会卸载冲突依赖" 是 pip 的语义,不是 bug
- 修复后**强烈建议 Runtime → Restart session**:即使 pip 装完新 torch,**当前 session 内存里仍是旧 torch import**,新代码加载时混用会再崩
- 报告里"Colab L4"setup 必须详细到这一步,否则别人复现 100% 踩同坑

---

### 14. Colab `Runtime → Restart session` 后 cwd 丢失,`!python -m src...` 失败

**症状**:
```
ModuleNotFoundError: No module named 'src'
```

`%cd` magic 在 Restart 之前是有效的,但重启后 cwd 回到默认 `/content`,后续 `!python -m src.training.run_one` 找不到 src 包。

**根因**:
`%cd` 是 IPython 进程内状态,Restart session 杀进程重启,状态全丢。用户重启后跳过 Cell 2 直接跑 Cell 7,cwd 还在 `/content`。

**修复**(commit `ff33e8c`):
所有跑 `!python ...` 的 cell 顶部加 `%cd /content/ft-recommender-experiment`(idempotent 安全)。共 7 个 cell 需要打补丁。

**经验教训**:
- IPython magic 的状态生命周期 = 进程生命周期,不要假设跨 Restart 持久
- **每个 `!shell` 命令都应该是 self-contained**,不依赖外部 cwd
- 这种坑只能在"模拟用户中途重启"时才能发现 — happy path 测试不会抓到

---

## 七、安全 / 流程(1 个问题)

### 15. 把 LLM Judge 留在云端跑,会暴露 API key

**症状**(被动发现):
原架构是 `evaluate_generation` 在云端跑训练时直接调 Anthropic API。这意味着:
- Colab notebook 必须 export `ANTHROPIC_API_KEY`(从 Colab Secrets 读)
- API key 暴露在 Colab session 环境里
- 任何 share 出去的 notebook(教授看代码 demo)都要小心 key 不被无意泄露

更糟糕的是用户在对话里贴了一个 OpenRouter key 想让我"帮你查余额",**这种 key 在对话日志、API log 等多处留痕**,撤销不彻底就是 sieve。

**根因**:
原始架构混合了"训练运行时"和"评分调用"在同一进程同一节点。这种耦合带来:
- 训练 GPU + judge API 串行依赖 → API 抖一下整个 run 失败
- 安全边界混淆 → key 必须能被云端进程读

**修复**(commit `d2261a9`):
把 generation eval 拆成两阶段:
1. **云端**:只产 prediction,标 `judge_pending=True`,eval_quality 用 ROUGE+format 占位
2. **本地**:用户在自己 Mac 上 `export ANTHROPIC_API_KEY` → `python scripts/run_judge.py` → 调 API 给 800 个 prediction 打分 → 真 eval_quality 写回

**经验教训**:
- **API key 不进云端**,这是个独立于实验本身的工程纪律
- 云端跑训练 / 本地跑评分,这种 "compute-heavy on cloud, key-heavy on local" 拆分是良好的安全实践
- 训练失败和 API 失败解耦,任一环节出问题不污染另一环节
- 报告里描述实验 setup 时,**不要写"我们用 ANTHROPIC_API_KEY 在 Colab 上跑评分"**,这种话审稿人会怀疑 reproducibility / security
- **任何在公开对话里暴露过的 key 立刻撤销**,即使你认为"对话很安全",也要假设最坏

---

## 总结表

| # | 问题域 | 核心症状 | Commit |
|---|---|---|---|
| 1 | 数据 | datasets 4.x 拒绝 script-based loader | (initial commits) |
| 2 | 数据 | trust_remote_code 现在硬要求 | (initial commits) |
| 3 | 数据 | CUAD 必须 contract-level split | (initial commits) |
| 4 | trl 框架 | trl 0.24 改用 prompt+completion / messages | (initial) |
| 5 | trl 框架 | tokenization mismatch warning | (initial) |
| 6 | 推理 | predict_batch 的 left-padding bug | `b8516c0` |
| 7 | 推理 | QLoRA dtype mismatch(bf16 vs fp32) | `b4057aa` |
| 8 | 配置 | max_length 太小,prompt 被截断 | `b8516c0` |
| 9 | jsonl | append 重复污染 composite | `b8516c0` |
| 10 | 日志 | 无醒目 summary,云端 log 难判 | `ecf3c92` |
| 11 | 日志 | 无 resume,断线重跑全部 | `47c08e7` |
| 12 | Colab | outputs/ 在临时盘 | `1c82387` |
| 13 | Colab | torchvision / torch ABI 冲突 | `76aa6d9` |
| 14 | Colab | Restart 后 cwd 丢失 | `ff33e8c` |
| 15 | 安全 | API key 不应进云端 | `d2261a9` |

**14 个工程问题,8 次 commit 修完,全部发生在"Initial commit → 实际开跑前"** 这段时间。**实验本身还没开始** — 这些都是"为了让实验能跑起来"的 setup 成本。

---

## 给后续报告 / 论文章节的写作建议

### "Experimental Setup" 章节模板

> We trained Qwen2.5-1.5B-Instruct with four PEFT methods (LoRA / DoRA / QLoRA /
> LoRA+) on three downstream tasks (BANKING77, Bitext Customer Support, CUAD)
> across two training-set scales (200 / 1000 samples), totalling 24 runs with
> a fixed seed=42.
>
> Training was performed on Google Colab L4 GPUs with `transformers==4.57.6`,
> `trl==0.24.0`, `peft==0.17.1`, `datasets==3.6.0`, `bitsandbytes>=0.43.0`. We
> note that `datasets` 4.x removes script-based loaders and was incompatible
> with `PolyAI/banking77` and `theatticusproject/cuad-qa`, requiring the 3.x
> pin. CUAD was split at the contract level (70/15/15) to avoid same-contract
> leakage between train/val/test, with split manifest stored alongside the
> data. Training-time `max_length` was empirically set per task (BANKING77:
> 512; CUAD: 768; Bitext: 1024) based on tokenizer-measured prompt length p95.
>
> The cloud-side runs produced predictions and deterministic metrics
> (macro-F1, perplexity, training time, peak memory). LLM-as-Judge scoring
> for the generation task (Bitext) was deferred to local post-hoc execution
> with `claude-haiku-4-5-20251001` to avoid exposing API keys to cloud
> environments and to decouple training failure from API availability.

### "Limitations and Reproducibility Notes" 章节内容建议

复制本文档第 6, 7, 8 项的工程教训作 limitation:
- "QLoRA inference requires explicit `torch.autocast`; we discovered this
  during evaluation, not training, and document it as a non-obvious gotcha
  for future replicators."
- "We measured that `predict_batch` with left padding can silently return
  the prompt as completion when batch lengths vary; we use token-level
  slicing to avoid this."
- "Pilot/larger sample sizes (200/1000) were chosen for compute budget
  reasons; results may not generalize to truly larger fine-tuning scales."

---

## 文档维护

本文档**应当随实验推进继续追加新问题**:

- 跑 24 runs 实际过程中的新错误(单 run 失败 / OOM / API 限流)
- 分析阶段发现的 ranking 反常(可能反推回去说明某个 method 本身设置问题)
- 论文审稿人提的 reproducibility 质疑(把 reviewer 的问题 + 我们的回应记下来)

**保存位置**:`docs/experiment_setup_issues.md`(本文件)
**git tracked**:✓
**与 README 的关系**:README 只讲 happy path,本文件讲完整 setup 旅程
