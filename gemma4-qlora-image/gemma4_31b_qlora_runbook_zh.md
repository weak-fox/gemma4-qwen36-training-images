# Gemma 4 31B QLoRA 远程跑通记录

## 结果概览

截至 2026-04-13，已在 `root@105.100.31.190` 上完成：

1. `Unsloth + Python 3.11 + CUDA` 环境准备
2. `unsloth/gemma-4-31b-it-unsloth-bnb-4bit` 本地完整下载
3. 本地目录加载成功，`device_map="balanced"` 成功分配到 `2x Tesla T4`
4. 一个最小 `QLoRA smoke test` 成功跑完 `max_steps=1`

## 机器信息

- 主机：`root@105.100.31.190`
- 系统：`Ubuntu 22.04.4`
- GPU：`Tesla T4` x2
- VRAM：
  - GPU0: `15360 MiB`
  - GPU1: `16384 MiB`
- Python：`3.11.15`
- 关键包：
  - `unsloth 2026.4.4`
  - `torch 2.10.0+cu130`
  - `numpy 2.4.4`
  - `socksio 1.0.0`

## 代理设置

安装和 Hugging Face 访问都依赖代理。常用设置：

```bash
export https_proxy=http://105.100.31.173:7897
export http_proxy=http://105.100.31.173:7897
export all_proxy=socks5://105.100.31.173:7897
```

注意：

- `huggingface_hub` 如果要识别 `all_proxy=socks5://...`，需要安装 `socksio`
- 对大文件下载，最终采用的是：
  - 保留 `http_proxy` / `https_proxy`
  - `unset all_proxy ALL_PROXY`
  - `HF_HUB_DISABLE_XET=1`

原因是 `hf_xet` 在这条链路上会卡在 `cas-server.xethub.hf.co`，出现 TLS EOF。

## 环境搭建

```bash
ssh root@105.100.31.190
mkdir -p /root/gemma4-qlora
cd /root/gemma4-qlora

export https_proxy=http://105.100.31.173:7897
export http_proxy=http://105.100.31.173:7897
export all_proxy=socks5://105.100.31.173:7897

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=/root/.local/bin:$PATH

uv python install 3.11
uv venv .venv --python 3.11
. .venv/bin/activate

uv pip install --upgrade pip setuptools wheel
uv pip install unsloth --torch-backend=auto
python -m pip install -U socksio
```

验证：

```bash
python - <<'PY'
import numpy, torch, unsloth
print("numpy", numpy.__version__)
print("torch", torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())
print("unsloth", unsloth.__version__)
PY
```

## 下载 31B 4bit 模型

模型最终目录：

```bash
/root/gemma4-qlora/models/gemma-4-31b-it-unsloth-bnb-4bit
```

推荐下载方式：

```bash
cd /root/gemma4-qlora
. .venv/bin/activate

export https_proxy=http://105.100.31.173:7897
export http_proxy=http://105.100.31.173:7897
unset all_proxy ALL_PROXY

export HF_HOME=/root/gemma4-qlora/hf
export HUGGINGFACE_HUB_CACHE=/root/gemma4-qlora/hf/hub
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=1800
export HF_HUB_ETAG_TIMEOUT=60

python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="unsloth/gemma-4-31b-it-unsloth-bnb-4bit",
    local_dir="/root/gemma4-qlora/models/gemma-4-31b-it-unsloth-bnb-4bit",
    max_workers=1,
)
PY
```

如果大文件中途断开，不要急着删目录，直接重跑。已完成的分片会复用。

## 下载完成后的补文件

这一步很重要。即使 6 个 `model-0000x-of-00006.safetensors` 都在，本地目录仍可能缺少索引或 tokenizer 文件，导致加载失败。

补齐：

```bash
cd /root/gemma4-qlora
. .venv/bin/activate

export https_proxy=http://105.100.31.173:7897
export http_proxy=http://105.100.31.173:7897
unset all_proxy ALL_PROXY

python - <<'PY'
from huggingface_hub import hf_hub_download
repo = "unsloth/gemma-4-31b-it-unsloth-bnb-4bit"
local_dir = "/root/gemma4-qlora/models/gemma-4-31b-it-unsloth-bnb-4bit"
for filename in [
    "model.safetensors.index.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
]:
    path = hf_hub_download(repo_id=repo, filename=filename, local_dir=local_dir)
    print(filename, path)
PY
```

## 本地加载验证

已验证可用：

```python
from unsloth import FastVisionModel

model, processor = FastVisionModel.from_pretrained(
    model_name="/root/gemma4-qlora/models/gemma-4-31b-it-unsloth-bnb-4bit",
    max_seq_length=512,
    dtype=None,
    load_in_4bit=True,
    device_map="balanced",
    local_files_only=True,
)
```

实测成功结果：

- 模型类：`Gemma4ForConditionalGeneration`
- Processor：`Gemma4Processor`
- `device_map="balanced"` 会把部分层放到 GPU0，其余大部分层放到 GPU1

## 最小训练 smoke test

### 关键说明

直接使用当前环境的 `TRL SFTTrainer` 会报错：

```text
AttributeError: 'NoneType' object has no attribute 'shape'
```

原因是：

- `Gemma4` 训练前向只返回 `loss`
- 当前 `TRL SFTTrainer` 还会访问 `outputs.logits` 计算 entropy 指标

因此这里用了一个最小 patched trainer，只改 `compute_loss`，直接复用 `transformers.Trainer.compute_loss`。

### 可运行脚本

```bash
cd /root/gemma4-qlora
. .venv/bin/activate

export https_proxy=http://105.100.31.173:7897
export http_proxy=http://105.100.31.173:7897
export all_proxy=socks5://105.100.31.173:7897

export HF_HOME=/root/gemma4-qlora/hf
export HUGGINGFACE_HUB_CACHE=/root/gemma4-qlora/hf/hub
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u - <<'PY'
import unsloth
from datasets import Dataset
from transformers import Trainer as HFTrainer
from trl import SFTTrainer, SFTConfig
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
import torch

MODEL_DIR = "/root/gemma4-qlora/models/gemma-4-31b-it-unsloth-bnb-4bit"
OUT_DIR = "/root/gemma4-qlora/runs/gemma4_31b_qlora_smoke_patch"

class PatchedSFTTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(inputs)
        inputs["use_cache"] = False
        loss, outputs = HFTrainer.compute_loss(
            self,
            model,
            inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )
        return (loss, outputs) if return_outputs else loss

examples = [
    {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Write one short sentence saying hello to a developer."}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Hello, developer, glad to work with you today."}]},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Answer in one sentence: what is QLoRA used for?"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "QLoRA is used to fine-tune large language models with low memory usage."}]},
        ]
    },
]

dataset = Dataset.from_list(examples)

model, processor = FastVisionModel.from_pretrained(
    model_name=MODEL_DIR,
    max_seq_length=512,
    dtype=None,
    load_in_4bit=True,
    device_map="balanced",
    local_files_only=True,
)

model = FastVisionModel.get_peft_model(
    model,
    r=8,
    lora_alpha=8,
    lora_dropout=0,
    bias="none",
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    max_seq_length=512,
)
FastVisionModel.for_training(model)

trainer = PatchedSFTTrainer(
    model=model,
    train_dataset=dataset,
    data_collator=UnslothVisionDataCollator(model, processor, max_seq_length=512),
    processing_class=processor,
    args=SFTConfig(
        output_dir=OUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=1,
        warmup_steps=0,
        learning_rate=2e-4,
        optim="adamw_8bit",
        fp16=True,
        bf16=False,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        dataset_text_field="",
        max_length=512,
        seed=3407,
    ),
)

result = trainer.train()
print(result.metrics)
print("CUDA_MAX_MEM", torch.cuda.max_memory_allocated(0), torch.cuda.max_memory_allocated(1))
PY
```

### 这一步的实测结果

- `max_steps=1` 成功完成
- 训练耗时约 `50.37s`
- 最终日志：

```text
{'train_runtime': 50.3718, 'train_samples_per_second': 0.02, 'train_steps_per_second': 0.02, 'train_loss': 9.61596393585205, 'epoch': 0.5}
```

- 峰值显存：

```text
GPU0: 11053898752
GPU1: 16106813952
```

## 当前结论

这台 `2xT4` 机器上，`31B 4bit` 路线已经实测验证到：

1. 权重可完整下载
2. 本地可完整加载
3. `device_map="balanced"` 可用
4. 经过一个最小 trainer 兼容性补丁后，`QLoRA 1-step smoke test` 可跑通

## 已整理好的可复用脚本

当前工作区已经整理出 3 个可直接复用的文件：

- `scripts/train_gemma4_31b_qlora.py`
- `scripts/run_remote_gemma4_31b_qlora.sh`
- `examples/smoke_messages.jsonl`

其中：

- `train_gemma4_31b_qlora.py` 是正式训练入口
- `run_remote_gemma4_31b_qlora.sh` 负责在远端统一代理、HF cache、CUDA 内存参数和默认路径
- `examples/smoke_messages.jsonl` 是 2 条最小可运行样本，用来做 smoke test

### 支持的数据格式

训练脚本当前支持这几类输入：

1. `json` / `jsonl`
2. `csv`
3. `parquet`
4. `datasets.load_from_disk()` 导出的目录

数据内容支持两种模式：

1. 直接提供 `messages` 列
2. 提供 `prompt` + `completion` 列，脚本会自动转换成对话格式

如果有系统提示词，也可以额外传：

```bash
--system-column system
```

### 推荐的数据格式示例

最稳妥的是直接用 `messages`：

```json
{"messages":[{"role":"user","content":[{"type":"text","text":"解释什么是 QLoRA"}]},{"role":"assistant","content":[{"type":"text","text":"QLoRA 是一种低显存微调大型模型的方法。"}]}]}
```

如果你原始数据更像指令数据，也可以这样：

```json
{"prompt":"解释什么是 QLoRA","completion":"QLoRA 是一种低显存微调大型模型的方法。"}
```

## 脚本化运行方式

### 1. 从本地工作区拷到远端

在当前目录执行：

```bash
scp scripts/train_gemma4_31b_qlora.py root@105.100.31.190:/root/gemma4-qlora/train_gemma4_31b_qlora.py
scp scripts/run_remote_gemma4_31b_qlora.sh root@105.100.31.190:/root/gemma4-qlora/run_remote_gemma4_31b_qlora.sh
scp examples/smoke_messages.jsonl root@105.100.31.190:/root/gemma4-qlora/examples/smoke_messages.jsonl
```

远端第一次执行前：

```bash
ssh root@105.100.31.190 'mkdir -p /root/gemma4-qlora/examples && chmod +x /root/gemma4-qlora/train_gemma4_31b_qlora.py /root/gemma4-qlora/run_remote_gemma4_31b_qlora.sh'
```

### 2. 先跑脚本化 smoke test

```bash
ssh root@105.100.31.190 '
cd /root/gemma4-qlora
./run_remote_gemma4_31b_qlora.sh \
  --max-steps 1 \
  --save-steps 0 \
  --output-dir /root/gemma4-qlora/runs/gemma4_31b_qlora_smoke_script
'
```

### 3. 跑你自己的真实数据

假设你的数据是：

```bash
/root/gemma4-qlora/data/train.jsonl
```

如果里面已经有 `messages` 列：

```bash
ssh root@105.100.31.190 '
cd /root/gemma4-qlora
DATASET_PATH=/root/gemma4-qlora/data/train.jsonl \
OUTPUT_DIR=/root/gemma4-qlora/runs/gemma4_31b_realdata \
./run_remote_gemma4_31b_qlora.sh \
  --num-train-epochs 1 \
  --save-steps 20 \
  --save-adapter
'
```

如果里面是 `prompt` / `completion`：

```bash
ssh root@105.100.31.190 '
cd /root/gemma4-qlora
DATASET_PATH=/root/gemma4-qlora/data/train.jsonl \
OUTPUT_DIR=/root/gemma4-qlora/runs/gemma4_31b_realdata \
./run_remote_gemma4_31b_qlora.sh \
  --prompt-column prompt \
  --completion-column completion \
  --num-train-epochs 1 \
  --save-steps 20 \
  --save-adapter
'
```

如果只是先试运行前 8 条：

```bash
ssh root@105.100.31.190 '
cd /root/gemma4-qlora
DATASET_PATH=/root/gemma4-qlora/data/train.jsonl \
OUTPUT_DIR=/root/gemma4-qlora/runs/gemma4_31b_first8 \
./run_remote_gemma4_31b_qlora.sh \
  --limit-examples 8 \
  --max-steps 2 \
  --save-steps 0
'
```

## 我帮你生成的人设数据

当前已经生成并同步到远端：

- `/root/gemma4-qlora/data/cogiot_xiaoge_train.jsonl`
- `/root/gemma4-qlora/data/cogiot_xiaoge_identity_seed.jsonl`
- `/root/gemma4-qlora/data/cogiot_xiaoge_dataset_notes.md`

设计目标是两层：

1. 问身份时稳定回答成 `我是 cogiot 的小舸`
2. 普通问答里带一点“小舸”的影子，比如更直接、先给结论、强调可执行性

### 两份数据怎么选

`cogiot_xiaoge_identity_seed.jsonl`

- 更短
- 更偏身份和风格锚定
- 更适合先做一个“人设效果明显”的演示版

`cogiot_xiaoge_train.jsonl`

- 更完整
- 除了身份锚定，还加入了一些普通问答风格
- 更适合做第一版可用助手

### 快速看身份效果

推荐先用 seed 版：

```bash
ssh root@105.100.31.190 '
cd /root/gemma4-qlora
DATASET_PATH=/root/gemma4-qlora/data/cogiot_xiaoge_identity_seed.jsonl \
OUTPUT_DIR=/root/gemma4-qlora/runs/cogiot_xiaoge_identity_seed \
./run_remote_gemma4_31b_qlora.sh \
  --num-train-epochs 1 \
  --save-steps 20 \
  --save-adapter
'
```

如果你更想要“身份 + 普通回答风格”一起有点变化，用完整版：

```bash
ssh root@105.100.31.190 '
cd /root/gemma4-qlora
DATASET_PATH=/root/gemma4-qlora/data/cogiot_xiaoge_train.jsonl \
OUTPUT_DIR=/root/gemma4-qlora/runs/cogiot_xiaoge_fullpersona \
./run_remote_gemma4_31b_qlora.sh \
  --num-train-epochs 1 \
  --save-steps 20 \
  --save-adapter
'
```

### 已验证的最小 smoke

我已经实测跑过这两个最小验证：

- 完整版：
  - `DATASET_PATH=/root/gemma4-qlora/data/cogiot_xiaoge_train.jsonl`
  - `--limit-examples 8 --max-steps 1 --save-steps 0`
  - 成功
- seed 版：
  - `DATASET_PATH=/root/gemma4-qlora/data/cogiot_xiaoge_identity_seed.jsonl`
  - `--limit-examples 8 --max-steps 1 --save-steps 0`
  - 成功

## 建议的下一步

如果要继续正式训练，不要直接扩大 batch 或序列长度，建议按这个顺序加：

1. 先保持 `per_device_train_batch_size=1`
2. 先保持 `max_length=512`
3. 先用极小训练集跑 `max_steps=5`
4. 再逐步增加数据量和训练步数
5. 如果要严格贴近官方 notebook，再把当前 patched trainer 替换为官方后续修复版本

## 常用检查命令

看显卡：

```bash
nvidia-smi
```

看模型文件：

```bash
ls -lh /root/gemma4-qlora/models/gemma-4-31b-it-unsloth-bnb-4bit/model-*.safetensors
```

看输出目录：

```bash
ls -lah /root/gemma4-qlora/runs/gemma4_31b_qlora_smoke_patch
```
