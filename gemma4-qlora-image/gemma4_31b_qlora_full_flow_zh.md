# Gemma 4 31B QLoRA 从零到验证全流程

## 适用范围

这份文档适用于下面这套环境和目标：

- 远端服务器：`root@105.100.31.190`
- 模型：`unsloth/gemma-4-31b-it-unsloth-bnb-4bit`
- 训练方式：`Unsloth + QLoRA + 4bit`
- 网络环境：在国内，需要代理
- 目标：
  1. 把 `31B 4bit` 模型下载并加载成功
  2. 用一份人设数据训练出“我是 cogiot 的小舸”这个效果
  3. 验证微调后确实比底座更像你想要的助手

这份流程是按已经验证过的实际路径整理的，不是照官方 notebook 原样抄写。

## 当前工作区里已经准备好的文件

如果你就在当前目录操作，下面这些文件已经可用：

- 训练脚本：`scripts/train_gemma4_31b_qlora.py`
- 远端启动脚本：`scripts/run_remote_gemma4_31b_qlora.sh`
- 人设数据生成器：`scripts/build_cogiot_persona_dataset.py`
- 最小 smoke 数据：`examples/smoke_messages.jsonl`
- 人设数据说明：`data/cogiot_xiaoge_dataset_notes.md`
- 这份全流程文档：`gemma4_31b_qlora_full_flow_zh.md`

## 目录约定

本流程默认使用这两个目录：

- 本地工作区：
  `/Users/cog/project/doucument-lyp/llm-train/gemma4-qlora`
- 远端工作目录：
  `/root/gemma4-qlora`

后面命令如果没有特别说明，都是按这个目录写的。

## 第 1 步：准备代理

远端和 Hugging Face 相关操作要用下面这组代理：

```bash
export https_proxy=http://105.100.31.173:7897
export http_proxy=http://105.100.31.173:7897
export all_proxy=socks5://105.100.31.173:7897
```

注意两点：

1. 日常 Python / pip / 训练过程可以保留这三个变量
2. 大文件下载模型时，建议：
   - 保留 `http_proxy` / `https_proxy`
   - 执行 `unset all_proxy ALL_PROXY`
   - 再配合 `HF_HUB_DISABLE_XET=1`

这是因为这条链路上 `hf_xet` 容易卡在 `cas-server.xethub.hf.co`。

## 第 2 步：初始化远端环境

先登录远端：

```bash
ssh root@105.100.31.190
```

创建工作目录并安装 Python 3.11 虚拟环境：

```bash
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

验证环境：

```bash
python - <<'PY'
import numpy, torch, unsloth
print("numpy", numpy.__version__)
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "devices", torch.cuda.device_count())
print("unsloth", unsloth.__version__)
PY
```

正常情况下你会看到：

- `numpy 2.4.4`
- `torch 2.10.0+cu130`
- `unsloth 2026.4.4`
- `cuda True`
- `devices 2`

## 第 3 步：把本地脚本同步到远端

在本地工作区执行：

```bash
cd /Users/cog/project/doucument-lyp/llm-train/gemma4-qlora

scp scripts/train_gemma4_31b_qlora.py root@105.100.31.190:/root/gemma4-qlora/train_gemma4_31b_qlora.py
scp scripts/run_remote_gemma4_31b_qlora.sh root@105.100.31.190:/root/gemma4-qlora/run_remote_gemma4_31b_qlora.sh
scp scripts/build_cogiot_persona_dataset.py root@105.100.31.190:/root/gemma4-qlora/build_cogiot_persona_dataset.py
scp examples/smoke_messages.jsonl root@105.100.31.190:/root/gemma4-qlora/examples/smoke_messages.jsonl
```

然后给远端脚本加执行权限：

```bash
ssh root@105.100.31.190 '
mkdir -p /root/gemma4-qlora/examples /root/gemma4-qlora/data /root/gemma4-qlora/runs
chmod +x /root/gemma4-qlora/train_gemma4_31b_qlora.py
chmod +x /root/gemma4-qlora/run_remote_gemma4_31b_qlora.sh
chmod +x /root/gemma4-qlora/build_cogiot_persona_dataset.py
'
```

## 第 4 步：下载 31B 4bit 模型

登录远端后执行：

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

如果中途断开，不要删目录，直接重跑。已经完成的分片会复用。

## 第 5 步：补齐索引和 tokenizer 文件

有时 6 个 safetensors 文件都在了，但目录里还缺少索引或 tokenizer 文件，这会导致加载失败。

继续在远端执行：

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

## 第 6 步：验证模型本地加载

在远端执行：

```bash
cd /root/gemma4-qlora
. .venv/bin/activate

export https_proxy=http://105.100.31.173:7897
export http_proxy=http://105.100.31.173:7897
export all_proxy=socks5://105.100.31.173:7897

python - <<'PY'
from unsloth import FastVisionModel

model, processor = FastVisionModel.from_pretrained(
    model_name="/root/gemma4-qlora/models/gemma-4-31b-it-unsloth-bnb-4bit",
    max_seq_length=512,
    dtype=None,
    load_in_4bit=True,
    device_map="balanced",
    local_files_only=True,
)
print(type(model).__name__)
print(type(processor).__name__)
print(getattr(model, "hf_device_map", None))
PY
```

你应该能看到：

- `Gemma4ForConditionalGeneration`
- `Gemma4Processor`
- `device_map="balanced"` 分到两张 T4

## 第 7 步：先跑一个最小 smoke test

这一步的意义是先确认训练链路没问题，不急着上正式数据。

```bash
ssh root@105.100.31.190 '
cd /root/gemma4-qlora
./run_remote_gemma4_31b_qlora.sh \
  --max-steps 1 \
  --save-steps 0 \
  --output-dir /root/gemma4-qlora/runs/gemma4_31b_qlora_smoke_script
'
```

这一步已经实测通过过，典型结果是：

- `train_runtime` 约 `27s`
- `train_loss` 约 `9.6159`

## 第 8 步：生成人设训练数据

你现在没有真实数据，所以这里用一份演示型种子数据来训练“小舸”效果。

在本地工作区执行：

```bash
cd /Users/cog/project/doucument-lyp/llm-train/gemma4-qlora
python3 scripts/build_cogiot_persona_dataset.py
```

会生成两份数据：

- `data/cogiot_xiaoge_train.jsonl`
- `data/cogiot_xiaoge_identity_seed.jsonl`

两者区别：

1. `cogiot_xiaoge_identity_seed.jsonl`
   更短，更偏身份锚定，适合最快看出“我是 cogiot 的小舸”效果

2. `cogiot_xiaoge_train.jsonl`
   更完整，除了身份，还加了一些普通问答风格

同步到远端：

```bash
scp data/cogiot_xiaoge_train.jsonl root@105.100.31.190:/root/gemma4-qlora/data/cogiot_xiaoge_train.jsonl
scp data/cogiot_xiaoge_identity_seed.jsonl root@105.100.31.190:/root/gemma4-qlora/data/cogiot_xiaoge_identity_seed.jsonl
scp data/cogiot_xiaoge_dataset_notes.md root@105.100.31.190:/root/gemma4-qlora/data/cogiot_xiaoge_dataset_notes.md
```

## 第 9 步：先用人设数据做 smoke 验证

先确认训练脚本能吃进这份数据。

完整版 smoke：

```bash
ssh root@105.100.31.190 '
cd /root/gemma4-qlora
DATASET_PATH=/root/gemma4-qlora/data/cogiot_xiaoge_train.jsonl \
OUTPUT_DIR=/root/gemma4-qlora/runs/cogiot_xiaoge_dataset_smoke \
./run_remote_gemma4_31b_qlora.sh \
  --limit-examples 8 \
  --max-steps 1 \
  --save-steps 0
'
```

身份 seed smoke：

```bash
ssh root@105.100.31.190 '
cd /root/gemma4-qlora
DATASET_PATH=/root/gemma4-qlora/data/cogiot_xiaoge_identity_seed.jsonl \
OUTPUT_DIR=/root/gemma4-qlora/runs/cogiot_xiaoge_identity_seed_smoke \
./run_remote_gemma4_31b_qlora.sh \
  --limit-examples 8 \
  --max-steps 1 \
  --save-steps 0
'
```

这两条都已经实测成功。

## 第 10 步：正式训练“小舸”身份版

如果你想先最快看出训练效果，优先用身份 seed 版：

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

这就是你刚才执行并成功完成的那条正式训练命令。

如果你更想兼顾身份和普通回答风格，就用完整版：

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

## 第 11 步：检查训练结果是否完整

如果训练正常结束，你应该在输出目录里看到：

- `adapter/`
- `processor/`
- `checkpoint-*`

例如你这次跑出来的是：

- `/root/gemma4-qlora/runs/cogiot_xiaoge_identity_seed/adapter`
- `/root/gemma4-qlora/runs/cogiot_xiaoge_identity_seed/processor`
- `/root/gemma4-qlora/runs/cogiot_xiaoge_identity_seed/checkpoint-40`
- `/root/gemma4-qlora/runs/cogiot_xiaoge_identity_seed/checkpoint-44`

查看目录：

```bash
ssh root@105.100.31.190 '
ls -lah /root/gemma4-qlora/runs/cogiot_xiaoge_identity_seed
printf "\n=====\n"
ls -lah /root/gemma4-qlora/runs/cogiot_xiaoge_identity_seed/adapter
'
```

查看训练状态：

```bash
ssh root@105.100.31.190 '
python3 - <<\"PY\"
import json
path = \"/root/gemma4-qlora/runs/cogiot_xiaoge_identity_seed/checkpoint-44/trainer_state.json\"
with open(path) as f:
    data = json.load(f)
print(\"global_step\", data.get(\"global_step\"))
print(\"epoch\", data.get(\"epoch\"))
for item in data.get(\"log_history\", [])[-5:]:
    print(item)
PY'
```

你这次这轮训练的实际结果是：

- `global_step = 44`
- `epoch = 1.0`
- 最后几步 loss 大致在 `1.1 ~ 2.6` 之间波动
- 最终 adapter 大小约 `234MB`

## 第 12 步：验证微调是否真的生效

最关键的不是“训练跑完了”，而是“问同一个问题时，微调模型和底座有没有明显差异”。

下面这段命令会做一组对比：

```bash
ssh root@105.100.31.190 '
cd /root/gemma4-qlora
. .venv/bin/activate
python -u - <<\"PY\"
from unsloth import FastVisionModel
from peft import PeftModel
import torch

BASE = \"/root/gemma4-qlora/models/gemma-4-31b-it-unsloth-bnb-4bit\"
ADAPTER = \"/root/gemma4-qlora/runs/cogiot_xiaoge_identity_seed/adapter\"

prompts = [
    \"你是谁？\",
    \"你和 cogiot 是什么关系？\",
    \"怎么开始学 Python？\",
]

def build_inputs(processor, prompt):
    messages = [{\"role\": \"user\", \"content\": [{\"type\": \"text\", \"text\": prompt}]}]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors=\"pt\",
        return_dict=True,
    )
    return {k: v.to(\"cuda:0\") if hasattr(v, \"to\") else v for k, v in inputs.items()}

def decode_answer(processor, outputs, input_ids):
    generated = outputs[0][input_ids.shape[-1]:]
    text = processor.decode(generated, skip_special_tokens=True)
    return \" \".join(text.split())

print(\"LOADING_BASE\", flush=True)
base_model, processor = FastVisionModel.from_pretrained(
    model_name=BASE,
    max_seq_length=512,
    dtype=None,
    load_in_4bit=True,
    device_map=\"balanced\",
    local_files_only=True,
)
FastVisionModel.for_inference(base_model)

for prompt in prompts:
    inputs = build_inputs(processor, prompt)
    with torch.inference_mode():
        outputs = base_model.generate(**inputs, max_new_tokens=64, do_sample=False, use_cache=True)
    answer = decode_answer(processor, outputs, inputs[\"input_ids\"])
    print(f\"BASE_Q: {prompt}\")
    print(f\"BASE_A: {answer}\")

print(\"LOADING_ADAPTER\", flush=True)
ft_model = PeftModel.from_pretrained(base_model, ADAPTER)
FastVisionModel.for_inference(ft_model)

for prompt in prompts:
    inputs = build_inputs(processor, prompt)
    with torch.inference_mode():
        outputs = ft_model.generate(**inputs, max_new_tokens=64, do_sample=False, use_cache=True)
    answer = decode_answer(processor, outputs, inputs[\"input_ids\"])
    print(f\"FT_Q: {prompt}\")
    print(f\"FT_A: {answer}\")
PY'
```

## 第 13 步：如何判断效果是否达到预期

你这次已经跑出的实际对比结果是这样的：

问 `你是谁？`

- 底座回答：
  `我是一个由 Google 训练的大型语言模型...`
- 微调后回答：
  `我是 cogiot 的小舸，一个偏向执行、注重结果的 AI 助手。`

问 `你和 cogiot 是什么关系？`

- 底座回答：
  `我与 Cogiot 之间没有直接的隶属或合作关系...`
- 微调后回答：
  `我是 cogiot 的小舸，一个偏向执行、注重把事情做完的 AI 助手。`

问 `怎么开始学 Python？`

- 底座更像通用百科式回答
- 微调后会更偏直接、务实、强调动手

这说明两件事已经成立：

1. 身份锚定成功
2. 普通回答里已经开始带一点“小舸”风格，但还没有过度自我介绍

## 第 14 步：如果你想继续优化

如果你觉得身份已经对了，但风格还不够像你，可以继续补这几类数据：

1. 更多真实业务问答
2. 更多更像你本人语气的表达
3. 更多多轮对话
4. 少量“不要过度自我介绍”的反例

更稳妥的第二轮做法：

```bash
ssh root@105.100.31.190 '
cd /root/gemma4-qlora
DATASET_PATH=/root/gemma4-qlora/data/cogiot_xiaoge_train.jsonl \
OUTPUT_DIR=/root/gemma4-qlora/runs/cogiot_xiaoge_round2 \
./run_remote_gemma4_31b_qlora.sh \
  --num-train-epochs 1 \
  --save-steps 20 \
  --save-adapter
'
```

## 常见问题

### 1. 下载特别慢或者卡住

优先确认你下载时用了：

```bash
unset all_proxy ALL_PROXY
export HF_HUB_DISABLE_XET=1
```

然后直接重跑下载命令，不要删模型目录。

### 2. 模型加载报缺少配置文件

说明你 safetensors 在，但 `index.json` / `tokenizer` / `processor_config` 没补齐。回到“第 5 步”补文件。

### 3. 训练时报 `outputs.logits` 相关错误

不要自己再写一份 trainer。直接用当前工作区里的 [train_gemma4_31b_qlora.py](/Users/cog/project/doucument-lyp/llm-train/gemma4-qlora/scripts/train_gemma4_31b_qlora.py)，这个脚本已经把兼容性补丁固化进去了。

### 4. 想最快验证人设效果，用哪份数据

先用：

```text
/root/gemma4-qlora/data/cogiot_xiaoge_identity_seed.jsonl
```

### 5. 想兼顾身份和普通问答风格，用哪份数据

用：

```text
/root/gemma4-qlora/data/cogiot_xiaoge_train.jsonl
```

## 最后结论

按这份文档执行，你可以完整复现这条已经验证通过的路径：

1. 远端环境准备
2. 31B 4bit 模型下载
3. 本地多 GPU 加载
4. 使用脚本跑通 QLoRA
5. 生成人设数据
6. 训练“小舸”身份版 adapter
7. 用底座和微调模型做对比，确认训练确实生效

如果你只想走最短路径，就记住下面这三条核心命令：

1. 生成数据

```bash
cd /Users/cog/project/doucument-lyp/llm-train/gemma4-qlora
python3 scripts/build_cogiot_persona_dataset.py
```

2. 训练身份版

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

3. 验证效果

执行“第 12 步”的底座 / adapter 对比推理命令。
