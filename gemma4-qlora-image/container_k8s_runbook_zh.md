# Gemma 4 31B QLoRA 容器与 K8s 运行手册

这套实现分成两个镜像：

- `train-nomodel`
  不内置基础模型，只负责训练。
- `runtime-with-model`
  内置 `gemma-4-31b-it-unsloth-bnb-4bit`，既能训练，也能加载 MinIO 上的 adapter 提供模型服务。

这两种镜像都通过环境变量驱动。按当前约束，镜像建议直接在服务器 `root@105.100.31.190` 上构建，因为这台机器已经有本地模型目录，而且外网链路特征已经摸清。

当前已经推送到内网仓库的正式镜像：

- `sz.cogiot.com/id348504946c01000/gemma4-qlora-train-nomodel:20260414-v1`
- `sz.cogiot.com/id348504946c01000/gemma4-qlora-train-nomodel:latest`
- `sz.cogiot.com/id348504946c01000/gemma4-qlora-runtime-with-model:20260414-v1`
- `sz.cogiot.com/id348504946c01000/gemma4-qlora-runtime-with-model:latest`

对应 digest：

- `gemma4-qlora-train-nomodel`: `sha256:dd9b28a147db57d5b3f1ad9b1a2b0cc20b215329ee93b7c2f8db982333c8a0de`
- `gemma4-qlora-runtime-with-model`: `sha256:cc0399b6ec3886c578d5907f0f7789ec00118279d26058c295ed8c763559abfc`

构建策略分两条：

- `train-nomodel`
  直接基于官方 `unsloth/unsloth` 基础镜像做源码 `docker build`。
- `runtime-with-model`
  不再走 `docker build` + 25GB build context。这条路在 `105.100.31.190` 上会把根盘打满。
  改成先构建 `train-nomodel`，再从它创建临时容器，把宿主机模型目录复制进 `/opt/models/...`，最后 `docker commit` 成最终镜像。

服务器上最推荐的做法不是直接 `docker pull`，而是先用仓库里的预拉取脚本把官方基础镜像拉到本地：

```bash
ssh root@105.100.31.190 '
cd /root/gemma4-qlora-image
export HTTP_PROXY=http://105.100.31.173:7897
export HTTPS_PROXY=http://105.100.31.173:7897
./scripts/prefetch_unsloth_base_image.sh
'
```

这个脚本会走一条更稳的链路：

1. `skopeo copy` 通过 `HTTP_PROXY/HTTPS_PROXY` 下载远端镜像到 OCI archive。
2. `ctr -n moby images import` 导入 containerd。
3. `ctr -n moby images export - | docker load` 导入 Docker。
4. 最后校验本地已经有 `unsloth/unsloth:2026.3.8-pt2.9.0-vllm-0.16.0-cu12.8-studio-release`。

如果你坚持直接 `docker pull`，那才需要给 Docker daemon 配代理。因为单纯在 shell 里 `export http_proxy/https_proxy/all_proxy` 不够，`docker pull` 走的是 `dockerd`。对应做法是：

```bash
sudo mkdir -p /etc/systemd/system/docker.service.d
sudo tee /etc/systemd/system/docker.service.d/http-proxy.conf >/dev/null <<'EOF'
[Service]
Environment="HTTP_PROXY=http://105.100.31.173:7897"
Environment="HTTPS_PROXY=http://105.100.31.173:7897"
Environment="ALL_PROXY=socks5://105.100.31.173:7897"
Environment="NO_PROXY=localhost,127.0.0.1,::1,sz.cogiot.com,.cogiot.com"
EOF
sudo systemctl daemon-reload
sudo systemctl restart docker
```

`sz.cogiot.com` 是内网仓库，推拉镜像时不要再走代理，也不需要额外限速。关键是把它放进 `NO_PROXY`，让 `dockerd` 直连。

如果走 `docker pull` 路线，执行 Docker 命令前，还要把当前 shell 里的代理变量去掉，避免 `docker` CLI 自己访问 `/run/docker.sock` 时被代理干扰：

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
```

构建脚本会先读取你当前 shell 的代理值并转换成 `--build-arg`，然后自动 `unset` 掉这些代理变量，再执行 `docker build`。这样既能让构建阶段的 `pip` 走代理，又不会干扰 Docker CLI 和 Docker daemon 通信。

## 1. 目录准备

假设仓库目录在服务器上是 `/root/gemma4-qlora-image`，模型目录是：

```bash
/root/gemma4-qlora/models/gemma-4-31b-it-unsloth-bnb-4bit
```

如果你还没有把当前仓库同步到服务器，可以先执行：

```bash
rsync -av /Users/cog/project/doucument-lyp/llm-train/gemma4-qlora/ \
  root@105.100.31.190:/root/gemma4-qlora-image/
```

## 2. 在服务器构建镜像

### 2.1 构建不带模型的训练镜像

```bash
ssh root@105.100.31.190 '
cd /root/gemma4-qlora-image
export HTTP_PROXY=http://105.100.31.173:7897
export HTTPS_PROXY=http://105.100.31.173:7897
./scripts/prefetch_unsloth_base_image.sh
IMAGE_TAG=gemma4-qlora:train-nomodel \
./scripts/build_train_nomodel_image.sh
'
```

### 2.2 构建带模型的训练/服务镜像

这一步依赖前一步已经存在的 `gemma4-qlora:train-nomodel`。构建脚本不会重新走大 build context，而是：

1. 从 `gemma4-qlora:train-nomodel` 创建临时容器。
2. 把 `/root/gemma4-qlora/models/gemma-4-31b-it-unsloth-bnb-4bit` 复制到容器内 `/opt/models/...`。
3. `docker commit` 成 `gemma4-qlora:runtime-with-model`。

这样做的目的很直接：减少一次 25GB context 打包和重复落盘，避免服务器磁盘在构建时被打满。

```bash
ssh root@105.100.31.190 '
cd /root/gemma4-qlora-image
export HTTP_PROXY=http://105.100.31.173:7897
export HTTPS_PROXY=http://105.100.31.173:7897
MODEL_SOURCE_DIR=/root/gemma4-qlora/models/gemma-4-31b-it-unsloth-bnb-4bit \
IMAGE_TAG=gemma4-qlora:runtime-with-model \
./scripts/build_runtime_with_model_image.sh
'
```

默认构建策略是：

1. 用 `unsloth/unsloth:2026.3.8-pt2.9.0-vllm-0.16.0-cu12.8-studio-release` 作为基础镜像。
2. 在其上只补少量运行时附加包。
3. `train-nomodel` 用普通 `docker build` 构建，不依赖 `buildx`。
4. `runtime-with-model` 复用 `train-nomodel`，通过临时容器复制模型再 `docker commit`。

也就是说，`cuda / torch / unsloth / vllm` 这套重包默认都复用官方基础镜像，不再在我们的 Dockerfile 里重复安装。

## 3. 训练模式

训练模式入口统一是：

```bash
APP_MODE=train
```

### 3.1 关键环境变量

- `MODEL_NAME_OR_PATH`
- `MODEL_LOCAL_FILES_ONLY`
- `TRAIN_DATASET_URI`
- `TRAIN_DATASET_PATH`
- `TRAIN_OUTPUT_DIR`
- `TRAIN_OUTPUT_URI`
- `TRAIN_DATASET_SPLIT`
- `TRAIN_MESSAGES_COLUMN`
- `TRAIN_PROMPT_COLUMN`
- `TRAIN_COMPLETION_COLUMN`
- `TRAIN_SYSTEM_COLUMN`
- `TRAIN_LIMIT_EXAMPLES`
- `TRAIN_MAX_SEQ_LENGTH`
- `TRAIN_BATCH_SIZE`
- `TRAIN_GRAD_ACCUM_STEPS`
- `TRAIN_NUM_EPOCHS`
- `TRAIN_MAX_STEPS`
- `TRAIN_LEARNING_RATE`
- `TRAIN_WARMUP_STEPS`
- `TRAIN_LOGGING_STEPS`
- `TRAIN_SAVE_STEPS`
- `TRAIN_SAVE_TOTAL_LIMIT`
- `TRAIN_SAVE_ADAPTER`
- `MINIO_ENDPOINT`
- `MINIO_ACCESS_KEY`
- `MINIO_SECRET_KEY`
- `MINIO_SECURE`
- `MINIO_REGION`

规则：

- 如果同时设置了 `TRAIN_DATASET_URI` 和 `TRAIN_DATASET_PATH`，优先使用 `TRAIN_DATASET_URI`。
- `TRAIN_OUTPUT_URI` 必填，训练完成后会把整个输出目录递归上传到 MinIO。
- `train-nomodel` 镜像默认从 Hugging Face 拉模型。
- `runtime-with-model` 镜像默认使用 `/opt/models/gemma-4-31b-it-unsloth-bnb-4bit`。

### 3.2 Docker 训练示例

#### 带模型镜像训练

```bash
docker run --rm --gpus all \
  -e APP_MODE=train \
  -e MODEL_NAME_OR_PATH=/opt/models/gemma-4-31b-it-unsloth-bnb-4bit \
  -e MODEL_LOCAL_FILES_ONLY=true \
  -e TRAIN_DATASET_URI=s3://llm-datasets/cogiot/cogiot_xiaoge_identity_seed.jsonl \
  -e TRAIN_OUTPUT_DIR=/workspace/output/cogiot_xiaoge_identity_seed \
  -e TRAIN_OUTPUT_URI=s3://llm-runs/gemma4/cogiot_xiaoge_identity_seed/run-001 \
  -e TRAIN_NUM_EPOCHS=1 \
  -e TRAIN_BATCH_SIZE=1 \
  -e TRAIN_GRAD_ACCUM_STEPS=1 \
  -e TRAIN_MAX_SEQ_LENGTH=512 \
  -e TRAIN_SAVE_STEPS=20 \
  -e TRAIN_SAVE_TOTAL_LIMIT=2 \
  -e TRAIN_SAVE_ADAPTER=true \
  -e MINIO_ENDPOINT=minio.example.internal:9000 \
  -e MINIO_ACCESS_KEY=replace-me \
  -e MINIO_SECRET_KEY=replace-me \
  -e MINIO_SECURE=false \
  gemma4-qlora:runtime-with-model
```

#### 不带模型镜像训练

```bash
docker run --rm --gpus all \
  -e APP_MODE=train \
  -e MODEL_NAME_OR_PATH=unsloth/gemma-4-31b-it-unsloth-bnb-4bit \
  -e MODEL_LOCAL_FILES_ONLY=false \
  -e TRAIN_DATASET_URI=s3://llm-datasets/cogiot/cogiot_xiaoge_identity_seed.jsonl \
  -e TRAIN_OUTPUT_DIR=/workspace/output/cogiot_xiaoge_identity_seed \
  -e TRAIN_OUTPUT_URI=s3://llm-runs/gemma4/cogiot_xiaoge_identity_seed/run-001 \
  -e MINIO_ENDPOINT=minio.example.internal:9000 \
  -e MINIO_ACCESS_KEY=replace-me \
  -e MINIO_SECRET_KEY=replace-me \
  -e MINIO_SECURE=false \
  -e https_proxy=http://105.100.31.173:7897 \
  -e http_proxy=http://105.100.31.173:7897 \
  -e all_proxy=socks5://105.100.31.173:7897 \
  gemma4-qlora:train-nomodel
```

## 4. 服务模式

服务模式只支持带模型镜像，入口是：

```bash
APP_MODE=serve
```

启动时会做这几步：

1. 从 `SERVE_ARTIFACT_URI` 下载整次训练产物目录。
2. 校验其中必须存在 `adapter/adapter_model.safetensors`。
3. 加载 baked-in base model。
4. 从下载下来的 `adapter/` 加载 LoRA。
5. 如果训练产物里有 `processor/`，优先用它。
6. 暴露 OpenAI 兼容的 `POST /v1/chat/completions`。

### 4.1 关键环境变量

- `MODEL_NAME_OR_PATH`
- `SERVE_ARTIFACT_URI`
- `SERVE_ARTIFACT_DIR`
- `SERVE_HOST`
- `SERVE_PORT`
- `SERVE_MAX_NEW_TOKENS_DEFAULT`
- `SERVE_TEMPERATURE_DEFAULT`
- `SERVE_TOP_P_DEFAULT`
- `SERVE_RESPONSE_MODEL_NAME`
- `MINIO_ENDPOINT`
- `MINIO_ACCESS_KEY`
- `MINIO_SECRET_KEY`
- `MINIO_SECURE`
- `MINIO_REGION`

### 4.2 Docker 服务示例

```bash
docker run --rm --gpus all -p 8000:8000 \
  -e APP_MODE=serve \
  -e MODEL_NAME_OR_PATH=/opt/models/gemma-4-31b-it-unsloth-bnb-4bit \
  -e SERVE_ARTIFACT_URI=s3://llm-runs/gemma4/cogiot_xiaoge_identity_seed/run-001 \
  -e SERVE_HOST=0.0.0.0 \
  -e SERVE_PORT=8000 \
  -e SERVE_RESPONSE_MODEL_NAME=gemma4-31b-cogiot-xiaoge \
  -e MINIO_ENDPOINT=minio.example.internal:9000 \
  -e MINIO_ACCESS_KEY=replace-me \
  -e MINIO_SECRET_KEY=replace-me \
  -e MINIO_SECURE=false \
  gemma4-qlora:runtime-with-model
```

### 4.3 OpenAI 兼容调用示例

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma4-31b-cogiot-xiaoge",
    "messages": [
      {"role": "user", "content": "你是谁？"}
    ],
    "temperature": 0.2,
    "max_tokens": 128
  }'
```

## 5. K8s 示例

仓库里已经放了 3 个示例文件：

- [minio-secret.example.yaml](/Users/cog/project/doucument-lyp/llm-train/gemma4-qlora/k8s/minio-secret.example.yaml)
- [train-job.example.yaml](/Users/cog/project/doucument-lyp/llm-train/gemma4-qlora/k8s/train-job.example.yaml)
- [serve-deployment.example.yaml](/Users/cog/project/doucument-lyp/llm-train/gemma4-qlora/k8s/serve-deployment.example.yaml)

这两个 workload 示例默认已经指向正式仓库地址：

- `sz.cogiot.com/id348504946c01000/gemma4-qlora-runtime-with-model:latest`
- 如果训练时不想在镜像里内置模型，可改成 `sz.cogiot.com/id348504946c01000/gemma4-qlora-train-nomodel:latest`

在 K8s 里先创建镜像拉取 Secret：

```bash
kubectl create secret docker-registry sz-cogiot-registry \
  --docker-server=sz.cogiot.com \
  --docker-username='<your-user>' \
  --docker-password='<your-password>' \
  --docker-email='devnull@cogiot.com'
```

如果集群节点也配置了 Docker/containerd 代理，内网仓库同样要加到 `NO_PROXY`，否则会出现 TLS EOF 或握手异常。

使用顺序：

1. 创建 `sz-cogiot-registry` 拉取 Secret。
2. 改好 MinIO 地址、桶路径和训练产物前缀。
3. 创建 MinIO Secret。
4. 提交训练 Job。
5. 训练完成后，把 `SERVE_ARTIFACT_URI` 改成这次训练的完整输出前缀。
6. 部署服务 Deployment。

## 6. 行为边界

- `train-nomodel` 不支持 `APP_MODE=serve`，会直接失败。
- `SERVE_ARTIFACT_URI` 必须是一次训练结果的完整前缀，不能只指向 `adapter/` 子目录。
- 服务模式当前只支持非流式 `stream=false`。
- 服务模式当前只支持文本消息，不支持图片输入。
- 训练输出上传的是整个 `TRAIN_OUTPUT_DIR`，不会只裁剪成 adapter。
