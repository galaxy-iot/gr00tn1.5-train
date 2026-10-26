# GR00T N1.5 双相机训练流程

本项目使用全局相机和腕部相机，微调 `nvidia/GR00T-N1.5-3B`。
GR00T 代码固定在 `4af2b622892f7dcb5aae5a3fb70bcb02dc217b96`，
GitHub 克隆和 FlashAttention wheel 下载统一使用 `https://gh-proxy.org/` 代理。

## 常用入口

所有命令在项目根目录执行，日常操作使用以下入口：

| 阶段 | 命令 | 执行脚本 |
| --- | --- | --- |
| 准备全部环境 | `./run.sh setup` | `scripts/pipeline.py` |
| 转换数据 | `./run.sh convert --task "任务指令"` | `scripts/process_bags.py` → `scripts/build_dataset.py` |
| 检查数据 | `./run.sh check-data` | `scripts/validate_dataset.py`、`scripts/check_training_data.py` |
| 正式训练 | `./run.sh train --output ./checkpoints/train_run1` | `scripts/train.py` |
| 量化 | `bash quantize.sh quantize ...` | `scripts/quantize_model.py` |
| 编译 QNN | `bash quantize.sh qnn ...` | `scripts/export_qnn.py` |
| 自动执行完整流程 | `./run.sh deploy --chipset qualcomm-qcs8275` | `scripts/deployment_workflow.py` |

内部职责：`pipeline_cli.py` 管理命令参数，`pipeline.py` 执行环境、数据和训练阶段，
`deployment_workflow.py` 串联部署流程，`workflow_state.py` 管理完成记录和重试。
`quantization_config.py` 统一注册机器人配置，`quantization_samples.py` 处理双相机校准样本。
`ort_qnn_standalone.py` 是独立推理工具，不参与训练和导出流程。
`all` 保留原有“环境准备 → 转换 → 10 步 smoke”行为；完整正式流程使用 `deploy`。

## 本机开发验证

```bash
./run.sh convert --task "吸取桌上的物体并放到指定位置"
./run.sh check-data
./run.sh smoke --dry-run
./run.sh train --dry-run
```

`convert` 使用 `gr00t-data` 环境解析录包、生成双相机 LeRobot v2 数据集，
并完整解码视频检查帧数、时间对齐、动作和归一化统计。
视频编码默认并行执行 2 个任务，每个 H.264 编码器使用 2 个线程。
可运行 `./run.sh convert --task "任务指令" --video-workers 4 --encoder-threads 2` 调整并发；
`--video-workers 1` 可让视频任务串行执行。并发越高，内存和磁盘读取压力越大。
录包解析和最终验证仍串行执行；`--workers` 仅控制训练数据加载。
默认直接扫描 `datas/*/metadata.yaml`，使用全部录包，不读取 `configs/episodes.json`。
转换必须通过 `--task "任务指令"` 为全部录包指定统一语言指令，或通过 `--manifest` 提供逐录包标注；两者互斥。
指令会写入 `episodes.source.json` 和训练/验证集的 `meta/tasks.jsonl`，由语言字段索引引用。
空白指令和 `__UNLABELED...` 占位文本会被拒绝，`--allow-draft` 也不会绕过语言检查。
按录包名称排序后每第 10 个留作验证，其余用于训练；时长从解析报告读取。
需要自定义标注时才传 `--manifest PATH`；构建采用的标注会保存在数据集的 `episodes.source.json`。
`check-data` 还使用 `gr00t-train` 环境里的真实 GR00T 数据加载器，
检查每个片段首尾样本的两路图像、变换后的状态/动作及批次拼接。
这两个阶段不需要 GPU，也不下载基础模型权重；首次加载处理器会生成 Hugging Face 本地模块缓存。
`--dry-run` 只打印命令，不训练，也不代表模型前向、反向或检查点保存已验证。

默认路径：

| 路径 | 用途 |
| --- | --- |
| `datas/` | 原始 ROS2 MCAP，两路相机都在这里 |
| `episode1_urdf_1113.urdf` | 机器人 URDF |
| `configs/episodes.json` | 可选人工标注示例，仅显式传入 `--manifest` 时使用 |
| `processed_dual/` | 两路图像时间戳、关节/FK、吸盘状态和报告 |
| `datasets/episode_pose_dual/` | 双相机训练集和验证集 |
| `episode_config.py` | GR00T 输入及变换配置 |

已完成输出可复用。输入配置变化时，使用新的 `--processed` / `--dataset` 路径；
不覆盖非空的未完成输出。旧的单相机中间文件和数据集已清理。

## 双相机与动作定义

- `/global_camera/image_raw/compressed` → `observation.images.front` → `video.front`。
- `/wrist_camera/image_raw/compressed` → `observation.images.wrist` → `video.wrist`。
- 两路图像分别按 header 时间戳，在同一 20 Hz 网格选择不晚于当前时刻的最近帧。
  任一路图像超过 75 ms、关节超过 50 ms 的样本均剔除，并在间断处切分片段。
- 每路视频保存为 640×480，模型输入缩放到 224×224；视角顺序固定为全局、腕部。
- 状态是 `base_link` 到 `link6` 的 FK 位置、旋转矩阵前两行的 6D 表示以及吸盘状态，共 10 维。
- 动作是下一帧（50 ms 后）的测量位姿和吸盘状态，并非录制的控制器目标。
  模型预测 16 步动作；位置按训练集统计归一化，6D 旋转保持原表示。
- 这是 `link6` 位姿，不是吸盘 TCP。部署控制器需要使用一致的坐标和动作语义。

## 一键转换、训练、量化和 QNN ONNX 编译

首次使用一条命令准备数据处理、训练和量化三个环境（需要已安装 Conda 和 uv）：

```bash
./run.sh setup
```

该命令依次准备 `gr00t-data`、`gr00t-train` 和 `.venv-quant`，并检查训练 GPU。
已有环境会复用并安装依赖；仅准备某个环境可使用 `setup-data`、`setup-train`、
`setup-quant`。AI Hub 账号凭据首次使用时另行配置：

```bash
.venv-quant/bin/qai-hub configure --api_token YOUR_API_TOKEN
```

运行完整流程（目标芯片需在 AI Hub 中受支持）：

```bash
./run.sh deploy \
  --chipset qualcomm-qcs8275 \
  --task "吸取桌上的物体并放到指定位置" \
  --dataset ./datasets/episode_pose_dual_labeled \
  --work-dir ./exports/workflow \
  --max-steps 10000 \
  --batch-size 1 \
  --num-samples 100
```

`deploy` 依次转换数据、正式训练、量化校准并上传 AI Hub，生成
`precompiled_qnn_onnx`。它复用 `convert` / `train` 的输入、数据集、环境和训练参数；
也可用 `--device "准确设备名称"` 替代 `--chipset`。加 `--dry-run` 只预览命令。

已有训练结果且其数据集已带真实任务指令时，使用以下命令跳过训练：

```bash
./run.sh deploy \
  --checkpoint ./checkpoint-10000 \
  --dataset ./datasets/episode_pose_dual_labeled \
  --chipset qualcomm-qcs8275 \
  --work-dir ./exports/existing_checkpoint
```

相同命令可以重复执行：

- 已有双相机数据集时跳过转换，但仍解码验证；修改原始录包、URDF 或数据构建配置后，应指定新的 `--dataset` / `--processed`。
- 训练、量化和编译成功后分别写入 `--work-dir` 下的 JSON 完成记录；再次执行会校验配置及产物并跳过成功阶段。
- 失败阶段保留已有文件，再次执行会使用新的尝试目录重新运行该阶段，不会从失败阶段内部断点续跑。
- 输入、脚本或相关配置变化时拒绝复用已有完成记录，请指定新的 `--work-dir`。手动生成的量化/编译目录不会仅凭存在就视为成功。

最终目录会打印为 `Precompiled QNN ONNX output: ...`，也记录在工作目录的
`qnn.json` 中。此流程需要本地 CUDA GPU、模型下载权限以及可用的 AI Hub 凭据；
导出仍跳过示例推理，需要在目标板用实际机器人输入验证。训练环境不会自动重装，
量化环境不存在时会自动运行 `quantize.sh setup`。

## 量化与 ONNX 导出的 uv 环境

使用独立的 `.venv-quant`，不需要激活 conda 或准备本地 Qualcomm 源码。
安装 uv 后运行：

```bash
bash quantize.sh setup
bash quantize.sh check
bash quantize.sh quantize \
  --checkpoint ./checkpoint-10000 \
  --dataset ./datasets/episode_pose_dual \
  --output ./exports/gr00t_quantized \
  --num-samples 100
```

`--num-samples` 是每个校准 / 评估分组的时间步数，不是完整轨迹数。
两组各抽取 100 个不重复的时间步；ViT 每个样本保留同一时间步的两路相机，
输入形状为 `(2, 3, 224, 224)`。量化入口会重新采集校准数据，避免复用其他
权重或数据集的旧缓存。若之前量化失败但已生成输出目录，请换一个新的 `--output`。

`setup` 创建 Python 3.12 环境，通过 uv 从配置的包索引安装正式发布的
`qai-hub-models` 及其 CLI，再执行官方 `grootn15` 依赖安装器（自动确认安装）。
默认安装最新稳定版，可设置 `QAI_HUB_MODELS_VERSION` 指定正式版本。
依赖快照保存在 `.venv-quant/requirements-installed.txt`；它用于记录环境，
不是可移植锁文件。需要更新依赖时重新运行 setup。

普通浮点 ONNX 导出可使用同一环境：

```bash
bash quantize.sh onnx \
  --checkpoint ./checkpoint-10000 \
  --dataset ./datasets/episode_pose_dual \
  --output ./exports/gr00t_float_onnx
```

量化需要 CUDA GPU，官方建议 3B 模型使用 40 GB 显存；此入口仍调用
官方量化流程，不解决小显存卡 OOM。浮点 ONNX 导出使用 CPU。
量化校准产物包含 ONNX、外部权重和 AIMET encodings，后续仍需匹配
QNN 的部署导出流程。安装和首次加载可能需要联网下载依赖、外部源码及模型资源。

## 导出离线语言预处理资源

已有训练 checkpoint 时无需重新训练，使用量化环境导出：

```bash
bash quantize.sh preprocess \
  --checkpoint ./checkpoint-10000 \
  --output ./exports/checkpoint-10000_preprocessing
```

此步骤在 CPU 上运行，从 checkpoint 提取词嵌入，并复制固定版本 GR00T 仓库自带的
Eagle Tokenizer、Processor、自定义处理代码、模型配置及 `experiment_cfg`。
自动查找 `third_party/Isaac-GR00T` 或本机 Qualcomm 缓存；找不到时使用
`--processor-dir /path/to/gr00t/model/backbone/eagle2_hg_model` 指定训练使用的资源。
不下载完整基础模型，输出目录必须不存在。

将整个输出目录复制到另一台设备，在兼容环境安装其中的 `requirements-preprocessing.txt`，然后运行：

```bash
python prepare_language.py \
  --bundle . \
  --instruction "吸取桌上的物体并放到指定位置" \
  --output ./instruction.npz
```

请在资源目录内执行以上命令。支持离线处理，输出 `input_ids`、`attention_mask`、
`input_embeds`、`image_token_mask`、`llm_attention_mask`。当前双相机配置的
`input_embeds` 形状为 `(1, 640, 2048)`。调用 LLM 前，必须用实时 ViT 输出替换
`input_embeds[image_token_mask]`；`ort_qnn_standalone.py` 的组件流水线已有这一替换步骤。
图像和机器人状态预处理、投影器/DiT mask、扩散调度及动作反归一化仍需在推理程序中实现。
此资源包单独执行 `preprocess` 导出，目前不会由 `qnn` / `deploy` 自动附带。

## 可选：完善任务标注

用真实任务指令生成数据集后可直接运行 `./run.sh train --dataset 数据集路径`，无需人工确认标志。
以下标注用于完善任务描述和片段质量，不会阻止训练。

旧数据集的 60 个录包自动生成草稿，按每第 10 个留作验证，
共 54 个训练录包和 6 个验证录包；每个录包暂用完整范围。
旧数据集任务文本为 `__UNLABELED_DO_NOT_TRAIN__`，所有人工确认标志为 false。
新转换流程不会再接受该占位文本。请提供真实指令并生成到新目录（中间解析数据可复用）：

```bash
./run.sh convert \
  --task "吸取桌上的物体并放到指定位置" \
  --dataset ./datasets/episode_pose_dual_labeled
```

以上指令仅为示例，请按录包实际任务填写。仅补充语言指令不会自动确认时间边界或同步标志。

填写每个片段的 `task`、`start_seconds`、`end_seconds`，确认片段边界后设置
该片段及顶层的 `boundaries_confirmed`。核对时钟和动作定义后，再设置
`synchronization_confirmed` 和 `feedback_as_action_confirmed`。
`camera_time_offsets_seconds` 可分别配置 `front`、`wrist` 的时间偏移；不应为了绕过检查随意设置。
`initial_suction: null` 表示未知，会舍弃首次吸盘事件之前的数据。
用户已确认吸盘初始状态默认为关闭，自动生成的草稿使用 `initial_suction: false`，
保留首次吸盘事件前的数据；首次事件及之后按实际记录的状态更新。
若某次录制初始状态不同，可通过显式标注设置 `initial_suction: true` 或 `null`。
旧的 37 个录包清单保存在 `configs/episodes.20261022.backup.json`，其中的初始状态确认不适用于新录包。
需要人工标注时，可复制数据集的 `episodes.source.json` 后编辑，再通过
`./run.sh convert --manifest configs/episodes.new.json --dataset datasets/episode_pose_reviewed` 指定清单。
恢复后的数据集位于 `datasets/episode_pose_dual/`。

到其他机器验证训练时，无需复制 16 GB 的原始 `datas/` 或 `processed_dual/`；
复制本项目脚本、`episode_config.py`、`third_party/Isaac-GR00T/` 和
`datasets/episode_pose_dual/` 即可运行 `check-data` / `smoke`。
若要在目标机器重新执行 `convert`，则需保留原始 `datas/` 和 `processed_dual/`。

修改标注后重建到新路径：

```bash
./run.sh convert --task "任务指令" --dataset datasets/episode_pose_dual_labeled
./run.sh check-data --dataset datasets/episode_pose_dual
```

## 目标 GPU 机器

复制项目代码、URDF、配置和生成的数据集到目标机器；只训练时不必复制原始 MCAP。
若需要重新转换，需携带录包并在目标机器重新生成中间报告（报告记录源录包绝对路径）。
训练环境要求支持当前 BF16 / FlashAttention 2 配方的 NVIDIA GPU。

```bash
./run.sh setup-train
./run.sh check-data --dataset datasets/episode_pose_dual
./run.sh smoke --dataset datasets/episode_pose_dual --batch-size 1
./run.sh train --dataset datasets/episode_pose_dual_labeled --batch-size 1 --max-steps 10000
```

`smoke` 运行真实 10 步训练，并检查有限 loss/梯度和检查点保存；允许草稿数据测试连通性。
`train` 允许当前生成的草稿数据；首次训练会下载基础模型。批量大小需按目标机器显存调整。
验证集独立保存，当前训练入口没有自动执行任务成功率评估。
