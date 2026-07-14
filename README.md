# DARE-Gate

DARE-Gate 是一个面向 AIG 组合电路的多表征学习模型。模型同时学习节点的结构向量、功能向量和行为响应向量，并由一个训练管理器统一完成三类任务的分阶段训练。

## 1. 模型目标

模型为每个电路节点生成三种核心表示：

- `structure`：聚合前驱与后继信息的双向结构向量；
- `function`：采用 DeepGate2 风格前向消息传递得到的功能向量；
- `behavior`：由固定探针、门级行为传播、重汇聚修正和跨探针交互得到的行为响应向量。

当前架构包括：

1. 结构分支分别沿前驱方向和后继方向进行拓扑消息传递；
2. 消息只在真实电路边和当前拓扑层活跃节点上计算注意力；
3. 功能分支使用 DeepGate2 风格的 PI 初始化、AND/NOT 消息传递和更新；
4. 行为分支使用8个可学习固定探针，分别传播不同探针下的节点行为；
5. AND 门同时建模扇入注意力和扇入对交互；
6. 可利用拓扑预处理得到的重汇聚源修正行为消息；
7. 跨探针 Transformer 融合固定探针响应；
8. 中心化解码器输出行为响应，并保证探针维度上的响应和为零。

默认输出形状如下：

```text
structure                    [N, 128]
function                     [N, 128]
behavior                     [N, 128]
probe_behavior               [N, 8, 128]
response                     [N, 8, 2]
signature                    [N, 16]
probability                  [N]
reconvergence_probability    [num_rc_pairs]
functional_distance          [num_tt_pairs]
```

## 2. 项目结构

```text
MyModel/
├── README.md
└── Code/
    ├── train.py
    ├── resume_train.py
    ├── daregate/
    │   ├── cli/             # 训练与恢复入口实现
    │   ├── data/            # NPZ 数据集、划分和批处理
    │   ├── encoders/        # 结构、功能和行为编码器
    │   ├── heads/           # DeepGate2 风格监督任务头
    │   ├── layers/          # 注意力、GRU 和拓扑检查点公共层
    │   ├── training/        # 联合损失、训练器和输出管理
    │   ├── utils/           # 拓扑层级与重汇聚预处理
    │   ├── config.py
    │   └── model.py
    ├── data/
    │   ├── train/
    │   │   ├── graphs.npz
    │   │   └── finalLabels.npz
    │   └── output/
    └── tests/
```

## 3. 数据格式

训练程序只读取两个文件：

```text
Code/data/train/graphs.npz
Code/data/train/finalLabels.npz
```

`graphs.npz` 的顶层成员为 `circuits`，保存每个电路的节点类型和边：

```text
x
edge_index
```

`finalLabels.npz` 的顶层成员为 `labels` 和 `meta`。每个电路的合并标签包含：

```text
prob
rc_pair_index
is_rc
tt_pair_index
tt_dis
min_tt_dis
B
R
S
mu
num_pis
sensitivity
sensitivity_sum
sensitivity_var
```

数据集会严格检查 `graphs.npz` 与 `finalLabels.npz` 的电路名称集合是否完全一致。当前共有10182个对齐电路。

加载数据时会先为所有电路计算并缓存：

- `forward_level`；
- `backward_level`；
- 拓扑重汇聚源 `rc_source`。

这一步是电路拓扑预处理，不是模型训练；完成后，同一训练进程的所有 epoch 会复用缓存。

## 4. 运行环境

项目使用 Conda 环境 `myModelEnv`，Python 版本为3.10.20。所有命令都应在项目根目录运行：

```bash
conda activate myModelEnv
cd /root/autodl-tmp/MyModel
```

也可以直接使用环境中的解释器：

```bash
/root/miniconda3/envs/myModelEnv/bin/python Code/train.py
```

## 5. 从头训练

直接运行：

```bash
python Code/train.py
```

当前默认配置为：

```text
epochs          60
prebehavior_batch_size  64
behavior_batch_size     32
num_workers     0
AMP             enabled on CUDA
save_every      5
```

等价的显式命令为：

```bash
python Code/train.py \
  --data-dir Code/data/train \
  --output-dir Code/data/output/default \
  --epochs 60 \
  --prebehavior-batch-size 64 \
  --batch-size 32 \
  --num-workers 0 \
  --device cuda:0 \
  --functional-start-epoch 10 \
  --behavior-start-epoch 20 \
  --save-every 5
```

`train.py` 始终启动一次全新训练。如果目标实验目录已经存在，程序会
直接删除该目录及其中的旧日志、指标和断点，然后创建干净的新运行；
不会备份旧目录。为防止误删，`--output-dir` 只允许指向
`Code/data/output/` 下的具体实验子目录。

训练 DataLoader 会自动按照行为阶段切换 batch：epoch 0–19 使用64，
epoch 20起使用32；切换点跟随 `--behavior-start-epoch`。

## 6. 三阶段训练任务

默认训练阶段如下：

1. epoch 0–9，`structure_function_basic`：训练节点逻辑1概率和结构重汇聚任务；
2. epoch 10–19，`structure_function_full`：增加真值表距离功能监督；
3. epoch 20–59，`joint`：增加行为响应重建、响应相似度和功能—行为解耦任务。

行为分支在 epoch 20 之前不会执行，从而避免计算尚未参与损失的行为编码器。

训练默认启用联合阶段早停，`early_stopping_patience=15`。因此60是目标上限；如果联合阶段的验证指标长期没有改善，训练可能提前结束。

## 7. 断点保存与恢复

训练只在每5个完整 epoch 后原子覆盖同一个断点文件：

```text
Code/data/output/default/checkpoints/model_last.pt
```

断点包含：

- 模型参数；
- 优化器与学习率调度器状态；
- AMP GradScaler 状态；
- 已完成 epoch 和当前训练阶段；
- 最佳指标与早停状态；
- 训练历史；
- 模型、损失和训练器配置。

程序不会生成 `model_best.pt` 或 `model_interrupted.pt`。主动按 `Ctrl+C` 时不会额外保存，也不会更新 `model_last.pt`。例如在第13个 epoch 中断，下次将从第10个 epoch 的周期断点恢复。

使用默认断点继续训练到总计60个 epoch：

```bash
python Code/resume_train.py
```

指定断点和目标总轮数：

```bash
python Code/resume_train.py \
  --checkpoint Code/data/output/default/checkpoints/model_last.pt \
  --epochs 60 \
  --prebehavior-batch-size 64 \
  --batch-size 32
```

`--epochs` 表示最终总 epoch 数，不是额外增加的轮数。恢复训练会复用原始 `data_split.json`，并恢复完整优化状态。未传入覆盖参数时，旧断点也会采用早期64、行为阶段32的默认调度。
`--prebehavior-batch-size` 控制行为阶段之前的 batch，`--batch-size` 控制
行为阶段开始后的 batch。

## 8. 数据划分

首次训练会生成：

```text
Code/data/output/default/logs/data_split.json
```

数据按照原始电路设计族划分，确保同一原始电路的不同综合变体不会同时出现在训练集、验证集或测试集中，避免数据泄漏。恢复训练必须复用该划分文件。

## 9. 进度显示

交互式终端默认显示训练和验证 `tqdm` 进度条，包括：

- 当前 epoch 和训练阶段；
- batch 进度、耗时、速度和 ETA；
- 总损失及当前阶段损失项；
- 学习率；
- PyTorch 已分配的 GPU 显存。

示例：

```text
2026-07-14 13:33:44 train E1/60 structure_function_basic: 42%|...|
loss=1.1023, prob=0.1412, rc=0.6787, lr=1.00e-04, gpu=2.31G
```

可调整刷新间隔或关闭进度条：

```bash
python Code/train.py --progress-min-interval 1.0
python Code/train.py --no-progress
```

## 10. 输出目录

每次运行的输出按类别保存：

```text
Code/data/output/default/
├── checkpoints/
│   └── model_last.pt
├── logs/
│   ├── data_split.json
│   ├── run_config.json
│   ├── resume_config.json
│   └── history.json
├── metrics/
├── predictions/
└── embeddings/
```

`history.json` 保存完整训练历史，`metrics` 按 epoch 保存指标。调用训练器的 `export_outputs` 后，模型预测与三种节点向量分别写入 `predictions` 和 `embeddings`。

## 11. Python 接口

模型可以独立调用：

```python
from Code.daregate import DAREGateModel

model = DAREGateModel()
output = model(
    gate_type,
    edge_index,
    forward_level,
    backward_level=backward_level,
    rc_source=rc_source,
    tt_pair_index=tt_pair_index,
    rc_pair_index=rc_pair_index,
)
```

训练管理器负责优化流程，模型类只负责前向计算：

```python
from Code.daregate import (
    DAREGateModel,
    DAREGateTrainer,
    ObjectiveConfig,
    TrainerConfig,
)

trainer = DAREGateTrainer(
    DAREGateModel(),
    trainer_config=TrainerConfig(
        learning_rate=1e-4,
        weight_decay=1e-5,
        output_dir="Code/data/output/api_run",
        save_every=5,
    ),
    objective_config=ObjectiveConfig(
        functionality_start_epoch=10,
        behavior_start_epoch=20,
    ),
)
trainer.fit(
    prebehavior_train_loader,
    validation_loader,
    num_epochs=60,
    late_train_loader=joint_train_loader,
    train_loader_switch_epoch=20,
)
```

DataLoader 应输出 `DAREGateBatch`，或字段名称相同的 mapping/对象。联合训练使用的主要字段为：

```text
gate_type          [N]
edge_index         [2, E]
forward_level      [N]
backward_level     [N]
rc_source          [N]
signature          [N, 16]
probability        [N]
tt_pair_index      [2, P_tt]
tt_distance        [P_tt]
rc_pair_index      [2, P_rc]
is_reconvergent    [P_rc]
batch_index        [N]
```

导出节点表示和预测：

```python
trainer.export_outputs(validation_loader, split="validation")
```

## 12. 显存与吞吐优化

当前实现包含以下不改变消息传递语义的计算优化：

- 注意力只投影当前拓扑层真实边涉及的活跃源节点和目标节点；
- softmax 始终只在同一目标节点的真实邻居之间计算；
- 拓扑递归按8个层级一段进行梯度检查点重计算；
- 跨探针 Transformer 按节点分块处理；
- 行为阶段开始前跳过完整行为分支，并使用 batch 64；
- 行为阶段开始后自动切换为 batch 32；
- CUDA 默认启用 AMP；
- 设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 降低显存碎片。

在 RTX 4090 24 GiB 上，包含32个最大训练电路、共101566个节点的联合训练压力 batch 已通过前向和反向验证，PyTorch 峰值分配显存约16.53 GiB。由于电路 DAG 的拓扑层存在严格顺序依赖，GPU 利用率会随层宽波动，不能保证持续稳定在85%以上。

当前 NPZ 使用对象字典，默认 `num_workers=0` 最稳妥，可避免 worker 进程复制完整归档对象。除非完成数据分片格式转换，否则不建议盲目增加 worker 数。

## 13. 测试与帮助

运行测试：

```bash
python -m unittest discover -s Code/tests -v
```

查看全部训练参数：

```bash
python Code/train.py --help
python Code/resume_train.py --help
```
