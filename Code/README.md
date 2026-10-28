# TriWave-GNN

TriWave-GNN 从标准单元引脚级异构图学习结构表征 `H_s`、稳定功能表征 `H_f` 和瞬态行为表征 `H_b`。输入包含 net edge、cell edge、物理特征和三类监督标签。本目录负责模型、训练、评估及嵌入导出；仿真与标签生成在另一台机器完成。

图数据使用 PyG `HeteroData`：`pin` 是统一的引脚节点类型，PI、PO 和单元输入/输出以角色属性区分；`('pin','net','pin')` 与 `('pin','cell','pin')` 保存两种关系。训练使用 PyG `DataLoader`，六个关系消息模块继承 `MessagePassing`，cell 注意力按目标入边使用 PyG `softmax`。一次拓扑前向扫描与三路 GRU 由模型显式调度。

## 项目结构

```text
Code/
├── pyproject.toml               安装与命令入口
├── setup.py                    包安装配置
├── install.sh                  安装入口
├── config.py                    项目级训练配置入口
├── debug.py                     单电路表征及预测检查
├── examples/
│   ├── data/train/             训练输入：graphs.npz、labels.npz
│   ├── exp/                    训练输出和检查点（Git 忽略）
│   └── *.py                    训练、恢复、评估、表征导出、发布入口
├── triwave_gnn/
│   ├── arch/                    MLP、TFMLP、消息、PI 初始化、拓扑传播、读出
│   ├── pretrained/              发布后的通用权重入口
│   ├── dataset.py               NPZ 读取、物理特征标准化、按设计取样
│   ├── parser.py                NPZ 解析与训练/验证/测试划分入口
│   ├── parser_func.py           PyG 图对象、配对批处理规则、输入输出引脚查询
│   ├── model.py                 直接声明消息聚合、GRU 更新和读出头；forward 返回 Hs/Hf/Hb
│   ├── trainer.py               数据划分、训练、验证、早停、恢复
│   └── utils/                   数据契约、DAG、标准化、损失、检查点、日志、种子、模型加载
```

模型使用可区分的 PI 结构初态和共享的 PI 功能初态，在引脚 DAG 上按拓扑深度传播 net/cell 两类消息。结构路消息读取前驱 `H_s`，功能路读取前驱 `H_s+H_f`，行为路读取前驱 `H_s+H_f+H_b`。每路消息经共享的关系网络生成、聚合，再由本路 GRU 更新；只有行为路读取物理边特征。cell 多输入消息使用目标更新前的本路状态参与 MLP 注意力打分。基础模型不加入连接两端逻辑类型的显式边上下文 `C_uv`。

训练入口先由 `NpzParser` 取得训练、验证、测试电路，再构造 `TriWaveGNN` 和 `TriWaveTrainer`；示例脚本逐阶段设置监督任务并调用 `trainer.train()`。`model.forward()` 返回 `H_s/H_f/H_b`，Trainer 的 `run_batch()` 调用模型读出头并计算阶段损失。具体代码见 [训练入口](examples/train.py)、[模型组装](triwave_gnn/model.py)、[训练器](triwave_gnn/trainer.py)。

## 数据与安装

`Code/` 是完整项目根目录。默认输入是上传到 `Code/examples/data/train/` 的 `graphs.npz` 与 `labels.npz`；当前正式数据含 11,087 个设计、60,182,921 个节点。两份文件须使用 `concatenated-ptr-1.0` 格式，且设计 ID、节点编号和网表哈希一致。训练启动时自动检查字段、指针、边界、拓扑、标签掩码与 `training_ready`。如果仿真机器改变了数据格式，需要同步更新解析器和契约。大数据包首次读取、验证和统计词表需要一定时间。

本机已有 PyTorch 时，进入项目根目录运行：

```bash
cd Code
python -m pip install -e .
python examples/validate.py
```

依赖要求见 `pyproject.toml`，包含 `torch-geometric>=2.8,<3`；本机 `myModelEnv` 已安装 PyTorch 2.5.0+cu124 和 PyG 2.8.0。

## 小规模训练检查

先在 VS Code 打开并运行 `examples/train_small.py`。其 `SMALL_CONFIG` 默认最多选取 30 个设计（包含训练、验证和测试），每个设计不超过 10,000 个节点，三个阶段各训练 3 轮。它复用 `train.py` 的训练流程，日志和检查点也保存在 `examples/exp/`。

抽样使用固定随机种子，并在运行目录保存 `split.json`。可在脚本中的 `SMALL_CONFIG` 调整规模，也可传入 `--max-designs`、`--max-nodes` 和各阶段轮数。小规模入口仍读取并验证完整 NPZ，启动时可能需要几分钟；实际训练只使用选中的设计。

此步骤用于观察损失是否下降、是否出现 NaN 或异常，不代表模型已收敛或完成对比实验。

## 正式训练

首次使用一批 NPZ 时，先运行 `examples/prepare_split.py`。它只读取元数据并计算文件哈希，生成 `examples/data/train/split.json`，不启动训练；已有名单验证通过后会保留，不会覆盖。

`train.py` 默认使用 `--split-mode fixed`，严格按该名单加载训练、验证、测试设计。同一网表哈希的设计始终放在同一集合，按网表组约 70%/15%/15% 划分。这是当前数据的内部评估划分，未验证子图的原电路分组。小规模入口显式使用 `--split-mode random`，不改变正式名单。

`train.py` 的 `--max-designs` 和 `--max-nodes` 默认均为 0，表示不限制设计数和节点数。

在 `Code/` 内运行：

```bash
python examples/train.py
```

各 `examples/*.py` 入口也支持从任意工作目录用绝对路径直接运行。例如在 AutoDL 上可执行 `/root/miniconda3/envs/myModelEnv/bin/python /root/autodl-tmp/MyModel/Code/examples/train.py`，无需先激活 Conda 环境或执行可编辑安装。

默认使用 NPZ 中**全部**设计，不会按节点数跳过大电路。默认隐藏维度 48，三个阶段最多分别训练 30、30、40 轮，每阶段验证损失连续 10 轮未改善则早停；每阶段可用 `--lr-step` 设置学习率衰减间隔。`--batch-size` 默认为 32，表示每次更新处理 32 个完整电路图；也可用 `utils/batch.py` 将多个完整引脚图合成互不连通的批图。交互终端使用两行动态面板：第一行为带方括号边框的连续播放轨道（粗线表示已完成、细线表示剩余，圆点表示当前位置）、阶段、轮次、批次计数、完成百分比、已用时间和 ETA；第二行为当前轮次累计损失及各项指标，准确率显示为百分比。面板在原位置刷新，最多每秒刷新 5 次，每个 batch 不新增日志。动态面板两行合为一条状态提示，仅第一行带黄色 `[INFO]`，第二行缩进显示指标；其他永久提示也使用该前缀；终端较窄时自动缩短指标名称，仍不足时省略尾部指标，完整指标保留在轮次汇总及 JSON 日志。训练和验证分别显示进度，结束后移除临时面板并保留轮次汇总。输出重定向到文件或不支持动态显示的终端时，每 `--progress-interval` 批次保存一次纯文本快照，默认 100；设为 0 时仅输出每次训练/验证/测试遍历完成时的快照。每轮分别汇总训练和验证指标，并记录在 `history.json`、`timeline.jsonl`。训练按完整电路划分训练、验证和测试集；物理特征标准化只使用训练设计。运行结果保存在自动创建的 `Code/examples/exp/triwave_gnn_<时间>/`，不会覆盖旧运行。

三个阶段依次为：

1. `H_s` 重汇聚关系 + `H_f` 逻辑 1 概率；
2. 保留第一阶段任务，加入真值表距离与 `H_f` 余弦距离的一致性；
3. 保留前两阶段任务，加入 `H_b` 的四项监督：总体翻转概率 `hb_tp`、首次响应时间 `hb_first_mean`、最终稳定时间 `hb_stable_mean`、毛刺发生率 `hb_glitch_rate`。方向翻转和其他行为字段保留在数据中，不计入初步模型的主损失。

阶段损失权重为概率:重汇聚:功能距离:行为总项 = `3:1:2:2`；行为总项内的四项任务等权。

每个阶段在验证损失改善时保存最佳检查点；默认每 **5 轮**保存一次 `last.pt`，阶段结束或早停时也保存。`--save-interval` 可调整间隔。`last.pt` 包含模型、优化器、阶段、数据哈希、划分、随机状态，以及该保存时刻的阶段最佳检查点副本；恢复时会还原最佳模型选择，避免未保存轮次产生的后续最佳文件影响续跑。中断后重跑最近保存点之后的轮次；第 5 轮保存前若中断且阶段尚未结束，还没有 `last.pt`。每轮日志仍照常记录。恢复同一份数据的中断训练：

```bash
python examples/resume.py examples/exp/<运行目录>
```

也可修改 `examples/resume.py` 顶部的 `RUN_DIR`，将其设为要恢复的实验目录，然后直接在 IDE 中运行该脚本，或执行 `python examples/resume.py`。脚本自动读取该目录下的 `last.pt`；命令行显式传入的运行目录会覆盖 `RUN_DIR`。

若显存不足，可缩小 batch 后从最近保存点恢复，例如：

```bash
python examples/resume.py examples/exp/<运行目录> --batch-size 16
```

恢复入口默认沿用检查点记录的 batch size，显式参数可覆盖；模型和优化器状态保留，后续更新使用新的 batch，因此训练轨迹会改变。无需沿用中断轮次的部分进度，该轮从最近保存点重新执行。若尚无 `last.pt`，应运行 `python examples/train.py --batch-size 16` 新建运行。

替换 NPZ 后，将旧 `split.json` 随旧数据归档，再运行 `examples/prepare_split.py` 生成新名单并启动新运行；恢复命令会拒绝数据哈希改变。PyG 迁移保留四任务架构的参数形状和顺序，并自动转换消息层参数名，因此同架构、同数据的检查点可继续加载。更早的十任务架构仍不兼容。小规模流程验证使用 `train_small.py`；也可给 `train.py` 显式添加 `--split-mode random --max-nodes 10000 --stage1-epochs 2 --stage2-epochs 2 --stage3-epochs 2`。`--max-nodes` 默认是 `0`，表示无上限。

## 评估与表征导出

```bash
python examples/evaluate.py --run examples/exp/<运行目录>
python examples/export.py --checkpoint examples/exp/<运行目录>/best.pt
```

评估只重新推理完整验证集和测试集，输出 `metrics_extended.json`，保留原来的 `metrics.json`；旧结果还会原样嵌入扩展文件的 `previous_metrics`。导出命令为每个设计生成一个 NPZ，保存 `node_id`、`hs`、`hf`、`hb` 和节点映射哈希；可以用 `--design <设计ID>` 只导出指定设计。

`evaluate.py` 默认加载运行目录的 `best.pt`；不存在时明确警告并回退到 `last.pt`。也可以通过 `--checkpoint <检查点路径>` 显式指定文件（相对路径以当前工作目录为基准）。评估不会训练模型或修改数据划分。

直接在 VS Code 中运行时，修改 `evaluate.py` 顶部的 `RUN_DIR` 即可，不需要填写命令行参数。`CHECKPOINT = None` 使用自动选择；也可改成 `CHECKPOINT = RUN_DIR / "last.pt"` 指定文件。运行目录还用于读取 `split.json` 和保存 `metrics_extended.json`。命令行的 `--run`、`--checkpoint` 可覆盖对应配置。

功能等价识别的 pooled/micro 结果位于 `metrics_extended.json` 的 `val`（或 `validation`）和 `test` 下的 `functional_equivalence`。仅使用功能配对掩码为真且距离标签有限的样本，以 `torch.isclose(distance, 0, atol=1e-8, rtol=0)` 标记正样本；相似度为 `1 - functional_distance(hf_i, hf_j)`。在完整验证集上选取 F1 最大的阈值，再固定阈值评价测试集，预测规则为 `score >= threshold`。相同分数整组分类；多个阈值的 F1 相同时选择更高的阈值。Precision、Recall、F1、Accuracy、ROC-AUC 使用 0～1 小数，同时记录 TP/FP/TN/FN 和正负样本数。ROC-AUC 对同分正负样本计半分，无额外依赖。

完整验证集或测试集没有有效功能配对时评估报错。验证集缺少任一类别时无法校准阈值，测试集的阈值相关指标写为 `null`；单类别电路的 AUC 写为 `null`。Precision 分母为预测正样本数，Recall 分母为实际正样本数，F1 分母为 `2TP+FP+FN`；对应分母为零时写为 `null`，并在 `undefined_reasons` 说明。正样本少于 100 个会提示样本过少（这是报告稳定性提示，不是 DeepGate2 的规定）；应重新生成评价配对，模型无需重新训练。

终端复用预训练的两行动态进度、已用时间、ETA 和 `[INFO]` 日志，完成测试后完整打印全部测试汇总。保留 `hf_prob_mae`、`hs_rc_error_rate`、`hf_tt_cosine_distance_mae`、`hf_tt_rank_accuracy` 及四项行为 MAE，新增 `hs_rc_accuracy = 1 - hs_rc_error_rate`，这些指标均保留有效 `count`。不统计模型推理效率。

扩展 JSON 的测试结果字段：

- `test.functional_equivalence`：所有测试配对合并的 pooled/micro 指标、混淆矩阵、正负样本数、正样本比例和验证集阈值。
- `test.per_circuit`：每个测试电路的全部原有指标和功能等价指标；无功能配对的电路仍保留，功能等价指标为 `null`。
- `test.functional_equivalence_macro`：逐电路算术平均，每项指标分别排除 `null`；`contributing_circuits` 记录每项指标实际参与平均的电路数。该平均不会按电路大小或配对数加权。单类别电路的 AUC 不参与平均；其他比率按各自分母判断是否定义。
- `test.similarity_diagnostics`：全部/正类/负类分数的 min、1%、5%、25%、50%、75%、95%、99%、max 分位数，分数恰为 1、超过 1 和接近 1 的比例。接近 1 定义为 `abs(score-1) <= atol`，容差分别为 `1e-8`、`1e-7`、`1e-6`、`1e-4`，均含恰为 1 的分数；不裁剪浮点超界值。
- `audit`：相似度方向、原始 NPZ 配对/标签/掩码与电路边界检查、验证阈值选择规则、标签来源，以及与原测试汇总的逐项差异。

`SAVE_PAIR_SCORES = True` 时同时保存 `equivalence_val_scores.npz` 和 `equivalence_test_scores.npz`，包含电路 ID、配对分段指针、相似度、正类布尔标签，供后续诊断使用。分数在收集后立即转到 CPU。

当前完整 NPZ 中每个设计的 `static_samples` 和 `hf_vector_count` 均为 15,000；项目生成方案描述随机稳态输入，研究记录说明距离来自采样响应汉明距离。当前包未包含激励向量、随机种子、生成执行日志或节点对 SAT/CEC 证明，且说明文档的一部分针对早期 30 设计包。因此 TT=0 在本评估中表示有限采样响应一致，不能宣称经过形式验证的严格功能等价。

`hf_prob_mae` 对应逻辑概率 MAE，结构重汇聚误差不能当作功能等价识别指标。与 DeepGate2 比较时还应说明评价数据、配对采样和宏平均口径是否一致。

运行契约测试：

```bash
python -m unittest discover -s triwave_gnn/tests -v
```

旧架构曾用全部 30 个设计、隐藏维度 16、每阶段一轮完成流程验证，结果仍保存在 `examples/exp/pilot_30_designs/`，但不能用当前模型继续训练或推理。当前架构已用全部 30 个设计、隐藏维度 8、每阶段一轮完成训练、评估和恢复流程验证；结果保存在 `examples/exp/revised_pilot_30_designs/`，第三阶段测试损失为 0.38336。该流程检查不代表模型已收敛。正式训练应新建运行目录。

所有文件的详细说明见 [文件导览](../explain/FILE_GUIDE.md)，PyG 数据布局与迁移验证见 [PyG 说明](../explain/PYG_MIGRATION.md)。完成正式训练后可用 `python examples/publish.py --run examples/exp/<运行目录>` 把所选检查点发布为 `triwave_gnn/pretrained/model.pth`；流程验证权重不自动发布为正式预训练模型。每轮的 UTC 时间、耗时和剩余时间估计记录在运行目录的 `timeline.jsonl`。
