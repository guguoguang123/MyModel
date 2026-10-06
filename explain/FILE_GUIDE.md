# TriWave-GNN 文件导览与代码执行说明

本文位于与 `Code/`、`related/` 同级的 `explain/`。它覆盖 `Code/` 下的源码、配置、文档、训练数据、实验产物和自动生成文件。下文的相对代码路径均从 `Code/` 起算；`related/` 保存研究记录及原始 NPZ，不属于代码文件树。

## 先看整体调用链

```text
examples/data/train/{graphs,labels}.npz
       │
       ▼
examples/train.py → parser.py → dataset.py → utils/schema.py
       │                            └→ utils/dag_utils.py、data_utils.py
       ▼
model.py → arch/initialization.py → arch/propagation.py
                                     ├→ arch/message.py → mlp.py、tfmlp.py
                                     └→ GRUCell 更新 Hs、Hf、Hb
       │
       ▼
trainer.py → utils/model_utils.py → utils/losses.py、metrics.py
       └→ utils/checkpoint.py、logger.py → examples/exp/<运行目录>/
```

**职责划分：**`examples/` 是数据与运行入口；`triwave_gnn/model.py` 组装网络；`arch/` 是可复用神经网络组件；`trainer.py` 组织优化；`utils/` 实现数据契约、批图、损失、指标、日志和检查点。图结构、传播规则、三路状态及监督任务的具体定义见下文。

## 一、项目根目录

| 文件 | 作用及内容 |
|---|---|
| `.gitignore` | Git 忽略规则。忽略 `examples/data/train/*.npz`、`examples/exp/` 中的运行产物、`__pycache__/`、`*.pyc`、打包中间目录等。数据和权重仍保留在本机，只是不纳入 Git。 |
| `README.md` | 面向使用者的项目说明：目录结构、数据契约、安装、训练、恢复、评估和导出命令。 |
| `MANIFEST.in` | 源码包清单补充规则：发布打包时包含 `triwave_gnn/pretrained/` 的说明及已发布的 `.pth`。它本身不生成模型权重。 |
| `pyproject.toml` | Python 构建系统、项目名称与版本、`Python >=3.10`、NumPy/PyTorch 依赖及包发现规则。 |
| `setup.py` | 兼容传统 setuptools 安装方式的配置，与 `pyproject.toml` 声明同一包。 |
| `install.sh` | 用当前 `python` 执行 `pip install -e`，将项目以可编辑方式安装到当前环境。脚本不选择 Conda 环境。 |
| `config.py` | 项目级参数入口。`get_parse_args()` 转调 `triwave_gnn/utils/config.py` 的训练参数解析器。 |
| `debug.py` | 单电路诊断入口：加载指定检查点和设计，打印三路表征形状、功能概率范围及重汇聚配对数量；不会训练或修改模型。 |

### 安装与直接运行的区别

安装命令为 `python -m pip install -e Code/`；它使 `triwave_gnn` 能被该 Python 环境中的其他脚本导入。`examples/*.py` 也会在启动时把 `Code/` 加入 Python 模块搜索路径，因此可直接通过绝对路径运行，不必先安装。两种方式使用的是同一套源码。

## 二、`examples/`：运行入口、数据和实验输出

| 文件 | 作用及关键代码 |
|---|---|
| `examples/__init__.py` | 将目录标记为 Python 包；不实现训练逻辑。 |
| `examples/train.py` | **主训练入口。**解析命令行参数，定位两份 NPZ，创建 `NpzParser`、`TriWaveGNN` 和 `TriWaveTrainer`；依次调用三个阶段的 `set_training_args()`、`train()`，最后 `evaluate()`。它展示实验执行顺序，不实现网络层或损失公式。 |
| `examples/resume.py` | **恢复入口。**先读取指定运行目录的 `last.pt`，取回原隐藏维度、随机种子、批大小和阶段轮数等配置，再重建解析器、模型、训练器。训练器进一步校验数据哈希、词表、数据划分并恢复优化器和随机状态。 |
| `examples/evaluate.py` | 从运行目录的 `best.pt` 加载模型，按 `split.json` 的训练/验证/测试划分计算各监督任务的掩码内误差及功能排序准确率，写出 `metrics.json`。评估不更新参数。 |
| `examples/export.py` | 将一个或多个设计的引脚表征导出为 NPZ；每个输出包含 `design_id`、局部 `node_id`、`hs/hf/hb`、节点映射及网表哈希。支持 `--design` 选择设计。 |
| `examples/feature_extract.py` | **单设计**表征导出入口：明确创建解析器和模型，再保存该设计的三路表征。与 `export.py` 的用途相近，前者更适合逐例查看，后者适合批量导出。 |
| `examples/publish.py` | 将一个完成第三阶段的运行发布为包内 `pretrained/model.pth`；实际完成状态和权重读取由 `pretrained/__init__.py` 检查。训练脚本不会自动执行发布。 |
| `examples/validate.py` | 不训练模型，只加载两份 NPZ 并运行完整数据契约检查，打印设计数、节点/边/配对数和文件 SHA-256。替换数据后应先执行。 |

### `examples/train.py` 的具体执行顺序

1. `DATA_DIR` 指向 `examples/data/train/`；`training_parser()` 定义轮数、维度、学习率、设备等参数。
2. `NpzParser` 创建 `CircuitDataset`，以**整个设计**为单位划分训练、验证和测试集。
3. 使用图文件提供的逻辑类型词表、角色词表，以及从引脚名提取的词表大小实例化 `TriWaveGNN`。
4. `TriWaveTrainer` 创建优化器、运行目录，记录数据来源和固定划分。
5. 阶段 1、2、3 顺序训练；每阶段结束从该阶段验证集最优检查点进入下一阶段。
6. 第三阶段完成后，加载其最佳参数评估测试集。

脚本默认使用全部可训练设计；`--max-nodes` 是显式的小规模流程检查选项，默认 `0` 不限制节点数。

### `examples/data/`

| 文件 | 作用及内容 |
|---|---|
| `examples/data/README.md` | 说明唯一默认训练数据入口及替换规则。 |
| `examples/data/train/graphs.npz` | 已由仿真机器生成的输入图包。含设计 ID、节点指针与属性、net/cell 两类有向边、拓扑顺序、物理原因特征及其有效位、词表和映射哈希。`dataset.py` 按指针切出每个设计。不是 Python 代码。 |
| `examples/data/train/labels.npz` | 与图包逐设计、逐节点对齐的监督包。含 `H_s` 重汇聚配对及标签、`H_f` 逻辑概率与真值表距离配对、`H_b` 翻转/时延/毛刺等目标及有效掩码。不是模型参数文件。 |

这两份 NPZ 必须一起替换，且维持当前 `concatenated-ptr-1.0`、`physical-1.0` 契约。文件的节点映射、设计顺序和网表哈希必须匹配；解析器只负责训练机读取，不会在此机重新仿真。

### `examples/exp/`：每次训练的产物

`examples/exp/README.md` 解释该目录；`examples/exp/pilot_30_designs/` 是小规模流程运行，`examples/exp/triwave_gnn_20260927_111754_733876/` 是另一次按默认参数启动的运行。两者采用相同文件约定；后者的文件集合会随训练进度增加。以下逐一说明：

| 运行目录内文件 | 作用及内容 |
|---|---|
| `run.json` | 启动时的参数、数据源 SHA-256、设计数量和 schema 版本；用于追溯一次实验。 |
| `split.json` | 训练、验证、测试三个集合中的设计 ID。恢复时必须与检查点一致。 |
| `history.json` | 每轮训练和验证的总损失、分项损失、指标及逐批结果；新版训练器把更多分项指标写入该文件。 |
| `timeline.jsonl` | 每行一轮的时间记录：UTC 时间、阶段、轮次、耗时、ETA、训练/验证总损失和分项指标。JSONL 可按行追加读取。较早轮次由旧版代码写入时可能只有总损失。 |
| `last.pt` | 最近**完成并保存**的训练状态：模型、优化器、阶段/轮次、随机状态、数据哈希、词表、标准化统计和划分。每轮覆盖，用于恢复；不代表验证效果最佳。 |
| `stage1_best.pt` | 第一阶段验证损失最低时保存的检查点；只在更好时更新。 |
| `stage2_best.pt` | 第二阶段验证损失最低时保存的检查点；进入第三阶段时用它作起点。第二阶段尚未完成的运行可能没有此文件。 |
| `stage3_best.pt` | 第三阶段验证损失最低时保存的检查点；作为最终选择的候选。第三阶段尚未完成时可能没有。 |
| `best.pt` | 完成三阶段后，由第三阶段最佳检查点复制出的最终选择权重；评估、导出和发布默认读取它。未完成运行可能没有。 |
| `test.json` | 使用第三阶段最佳参数在测试集得到的总损失、分项损失与逐批结果；完成最终评估后才出现。 |
| `metrics.json` | `examples/evaluate.py` 额外计算的可解释指标，按 train/val/test 划分，包括 MAE、重汇聚错误率、功能排序准确率等；运行该脚本后才出现。 |

`.pt` 文件是 PyTorch 序列化检查点，不是可直接阅读的源码；`.json`/`.jsonl` 是文本记录。`pretrained/model.pth` 与这些实验检查点的职责不同：它是显式发布后供其他程序复用的权重位置。

## 三、`triwave_gnn/`：数据、模型、训练器

| 文件 | 作用及关键代码 |
|---|---|
| `triwave_gnn/__init__.py` | 对外导出 `CircuitDataset`、`NpzParser`、`TriWaveGNN` 和版本号，支持 `from triwave_gnn import ...`。 |
| `triwave_gnn/__version__.py` | 包版本常量；当前为 `0.2.0`。 |
| `triwave_gnn/dataset.py` | **数据对象。**加载无 pickle 的两份 NPZ，验证契约，构造词表；`__getitem__()` 按指针取一个完整电路，转为张量，生成拓扑深度和各深度的边范围，并附上全部训练目标与掩码。`fit_normalization()` 只使用训练设计拟合物理特征统计量，避免验证/测试信息泄漏。 |
| `triwave_gnn/parser.py` | **数据入口。**`NpzParser` 创建 `CircuitDataset`，用固定种子划分设计，返回 PyTorch `Subset`。显式检查传入的图/标签路径就是该数据目录的两个约定文件。 |
| `triwave_gnn/parser_func.py` | 图辅助函数：取主输入引脚索引，以及没有出边的输出引脚索引。供后续单电路分析使用。 |
| `triwave_gnn/model.py` | **网络组装中心。**直接声明初态模块、六组关系消息聚合器、六组 GRU 更新器和三个任务读出头。`forward()` 返回三路表征；三个 `pred_*()` 方法从表征产生任务预测。详见下一节。 |
| `triwave_gnn/trainer.py` | **训练控制中心。**初始化运行、数据标准化、优化器、恢复；`run_batch()` 计算预测与损失，`_run_epoch()` 做逐批优化及指标统计，`train()` 负责阶段训练、验证、早停和检查点，`evaluate()` 做最终测试。详见下文。 |

### `model.py`：网络组装细节

`TriWaveGNN(logic_count, role_count, pin_count, hidden)` 把神经网络组件组装起来。设电路有 `N` 个引脚，隐层宽度为 `D=hidden`：

1. `PinInitialization` 为每个引脚生成 `H_s,H_f,H_b` 初态，三者形状都是 `[N,D]`。逻辑类型、角色、引脚名称先被嵌入；同一电路的 PI 在 `H_s` 中有可区分的初态，PI 在 `H_f/H_b` 中使用各自共享初态。
2. 对每路状态分别建立 net 和 cell 两种关系的消息组件，总共 `3 × 2 = 6` 个 `RelationMessage`；同时建立六个 `GRUCell` 更新器。这些模块在 `model.py` **显式声明**，`arch/propagation.py` 只负责按图深度调度它们。
3. `H_s` 消息使用 `H_s`，输入宽度 `D`；`H_f` 消息使用拼接的 `H_s+H_f`，宽度 `2D`；`H_b` 消息使用 `H_s+H_f+H_b`，宽度 `3D`。**只有 `H_b` 的消息接收物理特征。**net 特征为原始 6 列加 6 列有效位，cell 为 7+7 列；对 `H_s/H_f`，物理输入宽度为零。
4. `TopologicalPropagation` 按节点深度从浅到深传播。每个深度先更新结构路，再功能路，再行为路；每路依次聚合 net 和 cell 消息，使用关系专属 GRU 更新目标引脚状态。`forward(item)` 最终返回 `(hs,hf,hb)`，每个张量为 `[N,D]`。
5. 三个读出头不改变表征的定义：`pred_rc_logits(hs,pairs)` 用结构配对特征判断重汇聚；`pred_prob_logits(hf)` 为每个引脚预测逻辑 1 概率的 logit；`pred_behavior_raw(hb)` 为每个引脚输出十项行为目标的原始值。概率型目标在损失/指标中再做 sigmoid。

阶段 1、2 调用 `forward(..., include_behavior=False)`，只传播 `H_s/H_f`；返回的 `H_b` 此时是初态，训练损失不会使用它。阶段 3 才传播并监督 `H_b`。`load_pretrained()` 可从显式发布的 `.pth` 装载权重，并检查隐藏维度。

### `trainer.py`：训练、恢复和评估细节

1. **创建或恢复运行。**新运行在 `examples/exp/` 下创建独立目录，记录 `run.json`、`split.json`，并只从训练设计拟合物理特征归一化。恢复运行读取 `last.pt`，验证数据哈希、词表、划分与关键配置，然后恢复模型、优化器和随机状态。
2. **单批计算。**`run_batch(item,stage)` 调用 `model_outputs()` 得到三路表征及各读出结果，再用 `loss_terms()` 计算当前阶段的加权总损失和分项损失。
3. **一轮训练。**`_run_epoch()` 将完整电路送入 `DataLoader`；多电路批次由 `utils/batch.py` 合成互不连通的图。训练批次执行前向、反向、梯度裁剪和 Adam 更新；验证/测试不更新参数。进度条显示总损失、耗时和 ETA，并定期显示分项损失与指标。
4. **阶段控制。**`set_training_args(stage)` 选择阶段。`train()` 每轮训练并验证，保存验证最优 `stageN_best.pt` 及最新 `last.pt`；阶段结束加载本阶段最优模型进入下一阶段。阶段 1 监督结构重汇聚与功能概率；阶段 2 保留前者并加入功能距离；阶段 3 保留前两阶段并加入行为监督。
5. **结束评估。**`evaluate()` 加载 `stage3_best.pt`，生成 `best.pt`，在固定测试设计上求损失并写入 `test.json`。

`last.pt` 每轮覆盖；`stageN_best.pt` 只在验证损失改进时覆盖。`history.json` 记录逐轮、逐批的结果；它不保存每一轮独立的完整模型文件。

## 四、`triwave_gnn/arch/`：模型组件

| 文件 | 作用及关键代码 |
|---|---|
| `arch/__init__.py` | 导出初态、拓扑传播和结构配对特征函数，供 `model.py` 组装。 |
| `arch/initialization.py` | `PinInitialization`。将逻辑类型、引脚角色和引脚名映射为向量，再投影成三路初态；使用按设计名称固定的随机种子为 PI 生成稳定且可区分的结构初态。PI 功能和行为初态是各自共享的可训练参数。 |
| `arch/message.py` | `RelationMessage`。对一条边拼接源状态、源/目标逻辑嵌入及可选物理特征，经 MLP 形成消息。net 边对同一目标取均值；cell 边调用注意力聚合。源和目标都来自引脚图的有向边。 |
| `arch/mlp.py` | 通用两层 MLP：线性层、ReLU/Tanh、线性层；被消息模块、注意力打分和读出头复用。 |
| `arch/tfmlp.py` | `TFMlpAggr`。用目标状态和边上下文计算每条 cell 消息的分数，对同一目标的入边做 softmax，再加权求和；输入是引脚状态与有向边上下文。 |
| `arch/propagation.py` | `TopologicalPropagation`。依据预先分组的节点深度及 net/cell 边指针执行单次因果扫描；按结构、功能、行为顺序调度 `model.py` 声明的聚合器和 GRU。前两阶段可以跳过行为路传播。 |
| `arch/readout.py` | `pair_features()` 将一对结构向量的求和、绝对差和逐元素积拼接为 `[配对数,3D]`，作为重汇聚读出 MLP 的输入。 |

## 五、`triwave_gnn/utils/`：训练与数据工具

| 文件 | 作用及关键代码 |
|---|---|
| `utils/__init__.py` | 工具子包入口，目前仅含说明，不隐式执行训练逻辑。 |
| `utils/schema.py` | NPZ **数据契约**：要求字段齐全、两包的设计与哈希一致、指针单调、边不跨设计、边符合拓扑顺序、物理特征名称/单位/列数符合约定、标签和掩码形状及数值有效。数据错误在训练前报出。 |
| `utils/dag_utils.py` | `node_depth()` 从拓扑顺序和 net/cell 有向边计算每个引脚到 PI 的最长层深；供单次按层传播使用。 |
| `utils/data_utils.py` | 对有效物理特征做 `log1p` 和标准化、将有效位附加到特征后；按设计和固定随机种子划分训练/验证/测试。归一化统计在训练设计上拟合。 |
| `utils/batch.py` | `collate_circuits()` 合并多个完整引脚图：拼接节点和监督数组，给边及配对索引加节点偏移，再按目标深度重排边，防止不同设计互相传消息。 |
| `utils/config.py` | `training_parser()` 集中定义参数及默认路径：阶段轮数、隐藏维度、批大小、优化器学习率、早停、进度间隔、随机种子、设备等。 |
| `utils/losses.py` | 定义 `STAGE_WEIGHTS`、十项行为目标及对应掩码。`loss_terms()` 计算概率 L1、重汇聚 BCE、功能余弦距离一致性和行为目标损失；第三阶段保留前两阶段监督。无有效标签的任务在该批跳过。 |
| `utils/metrics.py` | `batch_metrics()` 在不反向传播的情况下计算概率 MAE、重汇聚准确率、真值表距离 MAE/排序准确率及各行为任务 MAE。返回“误差或正确数之和、有效标签数”，供训练器按真实标签数量汇总。 |
| `utils/utils.py` | `get_function_acc()`：抽样比较两组真值表距离的相对大小，衡量 `H_f` 预测距离能否保留功能相似性的排序。 |
| `utils/model_utils.py` | `model_outputs()` 一次调用模型及全部读出头，整理成 Trainer/评估共用的结果字典；`checkpoint_summary()` 提供检查点摘要。 |
| `utils/checkpoint.py` | 以临时文件加原子替换保存 PyTorch 检查点；读取检查点，并获取/恢复 Python、NumPy、PyTorch、CUDA 随机状态，用于中断后继续。 |
| `utils/inference.py` | 根据检查点重建模型，核对词表，加载模型参数和训练时的物理特征标准化统计；供评估和表征导出使用。 |
| `utils/logger.py` | `AverageMeter` 累计均值，`BatchProgress` 输出终端进度条与分项快照，`TrainingLogger` 按轮追加 UTC 时间、损失和指标到 `timeline.jsonl`。 |
| `utils/random_seed.py` | 统一设置 Python、NumPy 和 PyTorch（包括 CUDA）的随机种子。 |

### 损失与显示指标的区别

`losses.py` 中的量直接参与梯度更新；`metrics.py` 中的量用于观察模型表现，不参与反向传播。例如，功能任务训练损失对预测/目标距离分别做零均值归一化后比较，而 `tt_distance_mae` 显示未按该损失归一化的距离误差。`rc` 是分类损失，`rc_accuracy` 是阈值 0.5 后的分类正确率。阶段 3 的十项 `H_b` 目标各自使用对应有效掩码；概率型目标用 BCE，非负连续目标在 `log1p` 空间用 Smooth L1。

## 六、`pretrained/` 与 `tests/`

| 文件 | 作用及关键代码 |
|---|---|
| `pretrained/__init__.py` | 定义默认发布路径 `pretrained/model.pth`；`publish_pretrained()` 检查运行已完成第三阶段，再把 `best.pt` 原子保存到发布路径。 |
| `pretrained/README.md` | 说明发布权重与实验恢复检查点的区别。当前一轮流程验证不自动发布为正式预训练模型。 |
| `pretrained/model.pth`（目前不存在） | 只有明确运行 `examples/publish.py` 后才生成；是供特征提取或后续任务加载的已发布权重。不存在不影响训练或从 `last.pt` 恢复。 |
| `tests/__init__.py` | 测试包入口。 |
| `tests/test_contract.py` | 数据与模型契约测试：拒绝跨设计边、拒绝重排物理特征列、检查 PI 初态和阶段损失、确认物理特征只影响 `H_b`、确认批图与逐图推理一致，并检查不稳定窗口损失使用自己的标签掩码。 |

## 七、自动生成文件

`__pycache__/` 和其中的 `*.pyc` 分布在 `examples/`、`triwave_gnn/`、`arch/`、`utils/`、`tests/` 等目录。每个 `.pyc` 对应同名 `.py` 的 Python 字节码缓存；导入或执行脚本时自动生成，既不是额外的模型代码，也不是训练权重，可以由 Python 重新生成。`*.egg-info/`、`build/`、`dist/` 若在安装或打包时出现，也属于生成的安装元数据或构建产物。

## 八、常见操作应看哪些文件

| 想做的事 | 先看这些文件 |
|---|---|
| 更改三路状态、net/cell 消息或 GRU 更新 | `model.py`、`arch/initialization.py`、`arch/message.py`、`arch/propagation.py` |
| 更改监督任务或权重 | `utils/losses.py`，并同步检查 `utils/metrics.py` 与数据掩码 |
| 更改训练轮数、学习率、批大小和输出频率 | `utils/config.py`、`examples/train.py` |
| 排查数据读取与图标签错位 | `examples/validate.py`、`utils/schema.py`、`dataset.py`、`utils/batch.py` |
| 排查训练恢复、检查点和日志 | `examples/resume.py`、`trainer.py`、`utils/checkpoint.py`、`utils/logger.py` |
| 导出或复用训练后的表征 | `examples/export.py`、`utils/inference.py`、`examples/publish.py` |

本项目的训练输入始终是已生成的 `graphs.npz` 和 `labels.npz`。替换为同契约的新数据后，先运行 `examples/validate.py`，再新建训练运行；旧运行的 `last.pt` 会拒绝不同数据哈希，以免误把两批数据接在同一次训练中。

## 本说明目录 `explain/`

| 文件 | 作用 |
|---|---|
| `FILE_GUIDE.md` | 本文，逐项解释 `Code/` 中的文件、主要代码及生成产物。 |
