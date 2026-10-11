# TriWave-GNN

TriWave-GNN 从标准单元引脚级异构图学习结构表征 `H_s`、稳定功能表征 `H_f` 和瞬态行为表征 `H_b`。输入包含 net edge、cell edge、物理特征和三类监督标签。本目录负责模型、训练、评估及嵌入导出；仿真与标签生成在另一台机器完成。

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
│   ├── parser_func.py           输入输出引脚查询
│   ├── model.py                 直接声明消息聚合、GRU 更新和读出头；forward 返回 Hs/Hf/Hb
│   ├── trainer.py               数据划分、训练、验证、早停、恢复
│   └── utils/                   数据契约、DAG、标准化、损失、检查点、日志、种子、模型加载
```

模型使用可区分的 PI 结构初态和共享的 PI 功能初态，在引脚 DAG 上按拓扑深度传播 net/cell 两类消息。结构路消息读取前驱 `H_s`，功能路读取前驱 `H_s+H_f`，行为路读取前驱 `H_s+H_f+H_b`。每路消息经共享的关系网络生成、聚合，再由本路 GRU 更新；只有行为路读取物理边特征。cell 多输入消息使用目标更新前的本路状态参与 MLP 注意力打分。基础模型不加入连接两端逻辑类型的显式边上下文 `C_uv`。

训练入口先由 `NpzParser` 取得训练、验证、测试电路，再构造 `TriWaveGNN` 和 `TriWaveTrainer`；示例脚本逐阶段设置监督任务并调用 `trainer.train()`。`model.forward()` 返回 `H_s/H_f/H_b`，Trainer 的 `run_batch()` 调用模型读出头并计算阶段损失。具体代码见 [训练入口](examples/train.py)、[模型组装](triwave_gnn/model.py)、[训练器](triwave_gnn/trainer.py)。

## 数据与安装

`Code/` 是完整项目根目录。现有两份 NPZ 已从 `related/` 复制到 `Code/examples/data/train/`，原件保留。以后只需替换 `Code/examples/data/train/graphs.npz` 与 `Code/examples/data/train/labels.npz`。两份文件须使用当前 `concatenated-ptr-1.0` 格式，且设计 ID、节点编号和网表哈希一致。训练启动时自动检查字段、指针、边界、拓扑、标签掩码与 `training_ready`。如果仿真机器改变了数据格式，需要同步更新解析器和契约，不能只替换文件。

本机已有 PyTorch 时，进入项目根目录运行：

```bash
cd Code
python -m pip install -e .
python examples/validate.py
```

依赖要求见 `pyproject.toml`；GPU 版 PyTorch 应按本机 CUDA 环境安装。

## 正式训练

在 `Code/` 内运行：

```bash
python examples/train.py
```

各 `examples/*.py` 入口也支持从任意工作目录用绝对路径直接运行。例如在 AutoDL 上可执行 `/root/miniconda3/envs/myModelEnv/bin/python /root/autodl-tmp/MyModel/Code/examples/train.py`，无需先激活 Conda 环境或执行可编辑安装。

默认使用 NPZ 中**全部**设计，不会按节点数跳过大电路。默认隐藏维度 48，三个阶段最多分别训练 30、30、40 轮，每阶段验证损失连续 10 轮未改善则早停；每阶段可用 `--lr-step` 设置学习率衰减间隔。`--batch-size` 默认为 1，也可用 `utils/batch.py` 将多个完整引脚图合成互不连通的批图。交互终端逐批刷新细线式进度轨道：青色表示已完成，灰色表示剩余；上方显示阶段、轮次、已用时间与 ETA。每隔 `--progress-interval` 批次输出各项损失及概率、重汇聚、真值表距离、行为任务指标。每轮分别汇总训练和验证指标，并记录在 `history.json`、`timeline.jsonl`。训练按完整电路划分训练、验证和测试集；物理特征标准化只使用训练设计。运行结果保存在自动创建的 `Code/examples/exp/triwave_gnn_<时间>/`，不会覆盖旧运行。

三个阶段依次为：

1. `H_s` 重汇聚关系 + `H_f` 逻辑 1 概率；
2. 保留第一阶段任务，加入真值表距离与 `H_f` 余弦距离的一致性；
3. 保留前两阶段任务，加入 `H_b` 的四项监督：总体翻转概率 `hb_tp`、首次响应时间 `hb_first_mean`、最终稳定时间 `hb_stable_mean`、毛刺发生率 `hb_glitch_rate`。方向翻转和其他行为字段保留在数据中，不计入初步模型的主损失。

阶段损失权重为概率:重汇聚:功能距离:行为总项 = `3:1:2:2`；行为总项内的四项任务等权。

每个阶段保存最佳检查点；每轮保存 `last.pt`，其中包含模型、优化器、阶段、数据哈希、划分和随机状态。恢复同一份数据的中断训练：

```bash
python examples/resume.py examples/exp/<运行目录>
```

替换 NPZ 后应直接启动新运行；恢复命令会拒绝数据哈希改变。模型架构已经更新，旧架构的 `last.pt` / `best.pt` 也会被明确拒绝，须新建运行目录。小规模流程验证可以显式添加 `--max-nodes 10000 --stage1-epochs 2 --stage2-epochs 2 --stage3-epochs 2`。`--max-nodes` 默认是 `0`，表示无上限。

## 评估与表征导出

```bash
python examples/evaluate.py --run examples/exp/<运行目录>
python examples/export.py --checkpoint examples/exp/<运行目录>/best.pt
```

评估输出 `metrics.json`，包含不同监督任务在训练、验证、测试设计上的掩码内误差。导出命令为每个设计生成一个 NPZ，保存 `node_id`、`hs`、`hf`、`hb` 和节点映射哈希；可以用 `--design <设计ID>` 只导出指定设计。

运行契约测试：

```bash
python -m unittest discover -s triwave_gnn/tests -v
```

旧架构曾用全部 30 个设计、隐藏维度 16、每阶段一轮完成流程验证，结果仍保存在 `examples/exp/pilot_30_designs/`，但不能用当前模型继续训练或推理。当前架构已用全部 30 个设计、隐藏维度 8、每阶段一轮完成训练、评估和恢复流程验证；结果保存在 `examples/exp/revised_pilot_30_designs/`，第三阶段测试损失为 0.38336。该流程检查不代表模型已收敛。正式训练应新建运行目录。

所有文件的详细说明见 [文件导览](../explain/FILE_GUIDE.md)。完成正式训练后可用 `python examples/publish.py --run examples/exp/<运行目录>` 把所选检查点发布为 `triwave_gnn/pretrained/model.pth`；当前验证权重不作为正式预训练模型发布。每轮的 UTC 时间、耗时和剩余时间估计记录在运行目录的 `timeline.jsonl`。
