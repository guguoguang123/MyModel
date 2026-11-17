# F：功能 net 恒等传递实验

这是待验证的架构假设，不是已确认的低等价识别性能成因。

`TriWaveGNN(..., functional_net_identity=False)` 默认保持原传播。开启后，在拓扑扫描的功能 net 步骤使用当前源状态执行 `current.index_copy(0, dst, current[src])`。复制不 detach，不修改源张量，多个接收节点共享同一源值并在反向时累加梯度。

`aggregators.f_net` 和 `updates.f_net` 保留完整参数，严格加载原检查点，但开启时不参与前向。功能 cell 消息网络与 GRU、结构与行为传播公式均保持原样。行为分支读取 Hf，因此其数值可以随 Hf 改变。没有改变引脚、监督范围、标签、配对采样或损失。

## 边语义与限制

生成 `pin_model.json`/NPZ 的代码不在本服务器，用户已确认。可复核的是服务器的 NPZ→PyG 读取代码、图契约文档和完整现有图包。`net_edge_audit.json` 明确保存这一限制，不宣称检查过缺失的生成脚本。

预检查覆盖所有存储图：net 源/目标角色、不同驱动源与重复边、net/cell 入边混合、cell 边的门内归属、拓扑前后关系、实际出现的逻辑类型，以及 net 两端的静态 one-count 是否一致。one-count 相同是必要一致性观察，不是完整真值表相等的证明；NPZ 也没有逐边反相或状态操作标记。F 根据现有 net/cell 契约及实测约束执行。

开启时每次前向还检查角色/类型资格、唯一 net 入边、无 cell 入边覆盖、源深度更小和无跨电路连接。不合条件直接报错；不选择、平均多驱动源。NOT/NAND/NOR/AOI/OAI 的输入到输出仍通过 cell 路径，不作恒等化。

## 准备与运行

环境：`/root/miniconda3/envs/myModelEnv/bin/python`。

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /root/miniconda3/envs/myModelEnv/bin/python -u \
  /root/autodl-tmp/MyModel/Code/examples/finetune_net_identity.py \
  --prepare --device cuda \
  --output /root/autodl-tmp/MyModel/Code/examples/exp/functional_pair_sampling_stage3_F_net_identity_20261112
```

准备阶段检查完整图，并用临时模型完成真实检查点的前向、反向和一次参数更新检查；不运行正式训练或完整验证。目录存在时拒绝覆盖。

准备成功后手动运行：

```bash
bash /root/autodl-tmp/MyModel/Code/examples/exp/functional_pair_sampling_stage3_F_net_identity_20261112/run.sh
```

`run.sh` 中已填入实际环境、入口和实验目录；`plan.json` 固定从 B 记录继承的数据、原始 `stage3_best.pt`、划分、归一化、batch_size=32、seed=42、采样种子、cap=1024、lr=3e-5、5 个 epoch。新优化器不恢复旧进度。训练每个 batch 对照 B 的采样记录，核对图顺序和逐列配对哈希。

训练损失复用 B 的原四任务实现和有效分母，通常有效系数仍为 prob=0.375、rc=0.125、func=0.25、behavior=0.25。未启用任何新分类损失或梯度隔离。开始训练前独立计算开启恒等传递后的 epoch 0 全验证基准，不能复用旧传播的基准。

## 输出

- `net_edge_audit.json`、`self_check.json`、`plan.json`：审查、临时模型自检、固定配置与代码哈希。
- `original_propagation_reference.py`：与 E 记录哈希相同的原传播代码，用于关闭开关后的 CPU 逐位一致性检查；CUDA 对照另记录最大差异并检查浮点容差，避免把并行归约的末位变化误报为传播逻辑改变。
- `baseline.json`、`val_epoch_1.json`～`val_epoch_5.json`：原始完整验证配对，float64 端点转换后计算余弦，阈值仅在验证集选择。保留全部任务指标、三种 AUC、P/R/F1/TP/FP/FN、1～999/1000～9999/≥10000 配对电路分组。
- `sampling.jsonl`：学习率、四任务损失、梯度范数、类别采样数、重复率和与 B 的逐 batch 配对一致性。
- `best_loss.pt`、`best_equiv.pt`、`last.pt`：保存开关、实际 epoch、验证阈值和指标。最佳选择包含 epoch 0，指标相同保留较早轮次。普通 `load_model` 自动恢复开关，严格加载全部参数；旧检查点缺省关闭。
- `comparison_B_F.json`：分别列出 B/F 最后一轮和各自最佳 F1 对应的完整验证结果，避免混用轮次。保留全部任务指标以观察退化。

B 原验证逐对分数未保存，因此 B 分组 P/R/F1/混淆计数由原逐电路结果准确汇总；其分组 pooled AUC 留空并说明缺失，不用 macro/加权 AUC 代替。F 本次会直接计算分组 pooled AUC。没有因此重新推理 B。

不会运行测试集。正式运行由用户手动启动并查看终端/`console.log`；助手不轮询长期进度。
