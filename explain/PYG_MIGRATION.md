# TriWave-GNN 的 PyG 数据与计算接口

当前代码版本为 `0.4.0`。本机训练环境使用 Python 3.10.20、PyTorch 2.5.0+cu124、PyG 2.8.0，GPU 为 RTX 4090。项目依赖已在 `Code/pyproject.toml` 与 `Code/setup.py` 声明。

## 1. 图对象

`parser_func.py` 中的 `PinCircuitData` 继承 PyG `HeteroData`。所有节点仍按 NPZ 中的统一引脚编号排列；PI、PO、单元输入和单元输出通过角色区分。两种边使用独立关系存储：

| 存储 | 字段与用途 |
|---|---|
| `data['pin']` | `num_nodes`、`node_role`、`node_logic`、`node_pin`、`is_pi`、`depth`、`topological_order`；节点级功能概率、行为标签及掩码也放在这里。 |
| `data['pin','net','pin']` | `edge_index` 为驱动到负载的连接，`edge_attr` 为标准化 net 特征及有效位。 |
| `data['pin','cell','pin']` | `edge_index` 为单元输入到输出的连接，`edge_attr` 为标准化 cell 特征及有效位。 |
| 图级属性 | `name`、结构监督 `hs_pairs/hs_target/hs_mask`、功能监督 `hf_pairs/hf_tt_target/hf_tt_mask`。 |

这是一个统一引脚节点类型、两种有向边关系的异构图。该布局使用 [PyG HeteroData 的节点和关系存储接口](https://pytorch-geometric.readthedocs.io/en/latest/notes/heterogeneous.html)。

## 2. 批处理与拓扑顺序

训练器使用 `torch_geometric.loader.DataLoader`；图边的偏移由 PyG 处理。`PinCircuitData.__cat_dim__()` 和 `__inc__()` 还声明了监督节点对和拓扑顺序中的引脚编号规则，以免把第二个设计的配对错误地指向第一个设计。这对应 [PyG 的自定义批处理规则](https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html)。

PyG 把多个完整设计合成互不连通的 `Batch`。随后 `prepare_topology()` 为 net/cell 各自建立按目标深度排序的边视图和层指针。原始边不重排，保证 `Batch.to_data_list()` 仍能正确还原每个设计。派生调度字段在下一次批处理时排除并重建，支持不同深度、不同大小的设计。

`trainer.move()` 在图的浅拷贝上准备调度并调用 `.to(device)`，避免把数据集缓存中的 CPU 图整体迁入显存。

## 3. MessagePassing 与三路公式

六个 `RelationMessage` 都继承 PyG `MessagePassing`。在每个拓扑深度，外层扫描选择目标节点和当前关系的边；`propagate()` 提取源状态、目标旧状态并完成聚合。模型使用 [MessagePassing 的自定义消息与聚合接口](https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html)，并保留以下公式：

```text
z_u^s = h_u^s
z_u^f = concat(h_u^s, h_u^f)
z_u^b = concat(h_u^s, h_u^f, h_u^b)

m_uv^(k,r) = message_mlp_(k,r)(concat(z_u^k, e_uv^r))
score_uv^k = score_mlp_k(concat(z_u^k, h_v^k, e_uv^cell))
```

只有行为路读取物理特征；结构路和功能路的 `e` 是零宽度张量。cell 分数通过 PyG `softmax` 在同一目标的入边之间归一化，`MessagePassing` 对加权消息求和；net 使用均值聚合，现有图的每个 net 目标只有一个驱动。聚合消息为 `D` 维，再由对应的 `GRUCell(D,D)` 更新本路状态。

节点初态仍可编码逻辑类型、角色与引脚名，消息不显式加入连接两端类型上下文 `C_uv`。三阶段训练、仅行为路读取物理特征、四项行为目标及损失权重 `3:1:2:2` 均保持已确认的设计。

## 4. 检查点

四任务架构标识仍为 `pin-forward-crossstate-4behavior-v1`。PyG 的 `message()` 和 `aggregate()` 是框架方法，因此旧模块参数名中的 `.message.layers.` 改为 `.message_mlp.layers.`，`.aggregate.score.` 改为 `.attention.score.`。`model.load_state_dict()` 会自动转换这两种旧命名；参数形状和顺序不变，优化器状态可恢复。

新运行在检查点和 `run.json` 中记录 `graph_backend='pyg'`。同架构、同数据、同词表的旧检查点可以继续使用；更早的十任务架构仍不兼容。上传新数据后，数据哈希不同，应新建训练运行。

## 5. 正式数据与验证记录

上传后的正式 NPZ 已完成完整契约检查：

| 项目 | 数量 |
|---|---:|
| 设计 | 11,087 |
| 引脚节点 | 60,182,921 |
| net 边 | 41,878,992 |
| cell 边 | 41,546,168 |
| 结构监督节点对 | 41,369,973 |
| 功能监督节点对 | 29,363,280 |

两包均为 `concatenated-ptr-1.0`、`physical-1.0`；所有设计均 `training_ready=True`。校验记录的 SHA-256：

```text
graphs.npz  ad409af2eb70654783d2bfcd15dc7bf4692f84f3c7c01809b12dec1d90be0440
labels.npz  812edc2139942db01d991e01920114fbdae4fb8a752b1f6cad720c673bea0325
```

已验证：

- 10 项契约测试通过，包含 PyG 对象、批图索引、不同深度图的逐图/批图一致性、批图还原、CPU/CUDA 一致性、物理特征隔离和旧权重加载。测试使用自动生成的微型 NPZ，无需加载正式大数据集。
- 三个设计的迁移前后对照中，表征输出最大绝对差为 `0`，梯度最大绝对差约 `7.45e-9`；原检查点的 Adam 状态恢复后，下一步参数更新也一致。
- 从新正式数据前 12 个设计中按 `max_nodes=10000` 选取 5 个设计，使用 CUDA、隐藏维度 48、批大小 2，完成三个阶段各一轮及测试。这里只检查运行链路，不代表收敛或全量数据的训练效果。
- 从第二阶段检查点恢复后继续训练第三阶段，最终权重与连续训练一致；评估、批量表征导出、单设计表征导出和诊断入口均通过检查。

全量正式训练仍从以下入口启动，默认读取全部设计：

```bash
/root/miniconda3/envs/myModelEnv/bin/python /root/autodl-tmp/MyModel/Code/examples/train.py
```

大数据包启动时需要读取、校验并建立词表，这部分发生在第一轮训练之前。PyG 支持不需要改变仿真端生成的 NPZ 格式。
