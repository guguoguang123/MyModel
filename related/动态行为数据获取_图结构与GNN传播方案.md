# 3.关于动态行为数据获取、图结构与GNN传播方案的问题

## 问题

“以往的门级组合电路表征学习，大多是将门级网表转换为AIG，
学习其结构和逻辑功能，但是对于一些电路运行时的动态信息
例如延迟，毛刺现象等往往不曾学习，在时序电路的表征学习中
会将翻转概率视为时序表征，但是组合电路也应该存在这翻转概率
这一动态的属性。这些动态属性的缺失可能会让模型学习到的电路通用表征
在某些需要电路动态信息的下游任务中（例如PPA中的功耗预测）性能上存在不足，
也就是以往的组合电路表征学习仅仅学习了电路的静态属性即结构以及逻辑功能，
即一个理想状态下，但是真实的电路并非是在这种理想状态下，
因此在学习电路的静态属性的基础上，还需要学习电路的动态属性。”

这目前算是我的研究出发点，并且对于此我已经将目前的研究进展记录进Markdown文档中。

我认为除了Markdown文档中的内容外，还存在以下问题：

1. 如何使用仿真软件获取我所需要的动态信息？
2. 以往的AIG GNN貌似已经不满足学习动态行为的条件了，因为其将门级网表全部转换成了与非门。以前只关注结构以及功能，而转换成与非图后，虽然功能会一致，结构上可能会有所变化，但是这些都是静态的信息。对于动态信息，AIG貌似不能很好地表示出原电路的动态信息。我应该使用怎样的图结构才合适？
3. DeepGate2设计了一种单轮的GNN传播方案来优化训练效率，我貌似也应该设计一种GNN传播方案，来适应我要额外学习到的动态信息。

也就是我要更加接近真实电路。

## 回答

你的方向是成立的，但研究出发点需要进一步收紧：不是“过去没人学习组合电路动态信息”，而是“现有通用组合电路表征学习没有把多维瞬态行为统一组织成独立、可迁移的表征”。

更严谨的表述可以改成：

> 以DeepGate系列为代表的通用组合电路表征学习通常基于技术无关的AIG，主要编码逻辑拓扑与稳定布尔功能。虽然已有工作分别研究门级翻转率、时延或毛刺率预测，但这些动态指标通常服务于单一任务，尚未被统一建模为面向组合电路节点的通用瞬态行为表征。本文希望在结构和功能表征之外，学习一个由工作负载、实现结构和时序条件共同决定的行为表征，使其同时刻画节点的活动强度、响应时间和瞬态稳定性，并增强模型在动态功耗、时序和可靠性相关下游任务上的迁移能力。

这里需要承认：GRANNITE已经学习组合逻辑门的翻转率，2026年也已有GNN毛刺率预测，时延预测工作更多。因此不能声称这些信息“从未被学习”，只能说它们没有被统一为通用组合电路行为表征。[GRANNITE](https://research.nvidia.com/publication/2020-07_grannite-graph-neural-network-inference-transferable-power-estimation)、[GNN毛刺率预测](https://doi.org/10.1145/3760779)、[PreRoutGNN](https://ojs.aaai.org/index.php/AAAI/article/download/29653/31111)

### 一、如何获取动态信息

你需要的是延迟感知门级仿真，不是目前项目中的零延迟布尔仿真。完整数据流应当是：

$$
\text{RTL}
\rightarrow
\text{标准单元映射网表}
\rightarrow
\text{布局布线}
\rightarrow
\text{SDF/SPEF}
\rightarrow
\text{门级时序仿真}
\rightarrow
\text{内部节点波形}
\rightarrow
\text{行为标签}
$$

#### 1.异构固定门延迟

先将不同门设置为不同固定延迟，例如：

- INV：20 ps；
- NAND2：35 ps；
- XOR2：60 ps。

使用自己实现的事件驱动仿真器，或者标准单元Verilog `specify`延迟，验证翻转、首次响应、最终稳定和毛刺标签的定义。

这一层适合调试方法，但不能作为最终“真实电路”实验。

#### 2.Liberty标准单元映射

使用Yosys/ABC将RTL映射到指定Liberty库，而不是转换成AIG：

$$
\text{RTL}
\xrightarrow{\text{Yosys/ABC + Liberty}}
\text{standard-cell netlist}
$$

门延迟开始取决于：

- 标准单元类型；
- 驱动强度；
- 输入翻转方向；
- 输入slew；
- 输出负载。

Yosys支持使用Liberty进行ASIC technology mapping。[Yosys文档](https://yosyshq.readthedocs.io/_/downloads/yosys/en/latest/pdf/)

#### 3.布局布线后的SDF仿真

使用OpenROAD Flow完成布局布线，通过OpenSTA和寄生参数生成SDF，再做SDF回标门级仿真。OpenSTA能读取Liberty、Verilog、SPEF，并支持生成SDF；但它是静态时序分析器，不能自己生成毛刺波形。[OpenSTA](https://github.com/the-openroad-project/opensta)、[OpenSTA `write_sdf`](https://github.com/The-OpenROAD-Project/OpenSTA/blob/master/sdf/Sdf.tcl)、[OpenROAD Flow](https://github.com/the-openroad-project/openroad-flow-scripts)

开源仿真可先尝试Icarus Verilog：

- 编译时启用`-gspecify`；
- 测试平台中使用`$sdf_annotate`；
- 输出VCD/FST波形；
- 使用`-sdf-warn`检查未成功回标的路径。

Icarus官方文档明确提供specify timing和SDF back-annotation支持。[Icarus specify文档](https://steveicarus.github.io/iverilog/usage/command_line_flags.html)、[Icarus SDF文档](https://steveicarus.github.io/iverilog/usage/vvp_flags.html)

不过，正式实验前必须用若干小电路人工核对波形，因为不同仿真器对SDF、脉冲过滤和inertial/transport delay的支持可能不同。有条件的话，最终选取一个子集与VCS、Questa或Xcelium对照。

#### 4.单次仿真要增加输入时序条件

不能只定义：

$$
\mathbf a_k\rightarrow\mathbf b_k
$$

还需要考虑主输入是否在同一时刻切换。真实行为至少取决于：

$$
(\mathbf a_k,\mathbf b_k,\mathbf t_k^{PI},\mathbf s_k^{PI})
$$

其中：

- $\mathbf t_k^{PI}$：各主输入的切换时刻；
- $\mathbf s_k^{PI}$：输入slew。

如果所有PI永远同时、瞬时切换，你只能学习“同步理想输入条件下”的动态行为。建议同时保留：

- 同时切换探针；
- 随机PI skew；
- 不同slew；
- 不同输入逻辑偏置和相关性。

#### 5.不建议为每次实验保存完整VCD

数据规模会迅速失控。更好的方式是仿真过程中流式累计每个节点的：

$$
N^{01},N^{10},t^{first},t^{stable},N^{extra},T^{glitch}
$$

最后只保存统计标签。VCD/FST只用于少量样本的正确性验证。

### 二、AIG为什么不再合适

你的判断基本正确，但应准确表述为：

> AIG仍可表示布尔动态在某种抽象延迟模型下的传播，但无法表示原标准单元实现的真实动态行为。

AIG化会丢失：

- 原始门类型及多输入门结构；
- 引脚角色和引脚顺序；
- 单元驱动强度；
- rise/fall不对称；
- 输入slew与输出负载；
- 单元内部timing arc；
- 线网RC和物理位置；
- 原实现的路径级数与路径平衡关系。

例如一个XOR门被分解成多个AND/NOT后，稳定功能不变，但：

- 路径数量改变；
- 路径深度改变；
- 重汇聚位置改变；
- 毛刺产生条件改变；
- 延迟和翻转次数改变。

所以AIG上学习到的是“AIG实现的行为”，不是原门级实现的行为。

### 三、推荐的图结构

建议使用接近STA的引脚级异构时序图：

$$
G_T=(V_{\text{pin}},E_{\text{cell}},E_{\text{net}})
$$

其中节点是标准单元引脚，边分为两类。

| 图元素 | 含义 | 主要属性 |
|---|---|---|
| Pin节点 | 单元输入/输出引脚 | PI/PO、方向、所属单元、逻辑层级 |
| Cell arc | 单元输入引脚到输出引脚 | timing sense、rise/fall delay LUT、逻辑功能、驱动强度 |
| Net arc | 驱动输出引脚到负载输入引脚 | 电容、电阻、线长、扇出、线延迟 |

这种表示与已有timing-engine-inspired GNN的思路一致：引脚作为节点，cell arc和net arc作为不同边类型。[Timing-engine-inspired GNN](https://jqgu.net/publications/papers/PD_DAC2022_Gu.pdf)

行为标签可以挂在：

- 输出pin；
- 或由输出pin驱动的net。

#### 是否还保留AIG

推荐不要让三个表征分别建立在无法对齐的图上。更合理的是：

- 主模型统一使用标准单元引脚图；
- $H_s$仍用重汇聚任务约束；
- $H_f$仍用逻辑概率、真值表距离约束；
- $H_b$使用动态行为约束；
- AIG只作为DeepGate2基线或辅助逻辑视图。

标准单元的布尔功能可以从Liberty `function`或单元Verilog模型获得，因此不需要依靠AIG才能学习$H_f$。

### 四、适合动态行为的GNN传播方案

你确实需要重新设计传播，但不能只是“为了有创新而设计一种GNN”。传播机制必须对应动态行为的物理因果关系。

推荐采用以下传播过程。

#### 1.反向负载传播：PO到PI

门延迟与输出负载、扇出和线网电容相关，因此先从fanout方向汇聚电气环境：

$$
\mathbf l_v=
\operatorname{AGG}_{u\in fanout(v)}
\phi_{\text{load}}(\mathbf l_u,e_{vu})
$$

得到节点的负载上下文$\mathbf l_v$。

#### 2.正向事件传播：PI到PO

动态信号按照拓扑方向传播：

$$
\mathbf m_{u\rightarrow v}
=
\phi_{\text{event}}
(
\mathbf h_u^b,
\mathbf h_u^f,
e_{uv}^{timing},
\mathbf l_v,
\mathbf w
)
$$

其中$\mathbf w$表示PI活动、skew、slew等workload条件。

#### 3.同一门内的扇入交互

普通求和或注意力不足以建模毛刺，因为毛刺取决于多个扇入信号之间的到达时间差：

$$
\Delta t_{ij}
=
|t_i^{arrival}-t_j^{arrival}|
$$

因此在每个多输入门中加入pairwise interaction：

$$
\mathbf q_{ij}
=
\psi(
\mathbf m_i,
\mathbf m_j,
\Delta t_{ij},
\rho_{ij}
)
$$

其中$\rho_{ij}$表示两个扇入之间的重汇聚或逻辑相关性。

最终更新：

$$
\mathbf h_v^b=
U_b
\left(
\operatorname{AGG}_i\mathbf m_i,\,
\operatorname{AGG}_{i<j}\mathbf q_{ij},\,
\mathbf l_v
\right)
$$

这套传播方案的逻辑是：

$$
\boxed{
\text{负载反传}
\rightarrow
\text{事件前传}
\rightarrow
\text{扇入时差与重汇聚交互}
}
$$

它比直接套用DeepGate2单轮传播更适合时延和毛刺。

同时建议将行为状态分成rise/fall通道：

$$
\mathbf h_v^b=
[
\mathbf h_v^{rise},
\mathbf h_v^{fall},
\mathbf h_v^{glitch}
]
$$

因为标准单元上升与下降时延通常不对称。

### 五、一个非常关键的建模问题

动态行为不是只由电路图决定的：

$$
H_b\ne E_b(G)
$$

更准确的是：

$$
H_b=
E_b(G_{\text{mapped}},W,C)
$$

其中：

- $G_{\text{mapped}}$：标准单元及物理实现；
- $W$：输入转换分布、输入skew、slew；
- $C$：工艺角、电压、温度和延迟模型。

同一个网表在不同workload和timing corner下会产生不同标签。如果模型输入不包含$W,C$，却把多个条件下的结果都作为标签，学习问题会变成“一张图对应多个答案”。

你有两种选择：

1. 固定workload、工艺库和corner，学习该标准条件下的行为指纹；
2. 更推荐：将workload和corner作为条件输入，学习条件化表示

$$
\mathbf h_v^b(W,C)
$$

后者更适合动态功耗等下游任务。

### 六、当前项目需要面对的数据问题

目前工作区和`RelatedData.zip`中基本只有AIG/NPZ数据，没有可用于重新映射的RTL、标准单元网表、Liberty、SDF、SPEF或LEF文件。现有行为仿真也是稳定布尔仿真，不支持timing和glitch，参见[当前仿真说明](MyModel/MyModel-sim/simProject/README.md)。

因此，现有10,182个AIG不能直接升级成真实动态数据集。AIG化已经丢失的信息无法反向恢复。

需要重新建立一条数据链：

1. 获取原始RTL/Verilog设计；
2. 用多个综合策略映射到标准单元库；
3. 至少生成mapped netlist和Liberty；
4. 部分设计继续布局布线，产生SPEF/SDF；
5. 执行延迟感知门级仿真；
6. 构建引脚级时序图和节点行为标签。

特别建议让同一RTL产生多个功能等价、实现不同的网表。这会形成最有说服力的实验样本：

$$
\text{相同功能}
\quad\Rightarrow\quad
H_f\text{相似}
$$

$$
\text{不同映射/布局/延迟}
\quad\Rightarrow\quad
H_b\text{不同}
$$

这比单纯报告行为标签预测误差，更能证明真正学习到了“功能之外的动态行为”。

### 七、建议的下一步顺序

暂时不要立即设计完整大模型，优先完成以下最小闭环：

1. 选择5～10个小型组合RTL；
2. 映射到一个公开标准单元库；
3. 生成带延迟的门级网表或SDF；
4. 人工设计几个确定会产生毛刺的输入转换；
5. 验证仿真器能正确输出首次响应、最终稳定和毛刺波形；
6. 建立pin-level timing graph；
7. 比较三种基线：
   - AIG和普通GNN；
   - 标准单元cell graph和普通GNN；
   - pin timing graph和因果传播GNN。

只有当第二步到第五步跑通后，才值得正式确定$H_b$和GNN架构。现在真正的第一技术风险不是模型，而是能否建立一条可信、可复现、规模可扩展的动态行为数据生成链。
