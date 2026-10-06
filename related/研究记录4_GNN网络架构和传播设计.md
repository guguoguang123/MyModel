框架仍然是：
$G_{\mathrm{pin}}
\longrightarrow \mathrm{GNN传播}
\longrightarrow
\{H_s,H_f,H_b\}$

现在只需要把传播拆成三个容易区分的问题：

1. 图按照什么顺序传播；
2. 每到一个节点，前驱消息怎样聚合；
3. 聚合后，节点的三种状态怎样更新。

### 1. 图怎么传播

对组合电路采用一次拓扑前向扫描：
$\text{PI}
\rightarrow \text{net edge}
\rightarrow \text{cell input pin}
\rightarrow \text{cell edge}
\rightarrow \text{cell output pin}
\rightarrow \text{net edge}
\rightarrow\cdots\rightarrow\text{PO}$
例如：
$U_0/Y
\xrightarrow{\text{net}}
U_1/A
\xrightarrow{\text{cell}}
U_1/Y
\xrightarrow{\text{net}}
U_2/B$

这里的“一次拓扑扫描”不等于“一层同步GNN”。

传统同步GNN一层只能前进一跳，而拓扑异步传播是：前驱节点更新完成后，立即用新状态更新后继。因此一次从PI到PO的拓扑扫描，就可以覆盖任意长度的组合路径。

所以长程依赖并不要求堆叠几十层GNN，这正好符合你避免过平滑的要求。

### 2.net edge怎么传播

net edge连接：
$\text{驱动单元输出引脚}\rightarrow\text{负载单元输入引脚}$

例如：

$U_0/Y\rightarrow U_1/A$

消息函数可以写成：

$$
m_{u\to v}^{k,\mathrm{net}}
=
\phi_{\mathrm{net}}^{k}
\left(
h_u^{k},\,e_{uv}^{\mathrm{net}}
\right),
\qquad
k\in\{s,f,b\}
$$

其中net edge特征可以包括：

$$
e_{uv}^{\mathrm{net}}
=
[R,C,\text{等影响延迟的属性}]
$$

通常一个sink pin只有一个driver，所以这里不存在真正的多扇入竞争。net传播主要负责把上游状态经过互连特征变换后，送到下游输入引脚。

因此，net edge不必强行使用复杂注意力。

### 3.cell edge怎么传播

cell edge连接同一单元内部的输入引脚和输出引脚。例如NAND2单元：

$$
U_1/A\rightarrow U_1/Y
$$

$$
U_1/B\rightarrow U_1/Y
$$

当更新$ U_1/Y$ 时，同时收集$ U_1/A$ 和 $U_1/B$的消息：

$$
m_{A\rightarrow Y}^{k,\mathrm{cell}},
\qquad
m_{B\rightarrow Y}^{k,\mathrm{cell}}
$$

这里才是注意力聚合真正发挥作用的位置：

$$
\alpha_{uY}^{k}
=
\operatorname{softmax}_{u\in\mathrm{fanin}(Y)}
\left(
a_k(h_u^k,h_Y^k,e_{uY}^{\mathrm{cell}})
\right)
$$

$$
\bar m_Y^k
=
\sum_{u\in\mathrm{fanin}(Y)}
\alpha_{uY}^{k}
m_{u\rightarrow Y}^{k,\mathrm{cell}}
$$

> 这里就是使用注意力机制聚合 $U_1/A\rightarrow U_1/Y$和 $U_1/B\rightarrow U_1/Y$两条cell消息。
>
> 先计算每条消息的重要性分数：
>
> $$
> s_{AY}^k
> =
> a_k\left(
> h_A^k,h_Y^k,e_{AY}^{\mathrm{cell}}
> \right)
> $$
>
> $$
> s_{BY}^k
> =
> a_k\left(
> h_B^k,h_Y^k,e_{BY}^{\mathrm{cell}}
> \right)
> $$
>
> $a_k(·)$代表计算score的函数过程（内部使用QKV点积注意力还是GAT式MLP注意力未3规定）
>
> 然后在 $Y$的所有cell扇入之间做softmax：
>
> $$
> \alpha_{AY}^k
> =
> \frac{\exp(s_{AY}^k)}
> {\exp(s_{AY}^k)+\exp(s_{BY}^k)}
> $$
>
> $$
> \alpha_{BY}^k
> =
> \frac{\exp(s_{BY}^k)}
> {\exp(s_{AY}^k)+\exp(s_{BY}^k)}
> $$
>
> 因此：
>
> $$
> \alpha_{AY}^k+\alpha_{BY}^k=1
> $$
>
> 最后对两条消息加权求和：
>
> $$
> \bar m_Y^{k,\mathrm{cell}}
> =
> \alpha_{AY}^k m_{A\rightarrow Y}^{k,\mathrm{cell}}
> +
> \alpha_{BY}^k m_{B\rightarrow Y}^{k,\mathrm{cell}}
> $$

它让模型自己学习：

- 哪个输入引脚对当前输出更重要；
- 不同逻辑功能下各输入的贡献；
- 不同cell arc延迟对动态行为的影响；
- 哪条输入路径更可能影响首次翻转、稳定时间和毛刺。

具体传什么语义不需要人工规定，由监督任务学习；只规定cell edge和net edge使用不同参数，并把对应物理特征提供给模型。

> ****这里的“cell edge和net edge使用不同参数”不是指人为给每一条边指定一个可学习参数，而是指：两类边使用两套独立的消息生成网络。****
>
> $$
> m_{u\to v}^{k,\mathrm{net}}
> =
> \phi_{\mathrm{net}}^{k}
> \left(
> h_u^k,e_{uv}^{\mathrm{net}};
> \theta_{\mathrm{net}}^k
> \right)
> $$
>
> $$
> m_{u\to v}^{k,\mathrm{cell}}
> =
> \phi_{\mathrm{cell}}^{k}
> \left(
> h_u^k,e_{uv}^{\mathrm{cell}};
> \theta_{\mathrm{cell}}^k
> \right)
> $$
>
> 其中：
>
> $$
> \theta_{\mathrm{net}}^k\neq\theta_{\mathrm{cell}}^k
> $$
>
> 表示net edge和cell edge使用不同的网络权重。例如，它们可以是两个不同的MLP：
>
> $$
> \phi_{\mathrm{net}}^k=\operatorname{MLP}_{\mathrm{net}}^k
> $$
>
> $$
> \phi_{\mathrm{cell}}^k=\operatorname{MLP}_{\mathrm{cell}}^k
> $$
>
> 需要区分两个概念：
>
> | 符号                                               | 含义                 | 是否由训练学习      |
> | -------------------------------------------------- | -------------------- | ------------------- |
> | $\theta_{\mathrm{net}}^k,\theta_{\mathrm{cell}}^k$ | 消息函数的网络权重   | 是                  |
> | $e_{uv}^{\mathrm{net}},e_{uv}^{\mathrm{cell}}$     | 每条边携带的物理特征 | 通常不是，是EDA数据 |
>
> 例如：
>
> $$
> e_{uv}^{\mathrm{net}}
> =
> [R,C......]
> $$
>
> 
>
> 所有net edge通常共享同一套 $\theta_{\mathrm{net}}^k$，所有cell edge共享同一套$ \theta_{\mathrm{cell}}^k$。并不是每一条具体的边都有一套独立权重，否则模型难以迁移到新电路。

### 对2、3补充修改

所有net edge通常共享同一套 $\theta_{\mathrm{net}}^k$，所有cell edge共享同一套$ \theta_{\mathrm{cell}}^k$。并不是每一条具体的边都有一套独立权重，否则模型难以迁移到新电路。

这边所说的共享一套 $\theta_{\mathrm{net}}^k$，$ \theta_{\mathrm{cell}}^k$是：
$\boxed{\text{共享消息函数的主体参数，但根据源节点、目标节点和边属性进行条件化}}$

两种连接必须能够区分

例如：$\text{NAND}_1/Y\rightarrow \text{AND}_1/A$和$\text{AND}_2/Y\rightarrow \text{OR}_1/A$

它们至少存在三方面差别：

1. 驱动端状态不同：NAND输出的 $h_u$ 与AND输出的$ h_u $不同；
2. 目标pin上下文不同：目标分别属于AND和OR；
3. 互连物理属性不同：RC、负载、电容、线延迟等可能不同。

另外也不能为每一种不同的连接类型实例化一个 聚合器，组合太多，参数量爆炸

在消息传递的时候增加一个输入量

原来的消息：

$$
m_{u\rightarrow v}^{k,\mathrm{net}}
=
\phi_{\mathrm{net}}^k
\left(
h_u^k,e_{uv}^{\mathrm{net}}
\right)
$$

扩展为：

$$
\boxed{
m_{u\rightarrow v}^{k,\mathrm{net}}
=
\phi_{\mathrm{net}}^k
\left(
h_u^k,e_{uv}^{\mathrm{net}},c_{uv}^{\mathrm{net}}
\right)
}
$$

其中三部分各自负责：

- $h_u^k$：源节点当前学习到的隐藏状态(中间过程嵌入向量)；
- $e_{uv}^{\mathrm{net}}$：这条互连的物理属性；
- $c_{uv}^{\mathrm{net}}$：源端和目标端是什么cell、什么pin。

$$
\boxed{
h_u^k=\text{源节点当前的动态表征}
}
$$

$$
\boxed{
C_{uv}=\text{什么类型的pin传给什么类型的pin}
}
$$

$$
\boxed{
e_{uv}=\text{这条边自身的物理及时序属性}
}
$$

消息函数写成：

$$
m_{u\rightarrow v}^{k,r}
=
\phi_r^k
\left(
h_u^k,C_{uv}^r,e_{uv}^r
\right),
\qquad
r\in\{\mathrm{net},\mathrm{cell}\},
\quad k\in\{s,f,b\}
$$

#### $e_{uv}$中涵盖内容：

****$E_{uv}^{net}$****

$$
E_{uv}^{\mathrm{net}}
=
\left[
R_{uv}^{\mathrm{path}},
R_{n}^{\mathrm{total}},
C_{n}^{\mathrm{ground}},
C_{n}^{\mathrm{coupling}},
C_{n}^{\mathrm{total}},
C_{v}^{\mathrm{sink}}
\right]
$$

包括：

- `path resistance（驱动到负载的路径电阻）`；
- `total net resistance（线网总电阻）`；
- `ground capacitance（接地电容）`；
- `coupling capacitance（耦合电容）`；
- `total net capacitance（线网总电容）`；
- `sink pin capacitance（目标输入pin电容）`。

这些信息从SPEF和Liberty中获得，只告诉模型造成传播延迟的物理条件

于是模型需要学习：

$(R,C,\text{load})
\longrightarrow
\text{潜在net传播行为}$

$E_{uv}^{cell}$

$$
E_{uv}^{\mathrm{cell}}
=
\left[
C_{u}^{\mathrm{input}},
C_{v}^{\mathrm{load}},
D_{U}^{\mathrm{strength}},
V_{U}^{\mathrm{threshold}},
A_{U}^{\mathrm{cell}},
C_{U}^{\mathrm{max}},
S_{U}^{\mathrm{max}}
\right]
$$

包括：

- `input pin capacitance（输入pin电容）`：从Liberty获得；
- `output load capacitance（输出负载电容）`：由输出net的SPEF电容和所有sink pin电容计算；
- `drive strength（驱动能力）`：例如X1、X2、X4，可从标准单元名称或库属性取得；
- `threshold-voltage class（阈值电压类别）`：例如低阈值、标准阈值、高阈值，库提供时使用；
- `cell area（单元面积）`：Liberty或LEF中取得，可辅助描述单元尺寸；
- `maximum capacitance（最大允许负载电容）`：Liberty提供时使用；
- `maximum transition（最大允许转换时间）`：Liberty提供时使用。

#### $C_{uv}$设计

##### 建立基本逻辑类型表

$$
\mathcal T_{\mathrm{logic}}
=
\{\mathrm{AND},\mathrm{NAND},\mathrm{OR},\mathrm{NOR},
\mathrm{XOR},\mathrm{XNOR},\mathrm{NOT},\mathrm{BUF},\ldots\}
$$

类型数量为：$N=\left|\mathcal T_{\mathrm{logic}}\right|$

建立一个可学习的类型嵌入表：$\mathbf T_{\mathrm{logic}}
\in\mathbb R^{N\times d_t}$

其中：

$$
\operatorname{Emb}_{logic}(\mathrm{NAND2\_X1})
=
\operatorname{Emb}_{logic}(\mathrm{NAND3\_X4})
=
\mathbf T_{\mathrm{logic}}[\mathrm{NAND}]
$$

`2、3`和`X1、X4`都不参与类型分类。

##### net edge的 $C_{uv}$

定义：

$$
C_{uv}^{\mathrm{net}}
=
\left[
\operatorname{Emb}_{\mathrm{logic}}(g_u)
\mathbin{\Vert}
\operatorname{Emb}_{\mathrm{logic}}(g_v)
\right]
$$

其中：

- \(g_u\)：驱动输出pin所属cell的基本逻辑类型；
- \(g_v\)：负载输入pin所属cell的基本逻辑类型；
- 拼接顺序固定为“源类型在前、目标类型在后”，所以传播方向也被保留。

对于一条连接：

$$
\mathrm{NAND}_1/Y\rightarrow\mathrm{AND}_1/A
$$

有：

$$
C_{1}^{\mathrm{net}}
=
\left[
\operatorname{Emb}(\mathrm{NAND})
\mathbin{\Vert}
\operatorname{Emb}(\mathrm{AND})
\right]
$$

对于另一条连接：$\mathrm{AND}_2/Y\rightarrow\mathrm{OR}_1/A$

有：

$$
C_{2}^{\mathrm{net}}
=
\left[
\operatorname{Emb}(\mathrm{AND})
\mathbin{\Vert}
\operatorname{Emb}(\mathrm{OR})
\right]
$$

显然：$C_1^{net}\neq C_2^{net}$

所有net edge仍然使用同一个net消息函数，但是输入的 $C_{uv}^{net}$不同

##### cell edge的 $C_{uv}$

cell edge固定连接同一个cell内部的：$\text{输入pin}\rightarrow\text{输出pin}$

对于AND、NAND、OR等输入对称的简单逻辑门，可以直接定义：

$$
C_{uv}^{\mathrm{cell}}
=
\operatorname{Emb}_{\mathrm{logic}}(g_{\mathrm{cell}})
$$

### 4.三个状态怎么更新

#### 结构状态

$$
h_v^{s,+}
=
U_s\left(
h_v^s,\bar m_v^s
\right)
$$

只依赖结构消息，学习拓扑、逻辑锥、重汇聚等结构性质。

#### 功能状态

$$
h_v^{f,+}
=
U_f\left(
h_v^f,\bar m_v^f
\right)
$$

功能状态利用结构上下文，因为同样的局部逻辑在不同拓扑环境下可能具有不同功能作用。

#### 动态行为状态

$$
h_v^{b,+}
=
U_b\left(
h_v^b,\bar m_v^b
\right)
$$

动态行为同时需要：

- \(H_s\)：路径、扇入、重汇聚等结构条件；
- \(H_f\)：逻辑值传播和输入敏感性；
- \(H_b\)：延迟、翻转、毛刺等动态信息。

更新函数$ U_s,U_f,U_b $可以先用不同参数的GRU
