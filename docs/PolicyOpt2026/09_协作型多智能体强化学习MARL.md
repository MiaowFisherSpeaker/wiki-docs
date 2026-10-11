# 09|协作型多智能体强化学习(MARL):协调、通信、探索与鲁棒

> **本篇为什么放在一起读**:多智能体 RL 是"策略优化 + 社会情境"最直接的结合——**多个学习者同时在变,环境对每个人都是非平稳的**(这也正是 *ALSO* 论文在语言层面处理的问题)。本篇是**传统(非 LLM)MARL**的集中阅读清单;LLM 多智能体与博弈见 `10`。
> **阅读深度**:ICML 2026 论文为**速读**(读了本地 PDF 的引言、贡献陈述与结论,描述后标〔速读〕;个别重点论文已全文精读,见各自小节);ICLR / NeurIPS 论文因 PDF 无法脚本下载,为**摘要级**(只依据官方摘要)。
> **建议顺序**:先读 §0 的三个概念,再按兴趣选一个小节;初学者最友好的是 §1(协调)与 §2(通信)。

## 0. 给初学者:MARL 的 3 个关键词

- **CTDE**(集中训练、分散执行):训练时可看全局信息,执行时每个智能体只看局部观测。
- **价值分解**:把团队价值 $Q_{\text{tot}}$ 拆成个体价值,并要求 **IGM 原则**:每个个体贪心地选动作,合起来恰是团队最优联合动作。
- **非平稳**:队友也在学习,所以"同一个动作"的回报分布一直在变——和社会模拟里"对手会适应"是同一个难点。

## 1. 价值分解、协调与信用分配

| 论文 | 核心想法 | 类型 |
|---|---|---|
| [Retaining Suboptimal Actions to Follow Shifting Optima in Multi-Agent Reinforcement Learning](https://iclr.cc/virtual/2026/poster/10009300) 〔ICLR〕 | 价值分解依赖单一最优动作,价值函数漂移时易陷入次优;学习**多个子价值函数**(S2Q) | 方法 |
| [Potentially Optimal Joint Actions Recognition for Cooperative Multi-Agent Reinforcement Learning](https://iclr.cc/virtual/2026/poster/10008896) 〔ICLR〕 | 单调性约束限制表达力;POW 方法保证最优策略可恢复,优于近似加权 | 理论 + 方法 |
| [Factored Value Functions for Graph-Based Multi-Agent Reinforcement Learning](https://proceedings.mlr.press/v306/rashwan26a.html) 〔ICML〕 | 图结构 MDP 中全局评论家信号弱、局部评论家构造欠妥;提出**扩散价值函数(DVF)**与 DA2C 算法,在 30+ 种环境类型上常优于基线,平均奖励最高 +11%;局限:假设影响图固定、训练仍是 CTDE 〔速读〕 | 方法 |
| [Correlated Policy Optimization in Multi-Agent Subteams](https://iclr.cc/virtual/2026/poster/10009287) 〔ICLR〕 | 受人类团队启发,把智能体分成完全相关的**子团队**,用贝叶斯网络 / DAG 刻画相关联合策略 | 理论 + 方法 |
| [Multi-Agent Guided Policy Optimization](https://iclr.cc/virtual/2026/poster/10009778) 〔ICLR〕 | CTDE 常未充分利用集中训练或缺理论保证;用**集中式引导**训练分散策略(MAGPO) | 理论 + 方法 |
| [Rethinking Ratio-Based Trust Regions for Policy Optimization in Multi-Agent Reinforcement Learning](https://neurips.cc/virtual/2026/poster/148870) 〔NeurIPS〕 | 队友非平稳放大优势估计方差,进而放大 MAPPO 类方法的局部比率更新方差;重新设计比率式信赖域 | 方法 |
| [Towards Complete Multi-Agent Coordination Policy Learning via Denoising Maximum Entropy Optimization](https://proceedings.mlr.press/v306/li26it.html) 〔ICML〕 | 全参数共享的统一策略 + 把角色嵌入编码到**正交空间**("空间划分"),并用**扩散模型**学习共享策略(扩散软策略迭代)以增强表达;与 ROMA 的角色距离约束不同 〔速读〕 | 方法 |
| [HPS: Hyperspherical Parameter Sharing for Efficient Multi-Agent Reinforcement Learning](https://proceedings.mlr.press/v306/fu26k.html) 〔ICML〕 | **几何梯度分解分析**表明参数共享下的梯度冲突主要来自**尺度**而非方向(各 Agent 在方向上基本一致);HPS 用**黎曼骨干 + 各 Agent 专属尺度生成器**把方向共识与尺度调制结构性解耦;SMAC、VMAS、Predator-Prey 上优于基线;局限:强共享需求的环境可能略降 〔速读〕 | 理论分析 + 方法 |
| [Role-Level Inductive Bias for Cross-Task Generalization in Multi-Agent Reinforcement Learning](https://proceedings.mlr.press/v306/yao26i.html) 〔ICML〕 | **角色级**归纳偏置(介于实体级与任务级之间):GTR 用**高斯混合先验**构造潜在角色空间,配排列不变的角色编码器与 UW-DPP 保证角色分配多样性、意图-影响解码器;SMAC / SMACv2 上零样本与少样本迁移明显优于基线 〔速读〕 | 方法 |
| [TACTIC: Task-Aware Sparse Coordination Graphs for Multi-Task Multi-agent Reinforcement Learning](https://proceedings.mlr.press/v306/peng26f.html) 〔ICML〕 | **任务条件的 VQ-VAE 轨迹嵌入**调制协调图边权(时间上平滑),用**动作边际方差**作敏感度代理裁剪冗余交互;**轨迹类别感知策略**配预训练后冻结的目标预测器,避免任务识别与控制学习的梯度干扰;SMACv1/v2 与 SUMO 上验证 〔速读〕 | 方法 |
| [Axiomatic Reinforcement Learning for Open Multi-Agent Systems from Shapley Axioms](https://neurips.cc/virtual/2026/poster/148243) 〔NeurIPS〕 | 开放系统(只有部分 Agent 可控):由 **Shapley 公理**推导价值学习规则 | 理论 |
| [A Separation Principle for Cooperative Multi-Agent Reinforcement Learning](https://neurips.cc/virtual/2026/poster/149847) 〔NeurIPS〕 | 各 Agent 动力学解耦但被群体目标耦合(如车队调度);**分离原理**避免联合状态-动作爆炸 | 理论 + 方法 |
| [Multi-Agent Reinforcement Learning with Submodular Reward](https://proceedings.mlr.press/v306/chen26dk.html) 〔ICML〕 | 联合奖励**次模**(边际收益递减)的合作 MARL:提出首个形式化框架,并给出带样本效率保证的算法 〔速读〕 | 理论 |
| [Submodular Multi-Agent Reinforcement Learning for Effective Online Distributed Task Allocation](https://neurips.cc/virtual/2026/poster/149944) 〔NeurIPS〕 | 次模团队效用下的在线分布式任务分配(分区拟阵) | 理论 + 方法 |
| [Automata-Conditioned Cooperative Multi-Agent Reinforcement Learning](https://proceedings.mlr.press/v306/yalcinkaya26a.html) 〔ICML〕 | ACC-MARL:用**确定性有限自动机(DFA)**表达任务,学习多任务、多智能体的**自我中心**分散策略;证明方法最优,策略的价值函数可用于**最优任务分配**;提供 JAX 工具链并行化 DFA 运算 〔速读〕 | 理论 + 方法 |
| [HyPOLE: Hyperproperty-Guided Multi-Agent Reinforcement Learning under Partial Observation](https://proceedings.mlr.press/v306/rafieioskouei26a.html) 〔ICML〕 | 用 **HyperLTL 超性质规约**表达多智能体目标与约束,经 Skolem 化与连续**鲁棒性函数**转成优化,再由现成 CTDE 算法(VDN / QMIX / QTRAN / IQL)学习;难点在写出精确规约,目前仅支持离散动作 〔速读〕 | 形式化方法 |

## 2. 通信

| 论文 | 核心想法 | 类型 |
|---|---|---|
| [LLM-Guided Communication for Cooperative Multi-Agent Reinforcement Learning](https://proceedings.mlr.press/v306/bae26c.html) 〔ICML〕 | LMAC:用 LLM 的推理设计通信协议,以显式**状态感知准则**迭代精炼,让各 Agent 尽可能一致地还原底层状态;局限:额外的解码器训练与离线 LLM 调用开销,协议质量依赖 LLM 能力,目前限于结构化描述 〔速读〕 | 方法 |
| [Sparse Topology-Aware Pairwise Scoring for Large-Scale Multi-Agent Reinforcement Learning](https://proceedings.mlr.press/v306/deng26n.html) 〔ICML〕 | SOPS:固定的小直径**指数骨干**通信拓扑 + 任务条件的**轻量成对打分**得到稀疏、时变的链路分配;Gumbel 采样与线性退火端到端训练,**跨种群规模无需重设计**;消息落地目标避免退化通信;大规模协作场景回报更高、收敛更快,具备零样本迁移 〔速读〕 | 方法 |
| [Adaptive Communication Range for Scalable Cooperative Multi-Agent Reinforcement Learning](https://neurips.cc/virtual/2026/poster/149196) 〔NeurIPS〕 | 依赖全局状态与固定图的过滤不适应大规模、通信受限环境;自适应通信范围(ACR-PPO) | 方法 |
| [DDACOM: Dual-Decoupled Adaptive Communication for Multi-Agent Reinforcement Learning under Dynamic Networks](https://neurips.cc/virtual/2026/poster/154024) 〔NeurIPS〕 | 带宽随时间变化的动态网络下的双解耦自适应通信 | 方法 |
| [From Local to Global: Progressive Consensus via Hierarchical Communication in Multi-Agent Reinforcement Learning](https://neurips.cc/virtual/2026/poster/154042) 〔NeurIPS〕 | 局部视角难以形成全局理解;层次通信下**渐进共识**(STAGE) | 方法 |
| [Structural Entropy Optimized Communication for Multi-Agent Reinforcement Learning](https://neurips.cc/virtual/2026/poster/149682) 〔NeurIPS〕 | 通信开销与性能的两难;用**结构熵**设计有理论指导的通信图 | 理论 + 方法 |

## 3. 探索

| 论文 | 核心想法 | 类型 |
|---|---|---|
| [Cooperative Multi-Agent Reinforcement Learning via Epigraph-Form Guided Exploration](https://neurips.cc/virtual/2026/poster/153701) 〔NeurIPS〕 | 内在奖励与团队目标直接混合会扭曲信号;用**上境图**重写成"在保持任务性能约束下最大化探索"(EFXPLORER) | 理论 + 方法 |
| [Focusing Influence Mechanism for Multi-Agent Reinforcement Learning](https://neurips.cc/virtual/2026/poster/155238) 〔NeurIPS〕 | 稀疏奖励下 Agent 影响力分散;鼓励其聚焦于**未充分探索**的状态 | 方法 |
| [Policy-Level Exploration for Coordinated Multi-Agent Reinforcement Learning](https://neurips.cc/virtual/2026/poster/154020) 〔NeurIPS〕 | 动作级随机探索与时间扩展的协调不匹配;策略级探索 + 确定性动作执行 | 方法 |
| [Semantic-level Exploration for Multi-Agent Reinforcement Learning](https://neurips.cc/virtual/2026/poster/152298) 〔NeurIPS〕 | 在**语义空间**而非原始状态-动作空间探索 | 理论 + 方法 |
| [Events as Triggers for Behavioral Diversity in Multi-Agent Reinforcement Learning](https://neurips.cc/virtual/2026/poster/153421) 〔NeurIPS〕 | 当前框架把固定行为绑定到固定 Agent;用**事件**触发行为切换 | 方法 |
| [Scalable Multi-Agent Contrastive Reinforcement Learning](https://neurips.cc/virtual/2026/poster/151201) 〔NeurIPS〕 | 稀疏奖励下目标条件的**对比式**评论家(MACRL) | 方法 |

## 4. 鲁棒、安全、离线与持续学习

| 论文 | 核心想法 | 类型 |
|---|---|---|
| [Bayesian Robust Cooperative Multi-Agent Reinforcement Learning Against Unknown Adversaries](https://iclr.cc/virtual/2026/poster/10010539) 〔ICLR〕 | 部署期面对**目标未知的对手**:带连续对手类型的贝叶斯 Dec-POMDP,求完美贝叶斯均衡 | 理论 + 方法 |
| [Distributionally Robust Cooperative Multi-agent Reinforcement Learning with Value Factorization](https://iclr.cc/virtual/2026/poster/10011740) 〔ICLR〕 | sim-to-real 差距下,要求每个 Agent 的鲁棒贪心动作与鲁棒联合动作对齐(DrIGM) | 理论 + 方法 |
| [Sample-Efficient Distributionally Robust Multi-Agent Reinforcement Learning via Online Interaction](https://iclr.cc/virtual/2026/poster/10008076) 〔ICLR〕 | 分布鲁棒马尔可夫博弈的**在线**样本高效算法 | 理论 |
| [Interaction-Breaking Adversarial Learning Framework for Robust Multi-Agent Reinforcement Learning](https://proceedings.mlr.press/v306/lee26s.html) 〔ICML〕 | 此前稳健 MARL 多考虑价值型攻击;提出基于**互信息**的"交互破坏"攻击(遮挡跨组观测 + 扰动协同动作),证明其 POMDP 与扰动动力学的 Dec-POMDP 在价值函数上等价;IBAL 学习对其稳健的策略 〔速读〕 | 理论 + 方法 |
| [Vulnerable Agent Identification in Large-Scale Multi-Agent Reinforcement Learning](https://proceedings.mlr.press/v306/li26ee.html) 〔ICML〕 | 提出**易受攻击智能体识别(VAI)**:写成层级对抗分散平均场控制(HAD-MFC),上层选择最脆弱的 Agent 子集是 **NP-hard**,下层学最坏情形对抗策略;用 **Fenchel-Rockafellar 变换**解耦,上层用贪心 + 带稠密奖励的 MDP/RL 求解;还学到预测各 Agent 脆弱度的价值函数 〔速读〕 | 理论 + 方法 |
| [Secure Multi-agent Reinforcement Learning for Service Systems with Affinity and Byzantine Nodes: Stability Analysis and Protection Design](https://proceedings.mlr.press/v306/jiang26u.html) 〔ICML〕 | 含**无限缓冲队列、作业-服务器亲和状态**与**拜占庭节点**的分散控制:线性函数逼近 + **拜占庭鲁棒的参数共识 + 同步的稳定性约束策略混合**,有稳定排队与可证收敛;案例含 LLM 语义路由、边缘计算、智能制造 〔速读〕 | 理论 + 方法 |
| [Safe Continuous-time Multi-Agent Reinforcement Learning via Epigraph Form](https://iclr.cc/virtual/2026/poster/10009261) 〔ICLR〕 | 连续时间多智能体 + 上境图的安全 RL | 理论 + 方法 |
| [Continuous-Time Value Iteration for Multi-Agent Reinforcement Learning](https://iclr.cc/virtual/2026/poster/10011305) 〔ICLR〕 | 用 HJB 方程的粘性解处理高频 / 不规则时间间隔的多智能体 | 理论 + 方法 |
| [Safe-Fair MACPO: Burden-Fair Constrained Policy Optimization for Safe Multi-Agent Reinforcement  Learning](https://neurips.cc/virtual/2026/poster/151944) 〔NeurIPS〕 | 团队安全 ≠ 个体公平:同一 Agent 可能总是承担"等待/让行"的负担;**负担公平**约束 | 方法 |
| [SamaDICE: Safe Multi-Agent Reinforcement Learning with Stationary Distribution Correction Estimation](https://neurips.cc/virtual/2026/poster/150278) 〔NeurIPS〕 | 离线安全 MARL:平稳分布校正估计,处理联合动作的分布偏移 | 方法 |
| [Offline Multi-Agent Reinforcement Learning via Sequential Score Decomposition](https://proceedings.mlr.press/v306/qiao26d.html) 〔ICML〕 | 指出离线 MARL 难点的根源是在线-离线数据采集鸿沟导致的**多模态联合行为分布**:独立正则会让各 Agent 错配、联合分布偏移;OMSD **顺序分解行为策略**并学习扩散条件分数作行为正则,无需建模完整联合策略或规划器,促进协调的模式选择;声称首个针对多模态行为策略的策略分解式离线 MARL;代码 `github.com/qiaodan-cuhk/OMSD` 〔速读〕 | 方法 |
| [Decentralized and Disentangled Task–Role Representation Learning for Generalizable Offline Multi-Agent Meta Reinforcement Learning](https://proceedings.mlr.press/v306/yuan26m.html) 〔ICML〕 | 面向开放环境(队友数量与任务变化)的**离线多智能体元 RL**:学习分散且解耦的任务-角色表示,提升对未见任务的泛化 〔速读〕 | 方法 |
| [STAIRS-Former: Spatio-Temporal Attention with Interleaved Recursive Structure TransFormer for Offline Mulit-task Multi-agent Reinforcement Learning](https://iclr.cc/virtual/2026/poster/10010923) 〔ICLR〕 | 离线多任务 MARL:变化的 Agent 数;时空注意力 + 递归结构 | 方法 |
| [Joint Adaptive Neighborhood Constraint for Offline Multi-Agent Reinforcement Learning](https://neurips.cc/virtual/2026/poster/149793) 〔NeurIPS〕 | 联合动作空间指数增长导致单动作在分布内、联合却不在;联合自适应邻域约束 | 方法 |
| [DLM: Unified Decision Language Model for Offline Multi-Agent Sequential Decision Making](https://proceedings.mlr.press/v306/zhang26v.html) 〔ICML〕 | 把离线多智能体决策重写为**对话式序列建模**;两阶段 **SFT → GRPO**:SFT 只用观测与动作即可与强离线基线竞争,GRPO 进一步降低 OOD 动作率、提升稳健性;单个统一模型跨任务 〔速读〕 | 方法 |
| [Addressing Exogenous Variability in Cooperative Multi-Agent Reinforcement Learning](https://neurips.cc/virtual/2026/poster/149019) 〔NeurIPS〕 | 对手或环境机制变化(外生变量)使团队失效;形式化为 **ED-POMDP**,提出 LEICA | 理论 + 方法 |
| [CHAIN: Continual Heterogeneous Cooperation with Information Bottleneck for Multi-Agent Reinforcement Learning](https://neurips.cc/virtual/2026/poster/155574) 〔NeurIPS〕 | 连续变化环境中异构 MARL 退化,与表示冗余相关;信息瓶颈 | 方法 |
| [MEAL: A Benchmark for Continual Multi-Agent Reinforcement Learning](https://proceedings.mlr.press/v306/tomilin26a.html) 〔ICML〕 | **纯 JAX、硬件加速**的持续多智能体 RL 基准,支持按需程序化生成与长任务序列;发现**结论会随任务数增加而显著改变**,持续学习方法与 MARL 结合在简单设定下可行,但随布局变大、序列变长、Agent 增多、奖励稀疏、部分可观测而退化,Agent 数增加影响最大 〔速读〕 | 基准 |
| [TABX: A High-Throughput Sandbox Battle Simulator for Multi-Agent Reinforcement Learning](https://proceedings.mlr.press/v306/lee26az.html) 〔ICML〕 | **JAX 高吞吐**战斗模拟器沙盒:单位规格、地形区、启发式行为与物理动力学均可配置(扇形部分可观测、非定向交互、异构角色、地形区带来的战略需求),用于系统研究信息依赖的价值学习、长程探索与对未见配置的泛化 〔速读〕 | 环境 / 基准 |
| [Robust Instruction Compliance in Cooperative Multi-Agent Reinforcement Learning](https://neurips.cc/virtual/2026/poster/148746) 〔NeurIPS〕 | 人类自然语言指令会打断当前行动:**宏动作价值取消** | 方法 |
| [One Step is Enough: Multi-Agent Reinforcement Learning Based on One-Step Policy Optimization for Order Dispatch on Ride-Sharing Platforms](https://neurips.cc/virtual/2026/poster/151407) 〔NeurIPS〕 | 网约车派单:基于**一步策略优化**绕开难估计的价值函数 | 应用 |
| [Learning Decentralized LLM Collaboration with Multi-Agent Actor Critic](https://proceedings.mlr.press/v306/liu26k.html) 〔ICML〕 | 用**多智能体 actor-critic**(CoLLM-DC 与 CoLLM-CC)优化分散式 LLM 协作,相对蒙特卡洛法(如 MAGRPO)更适合**长程稀疏奖励**;协作写作、编码(Qwen2.5-Coder-3B 辅助 Qwen3-4B)与博弈任务上 CoLLM-CC 最优 〔速读〕 | 方法 |

## 5. 本篇小结

1. **共同母题**:当别人也在学,"最优"变成了**相对**概念。解法是:结构化(分解、图、角色)、通信、鲁棒(分布鲁棒、对手类型)、公理化(Shapley)。
2. **理论味较浓的**:Bayesian Robust(完美贝叶斯均衡)、DrIGM、Axiomatic RL(Shapley 公理)、Structural Entropy Communication、Submodular MARL。
3. **与社会情境的联系**:*Addressing Exogenous Variability* 中的"外生变量(对手变化)"与 ALSO 的非平稳对手直接对应;*Bayesian Robust* 的"未知对手类型"与 ALSO 的"策略臂"可对照阅读。
