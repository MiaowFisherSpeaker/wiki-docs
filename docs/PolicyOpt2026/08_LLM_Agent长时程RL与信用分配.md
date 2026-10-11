# 08|LLM Agent 的长时程强化学习:信用分配、工具调用、记忆与主动推理

> **本篇为什么放在一起读**:当 LLM 从"一问一答"变成**多轮、调工具、带记忆**的 Agent,奖励只在最后给出一次,**"哪一步功劳(或过错)最大"成了核心问题**——这就是信用分配(credit assignment)。本篇论文几乎都是在 GRPO 之上,为"多步"设计更细的优势估计。
> **阅读深度**:ICML 2026 论文为**速读**(读了本地 PDF 的引言、贡献陈述与结论,描述后标〔速读〕;个别重点论文已全文精读,见各自小节);ICLR / NeurIPS 论文因 PDF 无法脚本下载,为**摘要级**(只依据官方摘要)。
> **与前面文档的关系**:*WMA 与世界模型*(`03`)解决"如何预测后果";本篇解决"如何把最终结果分配给每一步"。*奖励与过程奖励模型*见 `06`。

## 0. 给初学者:为什么长时程这么难

- **稀疏奖励**:几十步只在最后给一个 0/1。
- **优势估计失真**:GRPO 把整条轨迹的优势平均分给所有 token,**好步骤可能被惩罚,坏步骤可能被奖励**。
- **步级(step-level)分组的新问题**:每一步的上下文不同,拿它们互相比较就是"苹果比橘子"——*Stratified GRPO*、*Hierarchy-of-Groups* 等都在修这个。

## 1. 信用分配的几条路线

### 路线 A:把优势按"步骤 / 层级 / 图"重新分组

| 论文 | 核心想法 | 类型 |
|---|---|---|
| [Stratified GRPO: Handling Structural Heterogeneity in Reinforcement Learning of LLM Search Agents](https://proceedings.mlr.press/v306/zhu26f.html) 〔ICML〕 | 形式化**跨层偏差**:搜索 Agent 的轨迹结构异质(工具调用次数与位置不同),用**全局基线**比较会"苹果比橘子";**分层优势归一(SAN)**把轨迹分成同质层、局部算优势,证明消除跨层偏差、层内条件无偏且单位方差,同时保持全局无偏与单位方差;QA 与深度研究基准上最多比 GRPO 高 12.6 分 〔速读〕 | 理论 + 方法 |
| [Hierarchy-of-Groups Policy Optimization for Long-Horizon Agentic Tasks](https://iclr.cc/virtual/2026/poster/10009347) 〔ICLR〕 | 步级相对优势存在**上下文不一致**(同一组里的步骤上下文不同);层级分组 | 方法 |
| [Group-Graph Policy Optimization for Long-Horizon Agentic Reinforcement Learning](https://neurips.cc/virtual/2026/poster/149985) 〔NeurIPS〕 | 长程 Agent 反馈延迟几十步;用**图**组织步骤间关系 | 方法 |
| [SSVPO: Effective Step-Level Credit Assignment for RL Training of Language Models](https://iclr.cc/virtual/2026/poster/10008182) 〔ICLR〕 | 用**序贯 Shapley 值**做步级信用分配 | 方法 + 博弈论工具 |
| [Credit Assignment with Resets in Language Model Reasoning](https://neurips.cc/virtual/2026/poster/151001) 〔NeurIPS〕 | 从轨迹中间状态**重置**来做针对性修正,而不是更新整条轨迹 | 方法 |
| [Transition-Aware Credit Assignment in Agentic Learning for LLM Reasoning](https://neurips.cc/virtual/2026/poster/154776) 〔NeurIPS〕 | 终端奖励下每个工具相关 token 都收到同样的轨迹优势,导致 GRPO 训练崩溃;转移感知 | 方法 + 问题诊断 |

### 路线 B:用不确定性、信息量、内部信号做稠密奖励

| 论文 | 核心想法 | 类型 |
|---|---|---|
| [Harnessing Uncertainty: Entropy-Modulated Policy Gradients for Long-Horizon LLM Agents](https://proceedings.mlr.press/v306/wang26ek.html) 〔ICML〕 | 形式化"**梯度幅度与策略熵耦合**":自信动作学得慢、不确定动作更新不稳;EMPG 用**自校准梯度缩放**(放大自信且正确、强惩罚自信但错误、衰减不确定步)加**未来清晰度奖励**;WebShop / ALFWorld / Deep Search 上 GRPO+EMPG、DAPO+EMPG 优于原版 〔速读〕 | 方法 |
| [AEM: Adaptive Entropy Modulation for Multi-Turn Agentic Reinforcement Learning](https://neurips.cc/virtual/2026/poster/154854) 〔NeurIPS〕 | 多轮 Agent RL 中自适应调制熵 | 方法 |
| [Information Gain-based Policy Optimization: A Simple and Effective Approach for Multi-Turn Search Agents](https://iclr.cc/virtual/2026/poster/10007215) 〔ICLR〕 | 多轮搜索 Agent:以**信息增益**作为过程奖励 | 方法 |
| [Attention Illuminates LLM Reasoning: The Uncovered Preplan-and-Anchor Rhythm Enables Fine-Grained Policy Optimization](https://proceedings.mlr.press/v306/li26ak.html) 〔ICML〕 | 用**注意力动态**揭示推理的"预规划-锚点"节律(局部聚焦头与全局聚焦头;两个度量 WAAD 与 FAI 找出预规划 token 与锚点 token),据此提出三种**重加权 token 级优势**的 RL 方法,兼容现有 RLVR 〔速读〕 | 方法 + 可解释性 |
| [Implicit Turn-Wise Policy Optimization for Proactive User-LLM Interaction](https://proceedings.mlr.press/v306/wang26x.html) 〔ICML〕 | ITPO:从稀疏结果奖励导出**隐式过程奖励模型**,得到逐轮奖励(并做归一化稳定);与 PPO / GRPO / RLOO 结合,在数学辅导、文档写作、医疗推荐上优于结果型基线;局限:依赖 LLM 用户模拟器 〔速读〕 | 方法 |
| [InT: Self-Proposed Interventions Enable Credit Assignment in LLM Reasoning](https://iclr.cc/virtual/2026/poster/10008910) 〔ICLR〕 | 模型**自提干预**来定位推理中出错的步骤 | 方法 |
| [Grounding Agent Reasoning with Structured Process Supervision for Multi-turn Reinforcement Learning](https://neurips.cc/virtual/2026/poster/155516) 〔NeurIPS〕 | 发现 74% 的失败轨迹存在"推理-观测不一致",且一旦出现会持续;结构化过程监督 | 方法 + 实证发现 |

### 路线 C:子问题、技能、记忆

| 论文 | 核心想法 | 类型 |
|---|---|---|
| [From Reasoning Chains to Verifiable Subproblems: Curriculum Reinforcement Learning Enables Credit Assignment for LLM Reasoning](https://neurips.cc/virtual/2026/poster/152795) 〔NeurIPS〕 | 把长推理拆成**可验证子问题**,以课程学习做信用分配(SCRL) | 方法 |
| [SAPO: Step-Level Skill-Augmented Policy Optimization for Multi-Turn LLM Agents](https://neurips.cc/virtual/2026/poster/149959) 〔NeurIPS〕 | 按步骤检索并更新**技能库** | 方法 |
| [Meta-Cognitive Memory Policy Optimization for Long-Horizon LLM Agents](https://neurips.cc/virtual/2026/poster/155887) 〔NeurIPS〕 | 记忆摘要质量逐步退化,而结果型 RL 无法定位;元认知记忆策略优化 | 方法 |
| [MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents](https://iclr.cc/virtual/2026/poster/10008961) 〔ICLR〕 | 让记忆与推理协同,使长程 Agent 内存**不随轮数无界增长** | 工程 |
| [Exploratory Memory-Augmented LLM Agent via Hybrid On- and Off-Policy Optimization](https://iclr.cc/virtual/2026/poster/10009229) 〔ICLR〕 | 记忆 + **在/离策略混合**以促进探索(EMPO²) | 方法 |
| [ResT: Reshaping Token-Level Policy Gradients for Tool-Use Large Language Models](https://iclr.cc/virtual/2026/poster/10008155) 〔ICLR〕 | 为工具调用任务**重塑 token 级策略梯度**以降方差 | 方法 |

## 2. 诊断与实证:这些问题到底在哪

| 论文 | 发现 | 类型 |
|---|---|---|
| [On Training Large Language Models for Long-Horizon Tasks: An Empirical Study of Horizon Length](https://proceedings.mlr.press/v306/kim26h.html) 〔ICML〕 | 以**任务长度**为中心的受控实证:任务长度本身(即使推理难度不变)是**独立的训练不稳定因素**;**缩短有效视野(horizon reduction)**能稳定 RL,并出现对更长视野的**视野泛化**;代码 Agent 的成功部分源于用循环/条件把长序列工具调用压成紧凑执行 〔速读〕 | 系统实证 |
| [On Information Self-Locking in Reinforcement Learning for Active Reasoning of LLM agents](https://proceedings.mlr.press/v306/zou26b.html) 〔ICML〕 | 主动推理里,**动作选择(AS)与信念追踪(BT)双向耦合**造成信用分配的结构性失败:结果型 RL 会被困在"信息自锁"(既不获取信息、也不吸收证据);提供理论与实证,提出 **AREW**(评论驱动的重加权,选择性重分配沿轨迹的优化信号),对噪声评论稳健、适用于多种 RL 机制 〔速读〕 | 理论 + 方法 |
| [Reducing Belief Deviation in Reinforcement Learning for Active Reasoning of LLM Agents](https://iclr.cc/virtual/2026/poster/10007172) 〔ICLR〕 | 主动推理的核心是**信念追踪**;LLM Agent 常出现信念偏差,据此改进 RL | 方法 |

## 3. 用 RL 训练协作 / 工具 / 推荐 / 其他 Agent 任务

| 论文 | 核心想法 | 类型 |
|---|---|---|
| [Tool-Integrated Reasoning via Hierarchical Multi-Agent Reinforcement Learning](https://neurips.cc/virtual/2026/poster/148956) 〔NeurIPS〕 | 把工具集成推理拆成规划者与执行者的**分层多智能体 RL** | 方法 |
| [Learning to Ask: Metacognitive Action Policy for Large and Small Language Model Collaboration](https://neurips.cc/virtual/2026/poster/148711) 〔NeurIPS〕 | 边缘-云协作中,大小模型间的"提问"作为元认知动作策略 | 方法 |
| [Adaptive Collaboration with Humans: Metacognitive Policy Optimization for Multi-Agent LLMs with Continual Learning](https://iclr.cc/virtual/2026/poster/10010313) 〔ICLR〕 | 多智能体 LLM 在知识边界外会集体失败;元认知策略优化,引入人的协作 | 方法 |
| [From Ideas to Code: Tree-structured Policy Optimization for Automated Algorithm Design with LLMs](https://neurips.cc/virtual/2026/poster/154407) 〔NeurIPS〕 | 自动算法设计:对"想法-计划"层级做**树结构**策略优化 | 方法 |
| [Rank-GRPO: Training LLM-based Conversational Recommender Systems with Reinforcement Learning](https://iclr.cc/virtual/2026/poster/10008949) 〔ICLR〕 | 会话式推荐:训练 LLM 时处理列表末尾排序质量下降(ConvRec-R1) | 应用 |

## 4. 更多长时程 Agent 训练:分层、里程碑、事后归因、框架与基准

### 4.1 分层与计划-执行

| 论文 | 核心想法 | 类型 |
|---|---|---|
| [HiPER: Hierarchical Plan–Execute RL for Multi-Turn LLM Agents](https://proceedings.mlr.press/v306/peng26h.html) 〔ICML〕 | 显式化成功轨迹里隐含的层级:**计划-执行接口**分离高层子目标规划与低层动作;**HAE(层级优势估计)**两时间尺度耦合,理论上**近似无偏且方差低于 GAE**;多个交互基准上优于基线 〔速读〕 | 理论 + 方法 |
| [StraTA: Incentivizing Agentic Reinforcement Learning with Strategic Trajectory Abstraction](https://neurips.cc/virtual/2026/poster/152317) 〔NeurIPS〕 | 用**策略性轨迹抽象**让 Agent 不再纯反应式,改善探索与信用分配 | 方法 |
| [Strict Subgoal Execution: Reliable Long-Horizon Planning in Hierarchical Reinforcement Learning](https://iclr.cc/virtual/2026/poster/10008513) 〔ICLR〕 | 分层 RL 中子目标不可行时,事后重标注难以纠正;**严格子目标执行** | 方法 |
| [MIRA: Memory-Integrated Reinforcement Learning Agent  with Limited LLM Guidance](https://iclr.cc/virtual/2026/poster/10007403) 〔ICLR〕 | 记忆整合的 RL Agent,只在需要时使用有限的 LLM 指导 | 方法 |
| [Milestone-Guided Policy Learning for Long-Horizon Language Agents](https://proceedings.mlr.press/v306/wang26jm.html) 〔ICML〕 | 识别**信用错配**与**样本低效**:超过 40% 的梯度更新含矛盾信号;BEACON 在**里程碑边界**切分轨迹,段内时间奖励塑形,**双尺度优势估计**;ALFWorld / WebShop / ScienceWorld 上相对 GRPO 的增益随任务长度从 26.2% 增至 73.6% 〔速读〕 | 方法 |
| [h1: Bootstrapping LLMs to Reason over Longer Horizons via Reinforcement Learning](https://proceedings.mlr.press/v306/ivanova26a.html) 〔ICML〕 | **把现有短问题链接成长依赖链**构造长程推理数据(无需额外标注),用**课程学习 + 仅结果奖励**的 RL;在组合的 GSM8K 上训练,迁移到 MATH-500、AIME、GSM Symbolic、ReasoningGym、LongBench-v2、Hash-hop;理论上证明**课程学习的样本复杂度相对全视野训练可有指数级改进**(类似稠密奖励) 〔速读〕 | 理论 + 方法 |

### 4.2 事后信息、图归因与不确定性

| 论文 | 核心想法 | 类型 |
|---|---|---|
| [From Outcomes to Actions: Leveraging Hindsight for Long-Horizon Language Agent Training](https://proceedings.mlr.press/v306/jiang26y.html) 〔ICML〕 | HPO:从**事后分布**视角重述蒙特卡洛策略梯度,指出离散语言动作空间的逐点比较会因忽略语义结构而**方差无界**;在**意图空间**构造事后分布得到有界方差、更平滑的学习信号,开销可忽略,甚至无结果奖励也能驱动学习 〔速读〕 | 理论 + 方法 |
| [Beyond Trajectory-Level Attribution: Graph-Based Credit Assignment for Agentic Reinforcement Learning](https://proceedings.mlr.press/v306/cheng26e.html) 〔ICML〕 | GraphGPO:把多条 rollout **聚合成统一的状态转移图**,依据"朝任务目标的全局进展"给每步分配信用,优于仅按最终结果归因的群组方法,且与现有群组 RL 互补 〔速读〕 | 方法 |
| [T$^2$PO: Uncertainty-Guided Exploration Control for Stable Multi-Turn Agentic Reinforcement Learning](https://proceedings.mlr.press/v306/wang26ah.html) 〔ICML〕 | 多轮 Agent RL 的不稳定源于**探索失控**;指出 token 熵与置信度各有局限,提出**不确定性引导的探索控制**,在多轮训练中稳定探索 〔速读〕 | 方法 |
| [Knowing When to Ask: Segment-Level Credit Assignment for LLM Tool Use](https://neurips.cc/virtual/2026/poster/149224) 〔NeurIPS〕 | 让模型学会"何时调用工具":**片段级**信用分配 | 方法 |
| [PiCA: Pivot-Based Credit Assignment For Search Agentic Reinforcement Learning](https://neurips.cc/virtual/2026/poster/155622) 〔NeurIPS〕 | 搜索 Agent 的**支点**式信用分配 | 方法 |
| [Memory-R2: Fair Credit Assignment for Long-Horizon Memory-Augmented LLM Agents](https://neurips.cc/virtual/2026/poster/152449) 〔NeurIPS〕 | 记忆让过去动作成为未来环境的一部分,使 rollout 之间不可比;**公平**信用分配 | 方法 |
| [MJ: Multi-turn LLM Jailbreaking via Decomposed Credit Assignment](https://neurips.cc/virtual/2026/poster/149174) 〔NeurIPS〕 | 多轮越狱攻击者的信用分解 | 方法 |
| [Selective Expert Guidance for Effective and Diverse Exploration in Reinforcement Learning of LLMs](https://iclr.cc/virtual/2026/poster/10008654) 〔ICLR〕 | RLVR 依赖基座能力;选择性使用专家引导,兼顾有效与多样的探索 | 方法 |
| [Bilevel Optimization of Synthetic Trajectories for Multi-Turn LLM Fine-Tuning](https://neurips.cc/virtual/2026/poster/155173) 〔NeurIPS〕 | 用**双层优化**改进多轮微调中的合成轨迹质量 | 方法 |

### 4.3 框架、数据规律与基准

| 论文 | 核心想法 | 类型 |
|---|---|---|
| [AgentGym-RL: An Open-Source Framework to Train LLM Agents for Long-Horizon Decision Making via Multi-Turn RL](https://iclr.cc/virtual/2026/poster/10008785) 〔ICLR〕 | 开源的多轮 RL 框架,覆盖多种真实环境 | 工程框架 |
| [MobileRL: Online Agentic Reinforcement Learning for Mobile GUI Agents](https://iclr.cc/virtual/2026/poster/10010890) 〔ICLR〕 | 移动 GUI Agent 的在线 RL,应对任务难度重尾与采样低效 | 工程 + 方法 |
| [ReTool: Reinforcement Learning for Strategic Tool Use in LLMs](https://iclr.cc/virtual/2026/poster/10006978) 〔ICLR〕 | 用 RL 教模型策略性使用**代码解释器** | 方法 |
| [VSearcher: Long-Horizon Multimodal Search Agent via Reinforcement Learning](https://neurips.cc/virtual/2026/poster/149459) 〔NeurIPS〕 | 长程**多模态**搜索 Agent | 应用 |
| [AstraFlow: Dataflow-Oriented Reinforcement Learning for Agentic LLMs](https://neurips.cc/virtual/2026/poster/152612) 〔NeurIPS〕 | 面向 Agent 的数据流式 RL 系统,支持多策略协同训练与异构资源 | 系统工程 |
| [How Data Scales in Agentic Reinforcement Learning: Laws and Synthesis Strategies](https://neurips.cc/virtual/2026/poster/155951) 〔NeurIPS〕 | Agent RL 里**任务、环境、轨迹**的数据缩放规律与合成策略 | 实证 |
| [ACON: Optimizing Context Compression for Long-horizon LLM Agents](https://proceedings.mlr.press/v306/kang26b.html) 〔ICML〕 | 在**自然语言空间优化压缩指南**(模型无关),同时压缩交互历史与环境观测;AppWorld / OfficeBench / 多目标 QA 上峰值 token 降 26–54% 且成功率优于其他压缩基线;压缩器可蒸馏进小模型,让 Qwen3-14B 等小 Agent 接近大模型 〔速读〕 | 工程 |
| [From Player to Master: Enhancing Test-Time Learning of LLM Agents via Reinforcement Learning over Memory](https://proceedings.mlr.press/v306/cai26b.html) 〔ICML〕 | MemoPilot:可插拔的**记忆领航员**,通过端到端训练**记忆更新过程**,提升冻结 LLM 在重复交互中的测试时学习;用带逐轮奖励与轮级优势的多轮 GRPO;局限:面对非平稳/自适应对手会退化,记忆预算仅 512 token 〔速读〕 | 方法 |
| [Whom to Query for What: Adaptive Group Elicitation via Multi-Turn LLM Interactions](https://proceedings.mlr.press/v306/ding26d.html) 〔ICML〕 | 在**有限提问与受访者预算**下做自适应群体信息获取:联合选择问题、受访者与信息传播机制——自适应问题选择 + **群体关系式受访者选择** + 基于图的响应插补;真实观点数据上在紧预算(尤其早期轮次)提升精度与校准 〔速读〕 | 方法 |
| [SafeHarbor: Defining Precise Decision Boundaries via Hierarchical Memory-Augmented Guardrail for LLM Agent Safety](https://proceedings.mlr.press/v306/liu26fn.html) 〔ICML〕 | 对有害轨迹做对抗增强并经自适应聚类自动生成**安全规则**;**对比学习投影**联合评估工具调用的语义与情境风险以减少过度拒绝;**自组织层级记忆**(自适应叶分裂)支持规则的可扩展管理与快速检索,无需重训;GPT-4o 上良性效用 63.6%,有害拒绝率 >93% 〔速读〕 | 安全 / 工程 |
| [Strategic Planning and Rationalizing on Trees Make LLMs Better Debaters](https://iclr.cc/virtual/2026/poster/10010721) 〔ICLR〕 | 用树上的战略规划与合理化提升辩论能力 | 应用 |
| [$\chi$-Bench: Can AI Agents Automate End-to-End, Long-Horizon, Policy-Rich Healthcare Workflows?](https://neurips.cc/virtual/2026/poster/139361) 〔NeurIPS〕 | 端到端、长程、策略密集的**医疗工作流**基准 | 基准 |
| [How Far Can LLM Agents Reason with Tables? Benchmarking Multi-Turn Agentic Table Question Answering in the Wild](https://proceedings.mlr.press/v306/huang26cf.html) 〔ICML〕 | TableAgent-Bench:基于 2,275 张工业表格的**多轮 Agentic TableQA 基准**,配 TAEF 框架诊断表定位、工具调用合理性等中间失败;评测 25 个模型,长程推理一致性差距明显 〔速读〕 | 评测 |
| [AlphaPareto: Formulaic Alpha Discovery with LLM-Guided Multi-Objective Reinforcement Learning](https://neurips.cc/virtual/2026/poster/153971) 〔NeurIPS〕 | LLM 引导的多目标 RL 发现公式化 alpha | 应用 |
| [Lean Refactor: Multi-Objective Controllable Proof Optimization via Agentic Strategy Search](https://neurips.cc/virtual/2026/poster/149326) 〔NeurIPS〕 | 对 Lean 证明做多目标可控重构的 Agent 框架 | 应用 |

## 4. 本篇小结

1. **统一视角**:所有方法都在回答 "$A_t$ 该怎么估?"。区别只是**用什么信号**(熵、信息增益、注意力、过程奖励、子问题、Shapley 值)和**如何分组**(分层、层级、图)。
2. **理论 vs 工程**:本组理论成分较少,多数是**实证诊断 + 工程配方**。较有理论味的有 *SSVPO*(Shapley 公理化)与 *Transition-Aware Credit Assignment*(失败机制分析)。
3. 对社会情境的启发:*ALSO* 把整段多轮对话压成一个"臂"来选(episode 级),本篇指出**逐轮 / 逐步的信用分配**才是更细的路径——两者可以互补。
