# 10|LLM 多智能体、博弈、自博弈与红队

> **本篇为什么放在一起读**:当 Agent 是 LLM,"多智能体"就变成**角色分工、讨论、谈判、对抗**。这里的共同方法是:**自博弈(self-play)**——让模型和"越来越强的自己"对弈,从而自动产生课程;以及**对抗共演化**——攻击者与防御者一起进化(红队)。
> **阅读深度**:ICML 2026 论文为**速读**(读了本地 PDF 的引言、贡献陈述与结论,描述后标〔速读〕;个别重点论文已全文精读,见各自小节);ICLR / NeurIPS 论文因 PDF 无法脚本下载,为**摘要级**(只依据官方摘要)。
> **与 `02` 的关系**:谈判 / 说服 / 心智理论(ToM)类论文在 `02`;本篇偏**训练方法与系统**。

## 1. 自博弈与对抗共演化

| 论文 | 核心想法 | 类型 |
|---|---|---|
| [SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning](https://iclr.cc/virtual/2026/poster/10011289) 〔ICLR〕 | 模型在多轮**零和博弈**中与不断变强的自己对弈,自动产生课程、无需人类监督,借此提升推理 | 方法 |
| [SPELL: Self-Play Reinforcement Learning for Evolving Long-Context Language Models](https://iclr.cc/virtual/2026/poster/10011237) 〔ICLR〕 | 多角色自博弈(提问者、回答者……)实现**无标签**长上下文推理优化 | 方法 |
| [MARSHAL: Incentivizing Multi-Agent Reasoning via Self-Play with Strategic LLMs](https://iclr.cc/virtual/2026/poster/10010522) 〔ICLR〕 | 多智能体 + 策略型 LLM 的自博弈,用于长程信用分配与 Agent 特定优势 | 方法 |
| [Strat-Reasoner: Reinforcing Strategic Reasoning of LLMs in Multi-Agent Games](https://proceedings.mlr.press/v306/he26u.html) 〔ICML〕 | 受博弈论启发的**递归推理**范式;RL 框架整合**集中式 CoT 比较**与**混合优势估计**,在多条推理序列间做细粒度信用分配,**无需奖励模型**;竞争与合作博弈上超过开源基线并接近闭源,泛化到未见博弈 〔速读〕 | 方法 |
| [Chasing Moving Targets with Online Self-Play Reinforcement Learning for Safer Language Models](https://proceedings.mlr.press/v306/liu26bn.html) 〔ICML〕 | Self-RedTeam:攻击者 $\pi_A$ 与防御者 $\pi_D$ 在**零和红队博弈**中以验证器奖励 $r_\phi$ 做 min–max 在线共演化,取代"先攻击、再打补丁"的被动循环;发现更多样的攻击并提升对对抗输入的鲁棒性 〔速读〕 | 方法 |
| [AdvEvo-MARL: Shaping Internalized Safety through Adversarial Co-Evolution in Multi-Agent Reinforcement Learning](https://proceedings.mlr.press/v306/pan26p.html) 〔ICML〕 | 攻击者与防御者**对抗共演化**,把安全意识**内化**进每个 Agent;引入**公共基线**(同功能组用组平均回报做优势基线)促进组内协作与更新稳定;多种 MAS 攻击场景下安全性最高 +18.33%,通用任务几乎不降甚至提升 〔速读〕 | 方法 |
| [Metis: Learning to Jailbreak LLMs via Self-Evolving Metacognitive Policy Optimization](https://proceedings.mlr.press/v306/zhou26bp.html) 〔ICML〕 | 把越狱重新表述为**对抗 POMDP 上的推理时策略优化**,自演化元认知循环(诊断→评审→策略转向),带可解释的推理轨迹;指出现有多轮越狱攻击存在**泛化鸿沟**,对强对齐前沿模型成功率下降 〔速读〕 | 方法 |
| [Auto-RT: Automatic Jailbreak Strategy Exploration for Red-Teaming Large Language Models](https://iclr.cc/virtual/2026/poster/10009656) 〔ICLR〕 | 自动探索越狱策略的强化学习红队,适应不断演化的防御 | 方法 |
| [There are Levels to It: Red Teaming LLMs with Hierarchical Reinforcement Learning](https://neurips.cc/virtual/2026/poster/153243) 〔NeurIPS〕 | 把红队写成 MDP,用**层次 RL** 最大化累计危害 | 方法 |
| [Generalized Intention Modeling in Multi-Agent Reinforcement Learning](https://neurips.cc/virtual/2026/poster/151588) 〔NeurIPS〕 | 对手建模的意图嵌入常是事先选定的(如下一动作);**广义意图建模** | 方法 |
| [What Preferences Can—and Cannot—Predict in Multi-Agent Online Learning](https://proceedings.mlr.press/v306/abbadi26a.html) 〔ICML〕 | 在一大类**正则化学习动力学**里,**序数偏好信息不一定决定稳定性**,博弈**真实的收益数值**可能至关重要——同一偏好顺序的不同基数实现可导致截然不同的长期结果;对偏好学习与奖励规约有警示意义 〔速读〕 | 理论 |
| [Learning-Augmented Mechanism Design for Facility Location under $L_p$-Norm Social Costs](https://neurips.cc/virtual/2026/poster/149389) 〔NeurIPS〕 | 设施选址:在不完美预测下设计防策略操纵(strategyproof)的机制,权衡一致性与鲁棒性 | 理论 |

## 2. 训练 / 优化多 LLM 系统

| 论文 | 核心想法 | 类型 |
|---|---|---|
| [Stronger-MAS: Multi-Agent Reinforcement Learning for Collaborative LLMs](https://iclr.cc/virtual/2026/poster/10010282) 〔ICLR〕 | On-policy RL(GRPO 式)训练**协作 LLM** 多智能体系统 | 方法 |
| [Dr. MAS: Stable Reinforcement Learning for Multi-Agent LLM Systems](https://neurips.cc/virtual/2026/poster/151327) 〔NeurIPS〕 | GRPO 全局归一化基线与异质 Agent 的奖励分布不匹配是不稳定的来源(理论定位);稳定的多智能体 LLM RL | 理论 + 方法 |
| [MARTI: A Framework for Multi-Agent LLM Systems Reinforced Training and Inference](https://iclr.cc/virtual/2026/poster/10010710) 〔ICLR〕 | 开源多智能体 LLM 强化训练与推理框架:集中交互 + 分布式训练 + 多轮异步 rollout | 工程框架 |
| [AgentPO: Enhancing Multi-Agent Collaboration via Reinforcement Learning](https://iclr.cc/virtual/2026/poster/10011478) 〔ICLR〕 | 用 RL 训练专门的"协作者"(Collaborator),直接优化 Agent 间协作 | 方法 |
| [Multi-Agent Design: Optimizing Agents with Better Prompts and Topologies](https://iclr.cc/virtual/2026/poster/10010348) 〔ICLR〕 | 自动设计多智能体系统的**提示与拓扑** | 方法 |
| [From Interaction Trajectories to Prompt Rules: Credit Assignment for Multi-Agent Prompt Optimization](https://proceedings.mlr.press/v306/wu26bp.html) 〔ICML〕 | TRUCE:多 Agent 系统里角色与交互结构固定,**提示就是隐式行为策略**;基于轨迹的信用分配把结果反馈归因到有信息的子轨迹,再做**单元级、局部**的提示修改;MultiAgentBench 与多智能体编程基准上一致提升 〔速读〕 | 方法 |
| [Context Learning for Multi-Agent Discussion](https://iclr.cc/virtual/2026/poster/10010672) 〔ICLR〕 | 多 LLM 讨论易**不一致**,源于各自上下文错位;上下文学习 | 方法 |
| [DECOR: Learning to Decompose and Collaborate in Deep Search via Multi-Agent Reinforcement Learning](https://proceedings.mlr.press/v306/chen26gp.html) 〔ICML〕 | 把深度搜索**功能性分解**为 Planner / Filter / Answerer 三个 Agent(Filter 过滤检索上下文以减轻"认知过载");**混合奖励**(团队结果信号 + 角色内在稠密奖励)解决多智能体信用分配;7 个基准显著优于单体基线;代价:推理延迟与 token 更高 〔速读〕 | 方法 |
| [GraphPlanner: Graph Memory-Augmented Agentic Routing for Multi-Agent LLMs](https://iclr.cc/virtual/2026/poster/10008792) 〔ICLR〕 | 图记忆增强的**智能体路由** | 方法 |
| [$\texttt{Multi}^2$: Hierarchical Multi-Agent Decision-Making with LLM-Based Agents in Interactive Environments](https://proceedings.mlr.press/v306/park26h.html) 〔ICML〕 | 角色专门化的**层级训练-适应框架**:System 1 生成情境化子目标,System 2 执行原子动作,以缓解长程"目标漂移"与 token 低效;System 2 用**策略锚定的离线学习 + KL 正则的在线精修**;发布层级数据集 〔速读〕 | 方法 |
| [Market Regime Council for Dynamic Credit Assignment in Multi-Agent LLM  Decision Systems](https://neurips.cc/virtual/2026/poster/148428) 〔NeurIPS〕 | 多 LLM 投资决策系统:用精确 **Shapley 信用**分配并应对市场状态切换 | 应用 |
| [Benefits and Limitations of Communication in Multi-Agent Reasoning](https://iclr.cc/virtual/2026/poster/10011921) 〔ICLR〕 | 给多智能体系统的**表达力**建立理论框架 | 理论 |
| [Eigen-Agent: Adaptive Multi-Agent Scientific Reasoning with Monitor-Based RAG](https://iclr.cc/virtual/2026/poster/10008628) 〔ICLR〕 | 隐式检索 + 结构化协作,避免"平均掉"强解 | 方法 |
| [MEMO: Memory-Augmented Model Context Optimization for Robust Multi-Turn Multi-Agent LLM Games](https://proceedings.mlr.press/v306/xie26x.html) 〔ICML〕 | **无权重更新**的自博弈:持久**记忆库**把轨迹提炼成可复用洞见 + 锦标赛式提示演化与优先回放;5 个文本博弈上胜率显著提升、比 RL 基线少用 **19 倍**对局、降低结果离散;并指出**提示的微小变化会改变有效策略与排名**,应做按提示分层的报告 〔速读〕 | 方法 + 评测反思 |

## 3. 应用型 / 任务型多智能体

| 论文 | 核心想法 | 类型 |
|---|---|---|
| [BFS-Prover-V2: Scaling up Multi-Turn Off-Policy RL and Multi-Agent Tree Search for LLM Step-Provers](https://proceedings.mlr.press/v306/xin26d.html) 〔ICML〕 | Lean 4 定理证明系统:训练端为**免蒸馏的多阶段专家迭代**(离策略 RL),用基于困惑度的 tactic 级数据过滤形成自动课程,并用周期性重训作"软重置"逃离局部最优;推理端用**规划者 Agent 分解子目标 + 共享子目标缓存**协作搜索;miniF2F 达 SOTA 级、ProofNet 上有稳健 OOD 泛化 〔速读〕 | 系统 + 方法 |
| [HiMAP-Travel: Hierarchical Multi-Agent Planning for Long-Horizon Constrained Travel](https://proceedings.mlr.press/v306/bui26b.html) 〔ICML〕 | **协调者**分配资源、**并行执行者**日级规划;**事务式约束监控器**(原子锁保证预算 / 多样性等耦合约束的确定性安全)+ **合作议价协议** + **GRPO 训练的统一角色条件策略**;相同骨干下比顺序规划高 8.67 pp,种子间方差降低 93% 〔速读〕 | 方法 + 应用 |
| [ATLAS: Constraints-Aware Multi-Agent Collaboration for Real-World Travel Planning](https://iclr.cc/virtual/2026/poster/10007591) 〔ICLR〕 | 约束感知的多 Agent 旅行规划 | 应用 |
| [MARS-SQL: A Multi-Agent Reinforcement Learning Framework For Text-To-SQL](https://proceedings.mlr.press/v306/yang26dh.html) 〔ICML〕 | 文本转 SQL 的可训练多智能体:**Grounding**(模式定位)/ **Generation**(Think–Act–Observe 循环,用真实数据库反馈自我纠错)/ **Validation**(生成概率式轨迹选择);BIRD-dev 执行准确率 77.84%、Spider-test 89.75% 〔速读〕 | 应用 + 方法 |
| [StitchCUDA: An Automated Multi-Agents End-to-End GPU Programing Framework with Rubric-based Agentic Reinforcement Learning](https://proceedings.mlr.press/v306/li26jc.html) 〔ICML〕 | 三个 Agent 生成**端到端 GPU 程序**;把多轮 Agentic RL **分解为原子技能**降低 rollout 开销,配**细粒度评分细则奖励**;发现**格式检查无法防止奖励欺骗**(如硬编码难以用正则发现) 〔速读〕 | 应用 + 方法 |

## 4. LLM 博弈能力评测

| 论文 | 发现 | 类型 |
|---|---|---|
| [How Far Are LLMs from Professional Poker Players? Revisiting Game-Theoretic Reasoning with Agentic Tool Use](https://iclr.cc/virtual/2026/poster/10006805) 〔ICLR〕 | 系统评估 LLM 在扑克中的博弈论推理,发现其难以对抗传统算法,并归纳出反复出现的失败模式 | 评测 |
| [LLMs as Rules Oracles: Exploring Real-World Multimodal Reasoning in Tabletop Strategy Game Environments](https://iclr.cc/virtual/2026/poster/10009317) 〔ICLR〕 | LudoBench:桌游规则理解的多模态推理基准(并非策略深度) | 评测 |
| [SafeDialBench: A Fine-Grained Safety Evaluation Benchmark for Large Language Models in Multi-Turn Dialogues with Diverse Jailbreak Attacks](https://iclr.cc/virtual/2026/poster/10010144) 〔ICLR〕 | 多轮对话 + 多种越狱攻击下的**细粒度安全**评测基准 | 评测 |
| [CoffeeBench: A Benchmark for Long-Horizon Strategic Decision-Making in Multi-Agent Economies](https://neurips.cc/virtual/2026/poster/138891) 〔NeurIPS〕 | 多智能体**经济**环境中的长程战略决策基准 | 评测 |

## 5. 本篇小结

1. **自博弈 = 自动课程**:*SPIRAL / SPELL / Self-RedTeam* 都用"和自己对弈"产生越来越难的训练分布。
2. **多 LLM 系统的 RL 有共同坑**:*Dr. MAS* 指出全局基线与异质 Agent 的奖励分布错配——与 `02` 中 *OmniSapiens / HARPO* 的"异质样本导致优势失衡"是**同一类问题**,可对照读。
3. **博弈论工具箱**:Shapley 信用、Nash 均衡、完美贝叶斯均衡、无悔动态,在这些论文里反复出现,值得初学者系统补课。
