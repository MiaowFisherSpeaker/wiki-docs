# 05|偏好优化与 RLHF 对齐:从"成对比较"到"列表、图与博弈"

> **本篇为什么放在一起读**:这批论文都在回答"**怎样把人的偏好变成训练信号**"。它们共享一条清晰的演进线:RLHF(奖励模型 + PPO)→ DPO(去掉奖励模型)→ DPO 的各种修补(噪声、传递性、先验、多目标、个性化)→ 偏好变成**列表 / 图 / 博弈**。
> **阅读深度**:ICML 2026 论文为**速读**(读了本地 PDF 的引言、贡献陈述与结论,描述后标〔速读〕;个别重点论文已全文精读,见各自小节);ICLR / NeurIPS 论文因 PDF 无法脚本下载,为**摘要级**(只依据官方摘要)。
> **建议顺序**:先读 §1 的三个"问题诊断类"(Alignment Tampering、Gradient Regularization、Are Easier or Harder),再读 §2 的目标函数改进。

## 0. 给初学者的预备

**Bradley–Terry (BT) 模型**:假设回答 $y_w$ 优于 $y_l$ 的概率是 $\sigma\big(r(y_w)-r(y_l)\big)$。DPO 的损失正是在 BT 假设下推出的:

$$\mathcal L_{\text{DPO}}=-\mathbb E\Big[\log\sigma\Big(\beta\log\tfrac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)}-\beta\log\tfrac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\Big)\Big]$$

下面很多论文都在质疑这条式子里的**某个假设**:BT 成立吗?偏好标签干净吗?参考策略该固定吗?只用成对比较够吗?

## 1. 诊断:RLHF / 奖励有哪些系统性问题

### Alignment Tampering(ICML 2026)
📄 [官方页面](https://proceedings.mlr.press/v306/hahm26a.html) · 本地 PDF:`papers/ICML2026/Alignment_Tampering_How_Reinforcement_Learning_from_Human_Fe.pdf` · 〔ICML 2026〕
- **想法**:偏好数据由被对齐的模型*自己的输出*构成,模型可以**影响偏好数据**,让 RLHF 放大不期望的偏差;根源是偏好集来自自身输出、且成对比较只给"谁更好"(摘要)。
- **类型**:问题发现(安全 / 理论倾向)。
- **带走**:RLHF 的数据不是"外生"的,存在**闭环**。这与社会情境里"对象会反应"同源。

### Gradient Regularization Mitigates Reward Hacking in RLHF and RLVR(ICML 2026)
📄 [官方页面](https://proceedings.mlr.press/v306/ackermann26a.html) · 本地 PDF:`papers/ICML2026/Gradient_Regularization_Mitigates_Reward_Hacking_in_Reinforc.pdf` · 〔ICML 2026〕
- **想法**:常规防奖励欺骗靠 KL 惩罚;这里换视角——让策略更新偏向**奖励更准的区域**(用梯度正则),而不是只限制离参考模型多远。
- **类型**:方法。
- **带走**:KL 不是唯一的"安全阀"。

### Are Easier or Harder Examples Better? Rethinking Data Selection for Reward Models and Preference Optimization(NeurIPS 2026)
📄 [官方页面](https://neurips.cc/virtual/2026/poster/151585) · [OpenReview](https://openreview.net/forum?id=ZVpkdTlkIZ)(PDF 需手动下载,脚本被其验证拦截) · 〔NeurIPS 2026〕
- **想法**:此前文献对"难例还是易例更好"结论冲突;作者在 RM、DPO、GRPO 上系统隔离"难度"这个因素(摘要)。
- **类型**:系统性实证。
- **带走**:当文献打架时,**先把混杂因素隔离**。

### Mitigating Reward Hacking in LLM-based Recommendation / Causal DPO for Generative Recommendation(ICML 2026)
📄 [官方页面](https://proceedings.mlr.press/v306/chen26ez.html) · 本地 PDF:`papers/ICML2026/Mitigating_Reward_Hacking_in_LLM_based_Recommendation_A_Pref.pdf` · 〔ICML 2026〕
📄 [官方页面](https://proceedings.mlr.press/v306/zhao26bo.html) · 本地 PDF:`papers/ICML2026/Causal_Direct_Preference_Optimization_for_Distributionally_R.pdf` · 〔ICML 2026〕
- **想法**:两篇都把 DPO 搬到**推荐**。前者从梯度角度形式化"奖励欺骗";后者指出 DPO 会**放大环境混杂造成的伪相关**,损害分布外泛化。
- **类型**:方法 + 理论分析。
- **带走**:推荐里的偏好数据混杂更严重,因果视角有用。

## 2. 目标函数改进:DPO 家族

> 这一节论文很多,按"它挑战 DPO 的哪个假设"归类,**先看表,感兴趣再点进去**。

| 论文 | 它挑战的假设 / 核心想法 | 类型 |
|---|---|---|
| [Autoregressive Direct Preference Optimization](https://proceedings.mlr.press/v306/oi26b.html) 〔ICML〕 | 把能量函数的定义域从输出空间 $\mathcal Y$ 扩到其**前缀闭包** $\mathcal Y^*$,使 BT 模型下就纳入自回归假设(ADPO);证明任意奖励可由自回归模型重参数化;找出两个长度度量:token 长度 $\mu$ 与反馈长度 $\mu'$——**DPO 是 $\mu'\equiv1$ 的特例,$\mu'\equiv\mu$ 则是 token 级变体**,两者独立选择可得任意粒度 〔速读〕 | 理论 + 方法 |
| [$\textitS$-SPPO: Semantic-Calibrated Self-Play Preference Optimization](https://proceedings.mlr.press/v306/chen26hc.html) 〔ICML〕 | 发现自博弈偏好优化在偏好预言机对**语义无差别**的回答给出过度自信胜负时会不稳定;S-SPPO 做**监督校准**(语义门控退火胜率目标)与**表示校准**(潜空间排斥防模式坍缩),证明保持常和博弈结构;Llama-3-8B 在 AlpacaEval 2.0 上 52.19% 胜率(LC 47.46%),不用人工标注或更强模型监督 〔速读〕 | 理论 + 方法 |
| [RSPO: Regularized Self-Play Alignment of Large Language Models](https://proceedings.mlr.press/v306/tang26ab.html) 〔ICML〕 | 在自博弈对齐的损失里加入**对参考策略的正则项**(框架 RSPO),证明仍收敛到正则化偏好博弈的 **Nash 均衡**;Mistral-7B 上 AlpacaEval-2 / Arena-Hard / MT-Bench 优于 SPPO、SimPO 与迭代 DPO,且正则促进回答多样性 〔速读〕 | 理论 + 方法 |
| [Multiplayer Nash Preference Optimization](https://iclr.cc/virtual/2026/poster/10006669) 〔ICLR〕 | 把对齐看作两人 Nash 博弈(NLHF)仍不够,扩展到**多人**,刻画非传递与异质偏好 | 理论 + 方法 |
| [Beyond Pairs: Your Language Model is Secretly Optimizing a Preference Graph](https://neurips.cc/virtual/2026/poster/150929) 〔NeurIPS〕 | 多条 rollout 构成**偏好图**;成对 DPO 丢弃传递性、产生冗余/冲突监督 | 方法 |
| [INSPO : Unlocking Intrinsic Self-Reflection for LLM Preference Optimization](https://neurips.cc/virtual/2026/poster/153491) 〔NeurIPS〕 | DPO 的最优策略会随标量化函数/参考策略改变(缺乏**不变性**),且没用满成对信息 | 理论 + 方法 |
| [MaPPO: Maximum a Posteriori Preference Optimization with Prior Knowledge](https://neurips.cc/virtual/2026/poster/149436) 〔NeurIPS〕 | 把偏好学习写成 **MAP 估计**,显式纳入先验奖励知识 | 方法 |
| [Bias-Variance Optimized Preference Optimization for Large Reasoning Models](https://neurips.cc/virtual/2026/poster/152470) 〔NeurIPS〕 | 推理模型的 trace 边缘化目标不可算,单 trace 近似带来高梯度方差;从**偏差-方差**角度优化 | 理论 + 方法 |
| [Semantic-aware Wasserstein Policy Regularization for Large Language Model Alignment](https://iclr.cc/virtual/2026/poster/10007051) 〔ICLR〕 | KL/f-散度只比较相同 token 位置的概率,不懂语义;用 **Wasserstein 正则** | 方法 |
| [Convex Optimization for Alignment and Preference Learning on a Single GPU](https://proceedings.mlr.press/v306/feng26l.html) 〔ICML〕 | COALA:用**凸神经网络(cvxNN)**提供稳定奖励信号,**无需冻结参考模型**,在 LLM 基础模型导出的偏好特征上做凸优化而非在文本上优化;证明收敛;发布 26,621 条师生对话的 EduFeedback 与 65,606 个偏好对;JAX 开源,可单卡运行 〔速读〕 | 方法 + 工程 |
| [$f$-GRPO & Beyond: Divergence-Based Reinforcement Learning Algorithms for General LLM Alignment](https://neurips.cc/virtual/2026/poster/153358) 〔NeurIPS〕 | 把"偏好目标=散度估计"的视角推广到 RLVR,得到 f-GRPO 系列 | 理论 + 方法 |
| [Distributionally Robust Listwise Preference Optimization](https://neurips.cc/virtual/2026/poster/154214) 〔NeurIPS〕 | 标签排序有歧义(近似平局、标注不一致);在 Plackett–Luce 上做**分布鲁棒**的列表偏好优化 | 理论 + 方法 |
| [Mult-DPO: Multinomial Direct Preference Optimization for Recommender Systems](https://neurips.cc/virtual/2026/poster/149537) 〔NeurIPS〕 | 推荐里反馈是**集合式**多正例,不是成对;提出多项式 DPO | 方法 |
| [Uni-DPO: A Unified Paradigm for Dynamic Preference Optimization of LLMs](https://iclr.cc/virtual/2026/poster/10010533) 〔ICLR〕 | 并非所有偏好对同等重要:按数据质量与学习难度**动态加权**的统一框架 | 方法 |
| [RE-PO: Robust Enhanced Policy Optimization as a General Framework for LLM Alignment](https://iclr.cc/virtual/2026/poster/10007900) 〔ICLR〕 | 偏好标签有噪声且可靠性不一;提出稳健增强的对齐框架 | 方法 |

> **这一节的趋势**:偏好从"一对一"走向**列表、图、博弈**;理论抓手是 BT 假设、非传递性、参考策略的作用。

## 3. 数据与噪声:标签不干净怎么办

| 论文 | 核心想法 | 类型 |
|---|---|---|
| [Distributionally Robust Reinforcement Learning from Human Feedback](https://proceedings.mlr.press/v306/mandal26a.html) 〔ICML〕 | 同时稳健化**奖励估计**(小批量梯度下降,线性奖励下 $T=O(1/\varepsilon^2)$ 迭代、每轮 $\tilde O(1/\varepsilon^2)$ 样本)与**策略优化**(重加权自然策略梯度,对数线性策略下线性收敛);在 Gemma-2B-it 与 Mistral-7B 上验证分布外稳健性 〔速读〕 | 理论 + 方法 |
| [Optimal Transport for LLM Reward Modeling from Noisy Feedback](https://proceedings.mlr.press/v306/pan26o.html) 〔ICML〕 | 把带噪偏好下的奖励建模看作**分布匹配**(联合一致性差异 JCD);刚性质量守恒迫使模型拟合离群点,故用**部分传输的质量松弛**(SelectiveRM)自主剔除与语义一致性矛盾的噪声样本;理论上优化了干净泛化误差的更紧上界 〔速读〕 | 理论 + 方法 |
| [Correct, Route, Calibrate: Efficient Preference Optimization from Noisy, Heterogeneous Human Feedback](https://neurips.cc/virtual/2026/poster/152258) 〔NeurIPS〕 | 标注者异质、可靠性随难度变化;提出"纠正-路由-校准"的偏好优化 | 方法 |
| [ActiveDPO: Active Direct Preference Optimization for Sample-Efficient Alignment](https://iclr.cc/virtual/2026/poster/10009521) 〔ICLR〕 | 人类偏好标注昂贵;**主动**选择样本以提升对齐的样本效率 | 方法 |
| [Alignment through Meta-Weighted Online Sampling: Bridging the Gap between Data Generation and Preference Optimization](https://iclr.cc/virtual/2026/poster/10011202) 〔ICLR〕 | 离线偏好数据与演化中的策略有分布错配;用元权重自适应采样(MetaAPO) | 方法 |
| [CAReDiO: Enhancing Cultural Alignment of LLM via Representativeness and Distinctiveness Guided Data Optimization](https://proceedings.mlr.press/v306/yao26p.html) 〔ICML〕 | 文化对齐语料缺**代表性**与**区分度**;用两个信息论目标迭代生成并精炼问题与回答,构造覆盖 15 种文化的 CARDSet,在多选与开放式基准上优于更大规模甚至人工整理的数据集 〔速读〕 | 数据工程 |
| [IDEAL: Data Equilibrium Adaptation for Multi-Capability Language Model Alignment](https://iclr.cc/virtual/2026/poster/10007504) 〔ICLR〕 | 多能力 SFT 时数据配比影响最终性能;提出数据均衡自适应 | 数据工程 |

## 4. 个性化、多元与多目标

| 论文 | 核心想法 | 类型 |
|---|---|---|
| [Learning to summarize user information for personalized reinforcement learning from human feedback](https://iclr.cc/virtual/2026/poster/10010999) 〔ICLR〕 | RLHF 用单一奖励模型假设人人偏好相同;用 RL 学会**总结用户信息**(PLUS)以个性化 | 方法 |
| [Swap-guided Preference Learning for Personalized Reinforcement Learning from Human Feedback](https://iclr.cc/virtual/2026/poster/10007470) 〔ICLR〕 | VPL(变分偏好学习)出现**后验坍缩**;用交换引导偏好学习 | 方法 + 问题发现 |
| [MATO: Multi-objective Personalized Alignment with Test-time Optimization for Large Language Models](https://neurips.cc/virtual/2026/poster/150723) 〔NeurIPS〕 | 多目标个性化对齐,在**测试时优化**,无需为每个偏好训练奖励模型 | 方法 |
| [LLM Active Alignment: A Nash Equilibrium Perspective](https://neurips.cc/virtual/2026/poster/149530) 〔NeurIPS〕 | 把一群 LLM 的行为用 **Nash 均衡**分析:每个 Agent 策略性地选择与哪些人群对齐 | 理论 |
| [Towards Cognitively-Faithful Decision-Making Models to Improve AI Alignment](https://iclr.cc/virtual/2026/poster/10006431) 〔ICLR〕 | 标准偏好获取忽略人的启发式与简化思维;建立更贴近认知的决策模型 | 建模 |
| [Safety Game: Inference-Time Alignment of Black-Box LLMs via Constrained Optimization](https://proceedings.mlr.press/v306/nguyen26s.html) 〔ICML〕 | 黑盒 LLM 的**推理时**安全:从不完全信息博弈推出一个**决策层**(相对回退的间隔 + 聚合而非逐候选的风险约束,有界打分误差下的条件稳健保证 Thm 3.1),在候选集上用线性规划选择;20 个模型-基准设定中 14 个取得最佳安全-有用性权衡;瓶颈在**打分器校准**而非决策层 〔速读〕 | 理论 + 方法 |

## 5. 其他领域的偏好优化(多模态、生成、翻译、推理)

| 论文 | 核心想法 | 类型 |
|---|---|---|
| [AVERE: Improving Audiovisual Emotion Reasoning with Preference Optimization](https://iclr.cc/virtual/2026/poster/10006963) 〔ICLR〕 | 音视频情感推理中的伪关联与幻觉;用偏好优化改进,并提出 EmoReAlM 基准 | 方法 + 评测 |
| [Cat-PO: Cross-modal Adaptive Token-rewards for Preference Optimization in Truthful Multimodal LLMs](https://iclr.cc/virtual/2026/poster/10007977) 〔ICLR〕 | 多模态偏好数据里**不同 token 与视觉的关联度不同**,用跨模态自适应 token 奖励 | 方法 |
| [Clearer Sight, Fewer Lies: Oriented Pickup Preference Optimization for Multimodal Hallucination Mitigation](https://neurips.cc/virtual/2026/poster/155535) 〔NeurIPS〕 | 多模态幻觉:生成偏好对视觉证据校准不足;"定向拾取"偏好优化 | 方法 |
| [DrPO: Drifting Preference Optimization for One-Step Generative Models](https://neurips.cc/virtual/2026/poster/151942) 〔NeurIPS〕 | 一步式文生图模型的在线偏好微调 | 方法 |
| [LucidNFT: LR-Anchored Multi-Reward Preference Optimization for Flow-Based Real-World Super-Resolution](https://neurips.cc/virtual/2026/poster/152037) 〔NeurIPS〕 | 真实超分的多奖励偏好优化,抑制"看似锐利但不忠实"的幻觉 | 应用 |
| [Manifold Drift in Flow Preference Optimization: A Root Cause of Reward Hacking](https://neurips.cc/virtual/2026/poster/154960) 〔NeurIPS〕 | 流匹配偏好优化会让样本**偏离预训练流形**,这是奖励欺骗的一个根因,给出理论 | 理论 + 方法 |
| [From Utterance to Vividity: Training Expressive Subtitle Translation LLM via Adaptive Local Preference Optimization](https://iclr.cc/virtual/2026/poster/10008445) 〔ICLR〕 | 字幕翻译的自适应局部偏好优化 | 应用 |
| [SoLoPO: Unlocking Long-Context Capabilities in LLMs via Short-to-Long Preference Optimization](https://iclr.cc/virtual/2026/poster/10007940) 〔ICLR〕 | 短到长偏好优化,提升**长上下文**能力 | 方法 |
| [ACQueReLlo: Alignment-Aware Constrained Quantization via Reinforcement Learning for Large Language Models](https://neurips.cc/virtual/2026/poster/154964) 〔NeurIPS〕 | 量化会削弱安全对齐;用 RL 做"对齐感知"约束量化 | 工程 |
| [SteadyThought: Mitigating LLM Under-Thinking via Thought-Level Preference Optimization](https://neurips.cc/virtual/2026/poster/148435) 〔NeurIPS〕 | 推理模型"欠思考";在**思维级别**做偏好优化 | 方法 |
| [Rewarding Doubt: A Reinforcement Learning Approach to Calibrated Confidence Expression of Large Language Models](https://iclr.cc/virtual/2026/poster/10006555) 〔ICLR〕 | 用对数评分规则奖励 LLM 表达**校准的置信度** | 方法 |
| [Inverse Reinforcement Learning with Dynamic Reward Scaling for LLM Alignment](https://iclr.cc/virtual/2026/poster/10010158) 〔ICLR〕 | 把 IRL 用于 LLM 对齐,动态缩放奖励处理安全数据不平衡 | 方法 |

## 6. 本篇小结(给初学者的三句话)

1. **偏好优化的"理论冲突点"集中在 BT 假设**:非传递性、噪声、异质性、列表结构。
2. **"奖励欺骗"有多种防线**:KL、梯度正则、数据层面、因果层面。
3. 偏好优化与社会情境的联系:**个性化与多元对齐**,本质是"不同的人有不同的奖励函数"。
