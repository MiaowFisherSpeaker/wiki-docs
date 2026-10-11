# 07|LLM 策略优化算法:GRPO / PPO 之后,大家在修什么

> **本篇为什么放在一起读**:这批论文几乎都在同一个"公共基线"上做手术:**PPO / GRPO(用可验证奖励训练推理模型,RLVR)**。读完你会清楚:这个基线有**哪几类系统性毛病**,每类毛病有**哪些思路**在修。
> **阅读深度**:ICML 2026 论文为**速读**(读了本地 PDF 的引言、贡献陈述与结论,描述后标〔速读〕;个别重点论文已全文精读,见各自小节);ICLR / NeurIPS 论文因 PDF 无法脚本下载,为**摘要级**(只依据官方摘要)。
> **与前面文档的关系**:*OmniSapiens / HARPO*(见 `02`)是同一类工作——也是在 GRPO 的**优势尺度**上动手;*ALSO*(见 `02`)则是另一条路线:**完全不改参数**。

## 0. 给初学者:5 个公式够用

**PPO 的裁剪目标**($r_t=\pi_\theta/\pi_{\theta_{\text{old}}}$):

$$L^{\text{clip}}=\mathbb E\Big[\min\big(r_tA_t,\ \operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t\big)\Big]$$

**GRPO 的优势**(同一问题采样一组回答,用组内统计量归一化,不需要价值网络):

$$\hat A_i=\frac{r_i-\mu_G}{\sigma_G+\varepsilon}$$

**KL 正则**:$J=\mathbb E[r]-\beta\,D_{\mathrm{KL}}(\pi_\theta\Vert\pi_{\text{ref}})$,防止策略跑得离参考模型太远。

**基线(baseline)与方差**:用 $A=r-b$ 代替 $r$ 不改变期望梯度,但能降方差;**$b$ 选得好不好决定训练稳不稳**。

**熵坍缩**:策略的熵迅速下降,只会输出少数几种"高分模式",探索死掉——这是下面第二类问题的共同背景。

## 1. 第一类毛病:优化不稳定(裁剪、信赖域、方差、离策略)

> 共同诊断:LLM-RL 的轨迹很长、词表很大,**经典 PPO 的"固定裁剪 + 单一基线"在这里不够用**。

| 论文 | 它指出的问题 → 它的修法 | 类型 |
|---|---|---|
| [Trust Region Masking for Long-Horizon LLM Reinforcement Learning](https://proceedings.mlr.press/v306/li26dp.html) 〔ICML〕 | 推出两个更紧的近似误差界:**Pinsker-Marginal** $O(T^{3/2})$ 与 **Mixed** $O(T)$,相对经典 $O(T^2)$ 分别改进 $O(\sqrt T)$ 与 $O(T)$ 倍;误差取决于序列中**最大 token 级 KL** $D^{tok,\max}_{KL}$,所以是**序列级量**,token 级裁剪/掩码控制不了;故提出 **TRM:整条违反信赖域的序列都掩掉**,给出首个非平凡的长程 LLM-RL 单调改进保证 〔速读〕 | 理论 + 方法 |
| [BandPO: Bridging Trust Regions and Ratio Clipping via Probability-Aware Bounds for LLM Reinforcement Learning](https://proceedings.mlr.press/v306/li26gm.html) 〔ICML〕 | 形式化指出:经典裁剪下**向上更新余量与动作概率成线性**,使低概率-正优势动作梯度几乎为零,抑制探索;**Band 算子**把 $f$-散度信赖域投影成**随概率动态变化的裁剪区间**(凸优化,特定散度有闭式解);数学基准上优于 GRPO、Clip-Higher、DAC,缓解熵坍缩;局限:只在数学推理上评估,未测 72B+ 〔速读〕 | 理论 + 方法 |
| [Beyond Euclidean Clipping: Overcoming Exploration Collapse in LLM RL via Riemannian Isometric Policy Optimization](https://proceedings.mlr.press/v306/cai26c.html) 〔ICML〕 | 指出 PPO-Clip 隐含用**欧氏度量**衡量策略差异,与策略**黎曼流形**几何不一致→病态更新与探索坍缩;RIPO 强制**等距更新**,并诱导统计同方差带来更好的偏差-方差权衡;相对 GRPO 在 AIME24 上最高 +60% 〔速读〕 | 理论 + 方法 |
| [BAPO: Stabilizing Off-Policy Reinforcement Learning for LLMs via Balanced Policy Optimization with Adaptive Clipping](https://iclr.cc/virtual/2026/poster/10007890) 〔ICLR〕 | 离策略训练中**负优势样本主导**,熵快速下降;用平衡 + 自适应裁剪稳定 | 理论 + 方法 |
| [STAPO: Stabilizing Reinforcement Learning for LLMs by Silencing Rare Spurious Tokens](https://neurips.cc/virtual/2026/poster/148439) 〔NeurIPS〕 | 后期性能崩溃源于**极少量的伪 token**;沉默这些稀有伪 token 来稳定 | 方法 |
| [Stabilizing Policy Gradients for Sample-Efficient Reinforcement Learning in LLM Reasoning](https://iclr.cc/virtual/2026/poster/10007976) 〔ICLR〕 | 现有实现为稳定性采取保守超参导致样本浪费;系统研究策略梯度的稳定性 | 方法 + 实证 |
| [Slow-Fast Policy Optimization: Reposition-Before-Update for LLM Reasoning](https://iclr.cc/virtual/2026/poster/10006661) 〔ICLR〕 | GRPO 早期低质量 rollout 带来噪声梯度;"**先重定位再更新**"(SFPO) | 方法 |
| [On the Design of KL-Regularized Policy Gradient Algorithms for LLM Reasoning](https://iclr.cc/virtual/2026/poster/10007225) 〔ICLR〕 | 把 KL 方向(前向/反向)、是否归一化、估计器 $k_1/k_2/k_3$ 这些散落的设计选择**系统化** | 理论 + 实证 |
| [Mode-Controlled Policy Optimization: A Geometry-Aware Recipe for LLM Post-Training](https://neurips.cc/virtual/2026/poster/150744) 〔NeurIPS〕 | KL 正则 RL 隐含地用**反向 KL**拟合"奖励倾斜"目标,固定在模式搜索点;提出可调的几何感知配方(MCPO) | 理论 + 方法 |
| [Listwise Policy Optimization: Group-based RLVR as Target-Projection on the LLM Response Simplex](https://neurips.cc/virtual/2026/poster/153692) 〔NeurIPS〕 | 把基于组的 RLVR 统一看作"**响应单纯形上的目标投影**" | 理论 |
| [Principled Policy Optimization for LLMs via Self-Normalized Importance Sampling](https://neurips.cc/virtual/2026/poster/152868) 〔NeurIPS〕 | GRPO 估计器无理论保证且高方差、GSPO 引入偏差;用自归一化重要性采样给出有原则的方法 | 理论 + 方法 |
| [Historical Relative Policy Optimization for Bootstrapping LLM Reasoning](https://neurips.cc/virtual/2026/poster/154697) 〔NeurIPS〕 | GRPO 仅在当前组内归一化,忽略训练过程中策略表现的演化;引入历史信息 | 方法 |
| [The Optimal Token Baseline: Variance Reduction for Long-Horizon LLM-RL](https://proceedings.mlr.press/v306/li26hy.html) 〔ICML〕 | **最优 token 基线(OTB)**:按每个 token 的"实现能量"(不确定性)加权,做无价值模型的方差缩减,且**无偏**;**Logit-梯度代理**只用前向 logits 估计梯度范数(无需额外反向);group 小到 N=4 仍保持性能,降低总 token 预算 〔速读〕 | 理论 + 方法 |
| [Hista and Numca: Estimate State Value Effectively for Large Language Model Reinforcement Learning](https://proceedings.mlr.press/v306/chen26w.html) 〔ICML〕 | PPO 的评论家会坍缩成粗糙的组均值基线;提出状态价值估计基准 SVEB,以及 Hista / Numca 两种状态价值估计;Hista 可与 DAPO 结合,在 0.5B–14B 多尺度上带来收益,并可无额外 rollout 地用于 Agentic RL 的信用分配 〔速读〕 | 方法 + 评测 |
| [DisPPO: Quantile-Based Distributional Reinforcement Learning for Large Language Models](https://proceedings.mlr.press/v306/zhou26z.html) 〔ICML〕 | 在 PPO 评论家里加入**非参数分位数回归**学习完整回报分布;证明**分布式 λ-Bellman 算子与分位数投影的复合在 Wasserstein 度量下是压缩映射**;Qwen / Llama 上在数学、SQL、多轮工具推理与 RLHF 中均优于 PPO、GRPO、DAPO,额外开销小 〔速读〕 | 理论 + 方法 |
| [Boosting Multi-Domain Reasoning of LLMs via Curvature-Guided Policy Optimization](https://iclr.cc/virtual/2026/poster/10009536) 〔ICLR〕 | 多领域 RL 的奖励曲面复杂、领域间冲突;用**曲率引导**的策略优化 | 方法 |
| [On Predictability of Reinforcement Learning Dynamics for Large Language Models](https://iclr.cc/virtual/2026/poster/10009393) 〔ICLR〕 | 发现 RL 引起的参数更新有 **Rank-1 主导**等规律,RL 训练动态可预测 | 实证 + 理论发现 |
| [Reinforcement Learning with Verifiable Rewards: GRPO’s Loss, Dynamics, and Success Amplification](https://proceedings.mlr.press/v306/mroueh26a.html) 〔ICML〕 | 证明可验证二值奖励下的 GRPO **等价于一个自适应加权的对比损失**(样本来自旧策略);推出多种 GRPO 变体的最优策略**闭式递推**与**成功概率(PoS)动态**:$p_n\to p^*\ge p_{\text{ref}}$(**成功放大**);要点:均值+方差归一化等价于均值归一化配一个自适应 KL 系数 $\beta_{\text{eff}}=\beta\sigma(q)$;**镜像 GRPO**(只对上一轮策略做 KL)保证单调改进并收敛到 PoS=1,而加参考 KL 则没有单调性保证 〔速读〕 | 理论 |
| [GDPO: Group reward-Decoupled Normalization Policy Optimization for Multi-reward RL Optimization](https://proceedings.mlr.press/v306/liu26aa.html) 〔ICML〕 | 分析 GRPO 在多奖励下的**奖励坍缩**:不同奖励组合被求和后塌成**相同优势值**,信号分辨率下降;GDPO 对**每个奖励单独做组内归一化**,再做批级优势归一;在工具调用、数学、代码推理三任务上均优于 GRPO 〔速读〕 | 理论分析 + 方法 |
| [Asymmetric Proximal Policy Optimization: mini-critics boost LLM reasoning](https://iclr.cc/virtual/2026/poster/10011887) 〔ICLR〕 | 主流 RL4LLM 弃用评论家;用**小型"迷你评论家"**重新启用价值函数 | 方法 |

> **这一节的趋势**:从"调超参"走向**理论诊断**——几何(黎曼、$f$-散度、反向 KL)、方差(最优基线)、实现差异(引擎不一致)。**对初学者最值得精读的是 *Trust Region Masking*(诊断一个常被忽视的问题)和 *GRPO's Loss, Dynamics*(把 GRPO 讲清楚)。**

## 2. 第二类毛病:探索坍缩与多样性

> 共同诊断:RLVR 常把 pass@1 提上去,却压低 pass@k 与解的多样性——**"蒸馏现有能力"而不是"发现新策略"**。

| 论文 | 核心想法 | 类型 |
|---|---|---|
| [Risk-Sensitive Reinforcement Learning for Alleviating Exploration Dilemmas in Large Language Models](https://iclr.cc/virtual/2026/poster/10011269) 〔ICLR〕 | 预训练 LLM 初始策略过尖;用**风险敏感**目标缓解探索困境 | 方法 |
| [RiskPO: Risk-based Policy Optimization with Verifiable Reward for LLM Post-Training](https://iclr.cc/virtual/2026/poster/10010102) 〔ICLR〕 | 均值型方法过度强调高概率序列、忽略稀有但有信息的推理路径;用**风险度量**替代均值 | 方法 |
| [Breaking $\textit{Winner-Takes-All}$: Cooperative Policy Optimization Improves Diverse LLM Reasoning](https://neurips.cc/virtual/2026/poster/150356) 〔NeurIPS〕 | rollout 之间"竞争"个体优势,而不是"合作"最大化全局多样性;提出**合作式**策略优化 | 方法 |
| [SDAE: Semantic-Diversity-Aware Exploration for Efficient Reinforcement Learning in Large Language Models](https://neurips.cc/virtual/2026/poster/148235) 〔NeurIPS〕 | 把响应当作原子单元的熵正则不够;做**语义多样性**感知的探索 | 方法 |
| [No Prompt Left Behind: Exploiting Zero-Variance Prompts in LLM Reinforcement Learning via Entropy-Guided Advantage Shaping](https://iclr.cc/virtual/2026/poster/10007755) 〔ICLR〕 | GRPO 忽略所有回答奖励相同的"零方差提示";用熵引导的优势塑形利用它们 | 方法 |
| [Experience Augmented Policy Optimization for LLM Reasoning](https://proceedings.mlr.press/v306/lu26n.html) 〔ICML〕 | EAPO:把 RL 经验表示成**动作级、随策略自适应的先验**(而非固定轨迹),只在**关键决策点**稀疏注入,再用经验感知的优化纳入在线训练;避免固定轨迹的策略错配 〔速读〕 | 方法 |
| [One-Way Policy Optimization for Self-Evolving LLMs](https://proceedings.mlr.press/v306/yang26ay.html) 〔ICML〕 | OWPO:解决 RLVR 的**稳定性-探索冲突**——对称 KL 会造成"**强制反转**":当策略偏离参考而变得更好时也被惩罚;把**优化方向与更新幅度解耦**,构造方向感知信赖域:"加速对齐"纠正劣质偏离、"增益锁定"保留优质偏离;优于 MOPD、DAPO 等基线 〔速读〕 | 方法 |
| [Adaptive Rollout Allocation for Online Reinforcement Learning with Verifiable Rewards](https://iclr.cc/virtual/2026/poster/10008842) 〔ICLR〕 | GRPO 对所有提示分配固定 rollout 数;按**方差预测**分配预算(VIP),提升采样效率 | 方法 |
| [Reinforcement Learning with Verifiable Rewards Implicitly Incentivizes Correct Reasoning in Base LLMs](https://iclr.cc/virtual/2026/poster/10007896) 〔ICLR〕 | 论证 RLVR 会**隐式激励正确推理**,而不只是提高采样效率 | 理论 + 实证 |
| [Scaf-GRPO: Scaffolded Group Relative Policy Optimization for Enhancing LLM Reasoning](https://iclr.cc/virtual/2026/poster/10008612) 〔ICLR〕 | 遇到远超能力的问题会出现"**学习悬崖**":始终零奖励,GRPO 优势塌成 0;用脚手架提示 | 方法 |
| [Curriculum Reinforcement Learning from Easy to Hard Tasks Improves LLM Reasoning](https://iclr.cc/virtual/2026/poster/10010141) 〔ICLR〕 | 难任务上 RLVR 奖励稀疏;**由易到难**课程 | 方法 |
| [Composition-RL: Compose Your Verifiable Prompts for Reinforcement Learning of Large Language Models](https://neurips.cc/virtual/2026/poster/148886) 〔NeurIPS〕 | 通过**组合**可验证提示扩充训练数据,尤其利用通过率为 1 的易题 | 方法 |
| [Scheduling Your LLM Reinforcement Learning with Reasoning Trees](https://iclr.cc/virtual/2026/poster/10009175) 〔ICLR〕 | 把 RLVR 视作对"**推理树**"逐步编辑,并据此做数据调度 | 方法 |
| [Tailoring the Training: Difficulty-Aware Learning Strategy Allocation for Large Language Models](https://proceedings.mlr.press/v306/zhou26bs.html) 〔ICML〕 | DALSA:从困惑度、熵等样本级学习动态出发,用**策略分配器**为每个样本选 SFT / RL / 丢弃,并配**反课程加权**(RL)与 ALS 损失(SFT);全量微调与 PEFT 下都有效 〔速读〕 | 方法 |
| [Breaking the Exploration Bottleneck: Rubric-Scaffolded Reinforcement Learning for Open-Ended LLM Reasoning](https://proceedings.mlr.press/v306/zhou26y.html) 〔ICML〕 | RuscaRL:把**清单式评分细则**既当作探索的**脚手架**(在任务指令中给出逐渐衰减的外部指导以促进内化),又当作**可验证奖励**;Qwen3-30B-A3B 在 HealthBench-500 上与 OpenAI-o3 相当 〔速读〕 | 方法 |
| [GeoAlign: Geometric Rollout Curation for Robust LLM Reinforcement Learning](https://proceedings.mlr.press/v306/zhou26ad.html) 〔ICML〕 | 发现**方向不一致**:少数高奖励 rollout 隐含的更新方向与批次共识严重冲突(角度长尾离群点);GeoAlign 用**仅分离的隐状态**按"潜空间方向共识"给 rollout 打分,对严重离群者做组内替换(保持每个提示的 rollout 数);在对话对齐(连续 RM 奖励)与数学推理(二值验证奖励)及奖励被污染时都更稳 〔速读〕 | 方法 |
| [Beyond Majority Voting: Self-Reflective Test-Time Reinforcement Learning for LLM Reasoning](https://proceedings.mlr.press/v306/wu26az.html) 〔ICML〕 | SR-TTRL:测试时 RL 用多数投票做伪标签,但正确答案常是逻辑少数;改为**分组、总结并批判性评估**不同推理轨迹来产生高保真伪标签;Qwen3-8B 的 AIME24 Pass@1 +26.7% 〔速读〕 | 方法 |
| [CTRL: Continual Test-Time Reinforcement Learning for Large Language Models](https://neurips.cc/virtual/2026/poster/153321) 〔NeurIPS〕 | 测试时 RL 在任务流上会累积错误;持续测试时 RL | 方法 |
| [Reward Is Enough: LLMs Are In-Context Reinforcement Learners](https://iclr.cc/virtual/2026/poster/10007762) 〔ICLR〕 | 发现 LLM 在**推理时**就会表现出 RL(ICRL 提示) | 实证 + 方法 |
| [$\textbf{Re}^{2}$: Unlocking LLM Reasoning via Reinforcement Learning with Re-solving](https://iclr.cc/virtual/2026/poster/10010421) 〔ICLR〕 | RLVR 后仍会"过度思考";引入**重新求解**(Re-solving) | 方法 |
| [Emergent Hierarchical Reasoning in LLMs through Reinforcement Learning](https://iclr.cc/virtual/2026/poster/10009834) 〔ICLR〕 | "顿悟时刻"、长度增长、熵动态其实都是**涌现的推理层级**(高层规划 vs 低层执行)的表现 | 分析 + 方法 |
| [Beyond English-Centric Training: How Reinforcement Learning Improves Cross-Lingual Reasoning in LLMs](https://iclr.cc/virtual/2026/poster/10008038) 〔ICLR〕 | 系统研究 RL 与 SFT 对**跨语言**推理泛化的影响 | 实证 |
| [DeepCompress: A Dual Reward Strategy for Dynamically Exploring and Compressing Reasoning Chains](https://iclr.cc/virtual/2026/poster/10010156) 〔ICLR〕 | 同时提升推理精度与效率,避免"过度思考/思考不足" | 方法 |
| [Generative Adversarial Reasoner: Enhancing LLM Reasoning with Adversarial Reinforcement Learning](https://iclr.cc/virtual/2026/poster/10007942) 〔ICLR〕 | 推理器与基于 LLM 的**判别器**对抗共训 | 方法 |
| [Incentivizing LLM Reasoning via Reinforcement Learning with Functional Monte Carlo Tree Search](https://iclr.cc/virtual/2026/poster/10007699) 〔ICLR〕 | 把可学习的**功能 token** 嵌入词表,结合 MCTS | 方法 |
| [RL of Thoughts: Navigating LLM Reasoning with Inference-time Reinforcement Learning](https://iclr.cc/virtual/2026/poster/10010732) 〔ICLR〕 | 推理时用 RL 引导推理结构(无需改 LLM 参数) | 方法 |
| [Reasoning Under 1 Billion: Memory-Augmented Reinforcement Learning for Large Language Models](https://neurips.cc/virtual/2026/poster/156787) 〔NeurIPS〕 | 小模型上 RL 初始探索弱;用**记忆增强** | 方法 |
| [Achieving Olympia-Level Geometry Large Language Model Agent via Complexity Boosting Reinforcement Learning](https://iclr.cc/virtual/2026/poster/10011798) 〔ICLR〕 | 用"复杂度提升"的 RL 训练奥赛级几何 Agent | 应用 |
| [SRPO: Self-Reflective Policy Optimization for Long-Horizon Reasoning](https://proceedings.mlr.press/v306/liu26ds.html) 〔ICML〕 | 让 LLM 分析自己已完成的轨迹,综合错误成简洁的"反思补丁",用**反思条件下的 rollout** 作为高质量 on-policy 数据把回合级结果变成事后稠密监督;作者强调这不是自蒸馏而是**反思引导探索的 on-policy RL**;局限:需可验证结果信号;人工评估发现 8–10% 的反思有害,过滤后 AIME'24 仅 +0.3 〔速读〕 | 方法 |
| [Detecting Data Contamination from Reinforcement Learning Post-training for Large Language Models](https://iclr.cc/virtual/2026/poster/10010649) 〔ICLR〕 | 检测 **RL 后训练阶段**的数据污染(此前只覆盖预训练/SFT) | 评测 |

> **这一节的趋势**:**把"没学到"当作"没探索到"**——于是出现课程、脚手架、风险目标、多样性奖励、测试时 RL。注意:这些方法多数是**启发式**,理论保证稀少。

## 3. 第三类:效率与系统

| 论文 | 核心想法 | 类型 |
|---|---|---|
| [AReaL-DTA: Dynamic Tree Attention for Efficient Reinforcement Learning of Large Language Models](https://proceedings.mlr.press/v306/zhang26hr.html) 〔ICML〕 | 把 rollout 组织成**前缀树**,用 **DFS 逐条根到叶路径**前向+反向,共享前缀只反传一次、过期后缀即反传即释放(活跃激活量只与最长序列深度成正比);异步 rollout 阶段做**负载均衡的分布式批处理**;在 τ²-bench 上验证 〔速读〕 | 系统工程 |
| [DARTS: Distribution-Aware Active Rollout Trajectory Shaping for Accelerating LLM Reinforcement Learning](https://proceedings.mlr.press/v306/wang26w.html) 〔ICML〕 | 把 rollout 低效归因于**响应长度的长尾分布本身**(组内长尾多为冗长内容),提出**分布感知的主动轨迹采样 + 自适应冗余分配**("主动分布塑形") 〔速读〕 | 系统工程 |
| [HiFloat4 Format for End-To-End Reinforcement Learning Post-Training of Large Language Models](https://neurips.cc/virtual/2026/poster/154752) 〔NeurIPS〕 | 首个端到端 **FP4** RL 后训练;发现主要损失来自 rollout 激活量化 | 系统工程 |
| [Not All Bits Are Equal: Scale-Dependent Memory Optimization Strategies for Reasoning Models](https://iclr.cc/virtual/2026/poster/10008643) 〔ICLR〕 | 4 比特对推理模型并不总是最优:KV 缓存可能主导内存;规模相关的内存优化策略 | 系统实证 |
| [AlphaRouter: Token-level Routing Between SLM and LLM with Reinforcement Learning and Tree Search](https://proceedings.mlr.press/v306/liao26c.html) 〔ICML〕 | 把 SLM-LLM **token 级路由**从"模仿静态 LLM"改为"搜索-更新":**CITS**(协作推理树搜索)构造紧凑协作状态空间,**TAPO**(树优势策略优化)用反事实子树算细粒度优势;精度-成本 Pareto 前沿前移,约 2.0B 激活参数达 7B 水平,某些配置超过单独 LLM 〔速读〕 | 方法 + 系统 |

## 4. 本篇小结(给初学者)

1. **GRPO 是新的"基线",正被一条条拆开**:信赖域(几何)、方差(基线)、探索(风险/多样性)、实现(训练-推理差异)。
2. 研究风格分化:一类是**有数学诊断**的(BandPO、黎曼裁剪、Trust Region Masking、Optimal Token Baseline),一类是**经验配方**(课程、脚手架、筛选)。看论文时先分清自己读的是哪一类。
3. 与社会情境的联系:当奖励来自**多目标、异质样本**时(如 *OmniSapiens*),GRPO 的"组内归一化"会让不同任务的梯度尺度失衡——这是本篇最容易迁移到社会行为建模的洞见。
