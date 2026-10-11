# 06|奖励从哪来:偏好强化学习、逆强化学习、模仿与奖励模型

> **本篇为什么放在一起读**:RL 的前提是"有奖励",而真实世界里奖励往往要从**偏好**或**示范**里学出来。本篇把"奖励"这件事的三种来源放在一起:偏好(PbRL / RLHF 式)、示范(IRL / 模仿学习)、以及 LLM 时代的**奖励模型、过程奖励与评审模型**。
> **优先论文**(你指定):Online Compatible Reward Identification、CoPT-AIL、TRIRL —— 都偏理论,下面给了更详细的说明(均已读 PDF 的设定、算法与主定理)。其余:ICML 篇为速读,ICLR / NeurIPS 篇为摘要级。
> **建议顺序**:Online Compatible Reward(问"要问多少次")→ CoPT-AIL(问"怎么初始化更好")→ TRIRL(问"怎样更稳地解 IRL")。

## 0. 给初学者的预备

- **偏好 RL(PbRL)**:没有数值奖励,只有"轨迹 A 比 B 好"的比较。
- **IRL**:从专家示范反推奖励;**模仿学习**:直接学专家策略(行为克隆 BC)或用判别器对抗学(GAIL/AIL)。
- **三者共同难点**:奖励**不唯一**(许多奖励对应同一最优策略),所以"识别"本身就是问题。

## 1. 优先论文:奖励识别与逆强化学习(理论)

### Online Compatible Reward Identification from Preference Feedback(ICML 2026)
📄 [官方页面](https://proceedings.mlr.press/v306/drago26a.html) · 本地 PDF:`papers/ICML2026/Online_Compatible_Reward_Identification_from_Preference_Feed.pdf` · 〔ICML 2026〕
**深度:全文读完(正文 §1–§4 的定理陈述;§5 随机偏好与附录证明未逐条核对)。** 作者 Drago、Mussi、Metelli(米兰理工)。

#### 问题:要问人类多少次,才能学到一个"处处准确"的奖励?
- 偏好反馈模型:轨迹 $\tau$ 优于 $\tau'$ 的概率 $P(\tau\succ\tau')=g\big(u(\tau)-u(\tau')\big)$,其中 $u$ 是效用(=轨迹奖励之和)。$g$ 取 **sigmoid** → Bradley–Terry 随机偏好;$g$ 取 **Heaviside 阶跃** → 确定性偏好。
- **兼容奖励集** $\mathcal R(P,g)=\{r:\ P(\tau\succ\tau')=g(u_r(\tau)-u_r(\tau'))\ \forall(\tau,\tau')\}$。BT 下有**平移不变性**($r^*+\beta$ 也兼容);确定性偏好下有**正尺度-平移不变性**($\alpha r^*+\beta,\ \alpha>0$)。
- 目标:用最少的**查询**(给人看的轨迹对)恢复**整个状态-动作空间**上都兼容的奖励,而不是只在 agent 常去的地方准。这给出**迁移性**(sim-to-real)与**可解释性**,是 RLHF 的"只要策略够好"目标做不到的。
- 评价:兼容集之间的 **Hausdorff 距离** $\mathcal H(\mathcal R,\hat{\mathcal R})$,$(\varepsilon,\delta)$-PAC。

#### 一个漂亮的技巧:查询向量
把轨迹 $\tau$ 表示为二值向量 $b(\tau)\in\{0,1\}^{SAH}$(出现过的 状态-动作-阶段 三元组为 1);查询 $(\tau,\tau')$ 对应**三值向量** $q=b(\tau)-b(\tau')\in\{-1,0,1\}^{SAH}$。于是一次偏好答案就是对 $q^\top r$ 符号(或值)的一次线性测量——**奖励识别变成线性代数问题**,"查询集是否充分"变成"查询矩阵能否确定兼容集"。**最小充分查询集叫"基"(basis)。**

#### 主要定理(确定性偏好)
| 结论 | 内容 |
|---|---|
| **Thm 4.1(离线下界)** | 任何 $(0,0)$-PAC 离线算法至少需要 $HSA(SA-1)/2$ 次查询——要**把每个阶段的所有状态-动作对两两比较** |
| **Thm 4.2(在线下界)** | 在线算法至少 $\Omega(HSA\log_2(SA))$——证明靠**归约到排序问题**:在线时可利用传递性,就像比较排序只需 $n\log n$ 次比较 |
| **Thm 4.3(不可近似性)** | 只要少问任何一个关键查询,存在 MDP 使 Hausdorff 距离达到最大值 $R_{\max}$——**少问一次可能导致任意大的错误** |

**直觉**:确定性偏好下,恢复兼容奖励就是对每个阶段的 $SA$ 个状态-动作对做**全序排序**。离线必须全比一遍;在线能"二分插入",省到 $n\log n$。

#### 随机偏好(BT)
引入**查询基及其指数**,刻画查询质量的两个相互竞争的因素:(i) 对恢复 $r^*$ 的**信息量**,(ii) 可靠估计 $r^*$ 所需的**样本数**。给出 $(\varepsilon,\delta)$-PAC 算法,以多项式级的人类查询复杂度找到近最优查询基(摘要,细节未逐条核对)。

#### 为什么需要"马尔可夫奖励"假设
对任意(非马尔可夫)效用,必须至少把每条轨迹都问一遍,而 $|\mathcal T|=(SA)^H$——对 $H$ **指数**。所以只在马尔可夫奖励下谈多项式复杂度。

#### 理论 vs 工程
**纯理论**(下界 + 不可近似性 + PAC 算法),只做表格 MDP,没有深度 RL 实验(附录有小规模数值模拟)。

#### 给初学者的带走
1. **"识别"与"优化"是两个目标**:只想要好策略,可以少问;想要**可迁移的奖励**,必须问够。
2. **把学习问题归约到经典问题**(这里是排序)是理论论文的常用套路,读到"by reduction to ..."就知道下界是怎么来的。
3. 与 `05` 的 DPO 家族对照:那些论文假设偏好数据"已经收集好";本文问**怎么收集**。

### CoPT-AIL:Provably Efficient Policy-Reward Co-Pretraining for Adversarial Imitation Learning(ICML 2026)
📄 [官方页面](https://proceedings.mlr.press/v306/xu26j.html) · 本地 PDF:`papers/ICML2026/Provably_Efficient_Policy_Reward_Co_Pretraining_for_Adversar.pdf` · 〔ICML 2026〕
**深度:全文读完(正文 §1–§4 的算法与定理;实验细节与附录证明未逐条核对)。** 代码 `github.com/LAMDA-RL/CoPT-AIL`(南大 LAMDA 组)。

#### 问题
**对抗模仿学习(AIL,GAIL 一类)**模仿质量高,但要大量在线交互;**行为克隆(BC)**纯离线却有复合误差。自然想法:先用 BC 预训练策略,再用 AIL 微调。可经验上"**几乎没收益**":策略在 AIL 开头会迅速变差。为什么?

#### 诊断:模仿差距 = 策略误差 + 奖励误差(Prop. 1)
AIL 是 $\max_\pi\min_r V^\pi_r-V^{\pi_E}_r$ 的博弈。仅预训练策略(算法 1)时:

$$V^{\pi_E}-V^{\bar\pi}\ \le\ \underbrace{\tfrac1K\Big[(V^{\pi_E}_{r^\star}-V^{\pi^1}_{r^\star})-(V^{\pi_E}_{r^1}-V^{\pi^1}_{r^1})\Big]}_{\text{奖励误差}}+\text{统计误差}+\underbrace{\tfrac{1}{\eta K}\mathbb E\big[\textstyle\sum_h D_{\mathrm{KL}}(\pi^E_h\Vert\pi^1_h)\big]}_{\text{策略误差}}+\tfrac\eta2H^3$$

BC 能把**策略误差**压得很小,但**奖励误差**依赖随机初始化的奖励 $r^1$,是**未被处理的主导瓶颈**——这就解释了经验现象。

#### 关键洞察:不必学回真实奖励,学一个**塑形奖励**就够
- **Prop. 2(塑形不变性)**:对任意势函数 $\{\Phi_h\}$ 与塑形奖励 $\tilde r_h(s,a)=r_h(s,a)-\Phi_h(s)+\mathbb E_{s'}[\Phi_{h+1}(s')]$,任意两策略的**价值差不变**:$V^{\pi'}_r-V^\pi_r=V^{\pi'}_{\tilde r}-V^\pi_{\tilde r}$。(telescoping:两者只差初始状态的势,价值差里抵消。)
- **奖励误差只依赖价值差**,所以学到**任何**一个塑形奖励都足以降低它——**绕开了 IRL 里"奖励不唯一"的老大难**。
- **关键等式**:假设专家是**软最优**的,$\pi^E_h(a|s)=\exp(Q^{\star,\text{soft}}_h(s,a)-V^{\star,\text{soft}}_h(s))$,取对数并代入软 Bellman 方程:

$$\log\pi^E_h(a|s)=r^\star_h(s,a)-V^{\star,\text{soft}}_h(s)+\mathbb E_{s'}\big[V^{\star,\text{soft}}_{h+1}(s')\big]$$

**即 $\log\pi^E$ 恰好是 $r^\star$ 以 $V^{\star,\text{soft}}$ 为势函数的塑形奖励**。

#### 算法 CoPT-AIL(算法 2,只多一行)
1. 行为克隆:$\pi^1\leftarrow\pi_{BC}$;
2. **把 BC 策略的对数似然直接当作初始奖励**:$r^1_h(s,a)=\log\pi^{BC}_h(a|s)$;
3. 之后和标准 AIL 一样(KL 正则策略优化 + 奖励更新)。

**为什么不怕 BC 的复合误差?** 复合误差是**执行**(自回归 rollout 偏离专家状态分布)的问题;而 $\log\pi^{BC}(a|s)$ 是在单个状态-动作对上**局部评估**,在专家分布附近依然准确。

#### 理论结果(Theorem 1)
奖励误差降至 $\tilde O\big(C|S||A|H^2/(KN)\big)$,其中 $C=\max_{s,h}d^{\pi_{BC}}_h(s)/d^{\pi_E}_h(s)$ 是 BC 与专家的状态访问比(浓度系数),$N$ 是专家轨迹数。总模仿差距
$$\tilde O\Big(\tfrac{C|S||A|H^2}{KN}+\sqrt{\tfrac{|S||A|H^2}{N}}+\sqrt{\tfrac{|S||A|H^4}{KN}}\Big)$$
对比无预训练的标准 AIL(Shani et al. 2021):$\tilde O\big(\sqrt{|S||A|H^2/K}+\sqrt{|S||A|H^3/N}+\sqrt{H^4/K}\big)$。**当 $N\gtrsim C\sqrt{|S||A|H^2/K^2}$ 时 CoPT-AIL 的界严格更好**——"首个说明 AIL 预训练收益的理论保证"(作者自述)。

#### 实验
MuJoCo / DeepMind Control 类任务(Walker、Hopper、Cheetah、Finger、Humanoid 等)上,与 IQLearn、PPIL、FILTER、HyPE、OLLIE、BC 等比较,作者报告环境交互效率与最终表现更优(曲线图,我只读到文字陈述,未逐图核对)。

#### 理论 vs 工程
- **理论创新(主要)**:误差分解诊断 + 塑形不变性 + 首个预训练收益界。
- **工程**:几乎零成本——**一次 BC 同时初始化策略和奖励**,代码改动一行。
- **局限**:界依赖**表格 / 有限状态**记号与浓度系数 $C$;"软最优专家"假设;实验是标准连续控制。

#### 给初学者的带走
**把"奖励不唯一"变成优势**:既然只需要价值差对,那么**任意塑形奖励都行**,而专家策略的对数概率恰好是一个。这个"$\log\pi$ 就是奖励"的观察与 `05` 里 DPO 的"隐式奖励 $\beta\log(\pi/\pi_{\text{ref}})$"是同一族想法,值得对照。

### TRIRL:Trust Region Inverse Reinforcement Learning(ICML 2026)
📄 [官方页面](https://proceedings.mlr.press/v306/diwan26a.html) · 本地 PDF:`papers/ICML2026/Trust_Region_Inverse_Reinforcement_Learning_Explicit_Dual_As.pdf` · 〔ICML 2026〕
**深度:全文读完(正文 §1–§4 的理论与算法;实验表格与附录证明未逐条核对)。** 作者来自达姆施塔特工大等;项目页 `anishhdiwan.github.io/trust-region-irl/`。

#### 背景:三种路线的矛盾
- **最大因果熵 IRL(MCE-IRL)**:求奖励 $r$ 与策略 $\pi$ 的鞍点。**对偶上升**每轮:(1)解一次完整的最大熵 RL 得最优策略;(2)按专家与策略的**特征期望差**更新奖励。**单调提升,但每轮都要完整解 RL,代价不可承受**。
- **对抗法(GAIL 及变体)**:把对偶上升改写成判别器 vs 策略的博弈,策略只走**几步**梯度,奖励是**局部**判别器信号。同一个鞍点,但**不稳定、难调、不优化对偶、学不出全局可重用的奖励**。
- **TRIRL 的目标**:既单调改进,又不必每轮解完整 RL。

#### 形式化:分布匹配 + 函数空间奖励更新
把 IRL 写成反向 KL 分布匹配:
$$\max_{\pi}\ \mathbb E_{\rho_\pi}[H(\pi(a|s))]-\beta\,\mathbb E_{\rho_\pi}\Big[\log\frac{\rho_\pi(s,a)}{\rho_E(s,a)}\Big]$$
$\beta\to\infty$ 时与 MCE-IRL 同一个鞍点,最优奖励为 $r^\star=\beta\log(\rho_E/\rho_{\pi^\star})$。Arenz 等人(2016)指出可以不沿参数梯度,而在**函数空间**迭代更新:
$$r^{(i+1)}=\mathcal U^{\rho^{(i)}}_{\epsilon}\,r^{(i)}=r^{(i)}-\epsilon\big(r^{(i)}-\beta\log\tfrac{\rho_E}{\rho_{\pi^{(i)}}}\big)$$
但这需要**当前奖励的全局最优策略** $\pi^{(i)}$。

#### 核心理论(本文的贡献)
用"**信赖域最优策略**"代替"全局最优策略":
$$\pi^{(i+1)}_{\mathrm{tr}}=\arg\max_\pi\ \mathbb E_{\rho_\pi}[H(\pi)]+\mathbb E_{\rho_\pi}\mathbb E_\pi[r^{(i)}]\quad\text{s.t.}\quad\mathbb E_{\rho_\pi}\big[\mathrm{KL}(\pi\Vert\pi^{(i)}_{\mathrm{tr}})\big]\le\zeta$$
- **Lemma 3.1**:信赖域最优最大熵策略,恰是**修改后的奖励** $r_\eta=\frac{1}{1+\eta}r+\frac{\eta}{1+\eta}\log\pi^{(i)}$ 的**全局**最大熵最优策略($\eta$ 为信赖域约束的拉格朗日乘子)。
- **Theorem 3.2**:对更新后的奖励 $\tilde r^{(i+1)}=\mathcal U_\epsilon r^{(i)}$,其信赖域最优策略,**同时也是更小步长** $\epsilon_{\mathrm{tr}}=\epsilon/(1+\eta)\le\epsilon$ **所得奖励** $r^{(i+1)}=\mathcal U_{\epsilon_{\mathrm{tr}}}r^{(i)}$ 的**全局**最大熵最优策略。

**读法**:先用大步长算一个"没修正的奖励",只在上一个策略附近**局部**优化出策略;然后把奖励**修正**到更小的步长——于是**(策略, 修正后奖励)恰是一对合法的最大熵对**。重复这个过程,**保留 MCE-IRL 的单调提升保证,却只做局部策略更新**。

#### 算法(Algorithm 1)
rollout $\pi^{(i)}$,训练判别器近似 $\log(\rho_E/\rho_{\pi^{(i)}})$ → $\tilde r=(1-\epsilon)r+\epsilon\beta D$ → 信赖域策略更新并得到 $\eta$ → $\epsilon_{\mathrm{tr}}=\epsilon/(1+\eta)$ → $r^{(i+1)}=(1-\epsilon_{\mathrm{tr}})r^{(i)}+\epsilon_{\mathrm{tr}}\beta D$。

**实用设计**:(1)判别器**logit** 直接当作对数密度比(GAIL 用 $\log\sigma$,不同);**判别器越准越好**(对抗法判别器太准会失败);(2)奖励依赖全部历史判别器 → 保留**最近 $k$ 个判别器的环形缓冲** + 一个已拟合的参数化奖励;(3)信赖域用**可微投影层(TRPL)**精确执行,取各状态 $\eta$ 的最大值(更保守);基于 PPO。三个变体:TR loss(把 $\eta$ 当惩罚系数)、$\max\eta$、retrospective-$\eta$。(4)支持**仅观测**的模仿($D(s,s')$),并能在**动力学变化后重新优化**学到的奖励。

#### 实验(文字陈述部分)
MuJoCo(HalfCheetah、Ant、Walker、Hopper、Humanoid)与 Unitree G1 走/跑、Go2 运动;基线 GAIL、AIRL、AMP、LSIQ、NEAR、SFM;聚合 IQM 达 **2.4 倍**;消融中 $\max\eta$ 变体在多数任务最好(稍偏离理论但执行严格信赖域)。硬件:基线在 RTX 3090 上训练,LSIQ 因大回放池用 RTX PRO 6000 Blackwell;用 **MuJoCo XLA** 的大规模并行环境(JAX)。

#### 理论 vs 工程
- **理论创新**:"信赖域最优 + 奖励修正 = 某个更小步长的全局最优"这一等价(Thm 3.2),使对偶上升可用**局部更新**实现。
- **工程创新**:判别器缓冲、TRPL 投影层、JAX 大规模并行实现。
- **局限**:理论假设判别器(密度比估计)精确;连续动作下用高斯策略 + 投影层,实际实现与理论略有偏离(作者承认)。

#### 给初学者的带走
**"局部最优 = 某个更小步长下的全局最优"**是一种可复用的证明模式——类似 PPO / TRPO 的信赖域思想,但这里用来**修正奖励**而不只是限制策略。读 `07` 里 BandPO、Trust Region Masking 时,可以对照思考:LLM-RL 的"信赖域"是否也能配一个"奖励修正"?


## 2. 偏好强化学习(PbRL)与多目标

> 共同关心:**怎样用尽量少的人类查询,学到可靠的策略?** 方向包括查询采样、离线数据、保守性、约束与多目标。

| 论文 | 核心想法 | 类型 |
|---|---|---|
| [From Reward-Free Representations to Preferences: Rethinking Offline Preference-Based Reinforcement Learning](https://proceedings.mlr.press/v306/yang26x.html) 〔ICML〕 | 揭示 PbRL 与**无奖励表示学习(前向-后向表示)**的联系:偏好目标可看作对比表示学习;FB-PbRL 两阶段(RFRL 预训练 + SimCLR 式对比微调),**无需学奖励或偏好模型**,在 DMC / Adroit / MetaWorld 上用更少偏好标签取得竞争或更优表现,对噪声稳健 〔速读〕 | 方法 |
| [Reinforcement Learning with Pairwise Preferences in Long-Term Decision Problems](https://proceedings.mlr.press/v306/carr26a.html) 〔ICML〕 | 提出**马尔可夫决策竞赛(Markov decision contest)**:三大结果——(1)**平稳策略在历史依赖策略中仍最优**(Thm 5.6);(2)精确求解**属于 P**,与 MDP 同一复杂度类(Thm 6.1);(3)**对冲策略迭代(HPI)**以 $1/\sqrt{\cdot}$ 速率收敛(Thm 7.5);13 个高维长时程任务上比最相关的 SPPO 更高效 〔速读〕 | 理论 + 方法 |
| [OPRIDE: Efficient Offline Preference-based Reinforcement Learning via In-Dataset Exploration](https://iclr.cc/virtual/2026/poster/10009600) 〔ICLR〕 | 离线 PbRL 查询效率低,原因是探索不足与过度优化学到的奖励;用数据集内探索 | 方法 |
| [Policy Likelihood-based Query Sampling and Critic-Exploited Reset for Efficient Preference-based Reinforcement Learning](https://iclr.cc/virtual/2026/poster/10010299) 〔ICLR〕 | 查询若来自过时经验就不再反映当前策略;按**策略似然**采样并"评论家重置" | 方法 |
| [Preference-based Policy Optimization from Sparse-reward Offline Dataset](https://iclr.cc/virtual/2026/poster/10006399) 〔ICLR〕 | 稀疏奖励离线数据上价值过于乐观;用偏好缓解 | 方法 |
| [Toward Conservative Planning from Human-AI Preferences in Reinforcement Learning](https://iclr.cc/virtual/2026/poster/10006499) 〔ICLR〕 | 数据覆盖差的离线 PbRL:模型化的**保守规划**(MCP) | 理论 + 方法 |
| [Fine-tuning Behavioral Cloning Policies with Preference‑Based Reinforcement Learning](https://iclr.cc/virtual/2026/poster/10007427) 〔ICLR〕 | 先用 BC 得安全初始策略,再用偏好在线微调;给出首个有原则的离线到在线分析(BRIDGE) | 理论 + 方法 |
| [Direct Preference Optimization for Primitive-Enabled Hierarchical RL: A Bilevel Approach](https://iclr.cc/virtual/2026/poster/10006704) 〔ICLR〕 | 把 DPO 用到分层 RL 的子目标生成(DIPPER),缓解下层策略演化的非平稳与不可行子目标 | 方法 |
| [Safe Reinforcement Learning with Preference-based Constraint Inference](https://proceedings.mlr.press/v306/li26ez.html) 〔ICML〕 | PbCRL:从轨迹偏好推断安全约束;在 BT 模型中加**死区(dead zone)**,证明其鼓励重尾代价分布、缓解约束低估;**信噪比(SNR)损失**促使代价有足够变化;两阶段训练减少在线标注;有收敛保证 〔速读〕 | 理论 + 方法 |
| [A Constrained Bi-level Optimization Framework for Constrained Preference-Based Reinforcement Learning](https://neurips.cc/virtual/2026/poster/152347) 〔NeurIPS〕 | 从偏好同时学奖励、代价与策略,写成约束双层优化(CB-PbRL) | 理论 + 方法 |
| [Human-in-the-Loop Policy Optimization for Preference-Based Multi-Objective Reinforcement Learning](https://proceedings.mlr.press/v306/hu26am.html) 〔ICML〕 | PBMORL:交互式学习决策者偏好,并用偏好信息引导多目标 RL 的策略搜索;四足机器人、MuJoCo 与 MMSD 任务验证 〔速读〕 | 方法 |
| [A Single Deep Preference-Conditioned Policy for Learning Pareto Coverage Sets](https://neurips.cc/virtual/2026/poster/150514) 〔NeurIPS〕 | 偏好条件的多目标 RL:证明每个偏好诱导唯一 Pareto 最优解(光滑 Tchebycheff) | 理论 |
| [BECON: Belief-Conditioned Constrained Multi-Objective Reinforcement Learning under Drifting Preferences and Budgets](https://neurips.cc/virtual/2026/poster/153172) 〔NeurIPS〕 | 偏好与安全预算会漂移、含噪;把约束多目标 RL 看成**部分可观测的任务规格问题** | 理论 + 方法 |
| [Imitation Learning for Multi-turn LM Agents via On-policy Expert Corrections](https://proceedings.mlr.press/v306/lauffer26a.html) 〔ICML〕 | 定量分析 SWE Agent 上学生与专家之间的**协变量偏移**;提出**在线专家纠正(OEC)**:学生先 rollout,再由专家接管纠正,借鉴 DAgger;Qwen2.5-Instruct 7B / 32B 在同级别 SWE-bench 上达最优,开源模型与数据;消融显示还需超越单元测试拒绝采样的数据过滤 〔速读〕 | 方法 + 实证 |

## 3. 奖励模型、过程奖励与 LLM 评审(LLM 时代的"奖励")

> 趋势:奖励模型不再只是一个标量分数,而是**会推理的评审**(J1、TIR-Judge、R1-Reward),或**逐步 / 清单式**的过程奖励。

| 论文 | 核心想法 | 类型 |
|---|---|---|
| [Alternating Reinforcement Learning for Rubric-Based Reward Modeling in Non-Verifiable LLM Post-Training](https://proceedings.mlr.press/v306/xu26ai.html) 〔ICML〕 | RUBRIC-ARM:把**评分细则生成视为隐动作**,生成器与评审通过共享正确性目标做**交替 RL**(交替调度有梯度方差分析作依据);9 个奖励建模基准平均 +4.7%,并改善下游策略后训练 〔速读〕 | 方法 |
| [R1-Reward: Training Multimodal Reward Model Through Stable Reinforcement Learning](https://iclr.cc/virtual/2026/poster/10011570) 〔ICLR〕 | 用稳定的 RL 训练**多模态奖励模型**,激活其长程推理 | 方法 |
| [J1: Incentivizing Thinking in LLM-as-a-Judge via Reinforcement Learning](https://iclr.cc/virtual/2026/poster/10008383) 〔ICLR〕 | 用 RL 教 LLM 评审"先想后判",统一可验证与不可验证任务 | 方法 |
| [Incentivizing Agentic Reasoning in LLM Judges via Tool-Integrated Reinforcement Learning](https://iclr.cc/virtual/2026/poster/10011026) 〔ICLR〕 | TIR-Judge:评审可调用 Python 执行器做精确验证 | 方法 |
| [How Neural Reward Models Learn Features for Policy Optimization: A Single-Index Analysis](https://neurips.cc/virtual/2026/poster/154598) 〔NeurIPS〕 | 在单指标高斯模型下分析**奖励模型如何学特征**;KL 正则下奖励被指数化,误差集中在"被奖励倾斜"的区域 | 理论 |
| [PRM-PBE: Process Reward Model for Reinforcement Learning in Programming-by-Example](https://proceedings.mlr.press/v306/fang26j.html) 〔ICML〕 | 示例编程(PBE)的**过程监督 RL**:反馈引导的推理树构造过程级监督数据;**以后继路径成功率**做偏好训练 PRM,并用按失败模式分层的三阶段课程;5 个基准显著提升通过率 〔速读〕 | 应用 + 方法 |
| [BOOSTAPR: Boosting Automated Program Repair via Execution-Grounded Reinforcement Learning with Dual Reward Models](https://proceedings.mlr.press/v306/li26gv.html) 〔ICML〕 | 三阶段:执行验证的 SFT → **双奖励模型**(序列级评估器 $R_{seq}$ + 行级编辑片段信用分配器 $R_{line}$,混合回归-偏好目标)→ PPO;Qwen2.5-Coder-32B 上 SWE-bench Verified 40.7%(+22.9pp);消融显示**主要收益来自序列级奖励**,行级奖励主要改善 OOD 泛化与训练效率 〔速读〕 | 应用 + 方法 |
| [Occupancy Reward Shaping: Improving Credit Assignment for Offline Goal-Conditioned Reinforcement Learning](https://iclr.cc/virtual/2026/poster/10010669) 〔ICLR〕 | 用生成式世界模型里的时间信息做**信用分配**(最优传输) | 方法 |
| [Reinforcement Learning for Long-Horizon Unordered Tasks: From Boolean to Coupled Reward Machines](https://neurips.cc/virtual/2026/poster/149595) 〔NeurIPS〕 | 奖励机(RM)在子任务可任意顺序时信息指数增长;提出数值 / 耦合奖励机 | 理论 + 方法 |
| [CM2: Reinforcement Learning with Checklist Rewards for Multi-Turn and Multi-Step Agentic Tool Use](https://neurips.cc/virtual/2026/poster/152001) 〔NeurIPS〕 | 多轮多步工具调用的 RL:用**清单奖励**替代可验证结果奖励 | 方法 |
| [PAIR: Prefix-Aware Internal Reward Model for Multi-Turn Agent Optimization](https://neurips.cc/virtual/2026/poster/156086) 〔NeurIPS〕 | 多轮 Agent 优化:前缀感知的**内部奖励模型**,避免昂贵的完整 rollout | 方法 |
| [S2T-RLHF: Hierarchical Credit Assignment for Stable Preference-Based RLHF](https://neurips.cc/virtual/2026/poster/149219) 〔NeurIPS〕 | 用层次化信用分配稳定偏好 RLHF 的训练 | 方法 |
| [KV-PRM: Efficient Process Reward Modeling via KV-Cache Transfer for Multi-Agent Test-Time Scaling](https://neurips.cc/virtual/2026/poster/150635) 〔NeurIPS〕 | 用 KV-Cache 迁移降低 PRM 的二次方打分成本(多智能体测试时扩展) | 工程 |
| [Solving the Granularity Mismatch: Hierarchical Preference Learning for Long-Horizon LLM Agents](https://iclr.cc/virtual/2026/poster/10007086) 〔ICLR〕 | 轨迹级 DPO 稳定但分不清信用,步级 DPO 细但噪声大;层次化偏好学习 | 方法 |
| [RLVMR: Reinforcement Learning with Verifiable Meta-Reasoning Rewards for Robust Long-Horizon Agents](https://iclr.cc/virtual/2026/poster/10008499) 〔ICLR〕 | 给长程 Agent 加**可验证的元推理奖励**,避免只奖励最终成功强化错误推理路径 | 方法 |

## 4. 本篇小结

1. **"奖励"是一等公民**:识别性(能不能唯一确定)、样本复杂度(要问几次)、稳定性(怎么优化)是三个独立的理论问题。
2. **LLM 时代的变化**:奖励从"标量分数"走向"**结构化的、可推理的、可验证的**"(清单、细则、过程、工具)。
3. 与前面的文档的联系:*ALSO* 用 LLM 评审给回合奖励(见 02),所以它继承了本篇的所有评审偏差问题。
