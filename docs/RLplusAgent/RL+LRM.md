:::info
**修改日志：**

[agent ai with RL_07.png](https://www.yuque.com/attachments/yuque/0/2025/png/43956489/1765960431003-066eb6ff-ad4e-4890-bb53-b1cdbd85db0f.png)[agent ai with RL_08.png](https://www.yuque.com/attachments/yuque/0/2025/png/43956489/1765960402779-a0aa64ff-3fcb-4ff6-856d-0e6c82ffa53d.png)[agent ai with RL_09.png](https://www.yuque.com/attachments/yuque/0/2025/png/43956489/1765960405573-8070d697-560d-421f-9245-04504aa2cd26.png)[agent ai with RL_10.png](https://www.yuque.com/attachments/yuque/0/2025/png/43956489/1765960443842-a2c9b577-7bdc-49a8-9a03-c506ade9167f.png)[agent ai with RL_11.png](https://www.yuque.com/attachments/yuque/0/2025/png/43956489/1765960402966-35bb0579-604b-4f45-a26d-96f89625171f.png)[agent ai with RL_12.png](https://www.yuque.com/attachments/yuque/0/2025/png/43956489/1765960428957-0d53ebc1-3f69-4e07-b7bf-54673d4a19fd.png)[agent ai with RL_13.png](https://www.yuque.com/attachments/yuque/0/2025/png/43956489/1765960425628-4817a89b-ff10-4a0b-b0fb-27886ae8ae7b.png)[agent ai with RL_14.png](https://www.yuque.com/attachments/yuque/0/2025/png/43956489/1765960439197-572ae63f-70b4-4ff5-b83c-1a0b94b6fe7a.png)[agent ai with RL_15.png](https://www.yuque.com/attachments/yuque/0/2025/png/43956489/1765960467970-3b210da3-f31c-43d2-9df1-3ce16cbd714c.png)[agent ai with RL_16.png](https://www.yuque.com/attachments/yuque/0/2025/png/43956489/1765960459987-bb152d60-2d5e-43ac-9682-566355922db7.png)[agent ai with RL_17.png](https://www.yuque.com/attachments/yuque/0/2025/png/43956489/1765960463856-d98a5731-3643-4dd7-9fad-51278c160ad8.png)[agent ai with RL_18.png](https://www.yuque.com/attachments/yuque/0/2025/png/43956489/1765960466748-2859c992-dc07-4010-83cd-14be4c9604e5.png)[agent ai with RL_19.png](https://www.yuque.com/attachments/yuque/0/2025/png/43956489/1765960466692-7781d7f8-f4e3-472a-99ee-c306f1c242fd.png)

<font style="color:#D22D8D;">2025.12.17前 初版从Notion迁移</font>

<font style="color:#D22D8D;">2025.12.17 16:30 增加预备知识Slides第一页；预备知识部分直接放附件了。</font>

:::

---

# RL+LRM总结
目前在Notion软件也有（其实就是把语雀当成国内备份了哈哈哈）

# 0v0
**参考文献**：Zhang K, Zuo Y, He B, et al. A survey of reinforcement learning for large reasoning models[J]. arXiv preprint arXiv:2509.08827, 2025. 

截至12.12 38篇引用。

[https://arxiv.org/abs/2509.08827](https://arxiv.org/abs/2509.08827)

**作者**：

![](https://cdn.nlark.com/yuque/0/2025/png/43956489/1765893016392-7f84aae4-d417-4cb3-a451-5c1621b089de.png)

⭐**Github** [https://github.com/TsinghuaC3I/Awesome-RL-for-LRMs](https://github.com/TsinghuaC3I/Awesome-RL-for-LRMs)

其中有关于整篇文章的slides，以及推理模型相关的RL论文列表。

本文章的笔墨着重于介绍大语言模型的RLVR而不是RLHF或者DPO。

---

# 一、预备知识
[预备知识](https://www.notion.so/2c6595a4a6f280e98ea6e8efc05f76cb?pvs=21)：（Notion原链接）

首次加载需要很久，因为是8倍分辨率导出的。为了阅读体验，直接隐藏了。可以在开头附件下载。

---

# 二、Datasets
## 1️⃣Static datasets for RL training of LLMs in **Agent domains**
+ 主要集中于两类能力：  1️⃣ search-as-action  2️⃣ tool use
+ signal: verifiable process signals → process rewards and offline evaluation  
信号：可验证的过程信号      以实现 过程奖励和离线评估
    1. search/browse traces   
  搜索/浏览轨迹（痕迹）
    2. evidence URLs  
  证据URLs
    3. tool-exectution logs  
  工具执行日志

![](RL+LRM%E6%80%BB%E7%BB%93/image%201.png)![Static datasets for RL training of LLMs, including Math, Code, STEM, and Agent domains. For data acquisition methods, “Distil” and “Anno” indicate distillation and annotation, respectively. “Merge” indicates the integration of existing datasets, including difficulty and quality filtering.](https://cdn.nlark.com/yuque/0/2025/png/43956489/1765893281353-1e74c9ab-fafd-47b4-8e11-917dfca6ad69.png)



### 介绍
1. **<font style="color:#117CEE;">Search-R1</font>**** **Search-R1是一个用于训练**多轮搜索交互推理能力**的数据集，通过将搜索引擎建模为环境的一部分，让语言模型在推理过程中自主生成搜索查询并与检索系统交互。该数据集的核心是**交错式推理-搜索行为**的**训练范式**，模型需要在多轮对话中逐步完善推理过程，同时动态调用搜索工具获取外部信息。 **适用场景**：
    - 开放域问答系统：需要结合内部知识和外部检索的复杂问答任务
    - 多跳推理任务：如HotpotQA等需要多步推理和事实验证的问答
    - 研究搜索与推理的交互机制：探索模型如何平衡内部计算和外部查询
    - 强化学习训练：研究多轮交互中的奖励设计和策略优化
2. **<font style="color:#117CEE;">ToRL</font>** ToRL是一个**工具集成强化学习**数据集，专注于让模型学习何时以及如何调用计算工具。该数据集将工具使用问题建模为强化学习任务，模型**需要在推理过程中生成代码并通过代码解释器执行**，根据执行反馈动态调整推理路径。 **适用场景**：
    - 代码生成与执行：让模型学会使用编程工具解决数学、逻辑问题
    - 计算密集型任务：需要调用外部计算工具（如数学计算器、代码解释器）的推理任务
    - 工具链编排：学习按合理顺序调用多个工具完成复杂任务
    - 强化学习中的工具使用策略学习：研究模型如何自主探索最优工具调用策略
3. **<font style="color:#117CEE;">ToolRL</font>** ToolRL是一个**研究工具选择奖励设计**的数据集，专注于工具推理中的**泛化问题**。该数据集将工具调用问题建模为Tool-Integrated Reasoning（TIR）任务范式，要求模型以**合理顺序和逻辑调用多个工具，并基于中间结果灵活调整思维路径**。 **适用场景**：
    - 工具选择策略研究：探索不同奖励设计对工具调用策略的影响
    - 多工具协同任务：需要按顺序调用多个工具完成复杂推理
    - 强化学习奖励设计：研究如何设计奖励函数以鼓励合理的工具使用
    - 泛化能力评估：测试模型在未见过的工具组合或任务类型上的表现
4. **<font style="color:#117CEE;">ZeroSearch</font>**** **ZeroSearch是一个**离线信息搜索**数据集，无需与真实搜索引擎交互即可激励LLM的搜索能力。该框架通过轻量级监督微调将LLM转换为检索模块，采用基于课程的推出策略逐步降低生成文档质量。 **适用场景**：
    - 离线搜索任务：在无法访问真实搜索引擎的环境下训练搜索能力
    - 低资源训练：大幅降低训练成本，避免真实API调用费用
    - 搜索能力模拟：研究如何让模型学会"想象"搜索过程
    - 检索增强生成：在检索增强的生成任务中训练模型的信息搜索能力
5. **<font style="color:#117CEE;">WebShaper</font>**<font style="color:#117CEE;"> </font>WebShaper是一个**信息搜索数据合成**框架，通过集合论形式化信息搜索任务，利用"知识投影"概念精确控制推理结构。该框架采用逐层扩展策略，通过Expander智能体基于形式化框架利用检索与验证工具将问题逐步复杂化。 **适用场景**：
    - 数据合成与增强：自动生成高质量的信息搜索训练数据
    - 形式化推理研究：基于集合论的形式化框架研究推理结构
    - 多步信息搜索：需要多轮检索和验证的复杂搜索任务
    - 基准测试构建：为信息搜索任务构建标准化的评估基准
6. **<font style="color:#117CEE;">MicroThinker</font>** MicroThinker提供了**多步智能体的完整执行轨迹**和丰富的工具使用日志，支持可扩展的策略学习、奖励建模和基于过程的强化学习。该数据集**涵盖了规划、检索、工具编排、证据验证和答案生成等关键能力**。 **适用场景**：
    - 多步推理轨迹分析：研究智能体在多轮交互中的决策过程
    - 工具使用日志挖掘：分析模型如何选择和调用工具
    - 策略学习研究：基于完整轨迹进行模仿学习或强化学习
    - 过程监督训练：利用中间步骤的监督信号进行训练
    - 智能体行为分析：研究智能体在复杂任务中的认知过程
7. **<font style="color:#117CEE;">ASeacher</font>** ASearcher是一个用于**长时程搜索智能体**的数据集，包含问题/答案字段和来源注释。该数据集采用完全异步强化学习训练框架，支持智能体进行超长时程的工具调用（超过40轮）和大量输出（超过150k token）。 **适用场景**：
    - 长时程搜索任务：需要多轮交互和大量工具调用的复杂搜索
    - 异步强化学习：研究异步训练框架在长时程任务中的表现
    - 大规模输出生成：需要生成大量内容的复杂任务
    - 工具链编排：需要按顺序调用大量工具完成复杂工作流
    - 智能体可扩展性研究：测试智能体在超长时程任务中的稳定性和性能

## 2️⃣Dynamic RL Environments for RL Training of LLMs

![](RL+LRM%E6%80%BB%E7%BB%93/image%202.png)![](https://cdn.nlark.com/yuque/0/2025/png/43956489/1765893501972-bedca0c6-f1df-4cf3-9905-8d6df606cd9d.png)
<font style="color:#000000;">提取Agent和Planning, multimodal相关的有如下几个：</font>

+ <font style="color:#000000;">In Code-based Enviroment:</font>
1. <font style="color:#000000;">agents(tool-using) </font>**<font style="color:#000000;">ReCall</font>**
2. <font style="color:#000000;">agents(computer-using) AgentCPM-GUI, ZeroGUI</font>
+ <font style="color:#000000;">In Game-based Environment:</font>
1. <font style="color:#000000;">agent planning field: </font>**<font style="color:#000000;">ALFWorld</font>**<font style="color:#000000;">, </font>**<font style="color:#000000;">ScienceWorld</font>**
    1. <font style="color:#000000;">multimodal reasoning capabilities: Code2Logic</font>
+ <font style="color:#000000;">In Model-based Environment:</font>
1. <font style="color:#000000;">Model-to-model interation Agent in front-/back-end programming: Sweet-RL</font>
+ <font style="color:#000000;">Ensemble-based Environment:  No</font>



### 介绍
1. **<font style="color:#117CEE;">Recall </font>** **ReCall** 是一个专注于**代码生成和工具集成**的强化学习环境。它的核心目标是训练LLM智能体（Agent）如何有效地与外部工具（如Python解释器）进行交互，以解决编程问题。
    - **核心机制**：ReCall利用先进的LLM自动构建一个基于Python的工具交互环境。它能够自主合成名为“SynTool”的训练数据，专门用于RL训练。这使得模型能够学习在何时、以及如何调用正确的工具来辅助代码编写、调试和验证。
    - **数据来源**：其主要数据是通过**模型合成（Model-based Synthesis, MS）** 生成的，这意味着环境中的任务和交互轨迹是由LLM自动创建的，具有很高的可扩展性。
    - **交互性**：支持**交互式（Interactive）** 训练，模型在环境中采取行动（如调用工具）后会获得反馈，从而学习正确的工具使用策略。
    - **应用场景**：主要用于**代码智能体（Code Agent）** 的训练，是帮助LLM从单纯的代码生成者进阶为能够使用工具解决问题的“程序员”的关键环境。
2. **<font style="color:#117CEE;">ALFWorld</font>** **ALFWorld** 是一个将文本指令与具身智能（Embodied AI）环境结合的基准测试和训练平台。它专注于训练智能体在模拟的文本化家庭环境中执行任务。
    - **核心机制**：ALFWorld提供了六个可交互的模拟家庭环境。智能体接收用自然语言描述的任务（例如“把冰箱里的苹果拿到茶几上”），然后需要在环境中通过一系列具体的动作（如`goto`, `take`, `put`）来完成任务。它将文本推理与在模拟空间中的行动规划结合起来。
    - **数据来源**：其环境主要是**基于规则合成（Rule-based Synthesis, RS）** 的，构建了一个遵循物理规则的文本冒险游戏式世界。
    - **交互性**：支持**交互式**训练，是研究**具身智能** 和**视觉语言导航（VLN）** 的经典环境之一。智能体通过与环境互动获得成功或失败的奖励信号。
    - **应用场景**：主要用于训练遵循指令的**交互式智能体**，是连接语言理解与物理行动的重要桥梁。
3. **<font style="color:#117CEE;">ScienceWorld</font>** **ScienceWorld** 是一个旨在评估和提升智能体**科学推理能力** 的交互式文本环境。它模拟了一个小学生的科学实验场景。
    - **核心机制**：该环境包含了30个不同类型的科学任务，涵盖生物、物理、化学等基础科学主题。智能体需要像科学家一样进行思考：提出假设、设计实验、操作实验设备、观察结果并得出结论。它测试的是模型对科学概念和因果关系的深层理解，而不仅仅是知识检索。
    - **数据来源**：环境同样是通过**规则合成（RS）** 构建的，确保了科学实验逻辑的正确性。
    - **交互性**：支持**交互式**训练，智能体的每一步操作都会导致环境状态的变化，并获得相应的反馈。这对于训练模型进行多步、长视野的科学推理至关重要。
    - **应用场景**：专门用于开发和评估**科学推理智能体**。它是检验模型是否真正“理解”科学原理，而非仅仅记忆知识的重要试金石。

---

# 三、开源RL框架
![](https://cdn.nlark.com/yuque/0/2025/png/43956489/1765893561738-d8fb00b1-26ec-4d9a-9412-501cbf5d942b.png)

## 1️⃣Primary Development
+ **<font style="color:#117CEE;">Verl </font>** **Verl** 是一个功能全面、算法丰富的开源RL框架，旨在为LLM的后训练提供高度灵活和可扩展的解决方案。它的设计哲学是支持广泛的训练范式和算法，以满足从学术研究到大规模生产应用的不同需求。
    - **核心架构与设计**：
        * **异步训练范式**：Verl的核心是其**HybridFlow**控制器，它支持高效的异步训练。这种设计将数据生成（Rollout）和模型训练（Learning）解耦，允许Rollout Worker（负责采样生成）和Trainer（负责参数更新）并行运作，极大提升了硬件利用率和训练效率。这对于需要生成长思维链（CoT）的RL训练至关重要。
        * **模块化设计**：Verl的各个组件（如环境、奖励函数、模型、算法）高度模块化，用户可以通过配置文件灵活组合，快速实验不同的训练配方。
    - **支持的算法**：Verl支持目前主流和前沿的RL算法，使其成为一个理想的算法试验平台。主要包括：
        * **PPO/GRPO系列**：这是当前RL训练LLM最主流的算法。Verl不仅支持标准的PPO和GRPO，还实现了其多种变体，如**GSPO**（一种序列级优化算法）、**ReMax**（一种高效的策略梯度算法）、**RLOO**（留一法基线）等。
        * **多轮与工具调用**：Verl对**Agentic RL**（智能体强化学习）有很好的支持，可以训练模型进行多轮对话、规划和工具调用（Tool Use）。
        * **离线与离线策略算法**：也支持**DPO**及其变体，方便与偏好学习技术结合。
    - **训练后端与性能**：
        * **分布式训练**：Verl集成了**FSDP**和**Megatron-LM**作为主要的训练后端，支持高效的大模型训练。用户可以根据模型规模和硬件条件选择最合适的并行策略。
        * **推理优化**：为加速耗时的采样过程，Verl紧密集成 **vLLM** 和 **SGLang** 这两个高性能推理引擎，确保在生成阶段也能达到高吞吐量。
    - **应用场景**：Verl非常适合需要进行大规模、多算法比较的研究工作，以及构建复杂的、需要多步推理和工具调用的智能体应用。其灵活性使得从数学推理、代码生成到搜索智能体等多种任务都能在其上有效开展。
+ **<font style="color:#117CEE;">ROLL</font>**  **ROLL** 的定位是一个面向**大规模生产级别**的RL训练系统。它特别强调在超大规模集群（如数千GPU）上的可扩展性、稳定性和易用性。如果说Verl是“算法实验室”，那么ROLL更像是“工业化生产线”。
    - **核心架构与设计**：
        * **Ray-Based多角色架构**：ROLL基于**Ray**这一成熟的分布式计算框架构建。它采用了清晰的多角色设计，包括**Rollout Worker**, **Trainer**, **Reward Worker**等，这些角色可以独立伸缩，从而实现近乎线性的扩展能力。这种设计使其能够轻松部署在大型Kubernetes集群上。
        * **生产级特性**：ROLL内置了检查点（Checkpointing）、容错（Fault Tolerance）、监控（Monitoring）等生产环境必需的特性，保证了长时间训练的稳定性和可恢复性。
    - **支持的算法与特性**：
        * **算法支持**：ROLL同样支持GRPO、PPO、REINFORCE++等核心算法。其重点不在于追求算法数量，而在于确保这些算法在大规模分布式环境下的稳定、高效运行。
        * **异步训练**：原生支持异步训练，这是其实现高扩展性的关键。
        * **Agentic RL支持**：同样支持训练多轮智能体。
        * **训练后端**：与Verl类似，ROLL主要采用**Megatron-Core**作为训练后端，并计划支持FSDP2，旨在为超大规模模型（如千亿参数）提供最优的并行训练方案。
        * **推理服务**：集成**SGLang**和vLLM，优化采样效率。
    - **应用场景**：ROLL非常适合需要训练极大模型（如百亿、千亿参数）的工业界团队和大型研究机构。当训练任务需要消耗数万甚至数十万GPU小时时，ROLL在稳定性和扩展性方面的优势会非常明显。
+ **<font style="color:#117CEE;">slime</font>**  **slime** 是一个相对较新、设计理念独特的框架。它的核心特点是**深度拥抱SGLang**，将自己定位为一个“SGLang-native”的RL训练框架，旨在实现极致的训练和采样效率。
    - **核心架构与设计**：
        * **SGLang-Native**：与Verl和ROLL将SGLang/vLLM作为可选的推理后端不同，slime与SGLang深度绑定。它充分利用SGLang在处理复杂生成模板（如思维链、工具调用）方面的性能优势，将整个RL训练流程构建在SGLang之上。
        * **专注效率**：slime的设计目标非常明确：通过紧密的栈集成来减少开销，实现可能更高的训练效率（Tokens/Second/GPU）。它可能不像Verl那样追求算法的全面性，而是专注于优化核心训练路径。
    - **支持的算法与特性**：
        * **算法**：目前主要支持GRPO等核心算法，并提供了多轮+工具调用（“Search-R1 lite”）的示例。其算法库可能不如Verl丰富，但针对支持的算法进行了深度优化。
        * **训练后端**：slime使用**Megatron-LM**作为训练后端，并通过Ray进行集群启动和管理。
        * **轻量级**：框架本身可能更轻量，旨在减少抽象层带来的开销。
    - **应用场景**：slime非常适合对训练效率有极致要求的场景，特别是当任务流程能很好地被SGLang的模板功能描述时。对于希望快速迭代、并且主要使用标准GRPO算法进行研究的团队来说，slime是一个非常有吸引力的选择。

### **综合对比与总结**
为了更清晰地展示三者的区别，下表进行了横向对比：

| **特性** | **Verl** | **ROLL** | **slime** |
| --- | --- | --- | --- |
| **设计理念** | **全面灵活**的算法试验平台 | **大规模生产级**训练系统 | **轻量高效**的SGLang原生框架 |
| **核心优势** | 算法支持广泛，模块化程度高，灵活性强 | 超大规模扩展性，生产级稳定性 | 与SGLang深度集成，追求极致效率 |
| **分布式架构** | HybridFlow控制器 | Ray-Based多角色 | Ray + Megatron-LM |
| **主要训练后端** | FSDP, Megatron-LM | Megatron-Core(主) | Megatron-LM |
| **主要推理后端** | vLLM, SGLang | SGLang, vLLM | **SGLang（深度集成）** |
| **算法覆盖面** | **非常广泛**（PPO, GRPO, GSPO, ReMax, DPO等） | 核心算法（GRPO, PPO等），强调稳定性 | 核心算法（如GRPO），专注优化 |
| **适用场景** | 学术研究，新算法探索，多模态/智能体应用 | 工业界千亿模型训练，超大规模集群 | 对效率敏感的研究，标准RL训练流程 |


## 2️⃣Secondary Development
是Primary development frameworks的拓展，可以支持更多的下游任务（应用）。其中框架集中在**agentic RL, multimodal RL, and multi-agent RL**

**Agentic RL:** 主要基于**veRL**框架，核心特征是异步生成和训练。

**Multimodal RL**：核心挑战在于数据处理和损失函数的设计。相关前沿研究可以关注RL surveys focused on vison models.

**Multi-agent RL**: 该部分框架关注为异步rollout和训练实现动态工作流。（rollout是轨迹采样）

---

# 四、应用Applicaitons
+ 一览图  ![](RL+LRM%E6%80%BB%E7%BB%93/image%204.png)

![](https://cdn.nlark.com/yuque/0/2025/png/43956489/1765893690208-ee2417c2-3db8-4043-aa93-51a110d6bd13.png)

由于不太关注Coding Tasks, Robotics Tasks, and Medical Tasks.下面着重介绍Agentic Tasks, Multimodal Tasks, Multi-Agent Systems.、

## 1️⃣Coding Tasks
CodingTasks中值得一提的是，CodeGeneration， 因为Agent在调用工具时可能  
1）自己生成调用代码 * （more flexible and scalable)  
2）提前写好，模型判断使用参数填入即可 ( one certain domain, that is **Domain-Sepecific Code**)

## 2️⃣Agentic Tasks
可以被分类为

+ **Coding Agent**
+ **Simple Search Agent**
+ **Browser-use Agent**
+ **DeepResearch**
+ **GUI&Computer-use Agent**
+ **others**

### 1. Coding Agent
1. code agents

> These developments suggest that combining RL with agentic coding is driving a shift from “single-step generation” toward “autonomous iteration.”  
 将RL与代理编码相结合正在推动从“单步生成”向“自主迭代”的转变。
>

 流程：RL 融合进code agents（各种方法）, 在 **SWE-Bench**上评估。

2. Tool-Integrated Reasoning, TIR **代表性工作**：**ARPO**，**AutoTiR**, **CoRT** and **ToRL** 
**其采用的策略如下：**
    - **post-train** models with SFT or RL(GRPO或其变体)
    - **结构化输出**触发工具执行。<code>...</code>
    - 结果**反馈**给推理循环

 其他**前沿工作**：

    - TARL： 多模态，沙箱环境， 融合任务训练循环
    - Tool-R1 ： 多步工具使用， 基于结果的奖励， 动态采样
3. Automated ML Programming   
让agent**自己数据分析、模型建模、和优化**   
已有工作： **MLE-STAR**, **ML-Agent**   
但是其性能表现更好的前提条件有点多，得具体再看论文才可以总结。

> Yang et al. [2025e] show that agents powered by relatively smaller models trained with RL can outperform agents using much larger but static models, especially when enhanced with durationaware updates and environment instrumentation to provide finer-grained reward signals.
>

### 2. Simple Search Agent
```plain
结构化prompt→ ｜ 多轮生成  ｜<- 数据来源：网上的搜索引擎、静态知识库
```

### 3. Browser-use Agent
also use web-browsing

### 4. DeepResearch Agent
**目标**是从不同online来源搜集信息来完成现实世界的问题，如汇报生成。

**工作流程**：

1. 任务规划（Task Planning）与拆解
2. 多轮、自适应的信息搜集  
Agent不会只进行一次搜索，而是会进行多轮交互式的信息搜集。
3. 信息的提取验证与合成
4. 自我反思与循环
5. 报告生成

其在RL相关的训练范式通常是人类反馈的，即RLHF

### 5. GUI& Computer-use Agent
### 6. Other tasks
+ 购物
+ 自动驾驶
+ 通用目的的agentic models，重在掌握使用工具解决现实世界任务的能力。

---

## 3️⃣Multimodal Tasks
在多模态任务中，RL加强模型能力，旨在**解决如下问题**：

+ 有限的数据设置
+ 长视频推理
+ 数值或者属性敏感的跨模态生成

而能力主要体现在**多模态理解**以及**生成**两个方面，具体技术细节不再列举。

### 1. 多模态理解
1. Image：   
1）推理和回答**不一致**：模型产生的思维无法映射到最终答案。   
2）**长链**探索崩溃：随着响应长度的增加，模型变得脆弱，容易产生幻觉。   
3）对数据质量的敏感性：RL样本选择至关重要，因为低质量的训练数据可能导致次优性能甚至负优化。
2. Video、3D   
比如spatial-temporal reasoning（前者），real-world dynamics（后者）



### 2. 多模态生成
1. Image
    - Diffusion Models将**去噪过程（steps）视作思维链轨迹**
    - GPRO在扩散模型中有所冲突，集中在和常微分方程采样之间的**内在冲突**  
 具体指的是，GRPO是随机，但是常微分方程的去噪过程是确定性的，限制了rollout采样。对此，有相关的研究。
2. Video 挑战集中**在时间相干性和物理现实**



然而在语音方面，存在两大方向，语音作为**输入模态**或者**输出模态**。其面临的独特挑战如下：

+ 序列长，信息密度高： 信息在时间轴上高度分散，对模型的长程以来建模能力要求高
+ 时序对齐： agent的actions如打断、回应等必须和语音流精准对齐，延迟或者提前都会导致交互失败。与此同时，在RL技术中，信用分配问题再次尤为突出。  
这里的信用分配问题指的是：
+ 奖励设计复杂：如何去量化“生成语音的自然度”、“交互的流畅性”……
+ 数据问题：稀缺，同时包含语音、视觉、动作轨迹的多模态交互数据构建成本也很高

方向： ~~具身智能~~ ；交互式对话agent(文本回复+语音；对话策略使用RL优化；是否能端到端)；…

但是带有思维链的推理本身很耗时，不适合实时对话。

## 4️⃣Multi-Agent Systems (MAS)
多智能体系统重在**提高合作、推理以及信用分配。**

> 信用分配，可以简单理解为将奖励分配到每一步，实际上就是奖励分配。  
常见的有  
1. **贪婪分配**，只奖励最关键的；  
2. **基于折扣的奖励**，由参数控制目光短浅还是长远；  
3. **优势函数**，目前的主流。  
当前存在的**问题**是我们不能等结果出来才给奖励，而过程没有奖励，这也就是**稀疏奖励**。而理想情况就是**信用分配**，结果是每一步共同努力的结果。
>

既然是一篇RL+LRM的综述，下面就会从传统MARL到LLM for MARL，再讲RL在其中的作用。

### 1. 传统MARL
**核心挑战**就是**信用分配的复杂性**、**环境的非平稳性**以及**agents之间的交流与合作**。  
此外，随着agents数目的增加，采样效率和可扩展性也面临挑战。

### 2. LLM for MARL
利用LLM的自然语言理解和生成能力提供有效的共享信息。

### 3. RL for LLM-based MAS
利用RL实现多个agents之间的高效协作和策略优化。框架有**LLaMAC**和**CTRL**

其中CTRL就使用了LLM**“自我批评”**去合成数据，再使用RL迭代优化模型输出，解决人工标注问题。

## 5️⃣Robotics Tasks
+ RL addresses data scarcity and generalization challenges in robotics by adapting LLM-style approaches to **Vision-Language-Action (VLA) models**.  （VLA = VLM + action modules）
+ Allowing VLAs to learn from environment interaction and simple rewards, recent RL methods (e.g., GRPO, RLOO, PPO) achieve superior performance and novel behaviors with minimal supervision.

主要关注**三个领域**： **机器人控制**、**视觉语言导航（VLN）**、**机器操作任务**

VLA RL研究目前集中在基于模拟的RL（simulation-based）。

VLA需要解决的**三个问题**：  
1）多轮环境交互以产生完整轨迹  
2）面临的是连续动作空间  
3）破除对人造过程奖励的依赖

## 6️⃣Medical Tasks
分为**可验证任务**和**不可验证任务**，其中

+ **可验证任务**：要求奖励设计稳定；一般使用基于规则的SFT+RL的范式
+ **不可验证任务**： 奖励的定义比较困难； 采用DPO、rubrics（某种意义的规则）、课程RL（curriculum RL）、离线RL。其扩展性是未来方向之一

目前的研究一样分为理解和生成

### 1. Medical Undestanding
两阶段的pipeline：SFT+RL，直接基于正确性的信号优化（确定性的，deterministric rewards)

除了textual QA, 基于规则的奖励的研究扩展到视觉和多模态任务

### 2. Medical Generation
生成任务包括放射学报告生成、多轮临床对话、治疗计划、诊断叙述等

# 五、未来方向
持续强化学习for LLMs、基于记忆的强化学习for LLMs、基于模型的、教LRMs有效推理的、教LRMs隐空间推理的、LLMs预训练的、扩散模型的、科学发现的、架构-算法一起设计的。

### **1️⃣** **持续强化学习（Continual RL, CRL）for LRMs**
+ **核心问题**：当前的RL训练通常针对一个固定任务或数据集进行，训练完成后模型参数就冻结了。这无法让模型在部署后持续从新交互中学习。
+ **未来方向**：研究**持续学习**框架,使LLM能够像人类一样，在不遗忘旧技能的情况下，持续学习新任务和新知识。
+ **关键挑战与目标**：
    - **克服灾难性遗忘**：这是核心挑战。需要将传统CRL技术（如**经验回放（Experience Replay）**、**策略复用（Policy Reuse）**、**奖励塑形（Reward Shaping）**）有效地应用于LLM的规模。
    - **实现终身学习**：目标是开发出能够进行**终身学习（Lifelong Learning）**的LRM，使其在动态变化的环境中保持适应性和进化能力。

### **2️⃣** **基于记忆的RL（Memory-based RL）for LLMs**
+ **核心问题**：当前智能体的记忆机制多是任务特定的，缺乏泛化性。记忆通常被视为一个临时缓冲区，而非可跨任务重用和迁移的经验知识库。
+ **未来方向**：将智能体的记忆从**任务特定缓冲区**转变为**结构化、可重用、可迁移的经验库**，使其成为支持更广泛适应性和终身学习的基础。
+ **关键挑战与目标**：
    - **记忆的泛化与迁移**：研究如何让智能体通过RL学会管理和操作记忆，实现经验的组合与跨任务泛化。
    - **迈向"经验时代"**：最终目标是进入一个"经验时代"，智能体集体互动的经验轨迹成为更广泛智能的基础。

### **3️⃣** **基于模型的RL（Model-based RL）for LLMs**
+ **核心问题**：RL的核心挑战之一是从环境中获取可扩展且稳健的奖励信号。对于语言智能体，构建能准确捕捉环境状态并生成可靠奖励的世界模型至关重要。
+ **未来方向**：深入研究如何将**世界模型（World Models）**与LLM智能体的RL训练无缝集成。利用LLM本身作为世界模型，或结合视频预训练等技术的生成式世界模型，是一个新兴趋势。
+ **关键挑战与目标**：
    - **世界模型与RL的集成**：解决如何将世界模型有效融入RL循环的开源问题。
    - **可扩展性**：基于模型的RL与LLM的结合被认为是一个特别有前景的可扩展方向。

### **4️⃣** **教授LRMs高效推理（Teaching LRMs Efficient Reasoning）**
+ **核心问题**：推理时缩放（如思维链CoT）提高了LRMs在难题上的准确性，但也带来了**系统性的过度思考（Over-thinking）**（对简单问题使用过长推理链）和**思考不足（Under-thinking）**（在激进截断下过早停止、依赖脆弱捷径）的问题。
+ **未来方向**：开发**计算资源分配策略**，使模型能根据实例难度和认知不确定性，自适应地调整推理深度和停止时机。
+ **关键挑战与目标**：
    - **资源理性（Resource Rationality）**：核心未解问题是教会LRMs成为"资源理性"的，即只在边际效用合理的情况下进行更长时间的推理。
    - **原则性的权衡**：将当前各种方法（如提示中硬编码推理级别、基于长度的奖励塑形、损失函数中的长度惩罚）归纳为一个**原则性的成本-性能权衡**框架。

### **5️⃣** **教授LLMs潜在空间推理（Teaching LLMs Latent Space Reasoning）**
+ **核心问题**：当前CoT和RL的结合通常在**离散的标记空间**中进行采样，可能丢失连续语义空间中的细微信息。**潜在空间推理（LSR）**在LLM的连续潜在空间中进行推理，可能更有利于RL优化。
+ **未来方向**：探索**RL与LSR的结合**，以开发更强大、适应性更强的推理模型。
+ **关键挑战与目标**：
    - **评估挑战**：评估连续潜在思维的质量比评估基于标记的思维更具挑战性。
    - **奖励信号**：这会使提供准确的监督信号（如奖励和优势）变得复杂，是RL与LSR结合面临的开放性挑战。

### **6️⃣** **RL用于LLMs预训练（RL for LLMs Pre-training）**
+ **核心问题**：传统预训练依赖于大量文本语料库和下一个标记预测。能否将RL更早地应用于流程中，甚至作为主要的预训练策略？
+ **未来方向**：探索将RL应用于**预训练阶段本身**，而不仅仅是后训练（post-training）微调。
+ **关键挑战与目标**：
    - **RL风格预训练**：例如**强化预训练（Reinforcement Pre-Training）**，将下一个标记预测重新定义为具有可从语料库派生的可验证奖励的RL问题。
    - **成本效益**：关键挑战是使RL风格的预训练在大规模应用时具有**成本效益**，这需要减少验证器负担和奖励设计相关的成本。
    - **无监督奖励设计**：这与无监督奖励设计密切相关。

### **7️⃣** **RL用于基于扩散的LLMs（RL for Diffusion-based LLMs）**
+ **核心问题**：扩散大语言模型（DLLMs）是一种新兴的生成范式，但将RL应用于DLLMs存在独特挑战，特别是**对数概率估计**和**存在多条可行解码轨迹**的问题。
+ **未来方向**：为DLLMs开发有效的RL算法。
+ **关键挑战与目标**：
    - **高效准确的ELBO估计**：解决由于DLLMs通过最大化证据下界（ELBO）来近似似然优化而导致的**高效准确对数概率估计**的开放性难题。
    - **引导最优采样轨迹**：利用RL来引导模型走向**最优采样轨迹**，这需要为中间去噪步骤设计有效的奖励函数。

### **8️⃣** **RL用于科学发现中的LLMs（RL for LLMs in Scientific Discovery）**
+ **核心问题**：在生物、化学等科学领域，RL面临的核心挑战是**大规模的结果验证**，这通常依赖于湿实验室实验，成本高、速度慢。
+ **未来方向**：推动LLMs在科学发现中的应用，从狭窄定义的任务扩展到具有开放目标的复杂交互。
+ **关键挑战与目标**：
    - **奖励制定与Oracle模型**：探索**奖励制定**和改进**Oracle模型**本身。
    - **构建RL环境**：构建支持快速实验-反馈循环的合适RL环境，例如**计算机模拟（in silico simulations）**。
    - **集成领域特定模型**：将领域特定模型集成到LLM训练中，并开发能处理一系列明确定义任务的**通用模型**。

### **9️⃣** **RL用于架构-算法协同设计（RL for Architecture-Algorithm Co-Design）**
+ **核心问题**：当前RL流程通常假设固定的模型架构（如密集Transformer或混合专家模型MoE），**架构的自由度及其硬件影响**被排除在学习循环之外。
+ **未来方向**：将**架构作为RL中的一级动作空间**，实现架构与算法的协同设计。
+ **关键挑战与目标**：
    - **强化MoE**：例如，让模型学习路由策略、专家激活、容量分配或稀疏模式，不仅优化任务奖励，还优化硬件感知目标（如延迟、内存流量、能耗）。
    - **关键问题**：设计稳健的**多目标奖励函数**，实现修改网络拓扑结构时的稳定**信用分配**，以及跨提示、任务和部署规模的**架构策略学习摊销**。

