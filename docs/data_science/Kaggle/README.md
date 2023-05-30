# Competition
## **Mayo Clinic - STRIP AI** <br>
 [Image Classification of Stroke Blood Clot Origin](https://www.kaggle.com/competitions/mayo-clinic-strip-ai/overview)
- 妙佑医疗国际(神经血管实验室) - “脑卒中血栓栓塞影像学和病理学登记"（全称Stroke Thromboembolism Registry of Imaging and Pathology,一个项目）（这里registry指的是数据库或信息集合）
- 脑卒中血凝块来源的图像分类
> 项目旨在对各种病因的血栓栓塞进行组织病理学表征，并检查血栓组成及其与机械性血栓切除再灌注的关系。 

<!-- <img src="https://www.hnu.edu.cn/images/logo.png" style="background-color: #ae4141">
<span style="color: #3574f0; ">汇报人：丁孝亮</span> -->


## Overview(Introduction)
### Description

> 竞赛目标

#### 目标

- 简要：缺血性脑卒中中血凝块来源的分类
- <details><summary>详细：使用全幻灯片数字病理图像...</summary>
      本次竞赛的目标是对缺血性中风中的血凝块起源进行分类。使用全幻灯片数字病理图像，您将建立一个区分两种主要急性缺血性卒中(AIS)病因亚型的模型:<b>心脏</b>和<b>大动脉粥样硬化</b>。您的工作将使医疗保健提供者能够更好地识别致命中风中血栓的来源，使医生更容易开出最佳的中风后治疗管理处方，并降低第二次中风的可能性。</details>

#### 讲的**故事**：

* <details><summary>中风仍然是全球第二大死亡原因...</summary>
    中风仍然是全球第二大死亡原因。在美国，每年有超过70万人因血凝块阻塞通往大脑的动脉而患上缺血性中风。第二次中风(总事件的23%是复发性的)恶化了患者的生存机会。<b>然而，如果医生能够确定中风的病因，那么随后的中风可能会减轻</b>，因为病因会影响中风事件后的治疗管理。在过去的十年中，机械祛栓已成为大血管闭塞引起的急性缺血性脑卒中的标准治疗方法。结果，回收的血块变得易于分析。医疗保健专业人员目前正在尝试应用基于深度学习的方法来预测缺血性中风的病因和凝块起源。然而，<b>独特的数据格式，图像文件大小，以及可用病理幻灯片的数量</b>创造了您可以帮助解决的挑战。梅奥诊所是一家非营利性的美国学术医疗中心，专注于综合医疗保健、教育和研究。卒中血栓栓塞成像和病理登记(STRIP)是一个独特的大型多中心项目，由梅奥诊所神经血管实验室领导，目的是对各种病因的血栓栓塞进行组织病理学鉴定，并检查凝块组成及其与机械取栓血运重建的关系。为了降低随后中风的几率，梅奥诊所神经血管研究实验室(Mayo Clinic Neurovascular Research Laboratory)鼓励数据科学家改进基于人工智能的病因分类，以便医生更好地开出正确的治疗方案。新的计算和人工智能方法可以帮助挽救中风幸存者的生命，并帮助我们更好地了解世界第二大死亡原因。
</details>



### evaluation

> 评估

#### 评估指标

通过**加权多类对数损失**进行评估。总体效果是每一类对最终分数的影响差不多。

每一张图片已经被标上了病因分类，CE（心源性栓塞）或者LAA（大动脉粥样硬化），**对于每个图像，提交每个类的概率**。公式如下：
$$
Log Loss = - (\frac{\sum^{M}_{i=1}{wi}\cdot\sum^{N_i}_{j=1}{\frac{y_{ij}}{N_i}}\cdot{lnp_{ij}}}{\sum^{M}_{i=1}{w_i}})
$$

- $N$表示类的集合(class set)中图片的数量,

- $M$表示类(classes)的数量

- $ln$自然对数

- $$
  y_{ij} =\left\{
  \begin{aligned}
  1 &,&观察的i属于j类 \\
  0 &,&观察的i不属于j类 
  \end{aligned}
  \right.
  $$

- $p_{ij}$表示图像i属于j类的预测的可能性

对于一个给定的图像所提交的可能性（预测）不需要求和到1，因为每一行除以了行和，在评分(score)前被重新缩放了。为了避免$log$函数的极限问题，每个预测的可能性$p$需要被替换成$max(min(p,1-10^{-15}),10^{-15})$



#### 提交文件(Submission File)格式

每个测试集病人id需要预测两种病因分类的可能性，也就是要求提交文件格式如下:

```text
patient_id,CE,LAA
01f2b3,0.5,0.5
04de22,0.5,0.5
0a47c9,0.5,0.5
0af8b6,0.5,0.5
...
```



### Timeline

> 时间线

<details>
    <summary>点击展开</summary>
    <pre>
<li> 2022年7月6日——开始日期。</li>
<li> 2022年9月28日——报名截止日期。您必须在此日期之前接受比赛规则才能参加比赛。</li>
<li> 2022年9月28日——团队合并截止日期。这是最后一天参加者可以参加或合并小队。</li>
<li> 2022年10月5日——最终提交截止日期。除非另有说明，所有截止日期均为UTC当天晚上11:59。如有必要，主办单位保留更新比赛时间的权利。</li></pre>
</details>



### Prizes

> 奖金

第一名 5000$

第二名 3000$

第三名 2000$



### Code Requirements

> 代码要求

参赛作品必须通过笔记本完成。为了使“提交”按钮在提交后激活，必须满足以下条件:CPU Notebook <= 9小时运行时间GPU Notebook <= 9小时运行时间禁止上网允许免费和公开的外部数据，包括预先训练的模型提交文件必须命名为`submission .csv`（机翻）





## Dataset Description

> 数据集描述



#### description

> 描述

数据集由超过一千张高分辨率全切片数字病理图像，每一张都描述了一个急性缺血性中风病人的血块。

上面描述过，一共**两种病因**，CE(心源性栓塞)orLAA(大动脉粥样硬化)。

任务是进行**分类**。



#### File and Data Field Descriptions

> 文件和数据字段说明

- train/ —— 包含TIFF(标签图像文件格式)的图片用于训练数据的文件夹。
- test/   ——包含用作测试数据的图像的文件夹。实际测试数据包括约280幅图像。
- other/——补充图像，也就是除了CE,LAA以外的，比如不明或者其他类型。也就是Unknown or Other
- train.csv   ——~~位于~~`train/` folder。包含`train/`文件夹里面图片的解释说明。包括这几个
  - `image_id`:唯一标志符，由"`{patient_id}_{image_num}`"组成
  - `center_id`：这个切片(幻灯片,也就是这个图片)是哪个医疗中心的。
  - `patient_id`：如名
  - `image_num`：如名
  - `label`：该字段表示分类目标。是CE还是LAA。

- test.csv     ——与train类似。字段一样。

- other.csv  —— 是`other/`文件夹的解释,和train具有相同字段。但是center_id不可获得。字段说明如下：

  - `label`：不是CE和LAA了。分为Unknown和Other
  - `other_specified`：如果标记为other则已知病因

  意思就是Unknown other_specified就为null,是Other就填上具体的病因。

- sample_submission.csv —— 正确格式的示例提交文件。注意的是**你需要做的是对每个病人id(`patiend_id`)进行预测，而不是对每个图片id(`image_id`)**

### Example Test Data

> 示例测试代码

这里讲的是由于这是个代码竞赛，所以给你的测试样例使用后，提交notebook时将替换样例，而不是给你的。Hidden Test data you know:)

