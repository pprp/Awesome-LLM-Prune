# Awesome-LLM-Prune

<div align='center'>
  <img src=https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg >
  <img src=https://img.shields.io/github/stars/pprp/Awesome-LLM-Prune.svg?style=social >
  <img src=https://img.shields.io/github/watchers/pprp/Awesome-LLM-Prune.svg?style=social >
  <img src=https://img.shields.io/badge/Release-v0.1-brightgreen.svg >
  <img src=https://img.shields.io/badge/License-GPLv3.0-turquoise.svg >
 </div>   
This repository is dedicated to the pruning of large language models (LLMs). It aims to serve as a comprehensive resource for researchers and practitioners interested in the efficient reduction of model size while maintaining or enhancing performance.

We encourage the community to engage with this resource; please leave comments on issues related to papers you’re interested in or corrections where misunderstandings may occur. For further inquiries or to contribute to this project, feel free to submit a pull request or open an issue.

| Taxonomy | Unstructured                                                 | Structured                                                  | Semi-Structured                                              | Benchmark                                                    |
| -------- | ------------------------------------------------------------ | ----------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Label    | <img src=https://img.shields.io/badge/unstructured-turquoise.svg > | <img src=https://img.shields.io/badge/structured-blue.svg > | <img src=https://img.shields.io/badge/semi_structured-brightgreen.svg > | <img src=https://img.shields.io/badge/benchmark-purple.svg > |



- SparseGPT: Massive Language Models Can be Accurately Pruned in One-shot 
    - Label: <img src=https://img.shields.io/badge/unstructured-turquoise.svg ><img src=https://img.shields.io/badge/semi_structured-brightgreen.svg >
    - Author: Elias Frantar, Dan Alistarh
    - Link: https://arxiv.org/pdf/2301.00774.pdf 
    - Code: https://github.com/IST-DASLab/sparsegpt 
    - Pub: ICML 2023
    - Summary: First to prune GPT with at least 50% sparsity without any training. SparseGPT is entirely local, which only focus on weight updates without any global gradient information.
    - 摘要:首次在没有任何训练的情况下，以至少50%的稀疏度修剪GPT。SparseGPT完全是局部的，它只关注权重更新，没有任何全局梯度信息。

- Wanda: A Simple and Effective Pruning Approach For Large Language Models
    - Label: <img src=https://img.shields.io/badge/unstructured-turquoise.svg ><img src=https://img.shields.io/badge/structured-blue.svg >
    - Author: Mingjie Sun, Zhuang Liu, Anna Bair, etc.
    - Link: https://arxiv.org/pdf/2306.11695.pdf 
    - Code: https://github.com/locuslab/wanda
    - Pub: ICML23 workshop 
    - Summary: Wanda simplify the SparseGPT with approximation thus just rely on weight and activation to compute the pruning metric. Wanda can be seen as a simplified version of SparseGPT, as it simplify the Hessian approximation, reducing computation greatly.
    - 摘要：Wanda通过近似简化了SparseGPT，因此只依赖权重和激活来计算修剪指标。Wanda可以被看作是SparseGPT的简化版本，因为它简化了Hessian近似，大大减少了计算量。

- Pruner-Zero: Evolving Symbolic Pruning Metric 

    - Label: <img src=https://img.shields.io/badge/unstructured-turquoise.svg ><img src=https://img.shields.io/badge/structured-blue.svg >
    - Author: Peijie Dong, Lujun Li, Zhenheng Tang, Xiang Liu, Xinglin Pan, Qiang Wang, Xiaowen Chu
    - Link: [arxiv.org/pdf/2406.02924v1](https://arxiv.org/pdf/2406.02924v1) 
    - Code: [pprp/Pruner-Zero: Evolving Symbolic Pruning Metric from scratch (github.com)](https://github.com/pprp/Pruner-Zero)
    - Pub: ICML24 
    - Summary: Pruner-Zero formulates the pruning metric as a symbolic discovery problem. They develop an automatic framework for searching symbolic pruning metrics using genetic programming. They model the pruning metric as tree-based symbols and employ genetic programming to automatically identify the optimal candidate symbolic pruning metric. Experiments on LLaMA, LLaMA-2, OPT demonstrate the superiority of Pruner-Zero.
    - 摘要：Pruner-Zero将修剪指标制定为一个符号发现问题。他们开发了一个使用遗传编程搜索符号修剪指标的自动化框架。他们将修剪指标建模为基于树的符号，并采用遗传编程自动识别最优的候选符号修剪指标。在LLaMA、LLaMA-2、OPT上的实验证明了Pruner-Zero的优越性。

- LLM-Kick: Compressing LLMs: The Truth Is Rarely Pure and Never Simple
    - Label: <img src=https://img.shields.io/badge/benchmark-purple.svg >
    - Author: Ajay Jaiswal, Zhe Gan, etc
    - Link: https://arxiv.org/pdf/2310.01382.pdf
    - Code: https://github.com/VITA-Group/llm-kick
    - Pub: ICLR 2024
    - Summary: The paper introduces LLM-KICK, a comprehensive benchmark to evaluate the performance of compressed large language models (LLMs) across various tasks, including language understanding, reasoning, generation, in-context retrieval, and in-context summarization. The authors find that existing state-of-the-art compression methods, such as pruning and quantization, often fail to maintain the performance of the original uncompressed LLMs, despite negligible changes in perplexity. LLM-KICK unveils several interesting observations, including significant performance degradation in pruned LLMs at trivial sparsity ratios, failure of pruning methods for structured N:M sparsity patterns, and the relatively better performance of quantization methods compared to pruning. The paper also investigates the ability of compressed LLMs in in-context settings, where pruned LLMs with high sparsity ratios (≥50%) are found to be robust retrieval systems and maintain similar performance in text summarization as their dense counterparts.
    - 摘要: 本文提出了LLM-KICK基准测试,用于评估压缩大型语言模型(LLMs)在各种任务上的性能,包括语言理解、推理、生成、上下文检索和上下文摘要。作者发现,现有的最先进压缩方法,如剪枝和量化,通常无法维持原始未压缩LLMs的性能,尽管困惑度的变化很小。LLM-KICK揭示了几个有趣的观察结果,包括剪枝LLMs在微小稀疏度下出现显著性能下降,剪枝方法在结构化N:M稀疏模式下失效,以及量化方法相对于剪枝的较佳性能。论文还研究了压缩LLMs在上下文设置中的能力,发现高稀疏度(≥50%)的剪枝LLMs是稳健的检索系统,在文本摘要方面与密集对应物保持相似的性能。
    - Comment: This paper question the performance of LLM after pruning, which provide us a new perspective besides pure perplexity. This paper is worth reading because its evaluation is comprehensive. 



- RIA: Plug-and-Play: An Efficient Post-Training Pruning Method for Large Language Models
    - Label: <img src=https://img.shields.io/badge/unstructured-turquoise.svg ><img src=https://img.shields.io/badge/semi_structured-brightgreen.svg >
    - Author: Yingtao Zhang, Haoli Bai, Haokun Lin, Jialin Zhao, Lu Hou, Carlo Vittorio Cannistraci
    - Link: https://openreview.net/pdf?id=Tr0lPx9woF
    - Code: https://github.com/biomedical-cybernetics/Relative-importance-and-activation-pruning
    - Pub: ICLR 2024
    - Summary: For post-training pruning method, this paper proposed two innovative and plug-and-play components, which is Relative Importance and Activations (RIA) and Channel Permutation (CP). (1) RIA re-evaluate the importance of each weight element based on all connections that originate from input and output. (2) CP aims to preserve important weights under N:M sparsity, which yields better N:M structures by permuting the input channels of weight.
    - 摘要：对于训练后修剪方法，这篇论文提出了两个创新且即插即用的组件，即相对重要性和激活（RIA）以及通道排列（CP）。(1) RIA基于源自输入和输出的所有连接重新评估每个权重元素的重要性。(2) CP旨在在N:M稀疏性下保留重要权重，通过对权重的输入通道进行排列来产生更好的N:M结构。
    - Comment: I have thoroughly reviewed the source code and can affirm its effectiveness. The code is indeed of superior quality, demonstrating excellent standards in development. 

- Pruning Large Language Models via Accuracy Predictor
    - Label: <img src=https://img.shields.io/badge/unstructured-turquoise.svg >
    - Author: Yupeng Ji, Yibo Cao, Jiucai Liu 
    - Link: https://arxiv.org/pdf/2309.09507.pdf 
    - Code: Not available 
    - Pub: Arxiv 
    - Summary: Formulate the pruning LLM as NAS problem. The search space is the prunining ratio, layer type, etc. By utilizing GBDT accuracy predictor, this paper take the layer-wise importance as input and predict the PPL.
    - 摘要：将LLM的修剪问题formulate为神经架构搜索（NAS）问题。搜索空间包括修剪比率、层类型等。通过利用GBDT准确性预测器，本文将逐层重要性作为输入，预测困惑度（PPL）。
    - Comment: With 525 architecture-accuracy pair, this paper train the GBDT with 7:3 ratio.

- LLM-Pruner: On the Strucutal Pruning of Large Language Models 
    - Label: <img src=https://img.shields.io/badge/structured-blue.svg >
    - Author: Xinyin Ma, Gongfan Fang, Xinchao Wang 
    - Link: https://arxiv.org/pdf/2305.11627.pdf 
    - Code: https://github.com/horseee/LLM-Pruner
    - Pub: NeurIPS 2023 
    - Summary: This paper endeavor find the copuled structures (Dependency Graph) in LLaMA and proposed Groupded Importance Estimation like Vector-wise, Element-wise, and Group Importance.
    -  摘要：这篇论文致力于在LLaMA中发现耦合结构（依赖图），并提出了分组重要性估计，如向量级、元素级和组级重要性。
    - Comment: Impressive work. This work is similar to MMRazor, which can handle CNN-based model. 

- The Emergence of Essential Sparsity in Large Pre-trained Models: The Weights that Matter
    - Label: <img src=https://img.shields.io/badge/benchmark-purple.svg >
    - Author: Ajay Jaiswal, Shiwei Liu, Tianlong Chen, Zhangyang Wang
    - Link: https://arxiv.org/pdf/2306.03805.pdf
    - Code: https://github.com/VITA-Group/essential_sparsity
    - Pub: NeurIPS 2023 
    - Summary: This paper proposes the existence of – “essential sparsity” defined with a sharp dropping point beyond which the performance declines much faster w.r.t the rise of sparsity level, when we directly remove weights with the smallest magnitudes in one-shot.
    - 摘要：这篇论文提出了"基本稀疏性"的存在，它定义为一个急剧下降点，当我们一次性直接移除最小幅度的权重时，超过这个点后，性能下降的速度会随着稀疏度的增加而变得更快。

- Compresso: Structured Pruning with Collaborative Prompting Learns Compact Large Language Models
    - Label: <img src=https://img.shields.io/badge/structured-blue.svg >
    - Author: Song Guo, Jiahang Xu, Li Lyna Zhang, Mao Yang 
    - Link: https://arxiv.org/pdf/2310.05015.pdf 
    - Code: https://github.com/microsoft/Moonlit/tree/main/Compresso
    - Pub: Under Review 
    - Summary: Combing instruction tuning with training-based Pruning. LoRA is incorporated to achieve memory-efficient. Collaborative pruning prompt encourage LLMs to better align with the pruning algorithm.
    - 摘要：将指令微调与基于训练的修剪相结合。引入LoRA以实现内存效率。协作修剪提示鼓励大型语言模型（LLMs）更好地与修剪算法对齐。
    - Comment: The prompt is really interesting, which is "Attention! LLM".

- The Truth is in There: Improving Reasoning in Language Models with Layer-Selective Rank Reduction
  - Label: <img src=https://img.shields.io/badge/structured-blue.svg >
  - Author: Pratyusha Sharma, Jordan T. Ash, Dipendra Misra 
  - Link: https://arxiv.org/pdf/2312.13558.pdf 
  - Code: Not available 
  - Pub: ICLR Under review 
  - Summary: This paper is not related to Pruning but to Low-rank decomposition. They find that removing higher-order component of weight matrics in MLP and attention can significantly improve the performance of LLMs.
  - 摘要：这篇论文与修剪无关，而是关于低秩分解。他们发现，移除MLP和注意力机制中权重矩阵的高阶分量可以显著提高大型语言模型（LLMs）的性能。

- Outlier Weighed Layerwise Sparsity (OWL): A Missing Secret Sauce for Pruning LLMs to High Sparsity
  - Label: <img src=https://img.shields.io/badge/unstructured-turquoise.svg >
  - Author:Lu Yin, You Wu, Zhenyu Zhang, Cheng-Yu Hsieh, Yaqing Wang, Yiling Jia, Mykola Pechenizkiy, Yi Liang, Zhangyang Wang, Shiwei Liu
  - Link:https://arxiv.org/abs/2310.05175
  - Code:https://github.com/luuyin/OWL 
  - Pub: ICML 2024
  - Summary: OWL challenges the assumption of uniform layer-wise assumption and tries to assign different layers with different pruning ratio by proposed OWL metric.
  - 摘要：OWL挑战了均匀层间假设，并尝试通过提出的OWL度量为不同层分配不同的修剪比率。

- The LLM Surgeon
  - Label: <img src=https://img.shields.io/badge/unstructured-turquoise.svg >
  - Author:Tycho F.A. van der Ouderaa, Markus Nagel, Mart van Baalen, Yuki M. Asano, Tijmen Blankevoort
  - Link:https://arxiv.org/pdf/2312.17244
  - Pub: ICLR24
  - Summary: This paper scales Kronecker-factored curvature approximations of the target loss landscape to large language models. The metric for this paper is Fisher information matrix.
  - 摘要：这篇论文将目标损失景观的Kronecker因子曲率近似扩展到大型语言模型。本文使用的度量是Fisher信息矩阵。

- Shortened LLaMA: A Simple Depth Pruning for Large Language Models
  - Label: <img src=https://img.shields.io/badge/structured-blue.svg >
  - Authors: Bo-Kyeong Kim, Geonmin Kim, Tae-Ho Kim, Thibault Castells, Shinkook Choi, Junho Shin, Hyoung-Kyu Song
  - Link: https://arxiv.org/abs/2402.02834 
  - Pub: ICLR24 Workshop (ME-FoMo) 
  - Summary: This paper delves into the naive implementation of structured pruning, specifically Depth Pruning, for Large Language Models (LLMs). Through experiments on zero-shot tasks, it is revealed that its performance is on par with width pruning techniques. However, the pruning ratio remains constrained to less than 35% (20%, 27%, 35%), and the performance on wikitext-2 (PPL) is somewhat less favorable compared to wanda. Nonetheless, this study demonstrates the feasibility of pruning by eliminating layers with lower block-level importance scores. Moreover, performance enhancement is observed after one-shot pruning via LoRA fine-tuning.
  - 摘要：这篇论文深入研究了大型语言模型（LLMs）结构化修剪的朴素实现，特别是深度修剪。通过零样本任务的实验，发现其性能与宽度修剪技术相当。然而，修剪比率仍然限制在35%以下（20%、27%、35%），在wikitext-2（困惑度PPL）上的性能相比wanda略差。尽管如此，这项研究证明了通过消除具有较低块级重要性分数的层来进行修剪是可行的。此外，通过LoRA微调，在一次性修剪后观察到性能提升。



- SliceGPT: Compress Large Language Models by Deleting Rows and Columns 
  - Label: <img src=https://img.shields.io/badge/structured-blue.svg >
  - Author: Saleh Ashkboos, Maximilian L. Croci, Marcelo Gennari do Nascimento, Torsten Hoefler, James Hensman
  - Link: https://arxiv.org/abs/2401.15024 
  - Pub: ICLR24
  - Summary: This paper focuses on structured pruning by removing rows and columns of a matrix to reduce parameters. However, this idea is similar to LLM-Pruner but weaker. The organization of this paper is somewhat peculiar: it dedicates two and a half pages to related works (too long). Additionally, in Table 1, SliceGPT (<30% sparsity) mainly compares its performance with SparseGPT under 2:4 structure pruning settings (50% sparsity), which is not quite fair. Please correct me if I am wrong.
  - 摘要：这篇论文关注通过移除矩阵的行和列来减少参数的结构化修剪。然而，这个想法与LLM-Pruner相似但较弱。这篇论文的结构有些特殊：它用了两页半的篇幅来介绍相关工作（太长了）。此外，在表1中，SliceGPT（<30%稀疏度）主要将其性能与SparseGPT在2:4结构修剪设置（50%稀疏度）下进行比较，这并不太公平。如果我有误解，请指正

- PERP: Rethinking the Prune-Retrain Paradigm in the Era of LLMs 
  - Label: <img src=https://img.shields.io/badge/unstructured-turquoise.svg >
  - Author: Max Zimmer, Megi Andoni, Christoph Spiegel, Sebastian Pokutta
  - Link: https://arxiv.org/pdf/2312.15230.pdf
  - Pub: Arxiv 
  - Summary: In the era of Large Language Models (LLMs), retraining becomes impractical due to memory and compute constraints. This paper proposes the use of Low-Rank Adaption to mitigate the expense of the retraining process. They explore four approaches, including BN-Recalibration, Biases, BN-Parameters, and Linear Probing. However, it's worth noting that most LLMs do not utilize Batch Normalization (BN). Indeed, this paper only conducts a few experiments on OPT and primarily focuses on works such as ResNet50 pruning. Furthermore, LoRA + Pruning is actually a component of SparseGPT (published in January 2023), so the novelty of this paper is somewhat limited.
  - 摘要：在大型语言模型（LLMs）的时代，由于内存和计算的限制，重新训练变得不切实际。这篇论文提出使用低秩适应来缓解重新训练过程的开销。他们探索了四种方法，包括BN-重校准、偏置、BN-参数和线性探测。然而，值得注意的是，大多数LLMs并不使用批量归一化（BN）。事实上，这篇论文只在OPT上进行了一些实验，主要关注的是ResNet50修剪等工作。此外，LoRA + 修剪实际上是SparseGPT（2023年1月发表）的一个组成部分，所以这篇论文的新颖性有些有限。
 

- Structural pruning of large language models via neural architecture search
  - Label: <img src=https://img.shields.io/badge/structured-blue.svg >
  - Author:Aaron Klein, Jacek Golebiowski, Xingchen Ma, Valerio Perrone, Cedric Archambeau 
  - Link: https://openreview.net/pdf?id=SHlZcInS6C
  - Pub: AutoML 
  - Summary: This paper discuss the relationship between NAS and structural pruning and employ multi-objective NAS to compress LLM. They view the pre-trained network as a super-network and search for the best sub-network that optimally balance between downstream tasks and parameter count. For training weight-sharing NAS, they employ sandwich rule to train sub-networks. After training, local search is utilized for finding the best sub-network.
  - 摘要：这篇论文讨论了神经架构搜索（NAS）和结构化修剪之间的关系，并使用多目标NAS来压缩大型语言模型（LLM）。他们将预训练网络视为超网络，搜索在下游任务和参数数量之间最佳平衡的子网络。对于权重共享NAS的训练，他们采用三明治规则来训练子网络。训练后，使用局部搜索来寻找最佳子网络。

- Not all Layers of LLMs are Necessary during Inference 
  - Label: <img src=https://img.shields.io/badge/structured-blue.svg >
  - Author: Siqi Fan, Xin JIang, Xiang Li, Xuying Meng, Peng Han, Shuo Shang, Aixin Sun, Yequan Wang, Zhongyuan Wang
  - Link: https://arxiv.org/pdf/2403.02181.pdf 
  - Pub: CoRR 
  - Summary: This paper analyse the activated layers across tasks and propose AdaInfer to determine the inference termination moments based on the input instance. Thus, they can use shallow layers for easy instance and deep layers for hard ones. In general, this technique can be treated as an early stopping strategy. The early stop signal is generated by two components: Feature Selection Module that crafts feature vector for current input instance; Classifier that utilize SVM or CRF to access the strength of stopping signal.
  - 摘要：这篇论文分析了不同任务中的激活层，并提出AdaInfer根据输入实例确定推理终止时刻。因此，他们可以对简单实例使用浅层，对困难实例使用深层。总的来说，这种技术可以被视为一种早停策略。早停信号由两个组件生成：特征选择模块为当前输入实例制作特征向量；分类器使用SVM或CRF来评估停止信号的强度。

- ShortGPT: Layers in Large Language Models are More Redundant Than You Expect
  - Label: <img src=https://img.shields.io/badge/structured-blue.svg >
  - Author: Xin Men, Mingyu Xu, Qingyu Zhang, Bingning Wang, Hongyu Lin, Yaojie Lu, Xianpei Han, Weipeng Chen
  - Link: https://arxiv.org/abs/2403.03853 
  - Pub: CoRR 
  - Summary: They discovered that the layers of LLMs exhibit high similarity and some layers are negligible. To remove the unimportant layers, they define a metric called Block Influence (BI) to gauge the significance of each layers in LLMs. Specifically, the BI score is actually the cosine similarity of two successive blocks. The experiments are limited as they didn't provide the results of ppl and there are various one-shot pruning for LLMs like SparseGPT and Wanda etc.
  - 摘要：他们发现LLMs的层展现出高度相似性，一些层是可以忽略的。为了移除不重要的层，他们定义了一个称为块影响（BI）的度量来衡量LLMs中每一层的重要性。具体来说，BI分数实际上是两个连续块的余弦相似度。实验有限，因为他们没有提供困惑度（ppl）的结果，而且已经存在各种针对LLMs的一次性修剪方法，如SparseGPT和Wanda等。


- LaCo: Large Language Model Pruning via Layer Collapse
  - Label: <img src=https://img.shields.io/badge/structured-blue.svg >
  - Author:Yifei Yang, Zouying Cao, Hai Zhao
  - Link: https://arxiv.org/pdf/2402.11187.pdf
  - Pub: CoRR 
  - Summary: These paper compute the different among layers (call it Reserving-Difference) and merge them (Call it Seeking-Common). Specifically, they merge m consecutive layers into one by using sum of parameter difference. Also, they employ trial-and-error by evaluating each merged  model with Cosine Similarity and make adjustment of the merge.
  - 摘要：这些论文计算层之间的差异（称为保留差异）并合并它们（称为寻求共同点）。具体来说，他们通过使用参数差异的总和将m个连续层合并成一个。此外，他们通过使用余弦相似度评估每个合并模型并对合并进行调整来采用试错法。
  - Comments: There is a lack of explanation of equation-1. Why it worked?

- Shortened LLaMA: A Simple Depth Pruning for Large Language Models 
  - Label: <img src=https://img.shields.io/badge/structured-blue.svg >
  - Author: Bo-Kyeong Kim, Geonmin Kim, Tae-Ho Kim, Thibault Castells, Shinkook Choi, Junho Shin, Hyoung-Kyu Song 
  - Link:https://arxiv.org/abs/2402.02834 
  - Pub: CoRR
  - Summary: This paper focus on depth pruning and analysis the influence of depth and width pruning on LLM inference efficiency. They explore various design factors including the choice of prunable units, the criteria and retraining frequency. (1) Prunable Units: width and depth; (2) Criteria: Magnitude, Taylor, Mag+ and Talyor+, PPL; (3)retrain: LoRA. Finally, they choose PPL as criteria and target Depth Pruning. They claim that depth pruning approach can compte with recent width pruning methods  on Zero-shot tasks performance.
  - 摘要：这篇论文关注深度修剪，并分析了深度和宽度修剪对LLM推理效率的影响。他们探索了各种设计因素，包括可修剪单元的选择、标准和重新训练频率。(1)可修剪单元：宽度和深度；(2)标准：幅度、Taylor、Mag+和Taylor+、PPL；(3)重新训练：LoRA。最终，他们选择PPL作为标准，并以深度修剪为目标。他们声称深度修剪方法在零样本任务性能上可以与最近的宽度修剪方法相媲美。

- FLAP: Fluctuation-based adaptive structured pruning for large language models
  - Label: <img src=https://img.shields.io/badge/structured-blue.svg >
  - Author: Yongqi An, Xu Zhao, Tao Yu, Ming Tang, Jinqiao Wang
  - Link: https://arxiv.org/abs/2312.11983 
  - Code: https://github.com/CASIA-IVA-Lab/FLAP 
  - Pub: AAAI 24
  - Summary: They proposed a retraining-free structured pruning framework for LLMs. (1) Structured Importance Metric: Fluctuation Pruning Metric (2) Adaptively Search Global Compress Ratio: (3) Compensation Mechanism: add additional biases to recover the output feature maps. Specifically, they observe that certain channels of hidden state features exhibits a low variation across different samples, indicating that if their corresponding input feature channels are pruned, the resulted change can be counterbalanced by the baseline value. Compared with Wanda, FLAP compute the sample variance of each input feature and weight it with the squared norm of the corresponding column of the weight matrics.
  - 摘要：他们提出了一个无需重新训练的LLMs结构化修剪框架。(1)结构化重要性度量：波动修剪度量 (2)自适应搜索全局压缩比：(3)补偿机制：添加额外的偏置以恢复输出特征图。具体来说，他们观察到隐藏状态特征的某些通道在不同样本间表现出低变异性，表明如果修剪了它们对应的输入特征通道，由此产生的变化可以通过基线值来平衡。与Wanda相比，FLAP计算每个输入特征的样本方差，并用权重矩阵对应列的平方范数进行加权。
  - Comment: This paper is well-written and the framework is clear. However, I have a question: they claim FLAP is a retraining-free framework but it still require retraining the biases.
 

- Bonsai: Everybody Prune Now: Structured Pruning of LLMs with only Forward Passes
  - Label: <img src=https://img.shields.io/badge/structured-blue.svg >
  - Author: Lucio Dery, Steven Kolawole, Jean-François Kagy, Virginia Smith, Graham Neubig, Ameet Talwalkar
  - Link: [arxiv.org/pdf/2402.05406.pdf](https://arxiv.org/pdf/2402.05406.pdf)
  - Code: https://github.com/ldery/Bonsai 
  - Summary: This work devoted to structured pruning of LLMs using only forward passes (gradient-free way). Bonsai can outperform gradient-based structured pruning methods and twice as fast as semi-structured pruning methods. Specifically, Bonsai measures the performance of each module's performance by generating sub-models, which require multiple forwards. Also, Bonsai use informative priors (Deep compression, a.k.a other unstructured pruning method) to drop modules. Bonsai adopts iterative pruning method. In each iteration, it will assess the prior of unpruned module and utilize them to select new sub-model.
  - 摘要：这项工作致力于仅使用前向传播（无梯度方式）对LLMs进行结构化修剪。Bonsai可以胜过基于梯度的结构化修剪方法，并且比半结构化修剪方法快两倍。具体来说，Bonsai通过生成子模型来测量每个模块的性能，这需要多次前向传播。此外，Bonsai使用信息先验（深度压缩，即其他非结构化修剪方法）来丢弃模块。Bonsai采用迭代修剪方法。在每次迭代中，它会评估未修剪模块的先验，并利用它们来选择新的子模型。


- The Unreasonable Ineffectiveness of the Deeper Layers 
  - Label: <img src=https://img.shields.io/badge/structured-blue.svg >
  - Author: Andrey Gromov, Kushal Tirumala, Hassan Shapourian, Paolo Glorioso, Daniel A. Roberts
  - Link: https://arxiv.org/pdf/2403.17887v1.pdf
  - Pub: Arxiv 
  - Summary: This paper aims to layer-pruning (structured pruning) by identify the optimal block of layers to prune by considering the similarity across layers. To recover performance, QLoRA is employed to make all experiments can be conducted on a A100. This paper claims that the shallow layers plays a more critical role than deeper layers of network.
  - 摘要：这篇论文旨在通过考虑层间的相似性来识别最佳的层块进行层修剪（结构化修剪）。为了恢复性能，使用QLoRA使所有实验可以在A100上进行。这篇论文声称浅层比网络的深层扮演更关键的角色。
  - Comment: good reference for studying the depth-dependence of neural networks.
 

- SLEB: Streamlining LLMs through Redundancy Verification and Elimination of Transformer Blocks
    - Label: <img src=https://img.shields.io/badge/structured-blue.svg >
    - Author: Jiwon Song, Kyungseok Oh, Taesu Kim, Hyungjun Kim, Yulhwa Kim, Jae-Joon Kim
    - Link: https://arxiv.org/pdf/2402.09025.pdf
    - Code: [https://github.com/leapingjagg-dev/SLEB](https://github.com/leapingjagg-dev/SLEB?tab=readme-ov-file)
    - Pub: Arxiv
    - Summary: This paper streamlines LLMs by identifying and removing redundant blocks. Specifically, cosine similarity is utilized to analyze the redundancy. Another metric3 is proposed for removing blocks.
    - 摘要：这篇论文通过识别和移除冗余块来简化LLMs。具体来说，利用余弦相似度来分析冗余性。另外提出了一个metric3用于移除块。
    - Comment: There should be more methods for comparison.

- Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning
  - Label: <img src=https://img.shields.io/badge/structured-blue.svg >
  - Tag: Structured Pruning
  - Author: Mengzhou Xia, Tianyu Gao, Zhiyuan Zeng, Danqi Chen;
  - Link: https://arxiv.org/abs/2310.06694
  - Code: https://github.com/princeton-nlp/LLM-Shearing
  - Pub: ICLR 2024 
  - Summary: To prune larger pre-trained model, this paper proposed (1) Targeted structured pruning: prune a LLM to specified target shape by removing layers, heads, and intermediate and hidden dimensions in an end-to-end manner; (2) Dynamic Batch Loading: update the composition of sampled data in each training batch based on varying losses across different domains.
  - 摘要：为了修剪更大的预训练模型，这篇论文提出了(1)目标结构化修剪：通过以端到端的方式移除层、头部以及中间和隐藏维度，将LLM修剪到指定的目标形状；(2)动态批次加载：根据不同领域的不同损失更新每个训练批次中采样数据的组成。

- Gradient-Free Adaptive Global Pruning for Pre-trained Language Models
    - Label: <img src=https://img.shields.io/badge/unstructured-turquoise.svg >
    - Author: Guangji Bai, Yijiang Li, Chen Ling, Kibaek Kim, Liang Zhao
    - Link: https://arxiv.org/pdf/2402.17946v1.pdf
    - Pub: Arxiv
    - Code: https://github.com/BaiTheBest/AdaGP
    - Summary: Due to the size of LLM, global pruning becomes impractical. However, local pruning often leads to suboptimal solutions. To address this issue, this paper propose Adaptive Global Pruning (AdaGP) to redefine the global pruning process into manageable, coordinated subproblems, allowing for resource-efficient optimization with global optimality.
    - 摘要：由于LLM的规模，全局修剪变得不切实际。然而，局部修剪常常导致次优解。为解决这个问题，这篇论文提出了自适应全局修剪（AdaGP），将全局修剪过程重新定义为可管理的、协调的子问题，允许在全局最优性的同时进行资源高效的优化。

- NutePrune: Efficient Progressive Pruning with Numerous Teachers for Large Language Models
    - Label: <img src=https://img.shields.io/badge/unstructured-turquoise.svg >
    - Author: Shengrui Li, Xueting Han, Jing Bai
    - Link: https://arxiv.org/pdf/2402.09773.pdf
    - Pub: Arxiv
    - Code: https://github.com/Luciuslsr/NutePrune
    - Summary: This work = structure pruning + progressive knowledge distillation; However, due to the memory constraints, knowledge distillation is hard in the context of LLM. To mitigate the memory cost, this paper propose to switch teacher and student by apply different sparsity ratio using various masks and LoRA modules.
    - 摘要：这项工作 = 结构修剪 + 渐进式知识蒸馏；然而，由于内存限制，在LLM环境下知识蒸馏变得困难。为了减轻内存成本，这篇论文提出通过使用不同的掩码和LoRA模块应用不同的稀疏比来切换教师和学生模型。


- BESA: Pruning Large Language Models with Blockwise Parameter-Efficient Sparity Allocation
    - Label: <img src=https://img.shields.io/badge/unstructured-turquoise.svg >
    - Author: Peng Xu, Wenqi Shao, Mengzhao Chen, Shitao Tang, Kaipeng Zhang, Peng Gao, Fengwei An, Yu Qiao, Ping Luo.
    - Link: https://openreview.net/pdf?id=gC6JTEU3jl
    - Code: https://github.com/LinkAnonymous/BESA
    - Pub: ICLR 2024
    - Summary: Existing pruning methods for LLM adopted a layer-wise approach but resulted in significant perturbation to the model’s output and required meticulous hyperparameter tuning(Pruning Ratio). This paper proposes BESA to handle it with block-wise adaptation. (1) Instead of pruning each Linear layer, BESA targets the overall pruning error w.r.t. one transformer block (2) it allocates layer-specific sparsity in a differentiable manner.
    - 摘要：现有的LLM修剪方法采用了逐层方法，但导致模型输出的显著扰动，并需要细致的超参数调整（修剪比率）。这篇论文提出BESA来通过块级适应处理这个问题。(1)BESA不是修剪每个线性层，而是针对一个transformer块的整体修剪误差；(2)它以可微分的方式分配层特定的稀疏性。

- Fast and Optimal Weight Update for Pruned Large Language Models
    - Label: <img src=https://img.shields.io/badge/unstructured-turquoise.svg ><img src=https://img.shields.io/badge/semi_structured-brightgreen.svg >
    - Authors: **Vladim ́ır Bozˇa**
    - Link: https://arxiv.org/pdf/2401.02938.pdf
    - Code: Not available
    - Code: https://github.com/fmfi-compbio/admm-pruning
    - Summary:  This paper focuses on the recovery process, which was first proposed in SparseGPT. This paper proposed an Alternating Direction Method of Multipliers (ADMM), with a simple iterative pruning mask selection.
    - 摘要：这篇论文关注于恢复过程，这一过程最初由SparseGPT提出。本文提出了一种交替方向乘子法（ADMM），并配合一个简单的迭代修剪掩码选择。

- COPAL: Continual Pruning in Large Language Generative Models
    - Label: <img src=https://img.shields.io/badge/unstructured-turquoise.svg ><img src=https://img.shields.io/badge/semi_structured-brightgreen.svg >
    - Authors: Srikanth Malla, Joon Hee Choi, Chiho Choi
    - Link: https://arxiv.org/pdf/2405.02347v1.pdf
    - Code: Not Available
    - Summary: This paper introduces COPAL, an algorithm for continual pruning of large language models under a model adaptation setting. The approach utilizes sensitivity analysis to guide the pruning process, enhancing model adaptability and computational efficiency without the need for retraining. The empirical evaluation demonstrates COPAL's effectiveness in maintaining performance across various datasets and model sizes.
    - 摘要：这篇论文介绍了COPAL，一种在模型适应设置下对大型语言模型进行持续修剪的算法。该方法利用敏感性分析来指导修剪过程，在无需重新训练的情况下提高模型的适应性和计算效率。实证评估证明了COPAL在各种数据集和模型规模上保持性能的有效性。

- DaSS: Dependency-Aware Semi-Structured Sparsity of GLU Variants in Large Language Models
    - Label: <img src=https://img.shields.io/badge/semi_structured-brightgreen.svg >
    - Authors: Zhiyu Guo, Hidetaka Kamigaito, Taro Wanatnabe
    - Link: https://arxiv.org/pdf/2405.01943v1.pdf
    - Code: Not available
    - Summary: This paper introduces Dependency-aware Semi-structured Sparsity (DaSS), a novel pruning method for SwiGLU-based Large Language Models (LLMs). DaSS integrates structural dependency into weight magnitude-based pruning, using an MLP-specific pruning metric that evaluates the importance of each weight by considering both its magnitude and the corresponding MLP intermediate activation norms. The method offers a balance between unstructured pruning flexibility and structured pruning consistency, achieving hardware-friendly N:M sparsity patterns. Empirical results show DaSS outperforms SparseGPT and Wanda in various tasks while maintaining computational efficiency.
    - 摘要：这篇论文介绍了依赖感知半结构化稀疏性（DaSS），这是一种针对基于SwiGLU的大型语言模型（LLMs）的新型修剪方法。DaSS将结构依赖性整合到基于权重幅度的修剪中，使用一种特定于MLP的修剪度量，通过考虑每个权重的幅度和相应的MLP中间激活范数来评估其重要性。该方法在非结构化修剪的灵活性和结构化修剪的一致性之间取得平衡，实现了硬件友好的N:M稀疏模式。实证结果表明，DaSS在保持计算效率的同时，在各种任务中的表现优于SparseGPT和Wanda。

- Structural Pruning of Pre-trained Language Models via Neural Architecture Search
    - Label: <img src=https://img.shields.io/badge/structured-blue.svg >
    - Authors: Aaron Klein, Jacek Golebiowski, Xingchen Ma, Valerio Perrone, Cedric Archambeau
    - Link: https://arxiv.org/pdf/2405.02267v1.pdf
    - Code: Not available
    - Summary: This paper explores the use of Neural Architecture Search (NAS) for structural pruning of pre-trained language models to address the challenges of high GPU memory requirements and inference latency. The authors propose a multi-objective approach that identifies the Pareto optimal set of sub-networks, enabling a flexible compression process without the need for retraining. The method leverages weight-sharing NAS techniques to accelerate the search for efficient sub-networks. Empirical evaluations demonstrate that their approach outperforms baseline models in terms of efficiency and adaptability, offering a promising strategy for deploying large language models in real-world applications.
    - 摘要：这篇论文探讨了使用神经架构搜索（NAS）对预训练语言模型进行结构化修剪，以解决高GPU内存需求和推理延迟的挑战。作者提出了一种多目标方法，识别帕累托最优子网络集，实现灵活的压缩过程，无需重新训练。该方法利用权重共享NAS技术加速搜索高效子网络。实证评估表明，他们的方法在效率和适应性方面优于基线模型，为在现实应用中部署大型语言模型提供了一种有前景的策略。
    - Note: All experiments are conducted on BERT not LLAMA. This NAS procedure requires massive computation when applying to LLaMA.


- Pruning Small Pre-Trained Weights Irreversibly and Monotonically Impairs "Difficult" Downstream Tasks in LLMs
    - Label: <img src=https://img.shields.io/badge/benchmark-purple.svg >
    - Authors: Lu Yin, Ajay Jaiswal, Shiwei Liu, Souvik Kundu, Zhangyang Wang
    - Link: https://arxiv.org/pdf/2310.02277v2.pdf
    - Code: https://github.com/VITA-Group/Junk_DNA_Hypothesis.git
    - Pub: ICML24
    - Summary: The paper presents the "Junk DNA Hypothesis," which challenges the notion that **small-magnitude weights in large language models (LLMs) are redundant and can be pruned without performance loss**. Contrary to common beliefs, the study argues that these weights encode essential knowledge for difficult downstream tasks. The authors demonstrate a monotonic relationship between the performance drop of downstream tasks and the magnitude of pruned weights, indicating that pruning can cause irreversible knowledge loss, even with continued training. The paper also contrasts pruning with quantization, showing that the latter does not exhibit the same monotonic effect on task difficulty. The findings suggest that small-magnitude weights are crucial for complex tasks and cannot be simply discarded. The study provides insights into the role of these weights and implications for LLM compression techniques.
    - 摘要：这篇论文提出了"垃圾DNA假说"，挑战了大型语言模型（LLMs）中小幅度权重是冗余的且可以在不损失性能的情况下被修剪的观点。与普遍认知相反，研究认为这些权重编码了困难下游任务的重要知识。作者证明了下游任务性能下降与被修剪权重幅度之间的单调关系，表明修剪可能导致不可逆的知识损失，即使继续训练也是如此。论文还将修剪与量化进行对比，显示后者对任务难度没有同样的单调影响。研究结果表明，小幅度权重对复杂任务至关重要，不能简单地被丢弃。这项研究为这些权重的作用及其对LLM压缩技术的影响提供了见解。
 
- Pruning via Merging: Compressing LLMs via Manifold Alignment Based Layer Merging
    - Label: <img src=https://img.shields.io/badge/structured-blue.svg>
    - Authors: Deyuan Liu, Zhanyue Qin, Hairu Wang, Zhao Yang, Zecheng Wang, Fangying Rong, Qingbin Liu, Yanchao Hao, Xi Chen, Cunhang Fan, Zhao Lv, Zhiying Tu, Dianhui Chu, Dianbo Sui
    - Link: https://arxiv.org/pdf/2406.16330
    - Code: Not available
    - Summary: The paper introduces a novel model compression technique known as Manifold-Based Knowledge Alignment and Layer Merging Compression (MKA). This method utilizes manifold learning and the Normalized Pairwise Information Bottleneck (NPIB) measure to merge similar layers within large language models (LLMs), effectively reducing the model size while maintaining performance. The key innovation lies in the alignment of knowledge across layers and the subsequent merging of those with high input-output similarity, which helps preserve the model's capabilities despite size reduction. Extensive evaluations on multiple benchmark datasets and various LLMs demonstrate that MKA achieves substantial compression ratios, outperforming traditional pruning methods. Notably, when combined with quantization, MKA can achieve even greater compression efficiencies. For instance, on the MMLU dataset using the Llama3-8B model, MKA attains a compression ratio of 43.75% with only a minimal performance decrease of 2.82%. The study underscores the potential of MKA as a resource-efficient and performance-preserving model compression approach for LLMs.
    - 摘要：这篇论文介绍了一种新的模型压缩技术，称为基于流形的知识对齐和层合并压缩（MKA）。该方法利用流形学习和归一化成对信息瓶颈（NPIB）度量来合并大型语言模型（LLMs）中的相似层，有效减小模型大小同时保持性能。关键创新在于跨层的知识对齐，以及随后合并那些具有高输入-输出相似性的层，这有助于在减小尺寸的同时保持模型的能力。在多个基准数据集和各种LLMs上的广泛评估表明，MKA实现了显著的压缩比，优于传统的修剪方法。值得注意的是，当与量化结合时，MKA可以实现更高的压缩效率。例如，在使用Llama3-8B模型的MMLU数据集上，MKA实现了43.75%的压缩比，性能仅下降2.82%。这项研究强调了MKA作为LLMs资源高效且保持性能的模型压缩方法的潜力。

- Token Fusion: Bridging the Gap between Token Pruning and Token Merging
    - Label: <img src=https://img.shields.io/badge/token-green.svg>
    - Authors: Minchul Kim*, Shangqian Gao, Yen-Chang Hsu, Yilin Shen, Hongxia Jin
    - Affiliation: Michigan State University, Samsung Research America
    - Link: [arXiv:2312.01026](https://arxiv.org/pdf/2312.01026)
    - Code: Not Available
    - Summary: The paper introduces "Token Fusion" (ToFu), a novel approach that combines the benefits of token pruning and token merging for Vision Transformers (ViTs). It is designed to address the high computational cost of deploying ViTs on edge devices by dynamically adapting the strategy based on the model's sensitivity to input interpolations. ToFu introduces a new merging technique called MLERP, which preserves the norm distribution during token merging, overcoming the limitations of average merging. The method is versatile and applicable to ViTs with or without additional training, establishing new benchmarks for computational efficiency and model accuracy in both classification and image generation tasks.
    - 摘要：这篇论文介绍了"Token Fusion"（ToFu），这是一种新方法，结合了视觉Transformer（ViTs）的token修剪和token合并的优点。它旨在解决在边缘设备上部署ViTs的高计算成本问题，通过根据模型对输入插值的敏感性动态调整策略。ToFu引入了一种新的合并技术，称为MLERP，在token合并过程中保持范数分布，克服了平均合并的局限性。该方法versatile，可应用于有或无额外训练的ViTs，在分类和图像生成任务中都建立了计算效率和模型准确性的新基准。

- CaM: Cache Merging for Memory-efficient LLMs Inference
    - Label: <img src=https://img.shields.io/badge/token-green.svg>
    - Authors: Yuxin Zhang, Yuxuan Du, Gen Luo, Yunshan Zhong, Zhenyu Zhang, Shiwei Liu, Rongrong Ji
    - Link: https://openreview.net/forum?id=LCTmppB165
    - Code: https://github.com/zyxxmu/cam
    - Summary: The paper introduces Cache Merging (CaM), an innovative approach to enhance the memory efficiency of Large Language Models (LLMs) during inference without compromising performance. CaM adaptively merges caches scheduled for eviction back into the remaining caches, using a novel sampling strategy based on attention score prominence. This method mitigates output perturbation caused by cache eviction, preserving critical token information. Extensive experiments with LLaMA, OPT, and GPTNeoX models across various benchmarks demonstrate CaM's effectiveness in improving the performance of memory-efficient LLMs.
    - 摘要：这篇论文介绍了缓存合并（CaM），这是一种创新方法，旨在提高大型语言模型（LLMs）在推理过程中的内存效率，同时不影响性能。CaM自适应地将计划淘汰的缓存合并回剩余缓存中，使用基于注意力分数显著性的新采样策略。这种方法减轻了由缓存淘汰引起的输出扰动，保留了关键的token信息。在LLaMA、OPT和GPTNeoX模型上跨多个基准的广泛实验证明了CaM在提高内存高效LLMs性能方面的有效性。

- Flash-LLM: Enabling Cost-Effective and Highly-Efficient Large Generative Model Inference with Unstructured Sparsity
  - Label: <img src=https://img.shields.io/badge/unstructured-turquoise.svg> <img src=https://img.shields.io/badge/structured-blue.svg>
  - Author: Haojun Xia, Zhen Zheng, Yuchao Li, Donglin Zhuang, Zhongzhu Zhou, Xiafei Qiu, Yong Li, Wei Lin, Shuaiwen Leon Song
  - Link: https://arxiv.org/pdf/2309.10285
  - Code: https://github.com/AlibabaResearch/flash-llm
  - Pub: VLDB23
  - Summary: The paper proposes Flash-LLM, an efficient GPU library to support unstructured sparsity on tensor cores for large generative model inference. Flash-LLM uses a "Load-as-Sparse and Compute-as-Dense" strategy to address the memory bandwidth bottleneck while tolerating redundant computations on tensor cores. It includes a new sparse format, sparse-to-dense transformation, and a two-level overlapping strategy to enable high-performance unstructured sparse matrix multiplication on tensor cores. Extensive evaluations show that Flash-LLM significantly outperforms state-of-the-art solutions at both the kernel and end-to-end framework levels.
  - 摘要: 本文提出了Flash-LLM,这是一个在张量核上支持非结构化稀疏性的高效GPU库,用于大型生成式模型推理。Flash-LLM采用"以稀疏方式加载,以密集方式计算"的策略,以解决内存带宽瓶颈,同时容忍张量核上的冗余计算。它包括一种新的稀疏格式、稀疏到密集的转换,以及一种双层重叠策略,以实现张量核上高性能的非结构化稀疏矩阵乘法。广泛的评估表明,Flash-LLM在内核和端到端框架级别都显著优于最先进的解决方案。


- NASH: A Simple Unified Framework of Structured Pruning for Accelerating Encoder-Decoder Language Models
  - Label: <img src=https://img.shields.io/badge/structured-blue.svg>
  - Author: Jongwoo Ko, Seungjoon Park, Yujin Kim, Sumyeong Ahn, Du-Seong Chang, Euijai Ahn, Se-Young Yun
  - Link: https://arxiv.org/abs/2310.10054
  - Code: https://github.com/jongwooko/NASH-Pruning-Official
  - Pub: EMNLP23 Findings
  - Summary: The paper proposes a structured pruning framework called NASH (Narrow encoder and Shallow decoder) for accelerating encoder-decoder language models. The key insights are: (1) the number of decoder layers is the dominant factor for inference speedup, and (2) low sparsity in the pruned encoder network enhances generation quality. Based on these findings, NASH narrows the encoder and shortens the decoder networks. The encoder uses gradual L0 regularization pruning to induce low sparsity, while the decoder employs uniform layer selection to achieve faster inference speed. Extensive experiments on diverse generation and inference tasks validate the effectiveness of NASH in both speedup and output quality.
  - 摘要: 该论文提出了一个称为NASH(窄编码器和浅解码器)的结构化剪枝框架,用于加速编码器-解码器语言模型。主要发现包括:(1)解码器层数是推理加速的主要因素,(2)编码器网络的低稀疏性可以提高生成质量。基于这些发现,NASH缩小了编码器并缩短了解码器网络。编码器使用渐进的L0正则化剪枝来诱导低稀疏性,而解码器采用统一的层选择来实现更快的推理速度。广泛的实验验证了NASH在加速和输出质量方面的有效性。
 
- LoRAPrune: Structured Pruning Meets Low-Rank Parameter-Efficient Fine-Tuning
  - Label: <img src=https://img.shields.io/badge/structured-blue.svg> <img src=https://img.shields.io/badge/semi__structured-brightgreen.svg>
  - Author: Mingyang Zhang, Hao Chen, Chunhua Shen, Zhen Yang, Linlin Ou, Xinyi Yu, Bohan Zhuang
  - Link: https://arxiv.org/abs/2305.18403
  - Code: https://github.com/aim-uofa/LoRAPrune
  - Pub: ACL24 Findings
  - Summary: LoRAPrune proposes a new framework that delivers an accurate structured pruned model in a highly memory-efficient manner. It first designs a LoRA-guided pruning criterion, which uses the weights and gradients of LoRA, rather than the gradients of pre-trained weights, for importance estimation. LoRAPrune then integrates this criterion into an iterative pruning process, effectively removing redundant channels and heads. Experiments on LLaMA series models show that LoRAPrune outperforms existing pruning methods in terms of performance and memory efficiency.
  - 摘要: LoRAPrune提出了一个新的框架,能够以高内存效率的方式生成精确的结构化剪枝模型。它首先设计了一个基于LoRA的剪枝准则,使用LoRA的权重和梯度,而不是预训练权重的梯度来估计重要性。LoRAPrune然后将这个准则集成到一个迭代剪枝过程中,有效地移除了冗余的通道和头。在LLaMA系列模型上的实验表明,LoRAPrune在性能和内存效率方面都优于现有的剪枝方法。

- Sparse Fine-tuning for Inference Acceleration of Large Language Models
  - Label: <img src=https://img.shields.io/badge/unstructured-turquoise.svg>
  - Author: Eldar Kurtic, Denis Kuznedelev, Elias Frantar, Michael Goin, Dan Alistarh
  - Link: [arxiv.org/pdf/2306.02924v1](https://arxiv.org/pdf/2306.02924v1)
  - Code: https://github.com/IST-DASLab/SparseFinetuning
  - Pub: Arxiv
  - Summary: The paper studies efficient sparse fine-tuning for large language models (LLMs) across three applications: speech transcription using Whisper, machine translation using T5, and reasoning using the open GPT-type MPT model. The key contributions are:(1) Investigating fine-tuning sparse models obtained via SparseGPT using various losses, including a type of per-token l2 knowledge distillation called SquareHead, which consistently recovers accuracy even at high sparsities. (2) Showing that the resulting sparse models can be executed with inference speedups on both CPUs and GPUs by leveraging sparsity, including a GPU-aware N:M sparse format for generative inference.
  - 摘要: 本文研究了在语音转录、机器翻译和推理等三个应用中,对大型语言模型(LLM)进行高效稀疏微调的方法。主要贡献包括: (1) 探索使用包括一种称为SquareHead的每个token的l2知识蒸馏在内的各种损失函数,来微调通过SparseGPT获得的稀疏模型,这种方法能够在高稀疏度下保持准确性。(2) 展示通过利用稀疏性,所得到的稀疏模型可以在CPU和GPU上实现推理加速,包括为生成式推理开发了一种GPU感知的N:M稀疏格式。

- The Cost of Down-Scaling Language Models: Fact Recall Deteriorates before In-Context Learning
  - Labels: <img src=https://img.shields.io/badge/unstructured-turquoise.svg> 
  - Authors: Tian Jin, Nolan Clement, Xin Dong, Vaishnavh Nagarajan, Michael Carbin, Jonathan Ragan-Kelley, Gintare Karolina Dziugaite
  - Link: https://arxiv.org/abs/2310.04680
  - Code: None
  - Pub: None
  - Summary: The paper studies the effects of scaling (pruning and dense scaling) on two core capabilities of large language models (LLMs): (1) recalling facts presented during pre-training, and (2) processing information presented in-context during inference. They find that reducing model size by more than 30% significantly decreases the ability to recall facts, while a 60-70% reduction largely preserves the ability to process in-context information, such as retrieving answers from a long context or learning parameterized functions from in-context examples. This disparity in the effects of scaling holds for both pruning and dense scaling, suggesting an inherent difference in how these two abilities are affected by model size.
  - 摘要: 该论文研究了缩放(修剪和密集缩放)对大型语言模型(LLM)两个核心能力的影响:(1)回忆预训练期间呈现的事实,(2)在推理过程中处理上下文中呈现的信息。他们发现,将模型大小减少30%以上会显著降低回忆事实的能力,而减少60-70%则基本保留了从上下文信息中处理信息的能力,如从长上下文中检索答案或从上下文示例中学习参数化函数。这种缩放效果的差异在修剪和密集缩放中都存在,表明这两种能力受模型大小影响的内在差异。
 
- One-Shot Sensitivity-Aware Mixed Sparsity Pruning for Large Language Models
  - Label: <img src=https://img.shields.io/badge/structured-blue.svg>
  - Author: Hang Shao, Bei Liu, Bo Xiao, Ke Zeng, Guanglu Wan, Yanmin Qian
  - Link: https://arxiv.org/pdf/2310.09499
  - Code: https://github.com/talkking/MixGPT
  - Pub: ICASSP24
  - Summary: The paper proposes a one-shot sensitivity-aware mixed sparsity pruning method for large language models. The key aspects are:
    1. An improved saliency criterion (ISC) that combines the OBS and OBD approaches to better select weights to prune.
    2. A sensitivity-aware mixed sparsity pruning strategy that allocates different sparsity levels to different layers and weight matrices based on their sensitivity estimated from the Hessian matrix.
    3. Experiments on LLaMA, LLaMA-2, and Baichuan models show the proposed method outperforms previous one-shot pruning approaches in terms of perplexity and zero-shot downstream task performance.
  - 摘要: 本文提出了一种一次性敏感性感知混合稀疏剪枝方法用于大型语言模型。主要包括以下几个关键点:
    1. 提出了一种改进的显著性准则(ISC),结合了OBS和OBD方法,更好地选择要剪枝的权重。
    2. 提出了一种基于敏感性的混合稀疏剪枝策略,根据来自Hessian矩阵估计的敏感性,为不同层和权重矩阵分配不同的稀疏度。
    3. 在LLaMA、LLaMA-2和Baichuan模型上的实验表明,所提出的方法在困惑度和零样本下游任务性能方面优于之前的一次性剪枝方法。

- Compact Language Models via Pruning and Knowledge Distillation
  - Labels: <img src=https://img.shields.io/badge/unstructured-turquoise.svg> <img src=https://img.shields.io/badge/structured-blue.svg>
  - Authors: Saurav Muralidharan, Sharath Turuvekere Sreenivas, Raviraj Joshi, Marcin Chochowski, Mostofa Patwary, Mohammad Shoeybi, Bryan Catanzaro, Jan Kautz, Pavlo Molchanov
  - Link: https://www.arxiv.org/pdf/2407.14679
  - Code: https://github.com/NVlabs/Minitron
  - Pub: None
  - Summary: The paper proposes a method to compress large language models (LLMs) by pruning and knowledge distillation. It explores structured pruning across multiple axes (depth, width, attention, and embeddings) and combines it with data-efficient retraining using knowledge distillation. The method is applied to compress the Nemotron-4 15B model, resulting in the MINITRON family of smaller models that outperform similarly-sized models while requiring significantly fewer training tokens.
  - 摘要: 该论文提出了一种通过修剪和知识蒸馏来压缩大型语言模型(LLM)的方法。它探索了跨多个维度(深度、宽度、注意力和嵌入)的结构化修剪,并将其与使用知识蒸馏的数据高效再训练相结合。该方法被应用于压缩Nemotron-4 15B模型,得到了MINITRON系列更小的模型,这些模型在需要的训练样本大大减少的情况下,仍然优于同等大小的模型。

- APT: Adaptive Pruning and Tuning Pretrained Language Models for Efficient Training and Inference
  - Label: <img src=https://img.shields.io/badge/structured-blue.svg> <img src=https://img.shields.io/badge/semi__structured-brightgreen.svg>
  - Author: Bowen Zhao, Hannaneh Hajishirzi, Qingqing Cao
  - Link: https://arxiv.org/pdf/2401.12200
  - Code: https://github.com/ROIM1998/APT
  - Pub: ICML 2024 Oral
  - Summary: APT adaptively prunes and tunes parameters of pretrained language models to improve both training and inference efficiency. It uses an outlier-aware salience scoring function to identify and prune less important parameter blocks during early fine-tuning. APT also dynamically adds more tuning parameters to recover the pruned model's performance. Experiments show APT can prune 60% of RoBERTa and T5 parameters while maintaining 98% task performance, and prune 70% of LLaMA parameters while preserving 86.4% performance. APT also speeds up fine-tuning by up to 8x and reduces training memory by up to 70%.
  - 摘要: APT自适应地剪枝和调整预训练语言模型的参数,以提高训练和推理效率。它使用一种考虑异常值的显著性评分函数,在早期微调期间识别并剪枝不重要的参数块。APT还动态添加更多调优参数来恢复被剪枝模型的性能。实验表明,APT可以在保持98%任务性能的情况下剪枝60%的RoBERTa和T5参数,并在保持86.4%性能的情况下剪枝70%的LLaMA参数。APT还可将微调速度提高高达8倍,并将训练内存减少高达70%。

- Not All Experts are Equal: Efficient Expert Pruning and Skipping for Mixture-of-Experts Large Language Models
  - Label: <img src=https://img.shields.io/badge/unstructured-turquoise.svg> <img src=https://img.shields.io/badge/structured-blue.svg>
  - Author: Xudong Lu, Qi Liu, Yuhui Xu, Aojun Zhou, Siyuan Huang, Bo Zhang, Junchi Yan, Hongsheng Li
  - Link: [arxiv.org/pdf/2402.14800v2](https://arxiv.org/pdf/2402.14800v2)
  - Code: https://github.com/Lucky-Lance/Expert_Sparsity
  - Pub: ACL2024 
  - Summary: The paper proposes post-training methods for expert pruning and dynamic expert skipping to enhance the deployment efficiency of Mixture-of-Experts (MoE) large language models (LLMs). The expert pruning method permanently removes less important experts to reduce memory usage, while the dynamic expert skipping method selectively skips certain experts during inference to improve inference speed, without compromising model performance. Experiments on the Mixtral 8x7B model demonstrate significant reductions in memory usage and inference speedups.
  - 摘要: 本文提出了用于专家修剪和动态专家跳过的后训练方法,以提高混合专家(MoE)大型语言模型(LLM)的部署效率。专家修剪方法永久删除不太重要的专家以减少内存使用,而动态专家跳过方法在推理过程中选择性地跳过某些专家以提高推理速度,而不会影响模型性能。在Mixtral 8x7B模型上的实验表明,内存使用量显著减少,推理速度也有所提高。

Part1:

- SQFT: Low-cost Model Adaptation in Low-precision Sparse Foundation Models
  - Label: <img src=https://img.shields.io/badge/structured-blue.svg>
  - Author: J. Pablo Muñoz, Jinjie Yuan, Nilesh Jain
  - Link: https://arxiv.org/pdf/2410.03750v1.pdf
  - Code: https://github.com/IntelLabs/Hardware-Aware-Automated-Machine-Learning
  - Pub: EMNLP 2024 Findings
  - Summary: SQFT is an end-to-end solution for low-precision sparse parameter-efficient fine-tuning of large pre-trained models. It includes stages for sparsification, quantization, fine-tuning with neural low-rank adapter search (NLS), and sparse parameter-efficient fine-tuning (SparsePEFT) with optional quantization-awareness. SQFT addresses the challenges of merging sparse/quantized weights with dense adapters by preserving sparsity and handling different numerical precisions.
  - 摘要: SQFT是一个端到端的解决方案,用于对大型预训练模型进行低精度稀疏参数高效微调。它包括稀疏化、量化、使用神经低秩适配器搜索(NLS)进行微调,以及可选的带有量化感知的稀疏参数高效微调(SparsePEFT)等阶段。SQFT通过保留稀疏性和处理不同的数值精度,解决了将稀疏/量化权重与密集适配器合并的挑战。


- Retrospective Sparse Attention for Efficient Long-Context Generation
    - Label: <img src=https://img.shields.io/badge/pruning-turquoise.svg >
    - Author: Seonghwan Choi, Beomseok Kang, Dongwon Jo, et al.
    - Link: http://arxiv.org/pdf/2508.09001v1 
    - Code: Not available
    - Pub: Arxiv 2025
    - Summary: This paper introduces RetroAttention, a novel KV cache update technique that retrospectively revises past attention outputs using new KV entries from subsequent decoding steps to improve long-context generation in LLMs. By maintaining a lightweight output cache, RetroAttention allows past queries to access more relevant context, outperforming state-of-the-art KV compression methods by increasing effective KV exposure and accuracy on long-generation benchmarks. #LLM #Pruning #SparseAttention #LongContext #KVcache #EfficientInference
    - 摘要: 该论文提出了一种名为RetroAttention的新型KV缓存更新技术，旨在解决长文本生成中因KV缓存线性增长导致的推理瓶颈问题。RetroAttention通过回顾性地利用后续解码步骤中新到达的KV条目来修正过去的注意力输出，从而提高有效KV暴露并显著提升生成精度，在长文本生成任务上优于现有KV压缩方法。#剪枝 #稀疏注意力 #长文本生成 #KV缓存 #大语言模型


- Classifier Language Models: Unifying Sparse Finetuning and Adaptive   Tokenization for Specialized Classification Tasks
    - Label: <img src=https://img.shields.io/badge/pruning-turquoise.svg >
    - Author: Adit Krishnan, Chu Wang, Chris Kong
    - Link: http://arxiv.org/pdf/2508.08635v1 
    - Code: Not available
    - Pub: Arxiv 2025
    - Summary: This paper introduces a token-driven sparse finetuning method for adapting small language models to specialized classification tasks by identifying and finetuning a sensitive subset of model parameters based on task-specific token constructs, avoiding the introduction of new parameters like LoRA. The proposed approach outperforms full finetuning, LoRA, layer selection, and prefix tuning on five diverse semantic classification tasks by identifying highly relevant semantic tokens. #Pruning #SparseFinetuning #LLM #Tokenization #SemanticClassification
    - 摘要: 这篇论文提出了一种针对特定分类任务的token驱动的稀疏微调策略，用于适配小型语言模型。该方法通过识别并微调数据集中与任务相关的关键token对应的模型参数子集，在不引入额外参数的情况下，实现了优于全参数微调、LoRA等方法的性能，并在多个语义分类任务上取得了显著效果。#剪枝 #稀疏微调 #大语言模型 #语义分类 #token驱动


- OverFill: Two-Stage Models for Efficient Language Model Decoding
    - Label: <img src=https://img.shields.io/badge/pruning-turquoise.svg >
    - Author: Woojeong Kim, Junxiong Wang, Jing Nathan Yan, et al.
    - Link: http://arxiv.org/pdf/2508.08446v1 
    - Code: Not available
    - Pub: Arxiv 2025
    - Summary: The paper introduces OverFill, a two-stage approach to LLM inference that decouples the prefill and decode stages, using a full model for the compute-intensive prefill and a dense pruned model for the memory-intensive decode. This strategy allows for improved generation quality with minimal latency overhead, achieving significant performance gains over similarly sized pruned models and matching the performance of models trained from scratch with less data. #Pruning #EfficientInference #LLM #TwoStageModel
    - 摘要: 这篇论文提出了OverFill，一种两阶段的大语言模型解码方法，旨在提高推理效率。OverFill利用完整模型进行预填充（prefill）阶段，并行处理输入，然后切换到剪枝后的模型进行解码阶段，从而在保证生成质量的同时，显著降低延迟。实验结果表明，OverFill在多个基准测试中优于同等大小的剪枝模型，甚至可以媲美从头训练的模型，同时减少了训练数据需求。#剪枝 #两阶段模型 #大语言模型 #推理优化 #效率提升


- Speed Always Wins: A Survey on Efficient Architectures for Large   Language Models
    - Label: <img src=https://img.shields.io/badge/pruning-turquoise.svg >
    - Author: Weigao Sun, Jiaxi Hu, Yucheng Zhou, et al.
    - Link: http://arxiv.org/pdf/2508.09834v1 
    - Code: Not available
    - Pub: Arxiv 2025
    - Summary: This survey paper provides a comprehensive overview of efficient LLM architectures, focusing on techniques that address the computational limitations of traditional transformers. It categorizes and examines methods like linear and sparse sequence modeling, efficient attention mechanisms, Mixture-of-Experts, and hybrid architectures, highlighting their applications across modalities to enable scalable and resource-aware foundation models. The paper serves as a blueprint for modern efficient LLM design, motivating future research in versatile AI systems. #EfficientLLM #LLMArchitectures #ModelEfficiency #SparseModels #MoE
    - 摘要: 这篇综述论文系统性地考察了提升大语言模型效率的创新架构，重点关注线性/稀疏序列建模、高效注意力机制变体、稀疏混合专家模型以及混合模型架构等技术。通过对这些技术的分类和讨论，论文旨在为构建更高效、通用的AI系统提供蓝图，并推动未来相关研究。#大语言模型 #高效架构 #模型压缩

- MoIIE: Mixture of Intra- and Inter-Modality Experts for Large Vision   Language Models
    - Label: <img src=https://img.shields.io/badge/pruning-turquoise.svg >
    - Author: Dianyi Wang, Siyuan Wang, Zejun Li, et al.
    - Link: http://arxiv.org/pdf/2508.09779v1 
    - Code: Not available
    - Pub: Arxiv 2025
    - Summary: This paper introduces MoIIE, a Mixture of Intra- and Inter-Modality Experts, to improve the efficiency of large vision-language models by routing tokens to modality-specific and shared cross-modal experts. Using a two-stage training strategy, MoIIE achieves comparable or superior performance to existing MoE-LLMs with fewer activated parameters, demonstrating effectiveness and generality across different data scales and LLM backbones. #MoE #VisionLanguage #SparseModels #ParameterEfficient #LLM
    - 摘要: 该论文提出了一种名为MoIIE (Mixture of Intra- and Inter-Modality Experts) 的新型视觉语言大模型稀疏化方法，通过混合模态内和模态间专家，有效提升了模型参数效率。MoIIE根据token的模态类型进行专家路由，使其分别进入模态内专家和共享的模态间专家，从而联合学习丰富的模态内特征和跨模态交互。实验结果表明，MoIIE模型在激活较少参数的情况下，性能可以媲美甚至超越现有的大型MoE视觉语言模型。#剪枝 #稀疏化 #混合专家模型 #视觉语言模型 #多模态学习

- HierMoE: Accelerating MoE Training with Hierarchical Token Deduplication   and Expert Swap
    - Label: <img src=https://img.shields.io/badge/pruning-turquoise.svg >
    - Author: Wenxiang Lin, Xinglin Pan, Lin Zhang, et al.
    - Link: http://arxiv.org/pdf/2508.09591v1 
    - Code: Not available
    - Pub: Arxiv 2025
    - Summary: This paper introduces HierMoE, a novel system that accelerates the training of Mixture-of-Experts (MoE) models by employing topology-aware techniques: hierarchical token deduplication to reduce communication overhead and expert swapping to balance workloads across GPUs. By theoretically modeling optimal strategies for token duplication and expert swapping, HierMoE achieves significant speedups in communication ($1.55\times$ to $3.32\times$) and end-to-end training ($1.18\times$ to $1.27\times$) compared to existing MoE training systems. #MoE #LLM #DistributedTraining #SparseModels #HierarchicalOptimization
    - 摘要: 该论文提出了HierMoE，一种通过层级Token去重和专家交换加速MoE模型训练的方法。 HierMoE通过减少通信流量和平衡GPU间负载，显著提升了MoE模型在分布式系统中的可扩展性，实验证明其在DeepSeek-V3和Qwen3-30B-A3B模型上相比现有系统实现了更快的通信速度和端到端训练速度。 #MoE #加速训练 #分布式训练 #通信优化 #负载均衡 #大语言模型

- NeuronTune: Fine-Grained Neuron Modulation for Balanced Safety-Utility   Alignment in LLMs
    - Label: <img src=https://img.shields.io/badge/pruning-turquoise.svg >
    - Author: Birong Pan, Mayi Xu, Qiankun Pi, et al.
    - Link: http://arxiv.org/pdf/2508.09473v1 
    - Code: Not available
    - Pub: Arxiv 2025
    - Summary: NeuronTune introduces a fine-grained neuron modulation framework for LLMs that balances safety and utility by selectively amplifying safety-critical neurons and suppressing utility-preserving neurons identified through attribution. This meta-learning approach outperforms existing layer-wise methods, achieving superior safety against malicious attacks and reducing unnecessary refusals, all while maintaining high-quality text generation and general task performance. #LLMPruning #NeuronModulation #SafetyAlignment #MetaLearning #SparseNetworks
    - 摘要: NeuronTune 提出了一种细粒度的神经元调制框架，旨在解决大语言模型在安全性和实用性对齐方面的挑战。该方法通过归因识别安全关键和实用性保留的神经元，并利用元学习自适应地增强安全神经元的激活，抑制实用神经元的激活，从而在提升模型安全性的同时保持其卓越的实用性，显著优于现有技术。#剪枝 #神经元调制 #安全性对齐 #实用性 #大语言模型

- EGGS-PTP: An Expander-Graph Guided Structured Post-training Pruning   Method for Large Language Models
    - Label: <img src=https://img.shields.io/badge/pruning-turquoise.svg >
    - Author: Omar Bazarbachi, Zijun Sun, Yanning Shen
    - Link: http://arxiv.org/pdf/2508.09471v1 
    - Code: Not available
    - Pub: Arxiv 2025
    - Summary: This paper introduces EGGS-PTP, a novel structured post-training pruning method for LLMs that uses expander graphs to guide N:M sparsity patterns, ensuring efficient information flow and preserving accuracy after pruning. EGGS-PTP achieves significant acceleration and memory savings while outperforming existing structured pruning techniques across various LLMs. #Pruning #Sparse #LLM #Expanders #ModelCompression
    - 摘要: 这篇论文提出了EGGS-PTP，一种基于扩展图引导的结构化后训练剪枝方法，旨在解决大语言模型部署时的计算和内存挑战。该方法利用图论指导N:M结构化剪枝，通过扩展图确保剪枝后网络的信息流动，从而在显著减少模型大小和计算量的同时，保持甚至超越现有结构化剪枝技术的精度表现。#剪枝 #结构化剪枝 #大语言模型 #后训练剪枝 #扩展图


- Communication Efficient LLM Pre-training with SparseLoCo
    - Label: <img src=https://img.shields.io/badge/pruning-turquoise.svg >
    - Author: Amir Sarfi, Benjamin Thérien, Joel Lidin, et al.
    - Link: http://arxiv.org/pdf/2508.15706v1 
    - Code: Not available
    - Pub: Arxiv 2025
    - Summary: This paper introduces SparseLoCo, a novel communication-efficient training algorithm for LLMs that combines Top-k sparsification and quantization to achieve extreme compression ratios (1-3% sparsity, 2-bit quantization). SparseLoCo approximates outer momentum locally with error feedback and sparse aggregation, outperforming full-precision DiLoCo while significantly reducing communication costs in bandwidth-constrained LLM pre-training scenarios. #Pruning #SparseTraining #LLM #DistributedTraining #CommunicationEfficiency
    - 摘要: 该论文提出了一种名为SparseLoCo的通信高效LLM预训练算法，通过结合Top-k稀疏化和量化技术，在带宽受限的环境下大幅降低通信成本。SparseLoCo的关键在于利用误差反馈近似局部外层动量，并发现稀疏聚合实际上可以提升模型性能，最终实现了优于全精度DiLoCo的性能，同时达到1-3%的稀疏度和2-bit量化的极端压缩率。#剪枝 #稀疏 #大语言模型 #通信高效 #分布式训练


- Flash Sparse Attention: An Alternative Efficient Implementation of   Native Sparse Attention Kernel
    - Label: <img src=https://img.shields.io/badge/pruning-turquoise.svg >
    - Author: Ran Yan, Youhe Jiang, Binhang Yuan
    - Link: http://arxiv.org/pdf/2508.18224v1 
    - Code: Not available
    - Pub: Arxiv 2025
    - Summary: This paper introduces Flash Sparse Attention (FSA), an optimized kernel implementation for Native Sparse Attention (NSA) that addresses the limitations of the original NSA kernel's efficiency with smaller Grouped Query Attention (GQA) sizes common in modern LLMs. FSA achieves significant kernel-level latency reduction (up to 3.5x) and end-to-end training/prefill speedups (up to 1.25x and 1.36x, respectively) compared to the vanilla NSA kernel, making sparse attention more practical for a wider range of LLM architectures. #Pruning #SparseAttention #LLM #KernelOptimization #EfficientComputation
    - 摘要: 这篇论文提出了Flash Sparse Attention (FSA)，一种改进的稀疏注意力核实现，旨在解决原生稀疏注意力(NSA)在小GQA组大小下效率低下的问题。FSA通过优化的内核设计，在现代GPU上实现了更广泛的LLM适用性，并显著提升了训练和推理速度，最高可达3.5倍的内核延迟降低和1.25倍的端到端训练加速。#剪枝 #稀疏注意力 #大语言模型 #GPU优化 #内核优化- Scaling Law for Post-training after Model Pruning


- Federated Fine-Tuning of Sparsely-Activated Large Language Models on   Resource-Constrained Devices
    - Label: <img src=https://img.shields.io/badge/pruning-turquoise.svg >
    - Author: Fahao Chen, Jie Wan, Peng Li, et al.
    - Link: http://arxiv.org/pdf/2508.19078v1 
    - Code: Not available
    - Pub: Arxiv 2025
    - Summary: This paper introduces FLUX, a novel system for federated fine-tuning of Mixture-of-Experts (MoE) based LLMs on resource-constrained devices. FLUX employs quantization-based local profiling, adaptive layer-aware expert merging, and dynamic expert role assignment to minimize time-to-accuracy, achieving significant speedups compared to existing methods. #FederatedLearning #LLMPruning #MoE #SparseTraining #ResourceConstrained
    - 摘要: 该研究提出了一种名为FLUX的系统，旨在解决资源受限设备上联邦微调MoE结构大语言模型的难题。FLUX通过量化局部剖析估计专家激活，自适应地进行分层专家合并以降低资源消耗，并采用动态专家角色分配策略平衡调优和非调优专家，从而显著提升了联邦微调的效率，实验表明FLUX在时间-精度方面优于现有方法，最高可达4.75倍的加速。#联邦学习 #MoE #剪枝 #稀疏激活 #大语言模型 #资源受限设备    - Label: <img src=https://img.shields.io/badge/semi_structured-brightgreen.svg > <img src=https://img.shields.io/badge/structured-blue.svg >
    - Authors: Xiaodong Chen, Yuxuan Hu, Jing Zhang, Xiaokang Zhang, Cuiping Li, Hong Chen
    - Link: https://arxiv.org/abs/2411.10272v1
    - Code: Not available
    - Summary: This paper investigates the post-training requirements of pruned Large Language Models (LLMs).  The authors propose a scaling law to predict the optimal amount of post-training data needed after pruning, considering factors like the pre- and post-pruning parameter counts and the post-training token count.  Experiments using Llama-3 and Qwen-2.5 series models with depth, width, and 2:4 semi-structured pruning demonstrate that higher pruning ratios require more post-training data, while larger LLMs need less.  The scaling law, derived from smaller LLMs, is shown to extrapolate effectively to larger models, offering a practical guideline for optimizing post-training data usage and reducing resource consumption.
    - 摘要：本文研究了剪枝大型语言模型 (LLM) 的后期训练需求。作者提出了一种缩放定律，用于预测剪枝后所需的最佳后期训练数据量，考虑了剪枝前后的参数计数和后期训练的标记计数等因素。使用 Llama-3 和 Qwen-2.5 系列模型进行深度、宽度和 2:4 半结构化剪枝的实验表明，较高的剪枝率需要更多的后期训练数据，而较大的 LLM 需要较少的数据。从较小的 LLM 推导出的缩放定律可以有效地推断到较大的模型，为优化后期训练数据的使用和减少资源消耗提供了实用的指导。


- Finding Transformer Circuits with Edge Pruning
    - Label: <img src=https://img.shields.io/badge/structured-blue.svg> 
    - Authors: Adithya Bhaskar, Alexander Wettig, Dan Friedman, Danqi Chen
    - Link: https://arxiv.org/pdf/2406.16778
    - Pub: NeurIPS 24 Spotlight
    - Code: https://github.com/princeton-nlp/Edge-Pruning
    - Summary: This paper introduces **Edge Pruning**, a novel and scalable method for automated **circuit discovery** in **Transformers**. The authors frame circuit discovery as an optimization problem and solve it using gradient-based pruning techniques, specifically targeting the **edges** between components rather than the components themselves. They replace the standard residual stream with a **disentangled residual stream**, allowing them to introduce **learnable binary masks** on the edges. These masks are optimized using techniques like **L0 regularization** to induce sparsity. The method is evaluated on its ability to faithfully describe full model behavior, recover ground-truth circuits in **Tracr** models, scale to large datasets (up to 100K examples), and find extremely sparse circuits in large models (up to 13 billion parameters). Edge Pruning outperforms previous methods like **ACDC** and **EAP** in terms of faithfulness and task performance on standard circuit-finding tasks, especially on complex tasks like multi-template **IOI**. It also perfectly recovers ground-truth circuits in two Tracr models. Furthermore, Edge Pruning is demonstrated to scale effectively to **CodeLlama-13B**, revealing highly sparse circuits (over 99.96% sparsity) that match the full model's performance in a case study comparing instruction prompting and in-context learning.
    - 摘要：本文介绍了一种名为**边缘剪枝（Edge Pruning）** 的新型可扩展方法，用于在 **Transformers** 中自动发现**回路（circuit）**。作者将回路发现视为一个优化问题，并使用基于梯度的剪枝技术来解决它，特别针对组件之间的**边**而不是组件本身。他们用一个**解耦的残差流（disentangled residual stream）** 替换了标准的残差流，允许他们在边上引入**可学习的二进制掩码（learnable binary masks）**。这些掩码使用 **L0 正则化**等技术进行优化以诱导稀疏性。该方法在以下几个方面进行了评估：忠实地描述完整模型行为的能力，恢复 **Tracr** 模型中已知的真实回路的能力，扩展到大型数据集（最多 10 万个示例）的能力，以及在大型模型（最多 130 亿个参数）中找到极其稀疏的回路的能力。在标准的回路发现任务上，边缘剪枝在忠实性和任务性能方面优于 **ACDC** 和 **EAP** 等先前的方法，特别是在像多模板 **IOI** 这样的复杂任务上。它还完美地恢复了两个 Tracr 模型中的真实回路。此外，边缘剪枝被证明可以有效地扩展到 **CodeLlama-13B**，在一个比较指令提示和上下文学习的案例研究中揭示了与完整模型性能相匹配的高度稀疏回路（超过 99.96% 的稀疏性）。


- Probe Pruning: Accelerating LLMs through Dynamic Pruning via Model-Probing
    - Label: <img src=https://img.shields.io/badge/structured-blue.svg >
    - Authors: Anonymous author 
    - Link: https://openreview.net/pdf?id=WOt1owGfuN
    - Code: Not available
    - Pub: Under review ICLR
    - Summary: This paper introduces **Probe Pruning (PP)**, a novel framework for online, dynamic, structured pruning of Large Language Models (LLMs) in a batch-wise manner. PP is motivated by the observation that not all samples and tokens contribute equally to the model's output. It identifies crucial weights by probing a small portion of each batch, enabling tailored dynamic pruning for different batches. PP comprises three main stages: **probing**, **history-informed pruning**, and **full inference**. In the probing stage, PP selects a small yet crucial set of hidden states, based on **residual importance**, to run a few model layers ahead. During history-informed pruning, PP strategically integrates the probing states with historical states collected from previous batches. It then structurally prunes weights based on the integrated states and the **PP importance score**, a metric designed to assess the importance of each weight channel in maintaining performance. In the final stage, full inference is conducted on the remaining weights. A key advantage of PP is its compatibility with existing models, as it operates without requiring additional neural network modules or fine-tuning. Comprehensive evaluations on **LLaMA-2/3** and **OPT** models show that even minimal probing (using just 1.5% of FLOPs) can substantially enhance the efficiency of structured pruning. For instance, when evaluated on **LLaMA-2-7B** with **WikiText2**, PP achieves a 2.56 × lower ratio of performance degradation per unit of runtime reduction compared to the state-of-the-art method at a 40% pruning ratio. The paper highlights that relying solely on calibration datasets for pruning can introduce biases and that PP effectively addresses batch-dependent outlier properties in LLMs without extensive fine-tuning.
    - 摘要：本文提出了一种名为 **Probe Pruning (PP)** 的新型框架，用于以批处理方式对大型语言模型 (LLMs) 进行在线、动态、结构化剪枝。PP 的灵感来自于观察到并非所有样本和标记对模型输出的贡献均等。它通过探测每个批次的一小部分来识别关键权重，从而实现针对不同批次的定制动态剪枝。PP 主要包括三个阶段：**探测**、**历史信息剪枝**和**完整推理**。在探测阶段，PP 根据**残差重要性**选择一小组但至关重要的隐藏状态，以提前运行几个模型层。在历史信息剪枝阶段，PP 将探测状态与从先前批次收集的历史状态进行策略性整合。然后，它根据整合的状态和 **PP 重要性分数**（一种专门用于评估每个权重通道在保持性能方面重要性的指标）对权重进行结构化剪枝。在最后阶段，对剩余的权重进行完整推理。PP 的一个关键优势是它与现有模型的兼容性，因为它无需额外的神经网络模块或微调即可运行。对 **LLaMA-2/3** 和 **OPT** 模型的综合评估表明，即使是最小的探测（仅使用 1.5% 的 FLOPs）也能显著提高结构化剪枝的效率。例如，当使用 **WikiText2** 在 **LLaMA-2-7B** 上进行评估时，在 40% 的剪枝率下，PP 的性能下降与运行时间减少的比率比最先进的方法低 2.56 倍。本文强调，仅依靠校准数据集进行剪枝可能会引入偏差，而 PP 可以有效地解决 LLMs 中批次相关的异常值属性，而无需进行大量的微调。
