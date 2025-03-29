<div align="center">
<img src="./assets/logo.png" width="250"/>
</div>

# Reason-RFT: Reinforcement Fine-Tuning for Visual Reasoning.

<p align="center">
        </a>&nbsp&nbsp⭐️ <a href="https://tanhuajie.github.io/ReasonRFT/">Project</a></a>&nbsp&nbsp | &nbsp&nbsp🤗 <a href="">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🌎 <a href="">Dataset</a>&nbsp&nbsp | &nbsp&nbsp📑 <a href="https://arxiv.org/abs/2503.20752">Paper</a>&nbsp&nbsp | &nbsp&nbsp💬 <a href="./assets/wechat.png">WeChat</a>
</p>

<p align="center">
</a>&nbsp&nbsp🤖 <a href="https://github.com/FlagOpen/RoboBrain/">RoboBrain</a>: Aim to Explore ReasonRFT Paradigm to Enhance RoboBrain's Embodied Reasoning Capabilities.
</p>

## 🔥 Overview
Visual reasoning abilities play a crucial role in understanding complex multimodal data, advancing both domain-specific applications and artificial general intelligence (AGI).
Existing methods improve VLM reasoning via Chain-of-Thought (CoT) supervised fine-tuning, using meticulously annotated training data to enhance visual reasoning capabilities.
However, this training paradigm may lead to overfitting and cognitive rigidity, restricting the model's ability to transfer visual reasoning skills across domains and limiting its real-world applicability.
To address these limitations, we propose **Reason-RFT**, a novel reinforcement fine-tuning framework that significantly enhances generalization capabilities in visual reasoning tasks.
**Reason-RFT** introduces a two-phase training framework for visual reasoning: (1) Supervised Fine-Tuning (SFT) with curated Chain-of-Thought (CoT) data activates the reasoning potential of Vision-Language Models (VLMs), followed by (2) Group Relative Policy Optimization (GRPO)-based reinforcement learning that generates multiple reasoning-response pairs, significantly enhancing generalization in visual reasoning tasks.
To evaluate \textit{Reason-RFT}'s visual reasoning capabilities, we reconstructed a comprehensive dataset spanning visual counting, structure perception, and spatial transformation, serving as a benchmark to systematically assess visual cognition, geometric understanding, and spatial generalization.
Experimental results demonstrate Reasoning-RFT's three key advantages: **(1) Performance Enhancement**: achieving state-of-the-art results across multiple tasks, outperforming most mainstream open-source and proprietary models; 
**(2) Generalization Superiority**: consistently maintaining robust performance across diverse tasks and domains, outperforming alternative training paradigms; 
**(3) Data Efficiency**: excelling in few-shot learning scenarios while surpassing full-dataset SFT baselines; 
**Reason-RFT** introduces a novel paradigm in visual reasoning, significantly advancing multimodal research.

<div align="center">
<img src="./assets/overview.png" />
</div>

## <a id="RoadMap"> 🎯 RoadMap</a>

- **`Support different VLMs`**: [RoboBrain](https://github.com/FlagOpen/RoboBrain/), [Qwen2-VL series](https://github.com/QwenLM/Qwen2.5-VL/), [Llava-VL series](https://github.com/LLaVA-VL/LLaVA-NeXT).
- **`Support General Visual Reasoning Tasks`**: 
    - Data generation and preparation: Please refer to [General Visual Reasoning Tasks](#GeneralVisualTasks).
    - Training and evaluating for **Visual Counting**: Please refer to [Visual Counting Section](#Visual_Counting).
    - Training and evaluating for **Struction Perception**: Please refer to [Struction Perception Section](#Struction_Perception).
    - Training and evaluating for **Spatial Transformation**: Please refer to [Spatial Transformation Section](#Spatial_Transformation).
- **`Support Embodied Visual Reasoning Tasks`**: 
    - Data generation and preparation: Please refer to [Embodied Visual Reasoning Tasks](#EmbodiedVisualReasoningTasks).
    - Training and evaluating for **Embodied Planning**: Please refer to [Embodied Planning Section](#Embodied_Planning).
    - Training and evaluating for **Embodied Affordance**: Please refer to [Embodied Affordance Section](#Embodied_Affordance).
    - Training and evaluating for **Embodied Trajectory**: Please refer to [Embodied Trajectory Section](#Embodied_Trajectory).
- **`Support HF/VLLM Inference`**: Please see [Inference Section](#Inference) for detail.


## 🗞️ News

- **`2025-03-29`**: 🌍 We have released the [repository](https://github.com/tanhuajie/Reason-RFT/) and [roadmap](#RoadMap) for **Reason-RFT**.
- **`2025-03-26`**: 📑 We have released our initial [ArXiv paper]((https://arxiv.org/abs/2503.20752/)) of **Reason-RFT**.


## <a id="GeneralVisualTasks"> 🎲 General Visual Reasoning Tasks</a>

***Coming Soon ...***


## <a id="EmbodiedVisualReasoningTasks"> 🤖 Embodied Visual Reasoning Tasks</a>
***Coming Soon ...***


## 📑 Citation
If you find this project useful, welcome to cite us.
```bib
@article{tan2025reasonrft,
    title={Reason-RFT: Reinforcement Fine-Tuning for Visual Reasoning}, 
    author={Huajie Tan and Yuheng Ji and Xiaoshuai Hao and Minglan Lin and Pengwei Wang and Zhongyuan Wang and Shanghang Zhang},
    journal={arXiv preprint arXiv:2503.20752},
    year={2025}
}
```
