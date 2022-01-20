---
layout: default
title: Inspiring Topics
nav_order: 5
toc_list: true
last_modified_date: Jan 19 2022
permalink: /analysis/
---

# Inspiring Topics about Structured Knowledge Grounding
{: .no_toc }

{: .fs-5 .fw-300 }
Here we present a collection of papers with inspiring topics that may help structured knowledge grounding. 


---

## Table of contents
{: .no_toc .text-delta .fs-4 style="font-weight:800"}

- TOC
{:toc}



## Multi-task learning

{: .fs-4 .fw-800 .text-blue-100}
**📜 A Joint Many-Task Model: Growing a Neural Network for Multiple NLP Tasks**. <br> ✍ Kazuma Hashimoto, Caiming Xiong, Yoshimasa Tsuruoka, Richard Socher
 *(EMNLP 2017)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/1611.01587){: .btn .btn-blue .mr-1 target="_blank" } 
   [Semantic Scholar](https://www.semanticscholar.org/paper/A-Joint-Many-Task-Model%3A-Growing-a-Neural-Network-Hashimoto-Xiong/ade0c116120b54b57a91da51235108b75c28375a){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

<details markdown="block">
  <summary>Abstract</summary>
  {: .fs-3 .text-delta .text-blue-100}
  ```
  Transfer and multi-task learning have traditionally focused on either a single source-target pair or very few, similar tasks. Ideally, the linguistic levels of morphology, syntax and semantics would benefit each other by being trained in a single model. We introduce such a joint many-task model together with a strategy for successively growing its depth to solve increasingly complex tasks. All layers include shortcut connections to both word representations and lower-level task predictions. We use a simple regularization term to allow for optimizing all model weights to improve one task's loss without exhibiting catastrophic interference of the other tasks. Our single end-to-end trainable model obtains state-of-the-art results on chunking, dependency parsing, semantic relatedness and textual entailment. It also performs competitively on POS tagging. Our dependency parsing layer relies only on a single feed-forward pass and does not require a beam search.
  ``` 
</details> 
{: .fs-5 .fw-600 .text-blue-300}

<details markdown="block">
  <summary>Comments</summary>
  {: .fs-3 .text-delta .text-red-100}
</details> 
{: .fs-3 .fw-600 .text-red-300}

---

{: .fs-4 .fw-800 .text-blue-100}
**📜 Which Tasks Should Be Learned Together in Multi-task Learning?**. <br> ✍ Trevor Standley, Amir R. Zamir, Dawn Chen, Leonidas Guibas, Jitendra Malik, Silvio Savarese
 *(ICML 2020)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/1905.07553){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://github.com/tstandley/taskgrouping){: target="_blank" .btn .btn-green .mr-1 }
   [Semantic Scholar](https://www.semanticscholar.org/paper/Which-Tasks-Should-Be-Learned-Together-in-Learning-Standley-Zamir/356941da708c6d5b06bce17463aca309fd33151a){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

<details markdown="block">
  <summary>Abstract</summary>
  {: .fs-3 .text-delta .text-blue-100}
  ```
  Transfer and multi-task learning have traditionally focused on either a single source-target pair or very few, similar tasks. Ideally, the linguistic levels of morphology, syntax and semantics would benefit each other by being trained in a single model. We introduce such a joint many-task model together with a strategy for successively growing its depth to solve increasingly complex tasks. All layers include shortcut connections to both word representations and lower-level task predictions. We use a simple regularization term to allow for optimizing all model weights to improve one task's loss without exhibiting catastrophic interference of the other tasks. Our single end-to-end trainable model obtains state-of-the-art results on chunking, dependency parsing, semantic relatedness and textual entailment. It also performs competitively on POS tagging. Our dependency parsing layer relies only on a single feed-forward pass and does not require a beam search.
  ``` 
</details> 
{: .fs-5 .fw-600 .text-blue-300}

<details markdown="block">
  <summary>Comments</summary>
  {: .fs-3 .text-delta .text-red-100}
</details> 
{: .fs-3 .fw-600 .text-red-300}

---

{: .fs-4 .fw-800 .text-blue-100}
**📜 AdapterFusion: Non-Destructive Task Composition for Transfer Learning**. <br> ✍ Jonas Pfeiffer, Aishwarya Kamath, Andreas Rücklé, Kyunghyun Cho, Iryna Gurevych
 *(EACL 2021)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2005.00247v3){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://adapterhub.ml/){: target="_blank" .btn .btn-green .mr-1 }
   [Semantic Scholar](https://www.semanticscholar.org/paper/AdapterFusion%3A-Non-Destructive-Task-Composition-for-Pfeiffer-Kamath/98ef0db84e62aef969629264c9de1f4d0013f3b9){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

<details markdown="block">
  <summary>Abstract</summary>
  {: .fs-3 .text-delta .text-blue-100}
  ```
  Sequential fine-tuning and multi-task learning are methods aiming to incorporate knowledge from multiple tasks; however, they suffer from catastrophic forgetting and difficulties in dataset balancing. To address these shortcomings, we propose AdapterFusion, a new two stage learning algorithm that leverages knowledge from multiple tasks. First, in the knowledge extraction stage we learn task specific parameters called adapters, that encapsulate the task-specific information. We then combine the adapters in a separate knowledge composition step. We show that by separating the two stages, i.e., knowledge extraction and knowledge composition, the classifier can effectively exploit the representations learned from multiple tasks in a non-destructive manner. We empirically evaluate AdapterFusion on 16 diverse NLU tasks, and find that it effectively combines various types of knowledge at different layers of the model. We show that our approach outperforms traditional strategies such as full fine-tuning as well as multi-task learning. Our code and adapters are available at this http URL.
  ``` 
</details> 
{: .fs-5 .fw-600 .text-blue-300}

<details markdown="block">
  <summary>Comments</summary>
  {: .fs-3 .text-delta .text-red-100}
</details> 
{: .fs-3 .fw-600 .text-red-300}

---

{: .fs-4 .fw-800 .text-blue-100}
**📜 Exploring and Predicting Transferability across NLP Tasks**. <br> ✍ Tu Vu, Tong Wang, Tsendsuren Munkhdalai, Alessandro Sordoni, Adam Trischler, Andrew Mattarella-Micke, Subhransu Maji, Mohit Iyyer
 *(EMNLP 2020)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2005.00770){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://github.com/tuvuumass/task-transferability){: target="_blank" .btn .btn-green .mr-1 }
   [Semantic Scholar](https://www.semanticscholar.org/paper/Exploring-and-Predicting-Transferability-across-NLP-Vu-Wang/d1206ccabd1980848f14472d6548251c2fab7963){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

<details markdown="block">
  <summary>Abstract</summary>
  {: .fs-3 .text-delta .text-blue-100}
  ```
  Recent advances in NLP demonstrate the effectiveness of training large-scale language models and transferring them to downstream tasks. Can fine-tuning these models on tasks other than language modeling further improve performance? In this paper, we conduct an extensive study of the transferability between 33 NLP tasks across three broad classes of problems (text classification, question answering, and sequence labeling). Our results show that transfer learning is more beneficial than previously thought, especially when target task data is scarce, and can improve performance even when the source task is small or differs substantially from the target task (e.g., part-of-speech tagging transfers well to the DROP QA dataset). We also develop task embeddings that can be used to predict the most transferable source tasks for a given target task, and we validate their effectiveness in experiments controlled for source and target data size. Overall, our experiments reveal that factors such as source data size, task and domain similarity, and task complexity all play a role in determining transferability.
  ``` 
</details> 
{: .fs-5 .fw-600 .text-blue-300}

<details markdown="block">
  <summary>Comments</summary>
  {: .fs-3 .text-delta .text-red-100}
</details> 
{: .fs-3 .fw-600 .text-red-300}

---

{: .fs-4 .fw-800 .text-blue-100}
**📜 Adapting Language Models for Zero-shot Learning by Meta-tuning on Dataset and Prompt Collections**. <br> ✍ Ruiqi Zhong, Kristy Lee, Zheng Zhang, Dan Klein
 *(EMNLP 2021)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2104.04670){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://github.com/ruiqi-zhong/Meta-tuning){: target="_blank" .btn .btn-green .mr-1 }
   [Semantic Scholar](https://www.semanticscholar.org/paper/Adapting-Language-Models-for-Zero-shot-Learning-by-Zhong-Lee/4b0ec90dc10e51c1fc983edcd57bb86636d7b3ca){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

<details markdown="block">
  <summary>Abstract</summary>
  {: .fs-3 .text-delta .text-blue-100}
  ```
  Large pre-trained language models (LMs) such as GPT-3 have acquired a surprising ability to perform zero-shot learning. For example, to classify sentiment without any training examples, we can “prompt" the LM with the review and the label description “Does the user like this movie?", and ask whether the next word is “Yes" or “No". However, the next word prediction training objective is still misaligned with the target zero-shot learning objective. To address this weakness, we propose meta-tuning, which directly optimizes the zero-shot learning objective by finetuning pre-trained language models on a collection of datasets. We focus on classification tasks, and construct the meta-dataset by aggregating 43 existing datasets and annotating 441 label descriptions in a question-answering (QA) format. When evaluated on unseen tasks, meta-tuned models outperform a samesized QA model and the previous SOTA zeroshot learning system based on natural language inference. Additionally, increasing parameter count from 220M to 770M improves AUC-ROC scores by 6.3%, and we forecast that even larger models would perform better. Therefore, measuring zero-shot learning performance on language models out-of-thebox might underestimate their true potential, and community-wide efforts on aggregating datasets and unifying their formats can help build models that answer prompts better.
  ``` 
</details> 
{: .fs-5 .fw-600 .text-blue-300}

<details markdown="block">
  <summary>Comments</summary>
  {: .fs-3 .text-delta .text-red-100}
</details> 
{: .fs-3 .fw-600 .text-red-300}

---

{: .fs-4 .fw-800 .text-blue-100}
**📜 Finetuned Language Models Are Zero-Shot Learners(FLAN)**. <br> ✍ Jason Wei, Maarten Bosma, Vincent Y. Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, Quoc V. Le
 *(arxiv 2021)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2109.01652){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://github.com/google-research/flan){: target="_blank" .btn .btn-green .mr-1 }
   [Semantic Scholar](https://www.semanticscholar.org/paper/Language-Models-are-Few-Shot-Learners-Brown-Mann/6b85b63579a916f705a8e10a49bd8d849d91b1fc){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

<details markdown="block">
  <summary>Abstract</summary>
  {: .fs-3 .text-delta .text-blue-100}
```
Recent work has demonstrated substantial gains on many NLP tasks and benchmarks by pre-training on a large corpus of text followed by fine-tuning on a specific task. While typically task-agnostic in architecture, this method still requires task-specific fine-tuning datasets of thousands or tens of thousands of examples. By contrast, humans can generally perform a new language task from only a few examples or from simple instructions - something which current NLP systems still largely struggle to do. Here we show that scaling up language models greatly improves task-agnostic, few-shot performance, sometimes even reaching competitiveness with prior state-of-the-art fine-tuning approaches. Specifically, we train GPT-3, an autoregressive language model with 175 billion parameters, 10x more than any previous non-sparse language model, and test its performance in the few-shot setting. For all tasks, GPT-3 is applied without any gradient updates or fine-tuning, with tasks and few-shot demonstrations specified purely via text interaction with the model. GPT-3 achieves strong performance on many NLP datasets, including translation, question-answering, and cloze tasks, as well as several tasks that require on-the-fly reasoning or domain adaptation, such as unscrambling words, using a novel word in a sentence, or performing 3-digit arithmetic. At the same time, we also identify some datasets where GPT-3's few-shot learning still struggles, as well as some datasets where GPT-3 faces methodological issues related to training on large web corpora. Finally, we find that GPT-3 can generate samples of news articles which human evaluators have difficulty distinguishing from articles written by humans. We discuss broader societal impacts of this finding and of GPT-3 in general.
``` 
</details> 
{: .fs-5 .fw-600 .text-blue-300}

<details markdown="block">
  <summary>Comments</summary>
  {: .fs-3 .text-delta .text-red-100}
</details> 
{: .fs-3 .fw-600 .text-red-300}

---

{: .fs-4 .fw-800 .text-blue-100}
**📜 Meta-learning via Language Model In-context Tuning**. <br> ✍ Tu Vu, Tong Wang, Tsendsuren Munkhdalai, Alessandro Sordoni, Adam Trischler, Andrew Mattarella-Micke, Subhransu Maji, Mohit Iyyer
 *(arxiv 2021)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2110.07814){: .btn .btn-blue .mr-1 target="_blank" }
</span> 

<details markdown="block">
  <summary>Abstract</summary>
  {: .fs-3 .text-delta .text-blue-100}
  ```
  The goal of meta-learning is to learn to adapt to a new task with only a few labeled examples. To tackle this problem in NLP, we propose in-context tuning, which recasts adaptation and prediction as a simple sequence prediction problem: to form the input sequence, we concatenate the task instruction, the labeled examples, and the target input to predict; to meta-train the model to learn from in-context examples, we fine-tune a pre-trained language model (LM) to predict the target label from the input sequences on a collection of tasks.
We benchmark our method on two collections of text classification tasks: LAMA and BinaryClfs. Compared to first-order MAML which adapts the model with gradient descent, our method better leverages the inductive bias of LMs to perform pattern matching, and outperforms MAML by an absolute 6% AUC ROC score on BinaryClfs, with increasing advantage w.r.t. model size. Compared to non-fine-tuned in-context learning (i.e. prompting a raw LM), in-context tuning directly learns to learn from in-context examples. On BinaryClfs, in-context tuning improves the average AUC-ROC score by an absolute 10%, and reduces the variance with respect to example ordering by 6x and example choices by 2x.
  ``` 
</details> 
{: .fs-5 .fw-600 .text-blue-300}

<details markdown="block">
  <summary>Comments</summary>
  {: .fs-3 .text-delta .text-red-100}
</details> 
{: .fs-3 .fw-600 .text-red-300}

---

{: .fs-4 .fw-800 .text-blue-100}
**📜 Multitask Prompted Training Enables Zero-Shot Task Generalization(T0)**. <br> ✍ Victor Sanh, Albert Webson, Colin Raffel, Stephen H. Bach, Lintang Sutawika, Zaid Alyafeai, Antoine Chaffin, Arnaud Stiegler, Teven Le Scao, Arun Raja, Manan Dey, M Saiful Bari, Canwen Xu, Urmish Thakker, Shanya Sharma Sharma, Eliza Szczechla, Taewoon Kim, Gunjan Chhablani, Nihal Nayak, Debajyoti Datta, Jonathan Chang, Mike Tian-Jian Jiang, Han Wang, Matteo Manica, Sheng Shen, Zheng Xin Yong, Harshit Pandey, Rachel Bawden, Thomas Wang, Trishala Neeraj, Jos Rozen, Abheesht Sharma, Andrea Santilli, Thibault Fevry, Jason Alan Fries, Ryan Teehan, Stella Biderman, Leo Gao, Tali Bers, Thomas Wolf, Alexander M. Rush
 *(arxiv 2021)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2110.08207){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://github.com/bigscience-workshop/promptsource/){: target="_blank" .btn .btn-green .mr-1 }
   [Weight](https://huggingface.co/bigscience){: target="_blank" .btn .btn-red .mr-1 }
</span> 

<details markdown="block">
  <summary>Abstract</summary>
  {: .fs-3 .text-delta .text-blue-100}
  ```
  Large language models have recently been shown to attain reasonable zero-shot generalization on a diverse set of tasks (Brown et al., 2020). It has been hypothesized that this is a consequence of implicit multitask learning in language models' pretraining (Radford et al., 2019). Can zero-shot generalization instead be directly induced by explicit multitask learning? To test this question at scale, we develop a system for easily mapping any natural language tasks into a human-readable, prompted form. We convert a large set of supervised datasets, each with multiple prompts with diverse wording. These prompted datasets allow for benchmarking the ability of a model to perform completely unseen tasks. We finetune a pretrained encoder-decoder model (Raffel et al., 2020; Lester et al., 2021) on this multitask mixture covering a wide variety of tasks. The model attains strong zero-shot performance on several standard datasets, often outperforming models up to 16x its size. Further, our approach attains strong performance on a subset of tasks from the BIG-bench benchmark, outperforming models up to 6x its size. All prompts and trained models are available at this https URL bigscience-workshop/promptsource/ and this https URL.
  ``` 
</details> 
{: .fs-5 .fw-600 .text-blue-300}

<details markdown="block">
  <summary>Comments</summary>
  {: .fs-3 .text-delta .text-red-100}
</details> 
{: .fs-3 .fw-600 .text-red-300}

---

{: .fs-4 .fw-800 .text-blue-100}
**📜 Balancing Average and Worst-case Accuracy in Multitask Learning**. <br> ✍ Paul Michel, Sebastian Ruder, Dani Yogatama
 *(arxiv 2021)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2110.05838){: .btn .btn-blue .mr-1 target="_blank" } 
   [Semantic Scholar](https://www.semanticscholar.org/paper/Balancing-Average-and-Worst-case-Accuracy-in-Michel-Ruder/9f0fe9197f080d042ad406b149ac03a1063c0351){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

<details markdown="block">
  <summary>Abstract</summary>
  {: .fs-3 .text-delta .text-blue-100}
  ```
  When training and evaluating machine learning models on a large number of tasks, it is important to not only look at average task accuracy—which may be biased by easy or redundant tasks—but also worst-case accuracy (i.e. the performance on the task with the lowest accuracy). In this work, we show how to use techniques from the distributionally robust optimization (DRO) literature to improve worst-case performance in multitask learning. We highlight several failure cases of DRO when applied off-the-shelf and present an improved method, Lookahead-DRO (L-DRO), which mitigates these issues. The core idea of L-DRO is to anticipate the interaction between tasks during training in order to choose a dynamic re-weighting of the various task losses, which will (i) lead to minimal worst-case loss and (ii) train on as many tasks as possible. After demonstrating the efficacy of L-DRO on a small controlled synthetic setting, we evaluate it on two realistic benchmarks: a multitask version of the CIFAR-100 image classification dataset and a large-scale multilingual language modeling experiment. Our empirical results show that LDRO achieves a better trade-off between average and worst-case accuracy with little computational overhead compared to several strong baselines.
  ``` 
</details> 
{: .fs-5 .fw-600 .text-blue-300}

<details markdown="block">
  <summary>Comments</summary>
  {: .fs-3 .text-delta .text-red-100}
</details> 
{: .fs-3 .fw-600 .text-red-300}

---

{: .fs-4 .fw-800 .text-blue-100}
**📜 Balancing Average and Worst-case Accuracy in Multitask Learning**. <br> ✍ Paul Michel, Sebastian Ruder, Dani Yogatama
 *(arxiv 2021)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2110.05838){: .btn .btn-blue .mr-1 target="_blank" } 
   [Semantic Scholar](https://www.semanticscholar.org/paper/Balancing-Average-and-Worst-case-Accuracy-in-Michel-Ruder/9f0fe9197f080d042ad406b149ac03a1063c0351){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

<details markdown="block">
  <summary>Abstract</summary>
  {: .fs-3 .text-delta .text-blue-100}
  ```
  When training and evaluating machine learning models on a large number of tasks, it is important to not only look at average task accuracy—which may be biased by easy or redundant tasks—but also worst-case accuracy (i.e. the performance on the task with the lowest accuracy). In this work, we show how to use techniques from the distributionally robust optimization (DRO) literature to improve worst-case performance in multitask learning. We highlight several failure cases of DRO when applied off-the-shelf and present an improved method, Lookahead-DRO (L-DRO), which mitigates these issues. The core idea of L-DRO is to anticipate the interaction between tasks during training in order to choose a dynamic re-weighting of the various task losses, which will (i) lead to minimal worst-case loss and (ii) train on as many tasks as possible. After demonstrating the efficacy of L-DRO on a small controlled synthetic setting, we evaluate it on two realistic benchmarks: a multitask version of the CIFAR-100 image classification dataset and a large-scale multilingual language modeling experiment. Our empirical results show that LDRO achieves a better trade-off between average and worst-case accuracy with little computational overhead compared to several strong baselines.
  ``` 
</details> 
{: .fs-5 .fw-600 .text-blue-300}

<details markdown="block">
  <summary>Comments</summary>
  {: .fs-3 .text-delta .text-red-100}
</details> 
{: .fs-3 .fw-600 .text-red-300}

---

{: .fs-4 .fw-800 .text-blue-100}
**📜 Balancing Average and Worst-case Accuracy in Multitask Learning**. <br> ✍ Paul Michel, Sebastian Ruder, Dani Yogatama
 *(arxiv 2021)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2110.05838){: .btn .btn-blue .mr-1 target="_blank" } 
   [Semantic Scholar](https://www.semanticscholar.org/paper/Balancing-Average-and-Worst-case-Accuracy-in-Michel-Ruder/9f0fe9197f080d042ad406b149ac03a1063c0351){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

<details markdown="block">
  <summary>Abstract</summary>
  {: .fs-3 .text-delta .text-blue-100}
  ```
  When training and evaluating machine learning models on a large number of tasks, it is important to not only look at average task accuracy—which may be biased by easy or redundant tasks—but also worst-case accuracy (i.e. the performance on the task with the lowest accuracy). In this work, we show how to use techniques from the distributionally robust optimization (DRO) literature to improve worst-case performance in multitask learning. We highlight several failure cases of DRO when applied off-the-shelf and present an improved method, Lookahead-DRO (L-DRO), which mitigates these issues. The core idea of L-DRO is to anticipate the interaction between tasks during training in order to choose a dynamic re-weighting of the various task losses, which will (i) lead to minimal worst-case loss and (ii) train on as many tasks as possible. After demonstrating the efficacy of L-DRO on a small controlled synthetic setting, we evaluate it on two realistic benchmarks: a multitask version of the CIFAR-100 image classification dataset and a large-scale multilingual language modeling experiment. Our empirical results show that LDRO achieves a better trade-off between average and worst-case accuracy with little computational overhead compared to several strong baselines.
  ``` 
</details> 
{: .fs-5 .fw-600 .text-blue-300}

<details markdown="block">
  <summary>Comments</summary>
  {: .fs-3 .text-delta .text-red-100}
</details> 
{: .fs-3 .fw-600 .text-red-300}

---

{: .fs-4 .fw-800 .text-blue-100}
**📜 Balancing Average and Worst-case Accuracy in Multitask Learning**. <br> ✍ Paul Michel, Sebastian Ruder, Dani Yogatama
 *(arxiv 2021)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2110.05838){: .btn .btn-blue .mr-1 target="_blank" } 
   [Semantic Scholar](https://www.semanticscholar.org/paper/Balancing-Average-and-Worst-case-Accuracy-in-Michel-Ruder/9f0fe9197f080d042ad406b149ac03a1063c0351){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

<details markdown="block">
  <summary>Abstract</summary>
  {: .fs-3 .text-delta .text-blue-100}
  ```
  When training and evaluating machine learning models on a large number of tasks, it is important to not only look at average task accuracy—which may be biased by easy or redundant tasks—but also worst-case accuracy (i.e. the performance on the task with the lowest accuracy). In this work, we show how to use techniques from the distributionally robust optimization (DRO) literature to improve worst-case performance in multitask learning. We highlight several failure cases of DRO when applied off-the-shelf and present an improved method, Lookahead-DRO (L-DRO), which mitigates these issues. The core idea of L-DRO is to anticipate the interaction between tasks during training in order to choose a dynamic re-weighting of the various task losses, which will (i) lead to minimal worst-case loss and (ii) train on as many tasks as possible. After demonstrating the efficacy of L-DRO on a small controlled synthetic setting, we evaluate it on two realistic benchmarks: a multitask version of the CIFAR-100 image classification dataset and a large-scale multilingual language modeling experiment. Our empirical results show that LDRO achieves a better trade-off between average and worst-case accuracy with little computational overhead compared to several strong baselines.
  ``` 
</details> 
{: .fs-5 .fw-600 .text-blue-300}

<details markdown="block">
  <summary>Comments</summary>
  {: .fs-3 .text-delta .text-red-100}
</details> 
{: .fs-3 .fw-600 .text-red-300}

---
## Prompt learning
## Few Shot Learning
## Task Unification
## Pre-training
## Analysis
## Others