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

## Prompt learning
## Few Shot Learning
## Task Unification
## Pre-training
## Analysis
## Others