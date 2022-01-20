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


---

## Multi-task learning
{: .fs-4 .fw-800 .text-blue-100}
**📜 A Comprehensive Exploration on WikiSQL with Table-Aware Word Contextualization**. <br> ✍ Wonseok Hwang, Jinyeong Yim, Seunghyun Park, Minjoon Seo
 *(KR@ML Workshop ar NIPS-19)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/1902.01069){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://github.com/naver/sqlova){: target="_blank" .btn .btn-green .mr-1 }
   [Semantic Scholar](https://www.semanticscholar.org/paper/A-Comprehensive-Exploration-on-WikiSQL-with-Word-Hwang-Yim/46b5d1bfe9bc72e056626c7f8cfd4936a4a00c0d){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

**Evaluation Tasks:** [WikiSQL](/datasets#wikisql)

<details markdown="block">
  <summary>Abstract</summary>
  {: .fs-3 .text-delta .text-blue-100}
  ```
  We present SQLOVA, the first Natural-language-to-SQL (NL2SQL) model to achieve human performance in WikiSQL dataset. We revisit and discuss diverse popular methods in NL2SQL literature, take a full advantage of BERT (Devlin et al., 2018) through an effective table contextualization method, and coherently combine them, outperforming the previous state of the art by 8.2% and 2.5% in logical form and execution accuracy, respectively. We particularly note that BERT with a seq2seq decoder leads to a poor performance in the task, indicating the importance of a careful design when using such large pretrained models. We also provide a comprehensive analysis on the dataset and our model, which can be helpful for designing future NL2SQL datsets and models. We especially show that our model’s performance is near the upper bound in WikiSQL, where we observe that a large portion of the evaluation errors are due to wrong annotations, and our model is already exceeding human performance by 1.3% in execution accuracy.
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

## Prompt learning
## Few Shot Learning
## Task Unification
## Pre-training
## Analysis
## Others