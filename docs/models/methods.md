---
layout: default
title: Reasoning Methods
nav_order: 4
toc_list: true
last_modified_date: Jan 11 2022
permalink: /methods/
---

# Methods to Structured Knowledge Grounding
{: .no_toc }

Editors: [Tianbao Xie](https://tianbaoxie.com/), ...

{: .fs-5 .fw-300 }
We present a collection of insightful research papers that focus on structured knowledge grounding tasks.



## Table of contents
{: .no_toc .text-delta .fs-4 style="font-weight:800"}

- TOC
{:toc}


## Linearized and concat

<table>
<td>
<span class="fs-4">
The following papers aim to use inearized the struc-tured knowledge and concatenated it with the text, some has been augmented by positional encoding(e.g., row/column embedding) for tables and template-based linearzation for tables and knowledge graphs.
</span>
</td>
</table>

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
**📜 K-BERT: Enabling Language Representation with Knowledge Graph**. <br> ✍ Weijie Liu, Peng Zhou, Zhe Zhao, Zhiruo Wang, Qi Ju, Haotang Deng, Ping Wang
 *(AAAI-20)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/1909.07606){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://github.com/autoliuweijie/K-BERT){: target="_blank" .btn .btn-green .mr-1 }
   [Semantic Scholar](https://www.semanticscholar.org/paper/K-BERT%3A-Enabling-Language-Representation-with-Graph-Liu-Zhou/06a73ad09664435f8b3cd90293f4e05a047cf375){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

**Evaluation Tasks:** Book, review,Chnsenticorp, Shopping, Weibo, XNLI, LCQMC, NLPCC-DBQA, MSRA-NER, Finance Q&A, Law Q&A, Finance NER, Medicine NER

<details markdown="block">
  <summary>Abstract</summary>
  {: .fs-3 .text-delta .text-blue-100}
  ```
  Pre-trained language representation models, such as BERT, capture a general language representation from large-scale corpora, but lack domain-specific knowledge. When reading a domain text, experts make inferences with relevant knowledge. For machines to achieve this capability, we propose a knowledge-enabled language representation model (K-BERT) with knowledge graphs (KGs), in which triples are injected into the sentences as domain knowledge. However, too much knowledge incorporation may divert the sentence from its correct meaning, which is called knowledge noise (KN) issue. To overcome KN, K-BERT introduces soft-position and visible matrix to limit the impact of knowledge. K-BERT can easily inject domain knowledge into the models by equipped with a KG without pre-training by-self because it is capable of loading model parameters from the pre-trained BERT. Our investigation reveals promising results in twelve NLP tasks. Especially in domain-specific tasks (including finance, law, and medicine), K-BERT significantly outperforms BERT, which demonstrates that K-BERT is an excellent choice for solving the knowledge-driven problems that require experts.
  ``` 
</details> 
{: .fs-5 .fw-600 .text-blue-300}

<details markdown="block">
  <summary>Comments</summary>
  {: .fs-3 .text-delta .text-red-100}
</details> 
{: .fs-3 .fw-600 .text-red-300}


## Others

