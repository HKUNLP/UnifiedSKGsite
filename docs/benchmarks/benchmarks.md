---
layout: default
title: Benchmark Datasets
nav_order: 3
toc_list: true
last_modified_date: Jan 19 2022
permalink: /benchmarks/
---

# Benchmark Datasets for Structured Knowledge Grounding
{: .no_toc }

{: .fs-5 .fw-300 }

We present a comprehensive collection of datasets of Structured Knowledge Grounding.

## TODO: Here is a task table for all information
## Table of contents
{: .no_toc .text-delta .fs-4 style="font-weight:800"}

- TOC
{:toc}

## Semantic Parsing

### Spider
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*[Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task](https://arxiv.org/abs/1809.08887)*. **EMNLP-18**

[comment]: <> ([Official Link]&#40;https://yale-lily.github.io/spider&#41;{: target="_blank" .btn .btn-green .mr-1 })

---

### GrailQA 
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*[Beyond I.I.D.: Three Levels of Generalization for Question Answering on Knowledge Bases](https://arxiv.org/abs/2011.07743)*. **WWW-21**

[comment]: <> ([Official Link]&#40;https://dki-lab.github.io/GrailQA/&#41;{: target="_blank" .btn .btn-green .mr-1 })

---

### WebQSP
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*[The Value of Semantic Parse Labeling for Knowledge Base Question Answering](https://aclanthology.org/P16-2033/)*. **ACL-2016**

[comment]: <> ([Official Link]&#40;http://aka.ms/WebQSP&#41;{: target="_blank" .btn .btn-green .mr-1 })

---

### MTOP
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*[MTOP: A Comprehensive Multilingual Task-Oriented Semantic Parsing Benchmark](https://aclanthology.org/2021.eacl-main.257/)*. **EACL-21**

[comment]: <> ([Dataset]&#40;https://fb.me/mtop_dataset&#41;{: target="_blank" .btn .btn-green .mr-1 })

> **Comments**
> We used the English part of MTOP as SKG benchmark.
{: .fs-4 .fw-600 .text-red-300}

---

## Question Answering


### WikiSQL
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*[Seq2SQL: Generating Structured Queries from Natural Language using Reinforcement Learning](https://arxiv.org/abs/1709.00103)*. **arixv-17**

[comment]: <> ([Official Link]&#40;https://github.com/salesforce/WikiSQL&#41;{: target="_blank" .btn .btn-green .mr-1 })


### WikiTableQuestion
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*Compositional Semantic Parsing on Semi-Structured Tables*.**ACL-15**

<span class="fs-1">
[Paper](https://aclanthology.org/P15-1142/){: .btn .btn-blue .mr-1 target="_blank" } [Official Link](https://ppasupat.github.io/WikiTableQuestions/){: target="_blank" .btn .btn-green .mr-1 } </span>

> **Comments**
> The 5-fold validation evaluation in origianl dataset is depracated by latest works. The 1st fold of train set and dev set are used as train set and dev set.
{: .fs-4 .fw-600 .text-red-300}


### CompWebQ
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*The Web as a Knowledge-base for Answering Complex Questions*.**NAACL-18**

<span class="fs-1">
[Paper](https://arxiv.org/abs/1911.11641){: .btn .btn-blue .mr-1 target="_blank" } [Official Link](https://www.tau-nlp.org/compwebq){: target="_blank" .btn .btn-green .mr-1 } {: target="_blank" .btn .btn-purple .mr-1 } </span>

### HybridQA
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*HybridQA: A Dataset of Multi-Hop Question Answering over Tabular and Textual Data*.**EMNLP-20**

<span class="fs-1">
[Paper](https://aclanthology.org/2020.findings-emnlp.91/){: .btn .btn-blue .mr-1 target="_blank" } [Official Link](https://hybridqa.github.io/){: target="_blank" .btn .btn-green .mr-1 } </span>

### OTT-QA
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*Open Question Answering over Tables and Text *.**ICLR-21**

<span class="fs-1">
[Paper](https://arxiv.org/abs/2010.10439){: .btn .btn-blue .mr-1 target="_blank" } [Official Link](https://github.com/wenhuchen/OTT-QA){: target="_blank" .btn .btn-green .mr-1 } </span>

### MultiModalQA
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*MultiModalQA: Complex Question Answering over Text, Tables and Images*.**ICLR-21**

<span class="fs-1">
[Paper](https://arxiv.org/abs/1709.00103){: .btn .btn-blue .mr-1 target="_blank" } [Official Link](https://allenai.github.io/multimodalqa/){: target="_blank" .btn .btn-green .mr-1 } </span>

### FeTaQA
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*FeTaQA: Free-form Table Question Answering*.**ICLR-18**

<span class="fs-1">
[Paper](https://arxiv.org/abs/2104.00369){: .btn .btn-blue .mr-1 target="_blank" } [Official Link](https://github.com/Yale-LILY/FeTaQA){: target="_blank" .btn .btn-green .mr-1 } </span>

### TAT-QA
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*TAT-QA: A Question Answering Benchmark on a Hybrid of Tabular and Textual Content in Finance*.**ACL-21**

<span class="fs-1">
[Paper](https://aclanthology.org/2021.acl-long.254/){: .btn .btn-blue .mr-1 target="_blank" } [Official Link](https://nextplusplus.github.io/TAT-QA/){: target="_blank" .btn .btn-green .mr-1 } </span>

---

## Data-to-Text

### E2E
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*The E2E Dataset: New Challenges For End-to-End Generation*.**SIGDIAL-17**

<span class="fs-1">
[Paper](https://aclanthology.org/W17-5525/){: .btn .btn-blue .mr-1 target="_blank" } [Official Link](http://www.macs.hw.ac.uk/InteractionLab/E2E/){: target="_blank" .btn .btn-green .mr-1 } </span>

> **Comments**
> There is another [E2E cleaned version](https://github.com/tuetschek/e2e-cleaning) released by authors. 
{: .fs-4 .fw-600 .text-red-300}

### WebNLG
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*The WebNLG Challenge: Generating Text from RDF Data*.**INLG-17**

<span class="fs-1">
[Paper](https://aclanthology.org/W17-3518/){: .btn .btn-blue .mr-1 target="_blank" } [Official Link](https://webnlg-challenge.loria.fr/){: target="_blank" .btn .btn-green .mr-1 } </span>

> **Comments**
> WebNLG challenge has many datasets available, the widely used dversion currently is the WebNLG challenge 2017. There is a useful [link](https://github.com/fuzihaofzh/webnlg-dataset) for summarization of this.
{: .fs-4 .fw-600 .text-red-300}

### DART
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*DART: Open-Domain Structured Data Record to Text Generation*.**ICLR-21**

<span class="fs-1">
[Paper](https://aclanthology.org/2021.naacl-main.37/){: .btn .btn-blue .mr-1 target="_blank" } [Official Link](https://github.com/Yale-LILY/dart){: target="_blank" .btn .btn-green .mr-1 } </span>


### ToTTo
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*ToTTo: A Controlled Table-To-Text Generation Dataset*.**ICLR-21**

<span class="fs-1">
[Paper](https://arxiv.org/abs/2004.14373){: .btn .btn-blue .mr-1 target="_blank" } [Official Link](https://github.com/google-research-datasets/ToTTo){: target="_blank" .btn .btn-green .mr-1 } </span>



### LogicNLG
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*Logical Natural Language Generation from Open-Domain Tables*.**ACL-20**

<span class="fs-1">
[Paper](https://aclanthology.org/2020.acl-main.708/){: .btn .btn-blue .mr-1 target="_blank" } [Official Link](https://github.com/wenhuchen/LogicNLG){: target="_blank" .btn .btn-green .mr-1 } </span>


---

## Conversational

### MultiWoZ21
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*MultiWOZ 2.1: A Consolidated Multi-Domain Dialogue Dataset with State Corrections and State Tracking Baselines*.**LREC-20**

<span class="fs-1">
[Paper](https://arxiv.org/abs/1907.01669){: .btn .btn-blue .mr-1 target="_blank" } [Official Link](https://github.com/budzianowski/multiwoz){: target="_blank" .btn .btn-green .mr-1 } </span>



> **Comments**
> There are many version of MultiWoZ, 2.1 and 2.2 are mostly used currently. We used the 2.1 version in our SKG benchmark. Some pre-procession on this dataset is needed, pls refer to [MultiWoZ](https://github.com/budzianowski/multiwoz) and [Trade-DST](https://github.com/jasonwu0731/trade-dst).
{: .fs-4 .fw-600 .text-red-300}




### KVRET(SMD)
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*Key-Value Retrieval Networks for Task-Oriented Dialogue*.**SIGdial-17**

<span class="fs-1">
[Paper](https://aclanthology.org/W17-5506/){: .btn .btn-blue .mr-1 target="_blank" } [Official Link](https://nlp.stanford.edu/blog/a-new-multi-turn-multi-domain-task-oriented-dialogue-dataset/){: target="_blank" .btn .btn-green .mr-1 } </span>

> **Comments**
> KVRET is also called SMD(Stanford Multi-Domain task-oriented dialogue dataset). The de-facto widely used version of this dataset is the pre-processed verison in [Mem2seq](https://github.com/HLTCHKUST/Mem2Seq).
{: .fs-4 .fw-600 .text-red-300}

### SParC
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*Submitted on 5 Jun 2019]
SParC: Cross-Domain Semantic Parsing in Context*.**ACL-19**

<span class="fs-1">
[Paper](https://arxiv.org/abs/1906.02285){: .btn .btn-blue .mr-1 target="_blank" } [Official Link](https://yale-lily.github.io/sparc){: target="_blank" .btn .btn-green .mr-1 } </span>


### CoSQL
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*CoSQL: A Conversational Text-to-SQL Challenge Towards Cross-Domain Natural Language Interfaces to Databases*.**EMNLP-19**

<span class="fs-1">
[Paper](https://arxiv.org/abs/1909.05378){: .btn .btn-blue .mr-1 target="_blank" } [Official Link](https://yale-lily.github.io/cosql){: target="_blank" .btn .btn-green .mr-1 } </span>


### SQA(MSR SQA)
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*Search-based Neural Structured Learning for Sequential Question Answering*.**ACL-17**

<span class="fs-1">
[Paper](https://aclanthology.org/P17-1167){: .btn .btn-blue .mr-1 target="_blank" } [Official Link](hhttps://www.microsoft.com/en-us/download/details.aspx?id=54253){: target="_blank" .btn .btn-green .mr-1 } </span>

---

## Fact Verification

### TabFact
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*TabFact: A Large-scale Dataset for Table-based Fact Verification*.**ICLR-20**

<span class="fs-1">
[Paper](https://arxiv.org/abs/1909.02164){: .btn .btn-blue .mr-1 target="_blank" } [Official Link](https://tabfact.github.io/){: target="_blank" .btn .btn-green .mr-1 } </span>

### FEVEROUS
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*The Fact Extraction and VERification Over Unstructured and Structured information (FEVEROUS) Shared Task*.**ICLR-21**

<span class="fs-1">
[Paper](https://aclanthology.org/2021.fever-1.1){: .btn .btn-blue .mr-1 target="_blank" } [Official Link](https://fever.ai/dataset/feverous.html){: target="_blank" .btn .btn-green .mr-1 } </span>

---

## Formal-Language-to-Text

### SQL2Text
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*Logic-Consistency Text Generation from Semantic Parses*.**ACL-21**

<span class="fs-1">
[Paper](https://aclanthology.org/2021.findings-acl.388/){: .btn .btn-blue .mr-1 target="_blank" } [Official Link](https://github.com/Ciaranshu/relogic){: target="_blank" .btn .btn-green .mr-1 } </span>

---

### Logic2Text
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*Logic2Text: High-Fidelity Natural Language Generation from Logical Forms* **EMNLP-20**

<span class="fs-1">
[Paper](https://arxiv.org/abs/2004.14579){: .btn .btn-blue .mr-1 target="_blank" } [Official Link](https://github.com/czyssrs/Logic2Text){: target="_blank" .btn .btn-green .mr-1 } </span>

---

## Other Related Datasets



