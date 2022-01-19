---
layout: default
title: Benchmark Datasets
nav_order: 3
toc_list: true
last_modified_date: Jan 19 2022
permalink: /datasets/
---

# Benchmark Datasets for Structured Knowledge Grounding
{: .no_toc }

{: .fs-5 .fw-300 }

We present a comprehensive collection of datasets of Structured Knowledge Grounding.


## Table of contents
{: .no_toc .text-delta .fs-4 style="font-weight:800"}

- TOC
{:toc}

---

<details markdown="block">
  <summary>🌟 Summary Table 🌟</summary>
{: .fs-4 .text-delta .text-red-200 style="font-weight:800"}

to be added.

</details>

---

## Semantic Parsing Tasks


### Spider
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task*. <br> Tao Yu, Rui Zhang, Kai Yang, Michihiro Yasunaga, Dongxu Wang, Zifan Li, James Ma, Irene Li, Qingning Yao, Shanelle Roman, Zilin Zhang, Dragomir Radev. **EMNLP-18**

<span class="fs-1">
[Paper](https://arxiv.org/abs/1809.08887){: .btn .btn-blue .mr-1 target="_blank" } [Official Link](https://yale-lily.github.io/spider){: target="_blank" .btn .btn-green .mr-1 }  </span>

> - **Knowledge**: Database
> - **User Input**: Question
> - **Output**: SQL
> - **Keywords**: Fully supervised; Cross-domain; Single turn

><details markdown="block">
>  <summary>Illustrative Example</summary>
>  {: .fs-3 .text-delta .text-red-100}
```
..
```
></details>


### GrailQA 
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*Beyond I.I.D.: Three Levels of Generalization for Question Answering on Knowledge Bases*.<br>Yu Gu, Sue Kase, Michelle Vanni, Brian Sadler, Percy Liang, Xifeng Yan, Yu Su. **Proceedings of the Web Conference-21**

<span class="fs-1">
[Paper](https://arxiv.org/abs/2011.07743){: .btn .btn-blue .mr-1 target="_blank" } [Official Link](https://dki-lab.github.io/GrailQA/){: target="_blank" .btn .btn-green .mr-1 } </span>

> - **Knowledge**: Knowledge Graph
> - **User Input**: Question
> - **Output**: S-Expression
> - **Keywords**: Large, 64k; Test generalization: i.i.d./compositional/zero-shot

><details markdown="block">
>  <summary>Illustrative Example</summary>
>  {: .fs-3 .text-delta .text-red-100}
```
..
```
></details> 

### WebQSP
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*The Value of Semantic Parse Labeling for Knowledge Base Question Answering*.<br> Wen-tau Yih, Matthew Richardson, Chris Meek, Ming-Wei Chang, Jina Suh. **ACL-2016**

<span class="fs-1">
[Paper](https://aclanthology.org/P16-2033/){: .btn .btn-blue .mr-1 target="_blank" } [Official Link](http://aka.ms/WebQSP){: target="_blank" .btn .btn-green .mr-1 }</span>

> - **Knowledge**: Knowledge Graph
> - **User Input**: Question
> - **Output**: S-Expression

><details markdown="block">
>  <summary>Illustrative Example</summary>
>  {: .fs-3 .text-delta .text-red-100}
```
..
```
></details> 




### MTOP
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*MTOP: A Comprehensive Multilingual Task-Oriented Semantic Parsing Benchmark*.<br>Haoran Li, Abhinav Arora, Shuohui Chen, Anchit Gupta, Sonal Gupta, Yashar Mehdad. **EACL-21**

<span class="fs-1">
[Paper](https://aclanthology.org/2021.eacl-main.257/){: .btn .btn-blue .mr-1 target="_blank" } [Dataset](https://fb.me/mtop_dataset){: target="_blank" .btn .btn-green .mr-1 } </span>

> - **Knowledge**: API
> - **User Input**: Question
> - **Output**: TOP-representation
> - **Keywords**: Spoken Language Understanding; TOP representation

><details markdown="block">
>  <summary>Illustrative Example</summary>
>  {: .fs-3 .text-delta .text-red-100}
```
..
```
></details> 

> **Comments**
> We used the English part of MTOP as SKG benchmark.
<!-- Mention the highlights or known issues of the dataset. -->
{: .fs-4 .fw-600 .text-red-300}

---

## Question Answering


### WikiSQL
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*Seq2SQL: Generating Structured Queries from Natural Language using Reinforcement Learning*.<br> Victor Zhong, Caiming Xiong, Richard Socher. **ICLR-18**

<span class="fs-1">
[Paper](https://arxiv.org/abs/1709.00103){: .btn .btn-blue .mr-1 target="_blank" } [Official Link](https://github.com/salesforce/WikiSQL){: target="_blank" .btn .btn-green .mr-1 } </span>
> - **Knowledge**: Table
> - **User Input**: Question
> - **Output**: Answer(adopted)/SQL
> - **Keywords**: Fully/weakly supervised semantic parsing(SQL provided); Large data

><details markdown="block">
>  <summary>Illustrative Example</summary>
>  {: .fs-3 .text-delta .text-red-100}
```
..
```
></details> 


### WikiTableQuestion
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*Compositional Semantic Parsing on Semi-Structured Tables*.<br> Panupong Pasupat, Percy Liang. **ACL-15**

<span class="fs-1">
[Paper](https://aclanthology.org/P15-1142/){: .btn .btn-blue .mr-1 target="_blank" } [Official Link](https://ppasupat.github.io/WikiTableQuestions/){: target="_blank" .btn .btn-green .mr-1 } </span>
> - **Knowledge**: Table
> - **User Input**: Question
> - **Output**: Answer
> - **Keywords**: Weakly supervised semantic parsing(using question-answer pairs as supervision); Row sensative(some qa related to row order)

><details markdown="block">
>  <summary>Illustrative Example</summary>
>  {: .fs-3 .text-delta .text-red-100}
```
..
```
></details> 

> **Comments**
> The 5-fold validation evaluation in origianl dataset is depracated by latest works. The 1st fold of train set and dev set are used as train set and dev set.
<!-- Mention the highlights or known issues of the dataset. -->
{: .fs-4 .fw-600 .text-red-300}


### CompWebQ
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*The Web as a Knowledge-base for Answering Complex Questions*.<br> Alon Talmor, Jonathan Berant. **NAACL-18**

<span class="fs-1">
[Paper](https://arxiv.org/abs/1911.11641){: .btn .btn-blue .mr-1 target="_blank" } [Official Link](https://www.tau-nlp.org/compwebq){: target="_blank" .btn .btn-green .mr-1 } {: target="_blank" .btn .btn-purple .mr-1 } </span>

> - **Knowledge**: Knowledge Graph
> - **User Input**: Question
> - **Output**: Answer
> - **Keywords**: Weakly supervised; Multihop

><details markdown="block">
>  <summary>Illustrative Example</summary>
>  {: .fs-3 .text-delta .text-red-100}
```
..
```
></details> 

### HybridQA
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*HybridQA: A Dataset of Multi-Hop Question Answering over Tabular and Textual Data*.<br> Wenhu Chen, Hanwen Zha, Zhiyu Chen, Wenhan Xiong, Hong Wang, William Wang. **EMNLP-20**

<span class="fs-1">
[Paper](https://aclanthology.org/2020.findings-emnlp.91/){: .btn .btn-blue .mr-1 target="_blank" } [Official Link](https://hybridqa.github.io/){: target="_blank" .btn .btn-green .mr-1 } </span>
> - **Knowledge**: Table + Text passages
> - **User Input**: Question
> - **Output**: Answer
> - **Keywords**: Multi-hop; Short-form entity/extractive

><details markdown="block">
>  <summary>Illustrative Example</summary>
>  {: .fs-3 .text-delta .text-red-100}
```
..
```
></details> 

### OTT-QA
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*Open Question Answering over Tables and Text *.<br> Wenhu Chen, Ming-Wei Chang, Eva Schlinger, William Yang Wang, William W. Cohen. **ICLR-21**

<span class="fs-1">
[Paper](https://arxiv.org/abs/2010.10439){: .btn .btn-blue .mr-1 target="_blank" } [Official Link](https://github.com/wenhuchen/OTT-QA){: target="_blank" .btn .btn-green .mr-1 } </span>
> - **Knowledge**: Table + Text passages
> - **User Input**: Question
> - **Output**: Answer
> - **Keywords**: More open table/text; Extractive ans

><details markdown="block">
>  <summary>Illustrative Example</summary>
>  {: .fs-3 .text-delta .text-red-100}
```
..
```
></details> 

### MultiModalQA
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*MultiModalQA: Complex Question Answering over Text, Tables and Images*.<br> Alon Talmor, Ori Yoran, Amnon Catav, Dan Lahav, Yizhong Wang, Akari Asai, Gabriel Ilharco, Hannaneh Hajishirzi, Jonathan Berant. **ICLR-21**

<span class="fs-1">
[Paper](https://arxiv.org/abs/1709.00103){: .btn .btn-blue .mr-1 target="_blank" } [Official Link](https://allenai.github.io/multimodalqa/){: target="_blank" .btn .btn-green .mr-1 } </span>
> - **Knowledge**: Table + Text + Images
> - **User Input**: Question
> - **Output**: Answer
> - **Keywords**: Short-form entity/extractive

><details markdown="block">
>  <summary>Illustrative Example</summary>
>  {: .fs-3 .text-delta .text-red-100}
```
..
```
></details> 


### FeTaQA
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*FeTaQA: Free-form Table Question Answering*.<br> Linyong Nan, Chiachun Hsieh, Ziming Mao, Xi Victoria Lin, Neha Verma, Rui Zhang, Wojciech Kryściński, Nick Schoelkopf, Riley Kong, Xiangru Tang, Murori Mutuma, Ben Rosand, Isabel Trindade, Renusree Bandaru, Jacob Cunningham, Caiming Xiong, Dragomir Radev. **ICLR-18**

<span class="fs-1">
[Paper](https://arxiv.org/abs/2104.00369){: .btn .btn-blue .mr-1 target="_blank" } [Official Link](https://github.com/Yale-LILY/FeTaQA){: target="_blank" .btn .btn-green .mr-1 } </span>
> - **Knowledge**: Table
> - **User Input**: Question
> - **Output**: Free-Form Answer
> - **Keywords**: Free-form answer

><details markdown="block">
>  <summary>Illustrative Example</summary>
>  {: .fs-3 .text-delta .text-red-100}
```
..
```
></details> 

### TAT-QA
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*TAT-QA: A Question Answering Benchmark on a Hybrid of Tabular and Textual Content in Finance*.<br> Fengbin Zhu, Wenqiang Lei, Youcheng Huang, Chao Wang, Shuo Zhang, Jiancheng Lv, Fuli Feng, Tat-Seng Chua. **ACL-21**

<span class="fs-1">
[Paper](https://aclanthology.org/2021.acl-long.254/){: .btn .btn-blue .mr-1 target="_blank" } [Official Link](https://nextplusplus.github.io/TAT-QA/){: target="_blank" .btn .btn-green .mr-1 } </span>
> - **Knowledge**: Table + Text
> - **User Input**: Question
> - **Output**: Answer(diverse form, including single span, multiple spans and free-form)
> - **Keywords**: Context hybrid; Numerical reasoning; Financial

><details markdown="block">
>  <summary>Illustrative Example</summary>
>  {: .fs-3 .text-delta .text-red-100}
```
..
```
></details> 

---



## Data-to-Text

### E2E
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*The E2E Dataset: New Challenges For End-to-End Generation*.<br> Jekaterina Novikova, Ondřej Dušek, Verena Rieser. **SIGDIAL-17**

<span class="fs-1">
[Paper](https://aclanthology.org/W17-5525/){: .btn .btn-blue .mr-1 target="_blank" } [Official Link](http://www.macs.hw.ac.uk/InteractionLab/E2E/){: target="_blank" .btn .btn-green .mr-1 } </span>
> - **Knowledge**: Table
> - **User Input**: None
> - **Output**: Text
> - **Keywords**: Text generation; Restaurant domain

><details markdown="block">
>  <summary>Illustrative Example</summary>
>  {: .fs-3 .text-delta .text-red-100}
```
..
```
></details> 
> **Comments**
> There is another [E2E cleaned version](https://github.com/tuetschek/e2e-cleaning) released by authors. 
<!-- Mention the highlights or known issues of the dataset. -->
{: .fs-4 .fw-600 .text-red-300}

### WebNLG
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*The WebNLG Challenge: Generating Text from RDF Data*.<br> Claire Gardent, Anastasia Shimorina, Shashi Narayan, Laura Perez-Beltrachini. **ICLR-21**

<span class="fs-1">
[Paper](https://aclanthology.org/W17-3518/){: .btn .btn-blue .mr-1 target="_blank" } [Official Link](https://webnlg-challenge.loria.fr/){: target="_blank" .btn .btn-green .mr-1 } </span>
> - **Knowledge**: Knowledge Graph(triples)
> - **User Input**: None
> - **Output**: Text
> - **Keywords**: Text generation; RDF Triples

><details markdown="block">
>  <summary>Illustrative Example</summary>
>  {: .fs-3 .text-delta .text-red-100}
```
..
```
></details> 
> **Comments**
> WebNLG challenge has many datasets available, the widely used dversion currently is the WebNLG challenge 2017. There is a useful [link](https://github.com/fuzihaofzh/webnlg-dataset) for summarization of this.
<!-- Mention the highlights or known issues of the dataset. -->
{: .fs-4 .fw-600 .text-red-300}

### DART
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*DART: Open-Domain Structured Data Record to Text Generation*.<br> Linyong Nan, Dragomir Radev, Rui Zhang, Amrit Rau, Abhinand Sivaprasad, Chiachun Hsieh, Xiangru Tang, Aadit Vyas, Neha Verma, Pranav Krishna, Yangxiaokang Liu, Nadia Irwanto, Jessica Pan, Faiaz Rahman, Ahmad Zaidi, Mutethia Mutuma, Yasin Tarabar, Ankit Gupta, Tao Yu, Yi Chern Tan, Xi Victoria Lin, Caiming Xiong, Richard Socher, Nazneen Fatema Rajani. **ICLR-21**

<span class="fs-1">
[Paper](https://aclanthology.org/2021.naacl-main.37/){: .btn .btn-blue .mr-1 target="_blank" } [Official Link](https://github.com/Yale-LILY/dart){: target="_blank" .btn .btn-green .mr-1 } </span>
> - **Knowledge**: Triples
> - **User Input**: None
> - **Output**: Text
> - **Keywords**: Text generation; Large data; E2E and WebNLG contained

><details markdown="block">
>  <summary>Illustrative Example</summary>
>  {: .fs-3 .text-delta .text-red-100}
```
..
```
></details> 

### ToTTo
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*ToTTo: A Controlled Table-To-Text Generation Dataset*.<br> Ankur P. Parikh, Xuezhi Wang, Sebastian Gehrmann, Manaal Faruqui, Bhuwan Dhingra, Diyi Yang, Dipanjan Das. **ICLR-21**

<span class="fs-1">
[Paper](https://arxiv.org/abs/2004.14373){: .btn .btn-blue .mr-1 target="_blank" } [Official Link](https://github.com/google-research-datasets/ToTTo){: target="_blank" .btn .btn-green .mr-1 } </span>
> - **Knowledge**: Highlighted Table
> - **User Input**: None
> - **Output**: Text
> - **Keywords**: Highlighted Table; Text generation

><details markdown="block">
>  <summary>Illustrative Example</summary>
>  {: .fs-3 .text-delta .text-red-100}
```
..
```
></details> 

### LogicNLG
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*Logical Natural Language Generation from Open-Domain Tables*.<br> Wenhu Chen, Jianshu Chen, Yu Su, Zhiyu Chen, William Yang Wang. **ACL-20**

<span class="fs-1">
[Paper](https://aclanthology.org/2020.acl-main.708/){: .btn .btn-blue .mr-1 target="_blank" } [Official Link](https://github.com/wenhuchen/LogicNLG){: target="_blank" .btn .btn-green .mr-1 } </span>
> - **Knowledge**: Table
> - **User Input**: None
> - **Output**: Logical Natural Language Generation
> - **Keywords**: Logical NL generation

><details markdown="block">
>  <summary>Illustrative Example</summary>
>  {: .fs-3 .text-delta .text-red-100}
```
..
```
></details>
---

## Conversational

### MultiWoZ21
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*MultiWOZ 2.1: A Consolidated Multi-Domain Dialogue Dataset with State Corrections and State Tracking Baselines*.<br> Mihail Eric, Rahul Goel, Shachi Paul, Abhishek Sethi, Sanchit Agarwal, Shuyang Gao, Adarsh Kumar, Anuj Kumar Goyal, Peter Ku, Dilek Hakkani-Tür. **LREC-20**

<span class="fs-1">
[Paper](https://arxiv.org/abs/1907.01669){: .btn .btn-blue .mr-1 target="_blank" } [Official Link](https://github.com/budzianowski/multiwoz){: target="_blank" .btn .btn-green .mr-1 } </span>
> - **Knowledge**: Ontology
> - **User Input**: Dialogue
> - **Output**: Dialogue State
> - **Keywords**: Dialog system

><details markdown="block">
>  <summary>Illustrative Example</summary>
>  {: .fs-3 .text-delta .text-red-100}
```
..
```
></details> 

> **Comments**
> There are many version of MultiWoZ, 2.1 and 2.2 are mostly used currently. We used the 2.1 version in our SKG benchmark. Some pre-procession on this dataset is needed, pls refer to [MultiWoZ](https://github.com/budzianowski/multiwoz) and [Trade-DST](https://github.com/jasonwu0731/trade-dst).
<!-- Mention the highlights or known issues of the dataset. -->
{: .fs-4 .fw-600 .text-red-300}




### KVRET(SMD)
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*Key-Value Retrieval Networks for Task-Oriented Dialogue*.<br> Mihail Eric, Christopher D. Manning. **SIGdial-17**

<span class="fs-1">
[Paper](https://aclanthology.org/W17-5506/){: .btn .btn-blue .mr-1 target="_blank" } [Official Link](https://nlp.stanford.edu/blog/a-new-multi-turn-multi-domain-task-oriented-dialogue-dataset/){: target="_blank" .btn .btn-green .mr-1 } </span>
> - **Knowledge**: Table
> - **User Input**: Statement
> - **Output**: Boolean
> - **Keywords**: Dialogue system; Each dialogue has a seperate table as kb

><details markdown="block">
>  <summary>Illustrative Example</summary>
>  {: .fs-3 .text-delta .text-red-100}
```
..
```
></details> 

> **Comments**
> KVRET is also called SMD(Stanford Multi-Domain task-oriented dialogue dataset). The de-facto widely used version of this dataset is the pre-processed verison in [Mem2seq](https://github.com/HLTCHKUST/Mem2Seq).
<!-- Mention the highlights or known issues of the dataset. -->
{: .fs-4 .fw-600 .text-red-300}

### SParC
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*Submitted on 5 Jun 2019]
SParC: Cross-Domain Semantic Parsing in Context*.<br> Tao Yu, Rui Zhang, Michihiro Yasunaga, Yi Chern Tan, Xi Victoria Lin, Suyi Li, Heyang Er, Irene Li, Bo Pang, Tao Chen, Emily Ji, Shreya Dixit, David Proctor, Sungrok Shim, Jonathan Kraft, Vincent Zhang, Caiming Xiong, Richard Socher, Dragomir Radev. **ACL-19**

<span class="fs-1">
[Paper](https://arxiv.org/abs/1906.02285){: .btn .btn-blue .mr-1 target="_blank" } [Official Link](https://yale-lily.github.io/sparc){: target="_blank" .btn .btn-green .mr-1 } </span>
> - **Knowledge**: Database
> - **User Input**: Multi-turn query
> - **Output**: SQL
> - **Keywords**: Fully supervised semantic parsing; Multi-turn

><details markdown="block">
>  <summary>Illustrative Example</summary>
>  {: .fs-3 .text-delta .text-red-100}
```
..
```
></details> 

### CoSQL
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*CoSQL: A Conversational Text-to-SQL Challenge Towards Cross-Domain Natural Language Interfaces to Databases*.<br> Tao Yu, Rui Zhang, He Yang Er, Suyi Li, Eric Xue, Bo Pang, Xi Victoria Lin, Yi Chern Tan, Tianze Shi, Zihan Li, Youxuan Jiang, Michihiro Yasunaga, Sungrok Shim, Tao Chen, Alexander Fabbri, Zifan Li, Luyao Chen, Yuwen Zhang, Shreya Dixit, Vincent Zhang, Caiming Xiong, Richard Socher, Walter S Lasecki, Dragomir Radev. **EMNLP-19**

<span class="fs-1">
[Paper](https://arxiv.org/abs/1909.05378){: .btn .btn-blue .mr-1 target="_blank" } [Official Link](https://yale-lily.github.io/cosql){: target="_blank" .btn .btn-green .mr-1 } </span>
> - **Knowledge**: Database
> - **User Input**: Dialog
> - **Output**: SQL
> - **Keywords**: Fully supervised semantic parsing; Dialogue

><details markdown="block">
>  <summary>Illustrative Example</summary>
>  {: .fs-3 .text-delta .text-red-100}
```
..
```
></details>

### SQA(MSR SQA)
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*Search-based Neural Structured Learning for Sequential Question Answering*.<br> Mohit Iyyer, Wen-tau Yih, Ming-Wei Chang. **ACL-17**

<span class="fs-1">
[Paper](https://aclanthology.org/P17-1167){: .btn .btn-blue .mr-1 target="_blank" } [Official Link](hhttps://www.microsoft.com/en-us/download/details.aspx?id=54253){: target="_blank" .btn .btn-green .mr-1 } </span>
> - **Knowledge**: Table
> - **User Input**: Multi-turn query
> - **Output**: Answer
> - **Keywords**: Weakly supervised semantic parsing; Sequential

><details markdown="block">
>  <summary>Illustrative Example</summary>
>  {: .fs-3 .text-delta .text-red-100}
```
..
```
></details> 

---

## Fact Verification

### TabFact
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*TabFact: A Large-scale Dataset for Table-based Fact Verification*.<br> Wenhu Chen, Hongmin Wang, Jianshu Chen, Yunkai Zhang, Hong Wang, Shiyang Li, Xiyou Zhou, William Yang Wang. **ICLR-20**

<span class="fs-1">
[Paper](https://arxiv.org/abs/1909.02164){: .btn .btn-blue .mr-1 target="_blank" } [Official Link](https://tabfact.github.io/){: target="_blank" .btn .btn-green .mr-1 } </span>
> - **Knowledge**: Table
> - **User Input**: Statement
> - **Output**: Boolean
> - **Keywords**: NL inference; Large data

><details markdown="block">
>  <summary>Illustrative Example</summary>
>  {: .fs-3 .text-delta .text-red-100}
```
..
```
></details> 

### FEVEROUS
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*The Fact Extraction and VERification Over Unstructured and Structured information (FEVEROUS) Shared Task*.<br> Rami Aly, Zhijiang Guo, Michael Sejr Schlichtkrull, James Thorne, Andreas Vlachos, Christos Christodoulopoulos, Oana Cocarascu, Arpit Mittal. **ICLR-21**

<span class="fs-1">
[Paper](https://aclanthology.org/2021.fever-1.1){: .btn .btn-blue .mr-1 target="_blank" } [Official Link](https://fever.ai/dataset/feverous.html){: target="_blank" .btn .btn-green .mr-1 } </span>
> - **Knowledge**: Table + Text
> - **User Input**: Statement
> - **Output**: Boolean
> - **Keywords**: NL inference; Large data

><details markdown="block">
>  <summary>Illustrative Example</summary>
>  {: .fs-3 .text-delta .text-red-100}
```
..
```
></details>


---

## Formal-Language-to-Text

### SQL2Text
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*Logic-Consistency Text Generation from Semantic Parses*.<br> Chang Shu, Yusen Zhang, Xiangyu Dong, Peng Shi, Tao Yu, Rui Zhang. **ACL-21**

<span class="fs-1">
[Paper](https://aclanthology.org/2021.findings-acl.388/){: .btn .btn-blue .mr-1 target="_blank" } [Official Link](https://github.com/Ciaranshu/relogic){: target="_blank" .btn .btn-green .mr-1 } </span>
> - **Knowledge**: Optional Database
> - **User Input**: SQL
> - **Output**: Text
> - **Keywords**: High-fidelity NLG

><details markdown="block">
>  <summary>Illustrative Example</summary>
>  {: .fs-3 .text-delta .text-red-100}
```
..
```
></details> 

### Logic2Text
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*Logic2Text: High-Fidelity Natural Language Generation from Logical Forms*.<br> Zhiyu Chen, Wenhu Chen, Hanwen Zha, Xiyou Zhou, Yunkai Zhang, Sairam Sundaresan, William Yang Wang. **EMNLP-20**

<span class="fs-1">
[Paper](https://arxiv.org/abs/2004.14579){: .btn .btn-blue .mr-1 target="_blank" } [Official Link](https://github.com/czyssrs/Logic2Text){: target="_blank" .btn .btn-green .mr-1 } </span>
> - **Knowledge**: Table Schema
> - **User Input**: Python-like program
> - **Output**: Text
> - **Keywords**: High-fidelity NLG

><details markdown="block">
>  <summary>Illustrative Example</summary>
>  {: .fs-3 .text-delta .text-red-100}
```
..
```
></details> 

---

## Other Related Datasets



