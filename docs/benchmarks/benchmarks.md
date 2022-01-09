---
layout: default
title: Benchmark Datasets
nav_order: 3
toc_list: true
last_modified_date: March 26 2021
permalink: /datasets/
---

# Benchmark Datasets for Structured Knowledge Grounding
{: .no_toc }

Editors: [Tianbao Xie](https://tianbaoxie.com){: target="_blank"}

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
[Paper](https://aclanthology.org/P16-2033/){: .btn .btn-blue .mr-1 target="_blank" } [Official Link](http://aka.ms/WebQSP){: target="_blank" .btn .btn-green .mr-1 } {: target="_blank" .btn .btn-purple .mr-1 } </span>

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

---



## Data-to-Text
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



## Conversational


## Fact Verification


## Formal-Language-to-Text


## Other Related Datasets

### Rainbow Benchmark
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*UNICORN on RAINBOW: A Universal Commonsense Reasoning Model on a New Multitask Benchmark*.<br> Nicholas Lourie, Ronan Le Bras, Chandra Bhagavatula, Yejin Choi. **AAAI-21**

<span class="fs-1">
[Paper](https://arxiv.org/abs/2103.13009){: .btn .btn-blue .mr-1 target="_blank" } [Official Link](https://allenai.org/data/rainbow){: target="_blank" .btn .btn-green .mr-1 } </span>

> - **Topics:** Rainbow is a universal commonsense reasoning benchmark spanning both social and physical common sense. Rainbow brings together 6 existing commonsense reasoning tasks: aNLI, Cosmos QA, HellaSWAG, Physical IQa, Social IQa, and WinoGrande.
- **Task format:**  text-to-text 
- **Size & Split:**  
- **Dataset creation:** reformatting specific versions of the above datasets to a text-to-text format so that models like T5 and BART.



### GLUE and SuperGLUE Benchmark 
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding*. <br> Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, Samuel R. Bowman. **ICLR-19**

<span class="fs-1">
[Paper](https://arxiv.org/abs/1804.07461){: .btn .btn-blue .mr-1 target="_blank" } [Official Link](https://gluebenchmark.com/){: target="_blank" .btn .btn-green .mr-1 } [Huggingface Card](https://huggingface.co/datasets/glue){: target="_blank" .btn .btn-purple .mr-1 }</span>



### LocatedNear Relation Extraction
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*Automatic Extraction of Commonsense LocatedNear Knowledge*. <br> Frank F. Xu, Bill Yuchen Lin, Kenny Q. Zhu. **ACL-18**

<span class="fs-1">
[Paper](https://arxiv.org/abs/1711.04204){: .btn .btn-blue .mr-1 target="_blank" } [Github Page](https://github.com/adapt-sjtu/commonsense-locatednear){: target="_blank" .btn .btn-grey .mr-1 } </span>

> - **Topics:** Objects. Mostly about physical objects that are typically found near each other in real life.
- **Task format:** Task 1 -- judge if a sentence describes two objects (mentioned in the sentence) being physically close by; Task 2 -- produce a ranked list of LOCATEDNEAR facts with the given classified results of large number of sentences.
- **Size & Split:** 5,000 sentences describe a scene of two physical objects and with a label indicating if the two objects are co-located in the scene --- train(4,000), test(1,000).
- **Dataset creation:** 
><details markdown="block">
>  <summary>Illustrative Example</summary>
>  {: .fs-3 .text-delta .text-red-100}
```
ID: 9888840
Sentence: In a few minutes more the mission ship was forsaken by her strange Sabbath congregation, and left with all the fleet around her floating quietly on the tranquil sea.	
Object 1: ship
Object 2: sea
Confidence: 1
```
></details> 
<!-- {: .fs-4 .fw-600 .text-red-300}
> **Comments** -->
