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
> - **Type**: Text2SQL
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
> - **Type**: Text2S-Expression
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
> - **Type**: Text2S-Expression

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
> - **Type**: Text2TOP-representation
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
> - **Type**: QA
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
> - **Type**: QA
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
> - **Type**: Text2SPARQL
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
> - **Type**: QA
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
> - **Type**: QA
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
> - **Type**: QA
> - **Keywords**: Free-form answer

><details markdown="block">
>  <summary>Illustrative Example</summary>
>  {: .fs-3 .text-delta .text-red-100}
```
..
```
></details> 

---



## Open-Ended QA

### ProtoQA
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*ProtoQA: A Question Answering Dataset for Prototypical Common-Sense Reasoning*. <br> Michael Boratko, Xiang Lorraine Li, Tim O’Gorman, Rajarshi Das, Dan Le, Andrew McCallum. **EMNLP-20**

<span class="fs-1">
[Paper](https://arxiv.org/abs/2005.00771){: .btn .btn-blue .mr-1 target="_blank" } [Github Page](https://github.com/iesl/protoqa-data){: target="_blank" .btn .btn-grey .mr-1 } [Huggingface Card](https://huggingface.co/datasets/proto_qa){: target="_blank" .btn .btn-purple .mr-1 } </span>

> - **Topics:** Prototypical situation.
- **Task format:** Given a question, a model is has to output a ranked list of answers covering multiple categories.
- **Size & Split:**  5,733 in total --- train (8,781), dev (1,030), test (102).
- **Dataset creation:** 
><details markdown="block">
>  <summary>Illustrative Example</summary>
>  {: .fs-3 .text-delta .text-red-100}
```
Question:
    Name a piece of equipment that you are likely to find at your office and not at home?
Categories: 
    printer/copier (37), office furniture (15), computer equipment (17), stapler (11), files (10), office appliances (5), security systems (1)
```
></details> 
<!-- {: .fs-4 .fw-600 .text-red-300}
> **Comments** -->


### OpenCSR
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*Differentiable Open-Ended Commonsense Reasoning*. <br> Bill Yuchen Lin, Haitian Sun, Bhuwan Dhingra, Manzil Zaheer, Xiang Ren, William W. Cohen. **NAACL-21**

<span class="fs-1">
[Paper](https://arxiv.org/abs/2010.14439){: .btn .btn-blue .mr-1 target="_blank" }</span>

> - **Topics:** Science. Most of the questions in the dataset are naturally occurring generic statements.
- **Task format:** Given an open-ended question, the model will output a weighted set of concepts.
- **Size & Split:**  19,520 in total --- train (15,800), dev (1,756), test (1,965).
- **Dataset creation:** 
><details markdown="block">
>  <summary>Illustrative Example</summary>
>  {: .fs-3 .text-delta .text-red-100}
```
Question: What can help alleviate global warming?
Supporting Facts: 
    f1: Carbon dioxide is the major greenhouse gas contributing to global warming.
    f2: Trees remove carbon dioxide from the atmosphere through photosynthesis.
    f3: The atmosphere contains oxygen, carbon dioxide, and water.
Weighted Answers: Renewable energy (w1), tree (w2), solar battery (w3)
```
></details> 
<!-- {: .fs-4 .fw-600 .text-red-300}
> **Comments** -->

---

## Constrained NLG

### CommonGen
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*CommonGen: A Constrained Text Generation Challenge for Generative Commonsense Reasoning*. <br> Bill Yuchen Lin, Wangchunshu Zhou, Ming Shen, Pei Zhou, Chandra Bhagavatula, Yejin Choi, Xiang Ren. **EMNLP-20 Findings**

<span class="fs-1">
[Paper](https://arxiv.org/abs/1911.03705){: .btn .btn-blue .mr-1 target="_blank" } [Official Link](https://inklab.usc.edu/CommonGen/){: target="_blank" .btn .btn-green .mr-1 } [Huggingface Card](https://huggingface.co/datasets/common_gen){: target="_blank" .btn .btn-purple .mr-1 } </span>

> - **Topics:** General. A wide range of concepts from everyday scenario. 
- **Task format:** Given a set of common concepts, the task is to generate a coherent sentence describing an everyday scenario using these concepts.
- **Size & Split:** 35,141 concept-sets in total --- train (32,651), dev (993), test (1,497). 
- **Dataset creation:** 
><details markdown="block">
>  <summary>Illustrative Example</summary>
>  {: .fs-3 .text-delta .text-red-100}
```
Common Concepts: {dog, frisbee, catch, throw}
Output:
GPT2 -- A dog throws a frisbee at a football player.
UniLM -- Two dogs are throwing frisbees at each other.
BART -- A dog throws a frisbee and a dog catches it.
T5 -- dog catches a frisbee and throws it to a dog.
```
></details> 
<!-- {: .fs-4 .fw-600 .text-red-300}
> **Comments** -->

### Cos-E 
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*Explain Yourself! Leveraging Language Models for Commonsense Reasoning*. <br> Nazneen Fatema Rajani, Bryan McCann, Caiming Xiong, Richard Socher. **ACL-19**

<span class="fs-1">
[Paper](https://arxiv.org/abs/1906.02361){: .btn .btn-blue .mr-1 target="_blank" } [Github Page](https://github.com/salesforce/cos-e){: target="_blank" .btn .btn-grey .mr-1 } [Huggingface Card](https://huggingface.co/datasets/cos_e){: target="_blank" .btn .btn-purple .mr-1 } </span>

> - **Topics:** General. Most questions in the dataset is based on everyday scenario and events. 
- **Task format:** Given a question, a model will return an explanation with the correct answer to the question.
- **Size & Split:** 10,952 in total --- train (9,741), dev (1,211).
- **Dataset creation:** 
><details markdown="block">
>  <summary>Illustrative Example</summary>
>  {: .fs-3 .text-delta .text-red-100}
```
Question: While eating a hamburger with friends, what are people trying to do?
Choices: have fun, tasty, or indigestion
CoS-E: Usually a hamburger with friends indicates a good time.
Correct Choice: have fun
```
></details> 
<!-- {: .fs-4 .fw-600 .text-red-300}
> **Comments** -->


### ComVE (SubTask C)
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*SemEval-2020 Task 4: Commonsense Validation and Explanation*. <br> Cunxiang Wang, Shuailong Liang, Yili Jin, Yilong Wang, Xiaodan Zhu, Yue Zhang. **SemEval-20**

<span class="fs-1">
[Paper](https://arxiv.org/abs/2007.00236){: .btn .btn-blue .mr-1 target="_blank" } [Github Page](https://github.com/wangcunxiang/SemEval2020-Task4-Commonsense-Validation-and-Explanation){: target="_blank" .btn .btn-grey .mr-1 } </span>

> - **Topics:** Commonsense explanation to nonsensical statement. 
- **Task format:** Given a nonsensical statement, the task is to generate the reason why this statement does not make sense.
- **Size & Split:** 11,997 8-sentence tuples in total --- train (10,000), dev (997), test (1,000). 
- **Dataset creation:** 
><details markdown="block">
>  <summary>Illustrative Example</summary>
>  {: .fs-3 .text-delta .text-red-100}
```
Task C: Commonsense Explanation (Generation)
Generate the reason why this statement is against common sense and we will use BELU to evaluate it.
    Statement: He put an elephant into the fridge.
    Referential Reasons:
        i. An elephant is much bigger than a fridge.
        ii. A fridge is much smaller than an elephant.
        iii. Most of the fridges aren’t large enough to contain an elephant.
```
></details> 
<!-- {: .fs-4 .fw-600 .text-red-300}
> **Comments** -->


---


## LM Probing Tasks

### LAMA Probes 
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*Language Models as Knowledge Bases?*. <br> Fabio Petroni, Tim Rocktäschel, Patrick Lewis, Anton Bakhtin, Yuxiang Wu, Alexander H. Miller, Sebastian Riedel. **EMNLP-19**

<span class="fs-1">
[Paper](https://arxiv.org/abs/1909.01066){: .btn .btn-blue .mr-1 target="_blank" } [Github Page](https://github.com/facebookresearch/LAMA){: target="_blank" .btn .btn-grey .mr-1 } [Huggingface Card](https://huggingface.co/datasets/lama){: target="_blank" .btn .btn-purple .mr-1 } </span>

> - **Topics:** General. LAMA is a probe for analyzing the factual and commonsense knowledge contained in pretrained language models.
- **Task format:** Given a pretrained language model knows a fact (subject, relation, object) such as (Dante,
born-in, Florence), the task should predict masked objects in cloze sentences such as “Dante was born in ___” expressing that fact.
- **Size & Split:** N/A
- **Dataset creation:** 
><details markdown="block">
>  <summary>Illustrative Example</summary>
>  {: .fs-3 .text-delta .text-red-100}
The ConceptNet config has the following fields:
```
masked_sentence: One of the things you do when you are alive is [MASK].
negated: N/A
obj: think
obj_label: think
pred: HasSubevent, 
sub: alive
uuid: d4f11631dde8a43beda613ec845ff7d1
```
></details> 
<!-- {: .fs-4 .fw-600 .text-red-300}
> **Comments** -->


### NumerSense
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*Birds have four legs?! NumerSense: Probing Numerical Commonsense Knowledge of Pre-trained Language Models*. <br> Bill Yuchen Lin, Seyeon Lee, Rahul Khanna, Xiang Ren. **EMNLP-20**

<span class="fs-1">
[Paper](https://arxiv.org/abs/2005.00683){: .btn .btn-blue .mr-1 target="_blank" } [Github Page](https://github.com/INK-USC/NumerSense){: target="_blank" .btn .btn-grey .mr-1 } [Huggingface Card](https://huggingface.co/datasets/numer_sense){: target="_blank" .btn .btn-purple .mr-1 } </span>

> - **Topics:** Numerical commonsense. The dataset contains probes from a wide range of categories, including objects, biology, geometry, unit, math, physics, geography, etc. 
- **Task format:** Given a masked sentence, the task is to choose the correct numerical answer from all provided choices. 
- **Size & Split:** 13.6k masked-word-prediction probes in total --- fine-tune (10.5k), test (3.1k). 
- **Dataset creation:** 
><details markdown="block">
>  <summary>Illustrative Example</summary>
>  {: .fs-3 .text-delta .text-red-100}
```
Question: A car usually has [MASK] wheels.
Choices: 
A) One  B) Two  C) Three  D) Four  E) Five
```
></details> 
<!-- {: .fs-4 .fw-600 .text-red-300}
> **Comments** -->

### RICA
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*RICA: Evaluating Robust Inference Capabilities Based on Commonsense Axioms*. <br> Pei Zhou, Rahul Khanna, Seyeon Lee, Bill Yuchen Lin, Daniel Ho, Jay Pujara, Xiang Ren. **arXiv 2020, accepted in EMNLP-Findings 2020**

<span class="fs-1">
[Paper](https://arxiv.org/abs/2005.00782){: .btn .btn-blue .mr-1 target="_blank" } [Projec Page](https://sites.google.com/usc.edu/rica){: target="_blank" .btn .btn-grey .mr-1 } [Leaderboard](https://eval.ai/web/challenges/challenge-page/832/overview){: target="_blank" .btn .btn-purple .mr-1 } </span>

> - **Topics:** General. The dataset contains probes from different types of commonsense, such as physical, material, and social properties. It focuses on logicaly-equivalent probes to test models' robust inference abilities. It also uses unseen strings as entities to separate fact-based recall from abstract reasoning capabilities.
- **Task format:** Given a masked sentence and two choices for the mask, the task is to selesct the correct choice. 
- **Size & Split:** Two evaluation settings with the same test data of 1.6k human-curated probes: 1. zero-shot setting; 2. models are fine-tuned on 9k of the human-verified RICA probes (8k for training and 1k for validation).
- **Dataset creation:** 
><details markdown="block">
>  <summary>Illustrative Example</summary>
>  {: .fs-3 .text-delta .text-red-100}
```
Question: A prindag is lighter than a fluberg, so a prindag should float [MASK] than a fluberg.
Choices: 
A) more  B) less
```
></details> 
<!-- {: .fs-4 .fw-600 .text-red-300}
> **Comments** -->

---

## Reading Comprehension 

### ReCoRD
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*ReCoRD: Bridging the Gap between Human and Machine Commonsense Reading Comprehension*.<br> Sheng Zhang, Xiaodong Liu, Jingjing Liu, Jianfeng Gao, Kevin Duh, Benjamin Van Durme. **arXiv, 2018**

<span class="fs-1">
[Paper](https://arxiv.org/abs/1810.12885){: .btn .btn-blue .mr-1 target="_blank" } [Official Link](https://sheng-z.github.io/ReCoRD-explorer){: target="_blank" .btn .btn-green .mr-1 } </span>

> - **Topics:** Commonsense-based reading comprehension with a focus on news articles. 
- **Task format:** Given a passage, a set of text spans marked in the passage, and a cloze-style query with a missing text span, a model must select a text span that best fits the query.
- **Size & Split:** Queries/Passages 120,730/80,121 in total --- train (100,730/65,709), dev (10,000/7,133), test (10,000/(7,279).
- **Dataset creation:** 
><details markdown="block">
>  <summary>Illustrative Example</summary>
>  {: .fs-3 .text-delta .text-red-100}
```
Passage: 
    (**CNN**) -- A lawsuit has been filed claiming that the iconic **Led Zeppelin** song "**Stairway to Heaven**" was far from original. The suit, filed on May 31 in the **United States District Court Eastern District of Pennsylvania**, was brought by the estate of the late musician **Randy California** against the surviving members of **Led Zeppelin** and their record label. The copyright infringement case alleges that the **Zeppelin** song was taken from the single "**Taurus**" by the 1960s band **Spirit**, for whom **California** served as lead guitarist. "Late in 1968, a then new band named **Led Zeppelin** began touring in the **United States**, opening for **Spirit**," the suit states. "It was during this time that **Jimmy Page**, **Led Zeppelin**'s guitarist, grew familiar with '**Taurus**' and the rest of **Spirit**'s catalog. **Page** stated in interviews that he found **Spirit** to be 'very good' and that the band's performances struck him 'on an emotional level.' "
• Suit claims similarities between two songs
• **Randy California** was guitarist for the group **Spirit**
• **Jimmy Page** has called the accusation "ridiculous"
(Cloze-style) Query:
    According to claims in the suit, "Parts of 'Stairway to Heaven,' instantly recognizable to the music fans across the world, sound almost identical to significant portions of ‘___.’”
Reference Answers:
    Taurus
```
></details> 

### Cosmos QA
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*Cosmos QA : Machine Reading Comprehension with Contextual Commonsense Reasoning*.<br> Lifu Huang, Ronan Le Bras, Chandra Bhagavatula, Yejin Choi. **EMNLP-19**

<span class="fs-1">
[Paper](https://arxiv.org/abs/1909.00277){: .btn .btn-blue .mr-1 target="_blank" } [Official Link](https://wilburone.github.io/cosmos){: target="_blank" .btn .btn-green .mr-1 } [Huggingface Card](https://huggingface.co/datasets/cosmos_qa){: target="_blank" .btn .btn-purple .mr-1 } </span>

> - **Topics:** Commonsense-based reading comprehension with a focus on people's everyday narratives, asking questions concerning on the likely causes or effects of events that require reasoning beyond the exact text spans in the context. 
- **Task format:** Given a paragraph and a question, a model must select the correct answer from a set of choices.
- **Size & Split:** Questions/Paragraphs 35,588/21,866 in total --- train (25,588/13,715), dev (26,534/2,460), test (25,263/5,711).
- **Dataset creation:** 
><details markdown="block">
>  <summary>Illustrative Example</summary>
>  {: .fs-3 .text-delta .text-red-100}
```
Paragraph: 
    It's a very humbling experience when you need someone to dress you every morning, tie your shoes, and put your hair up. Every menial task takes an unprecedented amount of effort. It made me appreciate Dan even more. But anyway I shan't dwell on this (I'm not dying after all) and not let it detact from my lovely 5 days with my friends visiting from Jersey.
Question:
    What's a possible reason the writer needed someone to dress him every morning?
Choices:
    A) The writer doesn't like putting effort into these tasks.
    B) The writer has a physical disability.
    C) The writer is bad at doing his own hair.
    D) None of the above choices.
Correct Choice: B
```
></details> 


### DREAM
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*DREAM: A Challenge Dataset and Models for Dialogue-Based Reading Comprehension*.<br> Kai Sun, Dian Yu, Jianshu Chen, Dong Yu, Yejin Choi, Claire Cardie. **TACL-19**

<span class="fs-1">
[Paper](https://arxiv.org/abs/1902.00164){: .btn .btn-blue .mr-1 target="_blank" } [Official Link](https://dataset.org/dream/){: target="_blank" .btn .btn-green .mr-1 } [Huggingface Card](https://huggingface.co/datasets/dream){: target="_blank" .btn .btn-purple .mr-1 } </span>

> - **Topics:** General. It focuses on in-depth multi-turn multi-party dialogues covering a variety of topics and scenarios in daily life. 34% of questions involve commonsense reasoning.
- **Task format:** Given a dialogue and a question, a model must select the correct answer from a set of choices.
- **Size & Split:** Questions/Dialogues 10,197/6,444 in total --- train (6,116/3,869), dev (2,040/1,288), test (2,041/(1,287).
- **Dataset creation:** The dialogues, questions, and answers are collected from English-as-a-foreign-language examinations designed by human experts.

><details markdown="block">
>  <summary>Illustrative Example</summary>
>  {: .fs-3 .text-delta .text-red-100}
```
Dialogue: 
    W: Tom, look at your shoes. How dirty they are! You must clean them.
    M: Oh, mum, I just cleaned them yesterday.
    W: They are dirty now. You must clean them again.
    M: I do not want to clean them today. Even if I clean them today, they will get dirty again tomorrow.
    W: All right, then.
    M: Mum, give me something to eat, please.
    W: You had your breakfast in the morning, Tom, and you had lunch at school.
    M: I am hungry again.
    W: Oh, hungry? But if I give you something to eat today, you will be hungry again tomorrow.
Question:
    Why did the woman say that she wouldn’t give him anything to eat?
Choices:
    (A) Because his mother wants to correct his bad habit.
    (B) Because he had lunch at school.
    (C) Because his mother wants to leave him hungry.
Correct Choice: (A)
```
></details> 

 

### MCScript
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*MCScript: A Novel Dataset for Assessing Machine Comprehension Using Script Knowledge*.<br> Simon Ostermann, Ashutosh Modi, Michael Roth, Stefan Thater, Manfred Pinkal. **LREC-18**

<span class="fs-1">
[Paper](https://www.aclweb.org/anthology/L18-1564/){: .btn .btn-blue .mr-1 target="_blank" } [Official Link](http://www.coli.uni-saarland.de/~simono/page.php?id=datacode){: target="_blank" .btn .btn-green .mr-1 } </span>

> - **Topics:** Assession of the contribution of commonsense-based script knowledge to machine comprehension. Scripts are sequences of events describing stereotypical human activities.
- **Task format:** Given a script and a subset of related questions, a model must select the correct answer from a set of choices to each question.
- **Size & Split:** Approximately 2,100 texts and a total of approximately 14,000 questions in total --- train (9,731 questions on 1,470 texts), dev (1,411 questions on 219 texts), and test (2,797 questions on 430 texts).
- **Dataset creation:** 

><details markdown="block">
>  <summary>Illustrative Example</summary>
>  {: .fs-3 .text-delta .text-red-100}
```
T: I wanted to plant a tree. I went to the home and garden store and picked a nice oak. Afterwards, I planted it in my garden.
Q1: What was used to dig the hole?
A) a shovel  B) his bare hands
Correct Answer: A
Q2 When did he plant the tree?
A) after watering it  B) after taking it home
Correct Answer: B
```
></details> 


---





## Text Game

### TWC
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*Text-based RL Agents with Commonsense Knowledge: New Challenges, Environments and Baselines*. <br> Keerthiram Murugesan, Mattia Atzeni, Pavan Kapanipathi, Pushkar Shukla, Sadhana Kumaravel, Gerald Tesauro, Kartik Talamadupula, Mrinmaya Sachan, Murray Campbell. **AAAI-21**

<span class="fs-1">
[Paper](https://arxiv.org/abs/2010.03790){: .btn .btn-blue .mr-1 target="_blank" } [Github Page](https://github.com/IBM/commonsense-rl){: target="_blank" .btn .btn-grey .mr-1 } </span>

> - **Topics:** Objects. A specific kind of commonsense knowledge about objects, their attributes, and affordances.  
- **Task format:**  A new text-based gaming environment for training and evaluating RL agents. 
- **Size & Split:** In TWC doamin, there are 928 total entities, 872 total objects, 190 unique objects, 56 supporters/containers, and 8 rooms. 30 unique games in total.
- **Dataset creation:** 
><details markdown="block">
>  <summary>Illustrative Example</summary>
>  {: .fs-3 .text-delta .text-red-100}
Example of a game walkthrough belonging to the <em>easy</em> difficulty level.
<img src="../../images/benchmarks/twc.png" width="100%" height="auto" />
></details> 
<!-- {: .fs-4 .fw-600 .text-red-300}
> **Comments** -->

### AFLWorld
{: .no_toc }

---



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
