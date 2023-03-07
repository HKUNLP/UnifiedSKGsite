---
layout: default
title: Benchmark Datasets
nav_order: 3
toc_list: true
last_modified_date: May 16 2022
permalink: /benchmarks/
---

# Benchmark Datasets for Structured Knowledge Grounding
{: .no_toc }
In progress
{: .label .label-yellow }

{: .fs-5 .fw-300 }

We present a comprehensive collection of datasets of Structured Knowledge Grounding.


## Table of contents
{: .no_toc .text-delta .fs-4 style="font-weight:800"}

- TOC
{:toc}

## Summary
{: .no_toc }

|   Dataset |   Knowledge   |   User Input  |   Output  |   Keywords    |   Contain in UnifiedSKG   |
|	[Spider](https://unifiedskg.com/benchmarks/#spider)	|	 Database	|	 Question	|	 SQL	|	 Fully supervised; Cross-domain; Single turn	|	v1	|
|	[Single Domain Text2SQL](https://unifiedskg.com/benchmarks/#single-domain-text2sql)	|	 Database	|	 Question	|	 SQL	|	 Fully supervised; Single turn	|		|
|	[Squall](https://unifiedskg.com/benchmarks/#squall)	|	 Database(converted from table)	|	 Question	|	 SQL	|	 Fully supervised from 80% of WikiTQ	|		|
|	[KaggleDBQA](https://unifiedskg.com/benchmarks/#kagglekbqa)	|	 Database	|	 Question	|	 SQL	|	 Fully supervised; Realistic	|		|
|	[Spider-Syn](https://unifiedskg.com/benchmarks/#spider-syn)	|	 Database	|	 Question	|	 SQL	|	 Fully supervised; Robustness	|		|
|	[Spider-DK](https://unifiedskg.com/benchmarks/#spider-dk)	|	 Database	|	 Question	|	 SQL	|	 Fully supervised; Annotation of types of domain knowledge text-to-SQL needed.	|		|
|	[SEDE](https://unifiedskg.com/benchmarks/#sede)	|	 Database	|	 Question	|	 SQL	|	 Fully supervised; Real usage on the Stack Exchange website	|		|
|	[EHRSQL](https://unifiedskg.com/benchmarks/#EHRSQL)	|	Database 	|	Question	|	SQL 	|	 Fully supervised; Electronic health records (EHRs)	|		|
|	[Break](https://unifiedskg.com/benchmarks/#break)	|	 	|	 	|	 	|	 	|		|
|	[GrailQA](https://unifiedskg.com/benchmarks/#grailqa)	|	 Knowledge Graph	|	 Question	|	 S-Expression	|	 Large, 64k; Test generalization: i.i.d./compositional/zero-shot	|	v1	|
|	[WebQSP](https://unifiedskg.com/benchmarks/#webqsp)	|	 Knowledge Graph	|	 Question	|	 S-Expression	|	 -	|	v1	|
|	[Russ](https://unifiedskg.com/benchmarks/#russ)	|	 API	|	 Question	|	 Query	|	-	|		|
|	[MTOP](https://unifiedskg.com/benchmarks/#mtop)	|	 API	|	 Question	|	 TOP-representation	|	 Spoken Language Understanding; TOP representation	|	v1	|
|	[WebAPI](https://unifiedskg.com/benchmarks/#webapi)	|	 	|	 	|	 	|	 	|		|
|	[WikiSQL](https://unifiedskg.com/benchmarks/#wikisql)	|	 Table	|	 Question	|	 Answer(adopted)/SQL	|	 Fully/weakly supervised semantic parsing(SQL provided); Large data	|	v1	|
|	[WikiTableQuestion](https://unifiedskg.com/benchmarks/#wikitablequestion)	|	 Table	|	 Question	|	 Answer	|	 Weakly supervised semantic parsing(using question-answer pairs as supervision); Row sensative(some qa related to row order)	|	v1	|
|	[CompWebQ](https://unifiedskg.com/benchmarks/#compwebq)	|	 Knowledge Graph	|	 Question	|	 Answer	|	 Weakly supervised; Multihop	|	v1	|
|	[HybridQA](https://unifiedskg.com/benchmarks/#hybridqa)	|	 Table + Text passages	|	 Question	|	 Answer	|	 Multi-hop; Short-form entity/extractive	|	v1	|
|	[OTT-QA](https://unifiedskg.com/benchmarks/#ott-qa)	|	 Table + Text passages	|	 Question	|	 Answer	|	 More open table/text; Extractive ans	|	v1	|
|	[MultiModalQA](https://unifiedskg.com/benchmarks/#multimodalqa)	|	 Table + Text + Images	|	 Question	|	 Answer	|	 Short-form entity/extractive	|	v1	|
|	[FeTaQA](https://unifiedskg.com/benchmarks/#fetaqa)	|	 Table	|	 Question	|	 Free-Form Answer	|	 Free-form answer	|	v1	|
|	[TAT-QA](https://unifiedskg.com/benchmarks/#tat-qa)	|	 Table + Text	|	 Question	|	 Answer(diverse form, including single span, multiple spans and free-form)	|	 Context hybrid; Numerical reasoning; Financial	|		|
|	[AIT-QA](https://unifiedskg.com/benchmarks/#ait-qa)	|	 Complex Table	|	Question	|	 Answer	|	Airline industry; Complex table 	|		|
|	[HiTab_qa](https://unifiedskg.com/benchmarks/#hitab)	|	 Hierarchical Table	|	Question	|	 Answer	|	 Hierarchical table; TableQA & Table-to-Text	|		|
|	[WikiSQL-TS_WikiTQ-TS](https://unifiedskg.com/benchmarks/#wikisql-ts_wikitq-ts)	|	 	|	 	|	 	|	 	|		|
|	[FinQA](https://unifiedskg.com/benchmarks/#fin-qa)	|	 Table	|	Question 	|	 Answer	|	 Numerical reasoning; Financial data	|		|
|	[MULTIHIERTT](https://unifiedskg.com/benchmarks/#multihiertt)	|	Multiple Hierarchical Table & Text 	|	Question 	|	Answer 	|	 Large-scale; Build from financial reportes 	|		|
|	[WebQA](https://unifiedskg.com/benchmarks/#web-qa)	|	 	|	 	|	 	|	 	|		|
|	[CFQ](https://unifiedskg.com/benchmarks/#cfq)	|	 	|	 	|	 	|	 	|		|
|	[E2E](https://unifiedskg.com/benchmarks/#e2e)	|	 Table	|	 None	|	 Text	|	 Text generation; Restaurant domain	|		|
|	[WebNLG](https://unifiedskg.com/benchmarks/#webnlg)	|	 Knowledge Graph(triples)	|	 None	|	 Text	|	 Text generation; RDF Triples	|		|
|	[DART](https://unifiedskg.com/benchmarks/#dart)	|	 Triples	|	 None	|	 Text	|	 Text generation; Large data; E2E and WebNLG contained	|	v1	|
|	[ToTTo](https://unifiedskg.com/benchmarks/#totto)	|	 Highlighted Table	|	 None	|	 Text	|	 Highlighted Table; Text generation	|	v1	|
|	[LogicNLG](https://unifiedskg.com/benchmarks/#logicnlg)	|	 Table	|	 None	|	 Logical Natural Language Generation	|	 Logical NL generation	|		|
|	[HiTab_NLG](https://unifiedskg.com/benchmarks/#hitab1)	|	  Hierarchical Table	|	 None	|	Text 	|	Hierarchical table; TableQA & Table-to-Text 	|		|
|	[MultiWoZ](https://unifiedskg.com/benchmarks/#multiwoz21)	|	 Ontology	|	 Dialogue	|	 Dialogue State	|	 Dialog system	|	v1	|
|	[KVRET(SMD)](https://unifiedskg.com/benchmarks/#kvretsmd)	|	 Table	|	 Statement	|	 Boolean	|	 Dialogue system; Each dialogue has a seperate table as kb	|	v1	|
|	[SParC](https://unifiedskg.com/benchmarks/#sparc)	|	 Database	|	 Multi-turn query	|	 SQL	|	 Fully supervised semantic parsing; Multi-turn	|	v1	|
|	[CoSQL](https://unifiedskg.com/benchmarks/#cosql)	|	 Database	|	 Dialog	|	 SQL	|	 Fully supervised semantic parsing; Dialogue	|	v1	|
|	[SQA(MSR SQA)](https://unifiedskg.com/benchmarks/#sqamsr-sqa)	|	 Table	|	 Multi-turn query	|	 Answer	|	 Weakly supervised semantic parsing; Sequential	|	v1	|
|	[SMCALFLOW](https://unifiedskg.com/benchmarks/#smcalflow)	|	 	|	 	|	 	|	 	|		|
|	[HybirdDial](https://unifiedskg.com/benchmarks/#hybirddial)	|	 	|	 	|	 	|	 	|		|
|	[TabFact](https://unifiedskg.com/benchmarks/#tabfact)	|	 Table	|	 Statement	|	 Boolean	|	 NL inference; Large data	|	v1	|
|	[FEVEROUS](https://unifiedskg.com/benchmarks/#feverous)	|	 Table + Text	|	 Statement	|	 Boolean	|	 NL inference; Large data	|	v1	|
|	[SQL2Text](https://unifiedskg.com/benchmarks/#sql2text)	|	 Optional Database	|	 SQL	|	 Text	|	 High-fidelity NLG	|	v1	|
|	[Logic2Text](https://unifiedskg.com/benchmarks/#logic2text)	|	 Table Schema	|	 Python-like program	|	 Text	|	 High-fidelity NLG	|	v1	|

---

## Semantic Parsing

### Spider
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*[Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task](https://arxiv.org/abs/1809.08887)*. **EMNLP-18**

[comment]: <> ([Official Link]&#40;https://yale-lily.github.io/spider&#41;{: target="_blank" .btn .btn-green .mr-1 })

### Single Domain Text2SQL
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*[Improving Text-to-SQL Evaluation Methodology](https://arxiv.org/abs/1806.09029)*. **ACL-18**


### Squall
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*[On the Potential of Lexico-logical Alignments for Semantic Parsing to SQL Queries](https://arxiv.org/abs/2010.11246)*. **EMNLP-20**


### KaggleKBQA
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*[KaggleDBQA: Realistic Evaluation of Text-to-SQL Parsers](https://arxiv.org/abs/2106.11455)*. **ACL-21**


### Spider-Syn
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*[Towards Robustness of Text-to-SQL Models against Synonym Substitution](https://arxiv.org/abs/2106.01065)*. **ACL-21**


### Spider-DK
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*[Exploring Underexplored Limitations of Cross-Domain Text-to-SQL Generalization](https://arxiv.org/abs/2109.05157)*. **EMNLP-21**


### SEDE
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*[Text-to-SQL in the Wild: A Naturally-Occurring Dataset Based on Stack Exchange Data](https://arxiv.org/abs/2106.05006)*. **NLP4Prog-21**


### EHRSQL
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*[EHRSQL: A Practical Text-to-SQL Benchmark for Electronic Health Records](https://openreview.net/forum?id=B2W8Vy0rarw)*. **NeurIPS-22 Datasets and Benchmarks**


### Break
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*[Break It Down: A Question Understanding Benchmark](https://arxiv.org/abs/2001.11770)*. **TACL-20**

### GrailQA 
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*[Beyond I.I.D.: Three Levels of Generalization for Question Answering on Knowledge Bases](https://arxiv.org/abs/2011.07743)*. **WWW-21**

[comment]: <> ([Official Link]&#40;https://dki-lab.github.io/GrailQA/&#41;{: target="_blank" .btn .btn-green .mr-1 })


### WebQSP
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*[The Value of Semantic Parse Labeling for Knowledge Base Question Answering](https://aclanthology.org/P16-2033/)*. **ACL-2016**

[comment]: <> ([Official Link]&#40;http://aka.ms/WebQSP&#41;{: target="_blank" .btn .btn-green .mr-1 })

### Russ
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*[Grounding Open-Domain Instructions to Automate Web Support Tasks](https://arxiv.org/abs/2103.16057)*. **NAACL-2021**


### MTOP
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*[MTOP: A Comprehensive Multilingual Task-Oriented Semantic Parsing Benchmark](https://aclanthology.org/2021.eacl-main.257/)*. **EACL-21**

[comment]: <> ([Dataset]&#40;https://fb.me/mtop_dataset&#41;{: target="_blank" .btn .btn-green .mr-1 })

### WebAPI
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*[Compositional Generalization for Natural Language Interfaces to Web APIs](https://arxiv.org/abs/2112.05209)*. **arxiv-21**

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
*[Compositional Semantic Parsing on Semi-Structured Tables](https://aclanthology.org/P15-1142/)*. **ACL-15**

[comment]: <> ([Official Link]&#40;https://ppasupat.github.io/WikiTableQuestions/&#41;{: target="_blank" .btn .btn-green .mr-1 })

**Comments**
The 5-fold validation evaluation in origianl dataset is depracated by latest works. The 1st fold of train set and dev set are used as train set and dev set.


### CompWebQ
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*[The Web as a Knowledge-base for Answering Complex Questions](https://arxiv.org/abs/1911.11641)*. **NAACL-18**

[comment]: <> ([Official Link]&#40;https://www.tau-nlp.org/compwebq&#41;{: target="_blank" .btn .btn-green .mr-1 } {: target="_blank" .btn .btn-purple .mr-1 } </span>)

### HybridQA
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*[HybridQA: A Dataset of Multi-Hop Question Answering over Tabular and Textual Data](https://aclanthology.org/2020.findings-emnlp.91/)*. **EMNLP-20**

[comment]: <> ([Official Link]&#40;https://hybridqa.github.io/&#41;{: target="_blank" .btn .btn-green .mr-1 })

### OTT-QA
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*[Open Question Answering over Tables and Text](https://arxiv.org/abs/2010.10439)*. **ICLR-21**

[comment]: <> ([Official Link]&#40;https://github.com/wenhuchen/OTT-QA&#41;{: target="_blank" .btn .btn-green .mr-1 })

### MultiModalQA
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*[MultiModalQA: Complex Question Answering over Text, Tables and Images](https://arxiv.org/abs/1709.00103)*. **ICLR-21**

[comment]: <> ([Official Link]&#40;https://allenai.github.io/multimodalqa/&#41;{: target="_blank" .btn .btn-green .mr-1 } </span>)

### FeTaQA
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*[FeTaQA: Free-form Table Question Answering](https://arxiv.org/abs/2104.00369)*. **ICLR-18**

[comment]: <> ([Official Link]&#40;https://github.com/Yale-LILY/FeTaQA&#41;{: target="_blank" .btn .btn-green .mr-1 })

### TAT-QA
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*[TAT-QA: A Question Answering Benchmark on a Hybrid of Tabular and Textual Content in Finance](https://aclanthology.org/2021.acl-long.254/)*. **ACL-21**

[comment]: <> ([Official Link]&#40;https://nextplusplus.github.io/TAT-QA/&#41;{: target="_blank" .btn .btn-green .mr-1 })


### AIT-QA
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*[AIT-QA: Question Answering Dataset over Complex Tables in the Airline Industry](https://arxiv.org/abs/2106.12944)*. **arxiv-21**


### HiTab
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*[HiTab: A Hierarchical Table Dataset for Question Answering and Natural Language Generation](http://arxiv.org/abs/2108.06712)*. **arxiv-21**


### WikiSQL-TS_WikiTQ-TS
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*[Topic Transferable Table Question Answering](https://arxiv.org/abs/2109.07377)*. **EMNLP-21**

### Fin-QA
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*[FinQA: A Dataset of Numerical Reasoning over Financial Data](https://arxiv.org/abs/2109.00122)*. **EMNLP-21**

### WebQA
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*[WebQA: Multihop and Multimodal QA](https://arxiv.org/abs/2109.00590)*. **arxiv-21**

### MULTIHIERTT
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*[MULTIHIERTT: Numerical Reasoning over Multi Hierarchical Tabular and Textual Data](https://aclanthology.org/2022.acl-long.454)*. **ACL-22**

### CFQ
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*[Measuring Compositional Generalization: A Comprehensive Method on Realistic Data](https://arxiv.org/abs/1912.09713)*. **ICLR-20**


---

## Data-to-Text

### E2E
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*[The E2E Dataset: New Challenges For End-to-End Generation](https://aclanthology.org/W17-5525/)*. **SIGDIAL-17**

[comment]: <> ([Official Link]&#40;http://www.macs.hw.ac.uk/InteractionLab/E2E/&#41;{: target="_blank" .btn .btn-green .mr-1 } </span>)

**Comments**
There is another [E2E cleaned version](https://github.com/tuetschek/e2e-cleaning) released by authors.

### WebNLG
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*[The WebNLG Challenge: Generating Text from RDF Data](https://aclanthology.org/W17-3518/)*. **INLG-17**

[comment]: <> ([Official Link]&#40;https://webnlg-challenge.loria.fr/&#41;{: target="_blank" .btn .btn-green .mr-1 })

**Comments**
WebNLG challenge has many datasets available。 There is a useful [link](https://gitlab.com/shimorina/webnlg-dataset) for summarization of this.

### Table-to-Text
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*[Table-to-Text: Describing Table Region with Natural Language](https://arxiv.org/abs/1805.11234)*. **AAAI-18**

### DART
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*[DART: Open-Domain Structured Data Record to Text Generation](https://aclanthology.org/2021.naacl-main.37/)*. **ICLR-21**

[comment]: <> ([Official Link]&#40;https://github.com/Yale-LILY/dart&#41;{: target="_blank" .btn .btn-green .mr-1 })


### ToTTo
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*[ToTTo: A Controlled Table-To-Text Generation Dataset](https://arxiv.org/abs/2004.14373)*. **ICLR-21**

[comment]: <> ([Official Link]&#40;https://github.com/google-research-datasets/ToTTo&#41;{: target="_blank" .btn .btn-green .mr-1 })


### LogicNLG
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*[Logical Natural Language Generation from Open-Domain Tables](https://aclanthology.org/2020.acl-main.708/)*. **ACL-20**

[comment]: <> ([Official Link]&#40;https://github.com/wenhuchen/LogicNLG&#41;{: target="_blank" .btn .btn-green .mr-1 })


### HiTab
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*[HiTab: A Hierarchical Table Dataset for Question Answering and Natural Language Generation](http://arxiv.org/abs/2108.06712)*. **arxiv-21**

---

## Conversational

### MultiWoZ
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*[MultiWOZ -- A Large-Scale Multi-Domain Wizard-of-Oz Dataset for Task-Oriented Dialogue Modelling](https://arxiv.org/abs/1810.00278)*. **EMNLP-18**

[comment]: <> ([Official Link]&#40;https://github.com/budzianowski/multiwoz&#41;{: target="_blank" .btn .btn-green .mr-1 })

[comment]: <> (**Comments**)

[comment]: <> (There are many version of [MultiWoZ]&#40;https://github.com/budzianowski/multiwoz&#41;. We used the [v2.1]&#40;&#41; in our SKG benchmark. Some pre-procession on this dataset is needed, please refer to [MultiWoZ]&#40;https://github.com/budzianowski/multiwoz&#41; and [Trade-DST]&#40;https://github.com/jasonwu0731/trade-dst&#41;.)


### KVRET(SMD)
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*[Key-Value Retrieval Networks for Task-Oriented Dialogue](https://aclanthology.org/W17-5506/)*. **SIGdial-17**

[comment]: <> ([Official Link]&#40;https://nlp.stanford.edu/blog/a-new-multi-turn-multi-domain-task-oriented-dialogue-dataset/&#41;{: target="_blank" .btn .btn-green .mr-1 })

**Comments**
KVRET is also called SMD(Stanford Multi-Domain task-oriented dialogue dataset). The de-facto widely used version of this dataset is the pre-processed verison in [Mem2seq](https://github.com/HLTCHKUST/Mem2Seq).

### SParC
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*[SParC: Cross-Domain Semantic Parsing in Context](https://arxiv.org/abs/1906.02285)*. **ACL-19**

[comment]: <> ([Official Link]&#40;https://yale-lily.github.io/sparc&#41;{: target="_blank" .btn .btn-green .mr-1 })


### CoSQL
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*[CoSQL: A Conversational Text-to-SQL Challenge Towards Cross-Domain Natural Language Interfaces to Databases](https://arxiv.org/abs/1909.05378)*. **EMNLP-19**

[comment]: <> ([Official Link]&#40;https://yale-lily.github.io/cosql&#41;{: target="_blank" .btn .btn-green .mr-1 })


### SQA(MSR SQA)
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*[Search-based Neural Structured Learning for Sequential Question Answering](https://aclanthology.org/P17-1167)*. **ACL-17**

[comment]: <> ([Official Link]&#40;hhttps://www.microsoft.com/en-us/download/details.aspx?id=54253&#41;{: target="_blank" .btn .btn-green .mr-1 })


### SMCALFLOW
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*[Task-Oriented Dialogue as Dataflow Synthesis](https://openreview.net/pdf?id=52j93pP1Bgd)*. **TACL-20**


### HybirdDial
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*[HYBRIDIALOGUE: An Information-Seeking Dialogue Dataset Grounded on Tabular and Textual Data](https://openreview.net/pdf?id=52j93pP1Bgd)*. **ACL-22**


---

## Fact Verification

### TabFact
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*[TabFact: A Large-scale Dataset for Table-based Fact Verification](https://arxiv.org/abs/1909.02164)*. **ICLR-20**

[comment]: <> ([Official Link]&#40;https://tabfact.github.io/&#41;{: target="_blank" .btn .btn-green .mr-1 })

### FEVEROUS
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*[The Fact Extraction and VERification Over Unstructured and Structured information (FEVEROUS) Shared Task](https://aclanthology.org/2021.fever-1.1)*. **EMNLP-21**

[comment]: <> ([Official Link]&#40;https://fever.ai/dataset/feverous.html&#41;{: target="_blank" .btn .btn-green .mr-1 })

---

## Formal-Language-to-Text

### SQL2Text
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*[Logic-Consistency Text Generation from Semantic Parses](https://aclanthology.org/2021.findings-acl.388/)*.**ACL-21**

[comment]: <> ([Official Link]&#40;https://github.com/Ciaranshu/relogic&#41;{: target="_blank" .btn .btn-green .mr-1 })

### Logic2Text
{: .no_toc }

{: .fs-4 .fw-800 .text-blue-100}
*[Logic2Text: High-Fidelity Natural Language Generation from Logical Forms](https://arxiv.org/abs/2004.14579)*. **EMNLP-20**

[comment]: <> ([Official Link]&#40;https://github.com/czyssrs/Logic2Text&#41;{: target="_blank" .btn .btn-green .mr-1 })

---

## Other Related Datasets
