---
layout: default
title: Methodology
nav_order: 4
toc_list: true
last_modified_date: Jan 19 2022
permalink: /methodology/
---

# Methods to Structured Knowledge Grounding
{: .no_toc }

{: .fs-5 .fw-300 }
We present a collection of insightful research papers that focus on structured knowledge encoding on structured knowledge grounding tasks.

## Table of contents
{: .no_toc .text-delta .fs-4 style="font-weight:800"}

- TOC
{:toc}


## Structured Knowledge Encoding
TODO: re-order papers by date.

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

{: .fs-4 .fw-800 .text-blue-100}
**📜 A Simple Language Model for Task-Oriented Dialogue**. <br> ✍ Ehsan Hosseini-Asl, Bryan McCann, Chien-Sheng Wu, Semih Yavuz, Richard Socher
 *(NIPS-20)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2005.00796){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://github.com/salesforce/simpletod){: target="_blank" .btn .btn-green .mr-1 }
   [Semantic Scholar](https://www.semanticscholar.org/paper/A-Simple-Language-Model-for-Task-Oriented-Dialogue-Hosseini-Asl-McCann/71d64c24dc0ac9726d2be57f4936ac4528430f64){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

**Evaluation Tasks:** [MultiWoZ](/datasets#multiwoz21)

<details markdown="block">
  <summary>Abstract</summary>
  {: .fs-3 .text-delta .text-blue-100}
  ```
  Task-oriented dialogue is often decomposed into three tasks: understanding user input, deciding actions, and generating a response. While such decomposition might suggest a dedicated model for each sub-task, we find a simple, unified approach leads to state-of-the-art performance on the MultiWOZ dataset. SimpleTOD is a simple approach to task-oriented dialogue that uses a single causal language model trained on all sub-tasks recast as a single sequence prediction problem. This allows SimpleTOD to fully leverage transfer learning from pre-trained, open domain, causal language models such as GPT-2. SimpleTOD improves over the prior state-of-the-art by 0.49 points in joint goal accuracy for dialogue state tracking. More impressively, SimpleTOD also improves the main metrics used to evaluate action decisions and response generation in an end-to-end setting for task-oriented dialog systems: inform rate by 8.1 points, success rate by 9.7 points, and combined score by 7.2 points.
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
**📜 TaBERT: Pretraining for Joint Understanding of Textual and Tabular Data**. <br> ✍ Pengcheng Yin, Graham Neubig, Wen-tau Yih, Sebastian Riedel
 *(ACL-20)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2005.08314){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](http://fburl.com/TaBERT){: target="_blank" .btn .btn-green .mr-1 }
   [Semantic Scholar](https://www.semanticscholar.org/paper/TaBERT%3A-Pretraining-for-Joint-Understanding-of-and-Yin-Neubig/a5b1d1cab073cb746a990b37d42dc7b67763f881){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

**Evaluation Tasks:** [WikiSQL](/datasets#wikisql), [Spider](/datasets#spider)

<details markdown="block">
  <summary>Abstract</summary>
  {: .fs-3 .text-delta .text-blue-100}
  ```
  Recent years have witnessed the burgeoning of pretrained language models (LMs) for text-based natural language (NL) understanding tasks. Such models are typically trained on free-form NL text, hence may not be suitable for tasks like semantic parsing over structured data, which require reasoning over both free-form NL questions and structured tabular data (e.g., database tables). In this paper we present TaBERT, a pretrained LM that jointly learns representations for NL sentences and (semi-)structured tables. TaBERT is trained on a large corpus of 26 million tables and their English contexts. In experiments, neural semantic parsers using TaBERT as feature representation layers achieve new best results on the challenging weakly-supervised semantic parsing benchmark WikiTableQuestions, while performing competitively on the text-to-SQL dataset Spider. Implementation of the model will be available at this [http URL](http://fburl.com/TaBERT) .
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
**📜 TAPAS: Weakly Supervised Table Parsing via Pre-training**. <br> ✍ Jonathan Herzig, Paweł Krzysztof Nowak, Thomas Müller, Francesco Piccinno, Julian Martin Eisenschlos
 *(ACL-20)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2004.02349){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://github.com/google-research/tapas){: target="_blank" .btn .btn-green .mr-1 }
   [Semantic Scholar](https://www.semanticscholar.org/paper/TaPas%3A-Weakly-Supervised-Table-Parsing-via-Herzig-Nowak/52cb05d721688cb766c6e282e9d55c3b8e3dc0cf){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

**Evaluation Tasks:** [WikiSQL](/datasets#wikisql), [WikiTableQuestion](/datasets#wikitablequestion), [SQA](/datasets#msr-sqa)

<details markdown="block">
  <summary>Abstract</summary>
  {: .fs-3 .text-delta .text-blue-100}
  ```
  Answering natural language questions over tables is usually seen as a semantic parsing task. To alleviate the collection cost of full logical forms, one popular approach focuses on weak supervision consisting of denotations instead of logical forms. However, training semantic parsers from weak supervision poses difficulties, and in addition, the generated logical forms are only used as an intermediate step prior to retrieving the denotation. In this paper, we present TAPAS, an approach to question answering over tables without generating logical forms. TAPAS trains from weak supervision, and predicts the denotation by selecting table cells and optionally applying a corresponding aggregation operator to such selection. TAPAS extends BERT's architecture to encode tables as input, initializes from an effective joint pre-training of text segments and tables crawled from Wikipedia, and is trained end-to-end. We experiment with three different semantic parsing datasets, and find that TAPAS outperforms or rivals semantic parsing models by improving state-of-the-art accuracy on SQA from 55.1 to 67.2 and performing on par with the state-of-the-art on WIKISQL and WIKITQ, but with a simpler model architecture. We additionally find that transfer learning, which is trivial in our setting, from WIKISQL to WIKITQ, yields 48.7 accuracy, 4.2 points above the state-of-the-art.
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
**📜 GraPPa: Grammar-Augmented Pre-Training for Table Semantic Parsing**. <br> ✍ Tao Yu, Chien-Sheng Wu, Xi Victoria Lin, Bailin Wang, Yi Chern Tan, Xinyi Yang, Dragomir Radev, Richard Socher, Caiming Xiong
 *(ICLR-21)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2009.13845){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://github.com/taoyds/grappa){: target="_blank" .btn .btn-green .mr-1 }
   [Pre-trained Model](https://huggingface.co/Salesforce/grappa_large_jnt){: .btn .btn-purple .mr-1 target="_blank" }
   [Semantic Scholar](https://www.semanticscholar.org/paper/GraPPa%3A-Grammar-Augmented-Pre-Training-for-Table-Yu-Wu/eedf45f62dea0eaef5643c42c84f7cc7b80ee782){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

**Evaluation Tasks:** [Spider](/datasets#spider), [WikiSQL fully-supervised-setting&weakly-supervised-setting](/datasets#wikisql), [WikiTableQuestion](/datasets#wikitablequestion)

<details markdown="block">
  <summary>Abstract</summary>
  {: .fs-3 .text-delta .text-blue-100}
  ```
  We present GraPPa, an effective pre-training approach for table semantic parsing that learns a compositional inductive bias in the joint representations of textual and tabular data. We construct synthetic question-SQL pairs over high-quality tables via a synchronous context-free grammar (SCFG) induced from existing text-to-SQL datasets. We pre-train our model on the synthetic data using a novel text-schema linking objective that predicts the syntactic role of a table field in the SQL for each question-SQL pair. To maintain the model's ability to represent real-world data, we also include masked language modeling (MLM) over several existing table-and-language datasets to regularize the pre-training process. On four popular fully supervised and weakly supervised table semantic parsing benchmarks, GraPPa significantly outperforms RoBERTa-large as the feature representation layers and establishes new state-of-the-art results on all of them.
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
**📜 SCoRe: Pre-Training for Context Representation in Conversational Semantic Parsing**. <br> ✍ Tao Yu, Rui Zhang, Alex Polozov, Christopher Meek, Ahmed Hassan Awadallah
 *(ICLR-20)*

<span class="fs-2">
   [Paper](https://openreview.net/forum?id=oyZxhRI2RiE){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://github.com/microsoft/SCoRE){: target="_blank" .btn .btn-green .mr-1 }
   [Semantic Scholar](https://www.semanticscholar.org/paper/SCoRe%3A-Pre-Training-for-Context-Representation-in-Yu-Zhang/ff1d3698b8d5f942e6a0775e173720210429b8ae){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

**Evaluation Tasks:** [SParC](/datasets#sparc), [CoSQL](/datasets#cosql), [MultiWoZ2.1](/datasets#multiwoz), [SQA](/datasets#sqa)

<details markdown="block">
  <summary>Abstract</summary>
  {: .fs-3 .text-delta .text-blue-100}
  ```
  Conversational Semantic Parsing (CSP) is the task of converting a sequence of natural language queries to formal language (e.g., SQL, SPARQL) that can be executed against a structured ontology (e.g.  databases, knowledge bases).  To accomplish  this  task,  a  CSP  system  needs  to  model  the  relation  between  the unstructured language utterance and the structured ontology while representing the multi-turn dynamics of the dialog. Pre-trained language models (LMs) are the state-of-the-art for various natural language processing tasks. However, existing pre-trained LMs that use language modeling training objectives over free-form text have limited ability to represent natural language references to contextual structural data. In this work, we present SCORE, a new pre-training approach for CSP tasks designed to induce representations that capture the alignment between the dialogue flow and the structural context. We demonstrate the broad applicability of SCORE to CSP tasks by combining SCORE with strong base systems on four different tasks (SPARC, COSQL, MWOZ, and SQA). We show that SCORE can improve the performance over all these base systems by a significant margin and achieves state-of-the-art results on three of them.
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
**📜 Structure-Grounded Pretraining for Text-to-SQL**. <br> ✍ Xiang Deng, Ahmed Hassan Awadallah, Christopher Meek, Oleksandr Polozov, Huan Sun, Matthew Richardson
 *(NAACL-20)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2010.12773){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://aka.ms/strug){: target="_blank" .btn .btn-green .mr-1 }
   [Semantic Scholar](https://www.semanticscholar.org/paper/Structure-Grounded-Pretraining-for-Text-to-SQL-Deng-Awadallah/346f54fd61f875ae348f0a38f189ccdacf232df4){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

**Evaluation Tasks:** Text2SQL datasets(Spider-Realistic, ATIS, GeoQuery, Restaurants, Academic, IMDB, Yelp, Scholar, Advising), [Spider](/datasets#spider), [WikiSQL(fully supervised setting)](/datasets#wikisql)

<details markdown="block">
  <summary>Abstract</summary>
  {: .fs-3 .text-delta .text-blue-100}
  ```
  Learning to capture text-table alignment is essential for tasks like text-to-SQL. A model needs to correctly recognize natural language references to columns and values and to ground them in the given database schema. In this paper, we present a novel weakly supervised Structure-Grounded pretraining framework (StruG) for text-to-SQL that can effectively learn to capture text-table alignment based on a parallel text-table corpus. We identify a set of novel prediction tasks: column grounding, value grounding and column-value mapping, and leverage them to pretrain a text-table encoder. Additionally, to evaluate different methods under more realistic text-table alignment settings, we create a new evaluation set Spider-Realistic based on Spider dev set with explicit mentions of column names removed, and adopt eight existing text-to-SQL datasets for cross-database evaluation. STRUG brings significant improvement over BERT-LARGE in all settings. Compared with existing pretraining methods such as GRAPPA, STRUG achieves similar performance on Spider, and outperforms all baselines on more realistic sets. All the code and data used in this work is public available at this https URL.
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
**📜 Understanding tables with intermediate pre-training**. <br> ✍ Julian Martin Eisenschlos, Syrine Krichene, Thomas Müller
 *(EMNLP-20)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2010.00571){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://github.com/google-research/tapas){: target="_blank" .btn .btn-green .mr-1 }
   [Semantic Scholar](https://www.semanticscholar.org/paper/Understanding-tables-with-intermediate-pre-training-Eisenschlos-Krichene/65be695739d0fa35212e49ccccd129535e6d9e15){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

**Evaluation Tasks:** [TabFact](/datasets#tabfact), [SQA](/datasets#msr-sqa)

<details markdown="block">
  <summary>Abstract</summary>
  {: .fs-3 .text-delta .text-blue-100}
  ```
Table entailment, the binary classification task of finding if a sentence is supported or refuted by the content of a table, requires parsing language and table structure as well as numerical and discrete reasoning. While there is extensive work on textual entailment, table entailment is less well studied. We adapt TAPAS (Herzig et al., 2020), a table-based BERT model, to recognize entailment. Motivated by the benefits of data augmentation, we create a balanced dataset of millions of automatically created training examples which are learned in an intermediate step prior to fine-tuning. This new data is not only useful for table entailment, but also for SQA (Iyyer et al., 2017), a sequential table QA task. To be able to use long examples as input of BERT models, we evaluate table pruning techniques as a pre-processing step to drastically improve the training and prediction efficiency at a moderate drop in accuracy. The different methods set the new state-of-the-art on the TabFact (Chen et al., 2020) and SQA datasets.
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
**📜 UniK-QA: Unified Representations of Structured and Unstructured Knowledge for Open-Domain Question Answering**. <br> ✍ Barlas Oguz, Xilun Chen, Vladimir Karpukhin, Stan Peshterliev, Dmytro Okhonko, Michael Schlichtkrull, Sonal Gupta, Yashar Mehdad, Scott Yih
 *(ACL-20)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2012.14610){: .btn .btn-blue .mr-1 target="_blank" } 
   [Semantic Scholar](https://www.semanticscholar.org/paper/UniK-QA%3A-Unified-Representations-of-Structured-and-O%C4%9Fuz-Chen/0ccf167707dddebe9bbfd2095256804698e3a81d){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

**Evaluation Tasks:**  NaturalQuestions, WebQuestions

<details markdown="block">
  <summary>Abstract</summary>
  {: .fs-3 .text-delta .text-blue-100}
  ```
  We study open-domain question answering with structured, unstructured and semi-structured knowledge sources, including text, tables, lists and knowledge bases. Departing from prior work, we propose a unifying approach that homogenizes all sources by reducing them to text and applies the retriever-reader model which has so far been limited to text sources only. Our approach greatly improves the results on knowledge-base QA tasks by 11 points, compared to latest graph-based methods. More importantly, we demonstrate that our unified knowledge (UniK-QA) model is a simple and yet effective way to combine heterogeneous sources of knowledge, advancing the state-of-the-art results on two popular question answering benchmarks, NaturalQuestions and WebQuestions, by 3.5 and 2.6 points, respectively.
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
**📜 Database Reasoning Over Text**. <br> ✍ James Thorne, Majid Yazdani, Marzieh Saeidi, Fabrizio Silvestri, Sebastian Riedel, Alon Halevy
 *(ACL-21)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2106.01074){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://github.com/facebookresearch/NeuralDB){: target="_blank" .btn .btn-green .mr-1 }
</span> 

**Evaluation Tasks:** WikiNLDB

<details markdown="block">
  <summary>Abstract</summary>
  {: .fs-3 .text-delta .text-blue-100}
  ```
  Neural models have shown impressive performance gains in answering queries from natural language text. However, existing works are unable to support database queries, such as "List/Count all female athletes who were born in 20th century", which require reasoning over sets of relevant facts with operations such as join, filtering and aggregation. We show that while state-of-the-art transformer models perform very well for small databases, they exhibit limitations in processing noisy data, numerical operations, and queries that aggregate facts. We propose a modular architecture to answer these database-style queries over multiple spans from text and aggregating these at scale. We evaluate the architecture using WikiNLDB, a novel dataset for exploring such queries. Our architecture scales to databases containing thousands of facts whereas contemporary models are limited by how many facts can be encoded. In direct comparison on small databases, our approach increases overall answer accuracy from 85% to 90%. On larger databases, our approach retains its accuracy whereas transformer baselines could not encode the context.
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
**📜 Turning Tables: Generating Examples from Semi-structured Tables for Endowing Language Models with Reasoning Skills**. <br> ✍ Ori Yoran, Alon Talmor, Jonathan Berant
 *(arxiv-21)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2107.07653){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://github.com/oriyor/turning_tables){: target="_blank" .btn .btn-green .mr-1 }
</span> 

**Evaluation Tasks:** DROP, IIRC, [MMQA](/datasets#multimodalqa)

<details markdown="block">
  <summary>Abstract</summary>
  {: .fs-3 .text-delta .text-blue-100}
  ```
  Models pre-trained with a language modeling objective possess ample world knowledge and language skills, but are known to struggle in tasks that require reasoning. In this work, we propose to leverage semi-structured tables, and automatically generate at scale question-paragraph pairs, where answering the question requires reasoning over multiple facts in the paragraph. We add a pre-training step over this synthetic data, which includes examples that require 16 different reasoning skills such as number comparison, conjunction, and fact composition. To improve data efficiency, we propose sampling strategies that focus training on reasoning skills the model is currently lacking. We evaluate our approach on three reading comprehension datasets that are focused on reasoning, and show that our model, PReasM, substantially outperforms T5, a popular pre-trained encoder-decoder model. Moreover, sampling examples based on current model errors leads to faster training and higher overall performance.
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
**📜 TAPEX: Table Pre-training via Learning a Neural SQL Executor**. <br> ✍ Qian Liu, Bei Chen, Jiaqi Guo, Morteza Ziyadi, Zeqi Lin, Weizhu Chen, Jian-Guang Lou
 *(arxiv-21)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2107.07653){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://github.com/microsoft/Table-Pretraining){: target="_blank" .btn .btn-green .mr-1 }
</span> 

**Evaluation Tasks:** [WikiSQL](/datasets#wikisql), [WikiTableQuestion](/datasets#wikitablequestion), [SQA](/datasets#msr-sqa), [TabFact](/datasets#TabFact)

<details markdown="block">
  <summary>Abstract</summary>
  {: .fs-3 .text-delta .text-blue-100}
  ```
  Recent progress in language model pre-training has achieved a great success via leveraging large-scale unstructured textual data. However, it is still a challenge to apply pre-training on structured tabular data due to the absence of large-scale high-quality tabular data. In this paper, we propose TAPEX to show that table pre-training can be achieved by learning a neural SQL executor over a synthetic corpus, which is obtained by automatically synthesizing executable SQL queries and their execution outputs. TAPEX addresses the data scarcity challenge via guiding the language model to mimic a SQL executor on the diverse, large-scale and high-quality synthetic corpus. We evaluate TAPEX on four benchmark datasets. Experimental results demonstrate that TAPEX outperforms previous table pre-training approaches by a large margin and achieves new state-of-the-art results on all of them. This includes improvements on the weakly-supervised WikiSQL denotation accuracy to 89.5% (+2.3%), the WikiTableQuestions denotation accuracy to 57.5% (+4.8%), the SQA denotation accuracy to 74.5% (+3.5%), and the TabFact accuracy to 84.2% (+3.2%). To our knowledge, this is the first work to exploit table pre-training via synthetic executable programs and to achieve new state-of-the-art results on various downstream tasks.
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
**📜 RnG-KBQA: Generation Augmented Iterative Ranking for Knowledge Base Question Answering**. <br> ✍ Xi Ye, Semih Yavuz, Kazuma Hashimoto, Yingbo Zhou, Caiming Xiong
 *(arxiv-21)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2109.08678){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://github.com/salesforce/rng-kbqa){: target="_blank" .btn .btn-green .mr-1 }
</span> 

**Evaluation Tasks:** [GrailQA](/datasets#grailqa), [WebQSP](/datasets#webqsp)

<details markdown="block">
  <summary>Abstract</summary>
  {: .fs-3 .text-delta .text-blue-100}
  ```
  Existing KBQA approaches, despite achieving strong performance on i.i.d. test data, often struggle in generalizing to questions involving unseen KB schema items. Prior ranking-based approaches have shown some success in generalization, but suffer from the coverage issue. We present RnG-KBQA, a Rank-and-Generate approach for KBQA, which remedies the coverage issue with a generation model while preserving a strong generalization capability. Our approach first uses a contrastive ranker to rank a set of candidate logical forms obtained by searching over the knowledge graph. It then introduces a tailored generation model conditioned on the question and the top-ranked candidates to compose the final logical form. We achieve new state-of-the-art results on GrailQA and WebQSP datasets. In particular, our method surpasses the prior state-of-the-art by a large margin on the GrailQA leaderboard. In addition, RnG-KBQA outperforms all prior approaches on the popular WebQSP benchmark, even including the ones that use the oracle entity linking. The experimental results demonstrate the effectiveness of the interplay between ranking and generation, which leads to the superior performance of our proposed approach across all settings with especially strong improvements in zero-shot generalization.
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
**📜 PICARD: Parsing Incrementally for Constrained Auto-Regressive Decoding from Language Models**. <br> ✍ Torsten Scholak, Nathan Schucher, Dzmitry Bahdanau
 *(EMNLP-21)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2109.05093){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://github.com/ElementAI/picard){: target="_blank" .btn .btn-green .mr-1 }
</span> 

**Evaluation Tasks:** [Spider](/datasets#spider), [CoSQL](/datasets#cosql)

<details markdown="block">
  <summary>Abstract</summary>
  {: .fs-3 .text-delta .text-blue-100}
  ```
  Large pre-trained language models for textual data have an unconstrained output space; at each decoding step, they can produce any of 10,000s of sub-word tokens. When fine-tuned to target constrained formal languages like SQL, these models often generate invalid code, rendering it unusable. We propose PICARD (code and trained models available at this https URL), a method for constraining auto-regressive decoders of language models through incremental parsing. PICARD helps to find valid output sequences by rejecting inadmissible tokens at each decoding step. On the challenging Spider and CoSQL text-to-SQL translation tasks, we show that PICARD transforms fine-tuned T5 models with passable performance into state-of-the-art solutions.
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
**📜 MATE: Multi-view Attention for Table Transformer Efficiency**. <br> ✍ Julian Martin Eisenschlos, Maharshi Gor, Thomas Müller, William W. Cohen
 *(EMNLP-21)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2109.04312){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://github.com/google-research/tapas){: target="_blank" .btn .btn-green .mr-1 }
</span> 

**Evaluation Tasks:** [HybridQA](/datasets#hybridqa), [SQA](/datasets#msr-sqa), [WikiTableQuestion](/datasets#wikitablequestion), [TabFact](/datasets#tabfact)

<details markdown="block">
  <summary>Abstract</summary>
  {: .fs-3 .text-delta .text-blue-100}
  ```
This work presents a sparse-attention Transformer architecture for modeling documents that contain large tables. Tables are ubiquitous on the web, and are rich in information. However, more than 20% of relational tables on the web have 20 or more rows (Cafarella et al., 2008), and these large tables present a challenge for current Transformer models, which are typically limited to 512 tokens. Here we propose MATE, a novel Transformer architecture designed to model the structure of web tables. MATE uses sparse attention in a way that allows heads to efficiently attend to either rows or columns in a table. This architecture scales linearly with respect to speed and memory, and can handle documents containing more than 8000 tokens with current accelerators. MATE also has a more appropriate inductive bias for tabular data, and sets a new state-of-the-art for three table reasoning datasets. For HybridQA (Chen et al., 2020b), a dataset that involves large documents containing tables, we improve the best prior result by 19 points.
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
**📜 Table Fact Verification with Structure-Aware Transformer**. <br> ✍ Hongzhi Zhang, Yingyao Wang, Sirui Wang, Xuezhi Cao, Fuzheng Zhang, Zhongyuan Wang
 *(ACL-20)*

<span class="fs-2">
   [Paper](https://aclanthology.org/2020.emnlp-main.126/){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://github.com/zhhongzhi/sat){: target="_blank" .btn .btn-green .mr-1 }
   [Semantic Scholar](https://www.semanticscholar.org/paper/Table-Fact-Verification-with-Structure-Aware-Zhang-Wang/38b3c835e272a25fca4fe523dad627feb6552bd3){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

**Evaluation Tasks:** [TabFact](/datasets#tabfact)

<details markdown="block">
  <summary>Abstract</summary>
  {: .fs-3 .text-delta .text-blue-100}
  ```
  Verifying fact on semi-structured evidence like tables requires the ability to encode structural information and perform symbolic reasoning. Pre-trained language models trained on natural language could not be directly applied to encode tables, because simply linearizing tables into sequences will lose the cell alignment information. To better utilize pre-trained transformers for table representation, we propose a Structure-Aware Transformer (SAT), which injects the table structural information into the mask of the self-attention layer. A method to combine symbolic and linguistic reasoning is also explored for this task. Our method outperforms baseline with 4.93% on TabFact, a large scale table verification dataset.
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
**📜 DoT: An efficient Double Transformer for NLP tasks with tables**. <br> ✍ Syrine Krichene, Thomas Müller, Julian Martin Eisenschlos
 *(ACL-21)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2106.00479){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://github.com/google-research/tapas){: target="_blank" .btn .btn-green .mr-1 }
</span> 

**Evaluation Tasks:** [WikiSQL](/datasets#wikisql),  [WikiTableQuestion](/datasets#wikitablequestion), [TabFact](/datasets#tabfact)

<details markdown="block">
  <summary>Abstract</summary>
  {: .fs-3 .text-delta .text-blue-100}
  ```
Transformer-based approaches have been successfully used to obtain state-of-the-art accuracy on natural language processing (NLP) tasks with semi-structured tables. These model architectures are typically deep, resulting in slow training and inference, especially for long inputs. To improve efficiency while maintaining a high accuracy, we propose a new architecture, DoT, a double transformer model, that decomposes the problem into two sub-tasks: A shallow pruning transformer that selects the top-K tokens, followed by a deep task-specific transformer that takes as input those K tokens. Additionally, we modify the task-specific attention to incorporate the pruning scores. The two transformers are jointly trained by optimizing the task-specific loss. We run experiments on three benchmarks, including entailment and question-answering. We show that for a small drop of accuracy, DoT improves training and inference time by at least 50%. We also show that the pruning transformer effectively selects relevant tokens enabling the end-to-end model to maintain similar accuracy as slower baseline models. Finally, we analyse the pruning and give some insight into its impact on the task model.
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
**📜 TUTA: Tree-based Transformers for Generally Structured Table Pre-training**. <br> ✍ Zhiruo Wang, Haoyu Dong, Ran Jia, Jia Li, Zhiyi Fu, Shi Han, Dongmei Zhang
 *(KDD-21)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2010.12537){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://github.com/microsoft/TUTA_table_understanding/){: target="_blank" .btn .btn-green .mr-1 }
   [Semantic Scholar](https://www.semanticscholar.org/paper/TUTA%3A-Tree-based-Transformers-for-Generally-Table-Wang-Dong/24a12899ce97bd4a56f7c6b49d3979b9465f0190){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

**Evaluation Tasks:** Cell Type Classification (CTC) tasks, Table Type Classification (TTC) tasks

<details markdown="block">
  <summary>Abstract</summary>
  {: .fs-3 .text-delta .text-blue-100}
  ```
  Tables are widely used with various structures to organize and present data. Recent attempts on table understanding mainly focus on relational tables, yet overlook to other common table structures. In this paper, we propose TUTA, a unified pre-training architecture for understanding generally structured tables. Noticing that understanding a table requires spatial, hierarchical, and semantic information, we enhance transformers with three novel structure-aware mechanisms. First, we devise a unified tree-based structure, called a bi-dimensional coordinate tree, to describe both the spatial and hierarchical information of generally structured tables. Upon this, we propose tree-based attention and position embedding to better capture the spatial and hierarchical information. Moreover, we devise three progressive pre-training objectives to enable representations at the token, cell, and table levels. We pre-train TUTA on a wide range of unlabeled web and spreadsheet tables and fine-tune it on two critical tasks in the field of table structure understanding: cell type classification and table type classification. Experiments show that TUTA is highly effective, achieving state-of-the-art on five widely-studied datasets.
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
**📜 TABBIE: Pretrained Representations of Tabular Data**. <br> ✍ Hiroshi Iida, Dung Thai, Varun Manjunatha, Mohit Iyyer
 *(NAACL-21)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2105.02584){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://github.com/SFIG611/tabbie){: target="_blank" .btn .btn-green .mr-1 }
   [Semantic Scholar](https://www.semanticscholar.org/paper/TABBIE%3A-Pretrained-Representations-of-Tabular-Data-Iida-Thai/386bfd0e411dee4f512a8737c55dd84846981182){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

**Evaluation Tasks:** Column population, Row population, and Column type prediction tasks

<details markdown="block">
  <summary>Abstract</summary>
  {: .fs-3 .text-delta .text-blue-100}
  ```
  Existing work on tabular representation learning jointly models tables and associated text using self-supervised objective functions derived from pretrained language models such as BERT. While this joint pretraining improves tasks involving paired tables and text (e.g., answering questions about tables), we show that it underperforms on tasks that operate over tables without any associated text (e.g., populating missing cells). We devise a simple pretraining objective (corrupt cell detection) that learns exclusively from tabular data and reaches the state-of-the-art on a suite of table based prediction tasks. Unlike competing approaches, our model (TABBIE) provides embeddings of all table substructures (cells, rows, and columns), and it also requires far less compute to train. A qualitative analysis of our model's learned cell, column, and row representations shows that it understands complex table semantics and numerical trends.
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
**📜 HittER: Hierarchical Transformers for Knowledge Graph Embeddings**. <br> ✍ Sanxing Chen, Xiaodong Liu, Jianfeng Gao, Jian Jiao, Ruofei Zhang, Yangfeng Ji
 *(EMNLP-21)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2008.12813){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://github.com/sanxing-chen/HittER){: target="_blank" .btn .btn-green .mr-1 }
   [Semantic Scholar](https://www.semanticscholar.org/paper/HittER%3A-Hierarchical-Transformers-for-Knowledge-Chen-Liu/7e7499b47fe57033768f26ef98a3b644688eb2a2){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

**Evaluation Tasks:** FB15K-237, WN18RR, FreebaseQA, [WebQuestionSP](/datasets#webqsp)

<details markdown="block">
  <summary>Abstract</summary>
  {: .fs-3 .text-delta .text-blue-100}
  ```
  This paper examines the challenging problem of learning representations of entities and relations in a complex multi-relational knowledge graph. We propose HittER, a Hierarchical Transformer model to jointly learn Entity-relation composition and Relational contextualization based on a source entity's neighborhood. Our proposed model consists of two different Transformer blocks: the bottom block extracts features of each entity-relation pair in the local neighborhood of the source entity and the top block aggregates the relational information from outputs of the bottom block. We further design a masked entity prediction task to balance information from the relational context and the source entity itself. Experimental results show that HittER achieves new state-of-the-art results on multiple link prediction datasets. We additionally propose a simple approach to integrate HittER into BERT and demonstrate its effectiveness on two Freebase factoid question answering datasets.
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
**📜 Constrained Language Models Yield Few-Shot Semantic Parsers**. <br> ✍ Richard Shin, Christopher H. Lin, Sam Thomson, Charles Chen, Subhro Roy, Emmanouil Antonios Platanios, Adam Pauls, Dan Klein, Jason Eisner, Benjamin Van Durme
 *(EMNLP-21)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2104.08768){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://github.com/microsoft/semantic_parsing_with_constrained_lm){: target="_blank" .btn .btn-green .mr-1 }
</span> 

**Evaluation Tasks:** Overnight, Break, SMCalFlow

<details markdown="block">
  <summary>Abstract</summary>
  {: .fs-3 .text-delta .text-blue-100}
  ```
  We explore the use of large pretrained language models as few-shot semantic parsers. The goal in semantic parsing is to generate a structured meaning representation given a natural language input. However, language models are trained to generate natural language. To bridge the gap, we use language models to paraphrase inputs into a controlled sublanguage resembling English that can be automatically mapped to a target meaning representation. Our results demonstrate that with only a small amount of data and very little code to convert into English-like representations, our blueprint for rapidly bootstrapping semantic parsers leads to surprisingly effective performance on multiple community tasks, greatly exceeding baseline methods also trained on the same limited data.
  ``` 
</details> 
{: .fs-5 .fw-600 .text-blue-300}

<details markdown="block">
  <summary>Comments</summary>
  {: .fs-3 .text-delta .text-red-100}
</details> 
{: .fs-3 .fw-600 .text-red-300}

---

## Pre-training

{: .fs-4 .fw-800 .text-blue-100}
**📜 CodeBERT: A Pre-Trained Model for Programming and Natural Languages**. <br> ✍ Zhangyin Feng, Daya Guo, Duyu Tang, Nan Duan, Xiaocheng Feng, Ming Gong, Linjun Shou, Bing Qin, Ting Liu, Daxin Jiang, Ming Zhou
 *(EMNLP 2020)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2002.08155){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://github.com/microsoft/CodeBERT){: target="_blank" .btn .btn-green .mr-1 }
   [Semantic Scholar](https://www.semanticscholar.org/paper/CodeBERT%3A-A-Pre-Trained-Model-for-Programming-and-Feng-Guo/0fe2636446cd686830da3d971b31a004d6094b3c){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

<details markdown="block">
  <summary>Abstract</summary>
  {: .fs-3 .text-delta .text-blue-100}
```
  We present CodeBERT, a bimodal pre-trained model for programming language (PL) and nat-ural language (NL). CodeBERT learns general-purpose representations that support downstream NL-PL applications such as natural language codesearch, code documentation generation, etc. We develop CodeBERT with Transformer-based neural architecture, and train it with a hybrid objective function that incorporates the pre-training task of replaced token detection, which is to detect plausible alternatives sampled from generators. This enables us to utilize both bimodal data of NL-PL pairs and unimodal data, where the former provides input tokens for model training while the latter helps to learn better generators. We evaluate CodeBERT on two NL-PL applications by fine-tuning model parameters. Results show that CodeBERT achieves state-of-the-art performance on both natural language code search and code documentation generation tasks. Furthermore, to investigate what type of knowledge is learned in CodeBERT, we construct a dataset for NL-PL probing, and evaluate in a zero-shot setting where parameters of pre-trained models are fixed. Results show that CodeBERT performs better than previous pre-trained models on NL-PL probing.
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
**📜 ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators**. <br> ✍ Kevin Clark, Minh-Thang Luong, Quoc V. Le, Christopher D. Manning
 *(ICLR 2020)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2003.10555){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://github.com/google-research/electra){: target="_blank" .btn .btn-green .mr-1 }
   [Semantic Scholar](https://www.semanticscholar.org/paper/ELECTRA%3A-Pre-training-Text-Encoders-as-Rather-Than-Clark-Luong/756810258e3419af76aff38c895c20343b0602d0){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

<details markdown="block">
  <summary>Abstract</summary>
  {: .fs-3 .text-delta .text-blue-100}
  ```
  While masked language modeling (MLM) pre-training methods such as BERT produce excellent results on downstream NLP tasks, they require large amounts of compute to be effective. These approaches corrupt the input by replacing some tokens with [MASK] and then train a model to reconstruct the original tokens. As an alternative, we propose a more sample-efficient pre-training task called replaced token detection. Instead of masking the input, our approach corrupts it by replacing some input tokens with plausible alternatives sampled from a small generator network. Then, instead of training a model that predicts the original identities of the corrupted tokens, we train a discriminative model that predicts whether each token in the corrupted input was replaced by a generator sample or not. Thorough experiments demonstrate this new pre-training task is more efficient than MLM because the model learns from all input tokens rather than just the small subset that was masked out. As a result, the contextual representations learned by our approach substantially outperform the ones learned by methods such as BERT and XLNet given the same model size, data, and compute. The gains are particularly strong for small models; for example, we train a model on one GPU for 4 days that outperforms GPT (trained using 30x more compute) on the GLUE natural language understanding benchmark. Our approach also works well at scale, where we match the performance of RoBERTa, the current state-of-the-art pre-trained transformer, while using less than 1/4 of the compute.
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
**📜 Structure-Grounded Pretraining for Text-to-SQL**. <br> ✍ Xiang Deng, Ahmed Hassan Awadallah, Christopher Meek, Oleksandr Polozov, Huan Sun, Matthew Richardson
 *(NAACL 2021)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2002.08155){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://aka.ms/strug){: target="_blank" .btn .btn-green .mr-1 }
   [Semantic Scholar](https://www.semanticscholar.org/paper/Structure-Grounded-Pretraining-for-Text-to-SQL-Deng-Awadallah/1e84152b10e48ef592917576ca74f814adadcdc7){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

<details markdown="block">
  <summary>Abstract</summary>
  {: .fs-3 .text-delta .text-blue-100}
  ```
  Learning to capture text-table alignment is essential for tasks like text-to-SQL. A model needs to correctly recognize natural language references to columns and values and to ground them in the given database schema. In this paper, we present a novel weakly supervised Structure-Grounded pretraining framework (StruG) for text-to-SQL that can effectively learn to capture text-table alignment based on a parallel text-table corpus. We identify a set of novel prediction tasks: column grounding, value grounding and column-value mapping, and leverage them to pretrain a text-table encoder. Additionally, to evaluate different methods under more realistic text-table alignment settings, we create a new evaluation set Spider-Realistic based on Spider dev set with explicit mentions of column names removed, and adopt eight existing text-to-SQL datasets for cross-database evaluation. STRUG brings significant improvement over BERT-LARGE in all settings. Compared with existing pretraining methods such as GRAPPA, STRUG achieves similar performance on Spider, and outperforms all baselines on more realistic sets. All the code and data used in this work is public available at this https URL.
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
**📜 KGPT: Knowledge-Grounded Pre-Training for Data-to-Text Generation**. <br> ✍ Wenhu Chen, Yu Su, Xifeng Yan, William Yang Wang
 *(EMNLP 2020)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2010.02307){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://github.com/wenhuchen/KGPT){: target="_blank" .btn .btn-green .mr-1 }
   [Semantic Scholar](https://www.semanticscholar.org/paper/KGPT%3A-Knowledge-Grounded-Pre-Training-for-Chen-Su/6f33bd4e62955f4d40424f8ae4ec83af4e97862c){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

<details markdown="block">
  <summary>Abstract</summary>
  {: .fs-3 .text-delta .text-blue-100}
  ```
  Data-to-text generation has recently attracted substantial interests due to its wide applications. Existing methods have shown impressive performance on an array of tasks. However, they rely on a significant amount of labeled data for each task, which is costly to acquire and thus limits their application to new tasks and domains. In this paper, we propose to leverage pre-training and transfer learning to address this issue. We propose a knowledge-grounded pre-training (KGPT), which consists of two parts, 1) a general knowledge-grounded generation model to generate knowledge-enriched text. 2) a pre-training paradigm on a massive knowledge-grounded text corpus crawled from the web. The pre-trained model can be fine-tuned on various data-to-text generation tasks to generate task-specific text. We adopt three settings, namely fully-supervised, zero-shot, few-shot to evaluate its effectiveness. Under the fully-supervised setting, our model can achieve remarkable gains over the known baselines. Under zero-shot setting, our model without seeing any examples achieves over 30 ROUGE-L on WebNLG while all other baselines fail. Under the few-shot setting, our model only needs about one-fifteenth as many labeled examples to achieve the same level of performance as baseline models. These experiments consistently prove the strong generalization ability of our proposed framework this https URL.
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
**📜 TABBIE: Pretrained Representations of Tabular Data**. <br> ✍ Hiroshi Iida, Dung Thai, Varun Manjunatha, Mohit Iyyer
 *(arixv 2021)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2105.02584){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://github.com/SFIG611/tabbie){: target="_blank" .btn .btn-green .mr-1 }
   [Semantic Scholar](https://www.semanticscholar.org/paper/TABBIE%3A-Pretrained-Representations-of-Tabular-Data-Iida-Thai/386bfd0e411dee4f512a8737c55dd84846981182){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

<details markdown="block">
  <summary>Abstract</summary>
  {: .fs-3 .text-delta .text-blue-100}
  ```
  Existing work on tabular representation-learning jointly models tables and associated text using self-supervised objective functions derived from pretrained language models such as BERT. While this joint pretraining improves tasks involving paired tables and text (e.g., answering questions about tables), we show that it underperforms on tasks that operate over tables without any associated text (e.g., populating missing cells). We devise a simple pretraining objective (corrupt cell detection) that learns exclusively from tabular data and reaches the state-of-the-art on a suite of table-based prediction tasks. Unlike competing approaches, our model (TABBIE) provides embeddings of all table substructures (cells, rows, and columns), and it also requires far less compute to train. A qualitative analysis of our model’s learned cell, column, and row representations shows that it understands complex table semantics and numerical trends.
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
**📜 Question Answering Infused Pre-training of General-Purpose Contextualized Representations**. <br> ✍ Robin Jia, Mike Lewis, Luke Zettlemoyer
 *(arixv 2021)*

<span classhttps://arxiv.org/abs/2106.08190){: .btn .btn-blue .mr-1 target="_blank" } 
   [Semantic Scholar](https://www.semanticscholar.org/paper/Question-Answering-Infused-Pre-training-of-Jia-Lewis/e4c13aadc6adeb8131bb08324e2688383fbb8ec9){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

<details markdown="block">
  <summary>Abstract</summary>
  {: .fs-3 .text-delta .text-blue-100}
  ```
  This paper proposes a pre-training objective based on question answering (QA) for learning general-purpose contextual representations, motivated by the intuition that the representation of a phrase in a passage should encode all questions that the phrase can answer in context. We accomplish this goal by training a bi-encoder QA model, which independently encodes passages and questions, to match the predictions of a more accurate cross-encoder model on 80 million synthesized QA pairs. By encoding QA-relevant information, the bi-encoder's token-level representations are useful for non-QA downstream tasks without extensive (or in some cases, any) fine-tuning. We show large improvements over both RoBERTa-large and previous state-of-the-art results on zero-shot and few-shot paraphrase detection on four datasets, few-shot named entity recognition on two datasets, and zero-shot sentiment analysis on three datasets.
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
**📜 Evaluating Large Language Models Trained on Code(Codex)**. <br> ✍ Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, Wojciech Zaremba
 *(arixv 2021)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2105.02584){: .btn .btn-blue .mr-1 target="_blank" } 
   [Website](https://openai.com/api/){: target="_blank" .btn .btn-red .mr-1 }
   [Semantic Scholar](https://www.semanticscholar.org/paper/Evaluating-Large-Language-Models-Trained-on-Code-Chen-Tworek/acbdbf49f9bc3f151b93d9ca9a06009f4f6eb269){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

<details markdown="block">
  <summary>Abstract</summary>
  {: .fs-3 .text-delta .text-blue-100}
  ```
  We introduce Codex, a GPT language model finetuned on publicly available code from GitHub, and study its Python code-writing capabilities. A distinct production version of Codex powers GitHub Copilot. On HumanEval, a new evaluation set we release to measure functional correctness for synthesizing programs from docstrings, our model solves 28.8% of the problems, while GPT-3 solves 0% and GPT-J solves 11.4%. Furthermore, we find that repeated sampling from the model is a surprisingly effective strategy for producing working solutions to difficult prompts. Using this method, we solve 70.2% of our problems with 100 samples per problem. Careful investigation of our model reveals its limitations, including difficulty with docstrings describing long chains of operations and with binding operations to variables. Finally, we discuss the potential broader impacts of deploying powerful code generation technologies, covering safety, security, and economics. Equal contribution OpenAI, San Francisco, California, USA. Anthropic AI, San Francisco, California, USA. Work performed while at OpenAI. Zipline, South San Francisco, California, USA. Work performed while at OpenAI. Correspondence to: Mark Chen <mark@openai.com>, Jerry Tworek <jt@openai.com>, Heewoo Jun <heewoo@openai.com>, Qiming Yuan <qiming@openai.com>. 
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
**📜 Learning Contextual Representations for Semantic Parsing with Generation-Augmented Pre-Training**. <br> ✍ Peng Shi, Patrick Ng, Zhiguo Wang, Henghui Zhu, Alexander Hanbo Li, Jun Wang, Cicero Nogueira dos Santos, Bing Xiang
 *(AAAI 2021)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2012.10309){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://github.com/awslabs/gap-text2sql){: target="_blank" .btn .btn-green .mr-1 }
   [Semantic Scholar](https://www.semanticscholar.org/paper/Learning-Contextual-Representations-for-Semantic-Shi-Ng/c75a2ee17056d2b8c14ac25f9f328a09eb4cf040){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

<details markdown="block">
  <summary>Abstract</summary>
  {: .fs-3 .text-delta .text-blue-100}
  ```
  Most recently, there has been significant interest in learning contextual representations for various NLP tasks, by leveraging large scale text corpora to train large neural language models with self-supervised learning objectives, such as Masked Language Model (MLM). However, based on a pilot study, we observe three issues of existing general-purpose language models when they are applied to text-to-SQL semantic parsers: fail to detect column mentions in the utterances, fail to infer column mentions from cell values, and fail to compose complex SQL queries. To mitigate these issues, we present a model pre-training framework, GenerationAugmented Pre-training (GAP), that jointly learns representations of natural language utterances and table schemas by leveraging generation models to generate pre-train data. GAP MODEL is trained on 2M utterance-schema pairs and 30K utterance-schema-SQL triples, whose utterances are produced by generative models. Based on experimental results, neural semantic parsers that leverage GAP MODEL as a representation encoder obtain new state-of-the-art results on both SPIDER and CRITERIA-TO-SQL benchmarks.
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
**📜 TAPEX: Table Pre-training via Learning a Neural SQL Executor**. <br> ✍ Qian Liu, Bei Chen, Jiaqi Guo, Morteza Ziyadi, Zeqi Lin, Weizhu Chen, Jian-Guang Lou
 *(arixv 2021)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2107.07653){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://github.com/microsoft/Table-Pretraining){: target="_blank" .btn .btn-green .mr-1 }
   [Website](https://table-pretraining.github.io/){: target="_blank" .btn .btn-green .mr-1 }
   [Semantic Scholar](https://www.semanticscholar.org/paper/TAPEX%3A-Table-Pre-training-via-Learning-a-Neural-SQL-Liu-Chen/8592953f1ebe38ba4cab05c28a088f5d5691a514){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

<details markdown="block">
  <summary>Abstract</summary>
  {: .fs-3 .text-delta .text-blue-100}
  ```
  Recent years pre-trained language models hit a success on modeling natural language sentences and (semi-)structured tables. However, existing table pre-training techniques always suffer from low data quality and low pre-training efficiency. In this paper, we show that table pre-training can be realized by learning a neural SQL executor over a synthetic corpus, which is obtained by automatically synthesizing executable SQL queries. By pre-training on the synthetic corpus, our approach TAPEX dramatically improves the performance on downstream tasks, boosting existing language models by at most 19.5%. Meanwhile, TAPEX has remarkably high pretraining efficiency and yields strong results when using a small pre-trained corpus. Experimental results demonstrate that TAPEX outperforms previous table pre-training approaches by a large margin, and our model achieves new state-of-the-art results on four well-known datasets, including improving the WIKISQL denotation accuracy to 89.6% (+4.9%), the WIKITABLEQUESTIONS denotation accuracy to 57.5% (+4.8%), the SQA denotation accuracy to 74.5% (+3.5%), and the TABFACT accuracy to 84.6% (+3.6%). Our work opens the way to reason over structured data by pre-training on synthetic executable programs. The project homepage is at https: //table-pretraining.github.io/.
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
**📜 CodeT5: Identifier-aware Unified Pre-trained Encoder-Decoder Models for Code Understanding and Generation**. <br> ✍ QYue Wang, Weishi Wang, Shafiq Joty, Steven C.H. Hoi
 *(EMNLP 2021)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2109.00859){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https: //github.com/salesforce/CodeT5){: target="_blank" .btn .btn-green .mr-1 }
   [Website](https://table-pretraining.github.io/){: target="_blank" .btn .btn-green .mr-1 }
   [Semantic Scholar](https://www.semanticscholar.org/paper/TAPEX%3A-Table-Pre-training-via-Learning-a-Neural-SQL-Liu-Chen/8592953f1ebe38ba4cab05c28a088f5d5691a514){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

<details markdown="block">
  <summary>Abstract</summary>
  {: .fs-3 .text-delta .text-blue-100}
  ```
  Pre-trained models for Natural Languages (NL) like BERT and GPT have been recently shown to transfer well to Programming Languages (PL) and largely benefit a broad set of code-related tasks. Despite their success, most current methods either rely on an encoder-only (or decoder-only) pre-training that is suboptimal for generation (resp. understanding) tasks or process the code snippet in the same way as NL, neglecting the special characteristics of PL such as token types. We present CodeT5, a unified pre-trained encoder-decoder Transformer model that better leverages the code semantics conveyed from the developer-assigned identifiers. Our model employs a unified framework to seamlessly support both code understanding and generation tasks and allows for multi-task learning. Besides, we propose a novel identifier-aware pre-training task that enables the model to distinguish which code tokens are identifiers and to recover them when they are masked. Furthermore, we propose to exploit the user-written code comments with a bimodal dual generation task for better NL-PL alignment. Comprehensive experiments show that CodeT5 significantly outperforms prior methods on understanding tasks such as code defect detection and clone detection, and generation tasks across various directions including PL-NL, NL-PL, and PL-PL. Further analysis reveals that our model can better capture semantic information from code. Our code and pre-trained models are released at https: //github.com/salesforce/CodeT5 .
  ``` 
</details> 
{: .fs-5 .fw-600 .text-blue-300}

<details markdown="block">
  <summary>Comments</summary>
  {: .fs-3 .text-delta .text-red-100}
</details> 
{: .fs-3 .fw-600 .text-red-300}

---

## Others

