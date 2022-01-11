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
**📜 Logical Natural Language Generation from Open-Domain Tables**. <br> ✍ Wenhu Chen, Jianshu Chen, Yu Su, Zhiyu Chen, William Yang Wang
 *(ACL-20)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2004.02349){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://github.com/wenhuchen/LogicNLG){: target="_blank" .btn .btn-green .mr-1 }
   [Semantic Scholar](https://www.semanticscholar.org/paper/Logical-Natural-Language-Generation-from-Tables-Chen-Chen/342ec2f1c1b3d29d3269a2566c44f239f0141aeb){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

**Evaluation Tasks:** [LogicNLG](datastes#logicnlg)

<details markdown="block">
  <summary>Abstract</summary>
  {: .fs-3 .text-delta .text-blue-100}
  ```
  Neural natural language generation (NLG) models have recently shown remarkable progress in fluency and coherence. However, existing studies on neural NLG are primarily focused on surface-level realizations with limited emphasis on logical inference, an important aspect of human thinking and language. In this paper, we suggest a new NLG task where a model is tasked with generating natural language statements that can be \emph{logically entailed} by the facts in an open-domain semi-structured table. To facilitate the study of the proposed logical NLG problem, we use the existing TabFact dataset \cite{chen2019tabfact} featured with a wide range of logical/symbolic inferences as our testbed, and propose new automatic metrics to evaluate the fidelity of generation models w.r.t.\ logical inference. The new task poses challenges to the existing monotonic generation frameworks due to the mismatch between sequence order and logical order. In our experiments, we comprehensively survey different generation architectures (LSTM, Transformer, Pre-Trained LM) trained with different algorithms (RL, Adversarial Training, Coarse-to-Fine) on the dataset and made following observations: 1) Pre-Trained LM can significantly boost both the fluency and logical fidelity metrics, 2) RL and Adversarial Training are trading fluency for fidelity, 3) Coarse-to-Fine generation can help partially alleviate the fidelity issue while maintaining high language fluency. The code and data are available at \url{this https URL}.
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
**📜 TaBERT: Pretraining for Joint Understanding of Textual and Tabular Data**. <br> ✍ Pengcheng Yin, Graham Neubig, Wen-tau Yih, Sebastian Riedel
 *(ACL-20)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2005.08314){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](http://fburl.com/TaBERT){: target="_blank" .btn .btn-green .mr-1 }
   [Semantic Scholar](https://www.semanticscholar.org/paper/TaBERT%3A-Pretraining-for-Joint-Understanding-of-and-Yin-Neubig/a5b1d1cab073cb746a990b37d42dc7b67763f881){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

**Evaluation Tasks:** [WikiSQL](/datasets#wikisql), [Spider　](/datasets#spider)

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
##  Manipulating transformers

{: .fs-4 .fw-800 .text-blue-100}
**📜 MATE: Multi-view Attention for Table Transformer Efficiency**. <br> ✍ Julian Martin Eisenschlos, Maharshi Gor, Thomas Müller, William W. Cohen
 *(EMNLP-21)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2109.04312){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://github.com/google-research/tapas){: target="_blank" .btn .btn-green .mr-1 }
</span> 

**Evaluation Tasks:** [HybridQA](/datasets#hybridqa), [SQA](/datasets#msr-sqa), WikiTableQuestion(/datasets#wikitablequestion), TabFact(/datasets#tabfact)

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
## Others

