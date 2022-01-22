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
We present a collection of research papers that related to structured knowledge grounding tasks.

## Table of contents
{: .no_toc .text-delta .fs-4 style="font-weight:800"}

- TOC
{:toc}


## Structured Knowledge Encoding
TODO: re-order papers by date.

{: .fs-4 .fw-800 .text-blue-100}
**A Comprehensive Exploration on WikiSQL with Table-Aware Word Contextualization**. 
 *(KR2ML Workshop at NIPS-19)*

<span class="fs-10">
   [Paper](https://arxiv.org/abs/1902.01069){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://github.com/naver/sqlova){: target="_blank" .btn .btn-green .mr-1 }
   [Semantic Scholar](https://www.semanticscholar.org/paper/A-Comprehensive-Exploration-on-WikiSQL-with-Word-Hwang-Yim/46b5d1bfe9bc72e056626c7f8cfd4936a4a00c0d){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

**Evaluation Tasks:** [WikiSQL](/benchmarks#wikisql)

---

{: .fs-4 .fw-800 .text-blue-100}
**K-BERT: Enabling Language Representation with Knowledge Graph**. 
 *(AAAI-20)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/1909.07606){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://github.com/autoliuweijie/K-BERT){: target="_blank" .btn .btn-green .mr-1 }
   [Semantic Scholar](https://www.semanticscholar.org/paper/K-BERT%3A-Enabling-Language-Representation-with-Graph-Liu-Zhou/06a73ad09664435f8b3cd90293f4e05a047cf375){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

**Evaluation Tasks:** Book, review,Chnsenticorp, Shopping, Weibo, XNLI, LCQMC, NLPCC-DBQA, MSRA-NER, Finance Q&A, Law Q&A, Finance NER, Medicine NER

{: .fs-4 .fw-800 .text-blue-100}
**A Simple Language Model for Task-Oriented Dialogue**. 
 *(NIPS-20)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2005.00796){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://github.com/salesforce/simpletod){: target="_blank" .btn .btn-green .mr-1 }
   [Semantic Scholar](https://www.semanticscholar.org/paper/A-Simple-Language-Model-for-Task-Oriented-Dialogue-Hosseini-Asl-McCann/71d64c24dc0ac9726d2be57f4936ac4528430f64){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

**Evaluation Tasks:** [MultiWoZ](/benchmarks#multiwoz21)

{: .fs-4 .fw-800 .text-blue-100}
**TaBERT: Pretraining for Joint Understanding of Textual and Tabular Data**. 
 *(ACL-20)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2005.08314){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](http://fburl.com/TaBERT){: target="_blank" .btn .btn-green .mr-1 }
   [Semantic Scholar](https://www.semanticscholar.org/paper/TaBERT%3A-Pretraining-for-Joint-Understanding-of-and-Yin-Neubig/a5b1d1cab073cb746a990b37d42dc7b67763f881){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

**Evaluation Tasks:** [WikiSQL](/benchmarks#wikisql), [Spider](/benchmarks#spider)

{: .fs-4 .fw-800 .text-blue-100}
**TAPAS: Weakly Supervised Table Parsing via Pre-training**. 
 *(ACL-20)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2004.02349){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://github.com/google-research/tapas){: target="_blank" .btn .btn-green .mr-1 }
   [Semantic Scholar](https://www.semanticscholar.org/paper/TaPas%3A-Weakly-Supervised-Table-Parsing-via-Herzig-Nowak/52cb05d721688cb766c6e282e9d55c3b8e3dc0cf){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

**Evaluation Tasks:** [WikiSQL](/benchmarks#wikisql), [WikiTableQuestion](/benchmarks#wikitablequestion), [SQA](/benchmarks#msr-sqa)

{: .fs-4 .fw-800 .text-blue-100}
**GraPPa: Grammar-Augmented Pre-Training for Table Semantic Parsing**. 
 *(ICLR-21)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2009.13845){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://github.com/taoyds/grappa){: target="_blank" .btn .btn-green .mr-1 }
   [Pre-trained Model](https://huggingface.co/Salesforce/grappa_large_jnt){: .btn .btn-purple .mr-1 target="_blank" }
   [Semantic Scholar](https://www.semanticscholar.org/paper/GraPPa%3A-Grammar-Augmented-Pre-Training-for-Table-Yu-Wu/eedf45f62dea0eaef5643c42c84f7cc7b80ee782){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

**Evaluation Tasks:** [Spider](/benchmarks#spider), [WikiSQL fully-supervised-setting&weakly-supervised-setting](/benchmarks#wikisql), [WikiTableQuestion](/benchmarks#wikitablequestion)

{: .fs-4 .fw-800 .text-blue-100}
**SCoRe: Pre-Training for Context Representation in Conversational Semantic Parsing**. 
 *(ICLR-20)*

<span class="fs-2">
   [Paper](https://openreview.net/forum?id=oyZxhRI2RiE){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://github.com/microsoft/SCoRE){: target="_blank" .btn .btn-green .mr-1 }
   [Semantic Scholar](https://www.semanticscholar.org/paper/SCoRe%3A-Pre-Training-for-Context-Representation-in-Yu-Zhang/ff1d3698b8d5f942e6a0775e173720210429b8ae){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

**Evaluation Tasks:** [SParC](/benchmarks#sparc), [CoSQL](/benchmarks#cosql), [MultiWoZ2.1](/benchmarks#multiwoz), [SQA](/benchmarks#sqa)

{: .fs-4 .fw-800 .text-blue-100}
**Structure-Grounded Pretraining for Text-to-SQL**. 
 *(NAACL-20)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2010.12773){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://aka.ms/strug){: target="_blank" .btn .btn-green .mr-1 }
   [Semantic Scholar](https://www.semanticscholar.org/paper/Structure-Grounded-Pretraining-for-Text-to-SQL-Deng-Awadallah/346f54fd61f875ae348f0a38f189ccdacf232df4){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

**Evaluation Tasks:** Text2SQL datasets(Spider-Realistic, ATIS, GeoQuery, Restaurants, Academic, IMDB, Yelp, Scholar, Advising), [Spider](/benchmarks#spider), [WikiSQL(fully supervised setting)](/benchmarks#wikisql)

{: .fs-4 .fw-800 .text-blue-100}
**Understanding tables with intermediate pre-training**. 
 *(EMNLP-20)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2010.00571){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://github.com/google-research/tapas){: target="_blank" .btn .btn-green .mr-1 }
   [Semantic Scholar](https://www.semanticscholar.org/paper/Understanding-tables-with-intermediate-pre-training-Eisenschlos-Krichene/65be695739d0fa35212e49ccccd129535e6d9e15){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

**Evaluation Tasks:** [TabFact](/benchmarks#tabfact), [SQA](/benchmarks#msr-sqa)

{: .fs-4 .fw-800 .text-blue-100}
**UniK-QA: Unified Representations of Structured and Unstructured Knowledge for Open-Domain Question Answering**. 
 *(ACL-20)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2012.14610){: .btn .btn-blue .mr-1 target="_blank" } 
   [Semantic Scholar](https://www.semanticscholar.org/paper/UniK-QA%3A-Unified-Representations-of-Structured-and-O%C4%9Fuz-Chen/0ccf167707dddebe9bbfd2095256804698e3a81d){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

**Evaluation Tasks:**  NaturalQuestions, WebQuestions

{: .fs-4 .fw-800 .text-blue-100}
**Database Reasoning Over Text**. 
 *(ACL-21)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2106.01074){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://github.com/facebookresearch/NeuralDB){: target="_blank" .btn .btn-green .mr-1 }
</span> 

**Evaluation Tasks:** WikiNLDB

{: .fs-4 .fw-800 .text-blue-100}
**Turning Tables: Generating Examples from Semi-structured Tables for Endowing Language Models with Reasoning Skills**. 
 *(arxiv-21)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2107.07653){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://github.com/oriyor/turning_tables){: target="_blank" .btn .btn-green .mr-1 }
</span> 

**Evaluation Tasks:** DROP, IIRC, [MMQA](/benchmarks#multimodalqa)

{: .fs-4 .fw-800 .text-blue-100}
**TAPEX: Table Pre-training via Learning a Neural SQL Executor**. 
 *(arxiv-21)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2107.07653){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://github.com/microsoft/Table-Pretraining){: target="_blank" .btn .btn-green .mr-1 }
</span> 

**Evaluation Tasks:** [WikiSQL](/benchmarks#wikisql), [WikiTableQuestion](/benchmarks#wikitablequestion), [SQA](/benchmarks#msr-sqa), [TabFact](/benchmarks#TabFact)

{: .fs-4 .fw-800 .text-blue-100}
**RnG-KBQA: Generation Augmented Iterative Ranking for Knowledge Base Question Answering**. 
 *(arxiv-21)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2109.08678){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://github.com/salesforce/rng-kbqa){: target="_blank" .btn .btn-green .mr-1 }
</span> 

**Evaluation Tasks:** [GrailQA](/benchmarks#grailqa), [WebQSP](/benchmarks#webqsp)

{: .fs-4 .fw-800 .text-blue-100}
**PICARD: Parsing Incrementally for Constrained Auto-Regressive Decoding from Language Models**. 
 *(EMNLP-21)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2109.05093){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://github.com/ElementAI/picard){: target="_blank" .btn .btn-green .mr-1 }
</span> 

**Evaluation Tasks:** [Spider](/benchmarks#spider), [CoSQL](/benchmarks#cosql)

{: .fs-4 .fw-800 .text-blue-100}
**MATE: Multi-view Attention for Table Transformer Efficiency**. 
 *(EMNLP-21)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2109.04312){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://github.com/google-research/tapas){: target="_blank" .btn .btn-green .mr-1 }
</span> 

**Evaluation Tasks:** [HybridQA](/benchmarks#hybridqa), [SQA](/benchmarks#msr-sqa), [WikiTableQuestion](/benchmarks#wikitablequestion), [TabFact](/benchmarks#tabfact)

{: .fs-4 .fw-800 .text-blue-100}
**Table Fact Verification with Structure-Aware Transformer**. 
 *(ACL-20)*

<span class="fs-2">
   [Paper](https://aclanthology.org/2020.emnlp-main.126/){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://github.com/zhhongzhi/sat){: target="_blank" .btn .btn-green .mr-1 }
   [Semantic Scholar](https://www.semanticscholar.org/paper/Table-Fact-Verification-with-Structure-Aware-Zhang-Wang/38b3c835e272a25fca4fe523dad627feb6552bd3){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

**Evaluation Tasks:** [TabFact](/benchmarks#tabfact)

{: .fs-4 .fw-800 .text-blue-100}
**DoT: An efficient Double Transformer for NLP tasks with tables**. 
 *(ACL-21)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2106.00479){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://github.com/google-research/tapas){: target="_blank" .btn .btn-green .mr-1 }
</span> 

**Evaluation Tasks:** [WikiSQL](/benchmarks#wikisql),  [WikiTableQuestion](/benchmarks#wikitablequestion), [TabFact](/benchmarks#tabfact)

{: .fs-4 .fw-800 .text-blue-100}
**TUTA: Tree-based Transformers for Generally Structured Table Pre-training**. 
 *(KDD-21)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2010.12537){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://github.com/microsoft/TUTA_table_understanding/){: target="_blank" .btn .btn-green .mr-1 }
   [Semantic Scholar](https://www.semanticscholar.org/paper/TUTA%3A-Tree-based-Transformers-for-Generally-Table-Wang-Dong/24a12899ce97bd4a56f7c6b49d3979b9465f0190){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

**Evaluation Tasks:** Cell Type Classification (CTC) tasks, Table Type Classification (TTC) tasks

{: .fs-4 .fw-800 .text-blue-100}
**TABBIE: Pretrained Representations of Tabular Data**. 
 *(NAACL-21)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2105.02584){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://github.com/SFIG611/tabbie){: target="_blank" .btn .btn-green .mr-1 }
   [Semantic Scholar](https://www.semanticscholar.org/paper/TABBIE%3A-Pretrained-Representations-of-Tabular-Data-Iida-Thai/386bfd0e411dee4f512a8737c55dd84846981182){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

**Evaluation Tasks:** Column population, Row population, and Column type prediction tasks

{: .fs-4 .fw-800 .text-blue-100}
**HittER: Hierarchical Transformers for Knowledge Graph Embeddings**. 
 *(EMNLP-21)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2008.12813){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://github.com/sanxing-chen/HittER){: target="_blank" .btn .btn-green .mr-1 }
   [Semantic Scholar](https://www.semanticscholar.org/paper/HittER%3A-Hierarchical-Transformers-for-Knowledge-Chen-Liu/7e7499b47fe57033768f26ef98a3b644688eb2a2){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

**Evaluation Tasks:** FB15K-237, WN18RR, FreebaseQA, [WebQuestionSP](/benchmarks#webqsp)

{: .fs-4 .fw-800 .text-blue-100}
**Constrained Language Models Yield Few-Shot Semantic Parsers**. 
 *(EMNLP-21)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2104.08768){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://github.com/microsoft/semantic_parsing_with_constrained_lm){: target="_blank" .btn .btn-green .mr-1 }
</span> 

**Evaluation Tasks:** Overnight, Break, SMCalFlow

## Pre-training

{: .fs-4 .fw-800 .text-blue-100}
**CodeBERT: A Pre-Trained Model for Programming and Natural Languages**. 
 *(EMNLP 2020)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2002.08155){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://github.com/microsoft/CodeBERT){: target="_blank" .btn .btn-green .mr-1 }
   [Semantic Scholar](https://www.semanticscholar.org/paper/CodeBERT%3A-A-Pre-Trained-Model-for-Programming-and-Feng-Guo/0fe2636446cd686830da3d971b31a004d6094b3c){: .btn .btn-purple .mr-1 target="_blank" }
</span>

{: .fs-4 .fw-800 .text-blue-100}
**Structure-Grounded Pretraining for Text-to-SQL**. 
 *(NAACL 2021)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2002.08155){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://aka.ms/strug){: target="_blank" .btn .btn-green .mr-1 }
   [Semantic Scholar](https://www.semanticscholar.org/paper/Structure-Grounded-Pretraining-for-Text-to-SQL-Deng-Awadallah/1e84152b10e48ef592917576ca74f814adadcdc7){: .btn .btn-purple .mr-1 target="_blank" }
</span>

{: .fs-4 .fw-800 .text-blue-100}
**KGPT: Knowledge-Grounded Pre-Training for Data-to-Text Generation**. 
 *(EMNLP 2020)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2010.02307){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://github.com/wenhuchen/KGPT){: target="_blank" .btn .btn-green .mr-1 }
   [Semantic Scholar](https://www.semanticscholar.org/paper/KGPT%3A-Knowledge-Grounded-Pre-Training-for-Chen-Su/6f33bd4e62955f4d40424f8ae4ec83af4e97862c){: .btn .btn-purple .mr-1 target="_blank" }
</span>

{: .fs-4 .fw-800 .text-blue-100}
**TABBIE: Pretrained Representations of Tabular Data**. 
 *(arixv 2021)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2105.02584){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://github.com/SFIG611/tabbie){: target="_blank" .btn .btn-green .mr-1 }
   [Semantic Scholar](https://www.semanticscholar.org/paper/TABBIE%3A-Pretrained-Representations-of-Tabular-Data-Iida-Thai/386bfd0e411dee4f512a8737c55dd84846981182){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

{: .fs-4 .fw-800 .text-blue-100}
**Question Answering Infused Pre-training of General-Purpose Contextualized Representations**. 
 *(arixv 2021)*

<span classhttps://arxiv.org/abs/2106.08190){: .btn .btn-blue .mr-1 target="_blank" } 
   [Semantic Scholar](https://www.semanticscholar.org/paper/Question-Answering-Infused-Pre-training-of-Jia-Lewis/e4c13aadc6adeb8131bb08324e2688383fbb8ec9){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

{: .fs-4 .fw-800 .text-blue-100}
**Evaluating Large Language Models Trained on Code(Codex)**. 
 *(arixv 2021)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2105.02584){: .btn .btn-blue .mr-1 target="_blank" } 
   [Website](https://openai.com/api/){: target="_blank" .btn .btn-red .mr-1 }
   [Semantic Scholar](https://www.semanticscholar.org/paper/Evaluating-Large-Language-Models-Trained-on-Code-Chen-Tworek/acbdbf49f9bc3f151b93d9ca9a06009f4f6eb269){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

{: .fs-4 .fw-800 .text-blue-100}
**Learning Contextual Representations for Semantic Parsing with Generation-Augmented Pre-Training**. 
 *(AAAI 2021)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2012.10309){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://github.com/awslabs/gap-text2sql){: target="_blank" .btn .btn-green .mr-1 }
   [Semantic Scholar](https://www.semanticscholar.org/paper/Learning-Contextual-Representations-for-Semantic-Shi-Ng/c75a2ee17056d2b8c14ac25f9f328a09eb4cf040){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

{: .fs-4 .fw-800 .text-blue-100}
**TAPEX: Table Pre-training via Learning a Neural SQL Executor**. 
 *(arixv 2021)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2107.07653){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://github.com/microsoft/Table-Pretraining){: target="_blank" .btn .btn-green .mr-1 }
   [Website](https://table-pretraining.github.io/){: target="_blank" .btn .btn-green .mr-1 }
   [Semantic Scholar](https://www.semanticscholar.org/paper/TAPEX%3A-Table-Pre-training-via-Learning-a-Neural-SQL-Liu-Chen/8592953f1ebe38ba4cab05c28a088f5d5691a514){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

{: .fs-4 .fw-800 .text-blue-100}
**CodeT5: Identifier-aware Unified Pre-trained Encoder-Decoder Models for Code Understanding and Generation**. 
 *(EMNLP 2021)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2109.00859){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https: //github.com/salesforce/CodeT5){: target="_blank" .btn .btn-green .mr-1 }
   [Website](https://table-pretraining.github.io/){: target="_blank" .btn .btn-green .mr-1 }
   [Semantic Scholar](https://www.semanticscholar.org/paper/TAPEX%3A-Table-Pre-training-via-Learning-a-Neural-SQL-Liu-Chen/8592953f1ebe38ba4cab05c28a088f5d5691a514){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

## Others

