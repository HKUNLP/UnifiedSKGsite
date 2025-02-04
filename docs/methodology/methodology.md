---
layout: default
title: Methodology
nav_order: 4
toc_list: true
last_modified_date: November 24 2022
permalink: /methodology/
---

# Methods to Structured Knowledge Grounding
{: .no_toc }
In progress
{: .label .label-yellow }

{: .fs-5 .fw-300 }
We present a collection of research papers that related to structured knowledge grounding tasks.

[comment]: <> (## Table of contents)

[comment]: <> ({: .no_toc .text-delta .fs-4 style="font-weight:800"})

[comment]: <> (- TOC)

[comment]: <> ({:toc})

`sk-encoding`: Exploring structured knowledge encoding methods(concatenation of text and structured knowledge, positional embeddings design, manipulation in transformers etc.) on structured knowledge grounding tasks.

`pre-training`: Exploring pre-train(unsupervised training data source, self-supervised tasks etc.) on structured knowledge grounding tasks.

`constrained-decoding`: Exploring decoding methods(constrained decoding etc.) on structured knowledge grounding tasks.

`unifying`: Exploring unification of structured knowledge grounding tasks.

`prompt-learning`: Exploring prompt-learning methods on structured knowledge grounding tasks.

---

{: .fs-4 .fw-800 .text-blue-100}
[A Comprehensive Exploration on WikiSQL with Table-Aware Word Contextualization](https://arxiv.org/abs/1902.01069). 
 **NIPS-19** `sk-encoding`

[comment]: <> (<span class="fs-1">)

[comment]: <> (   [Code]&#40;https://github.com/naver/sqlova&#41;{: target="_blank" .btn .btn-green .mr-1 })

[comment]: <> (   [Semantic Scholar]&#40;https://www.semanticscholar.org/paper/A-Comprehensive-Exploration-on-WikiSQL-with-Word-Hwang-Yim/46b5d1bfe9bc72e056626c7f8cfd4936a4a00c0d&#41;{: .btn .btn-purple .mr-1 target="_blank" })

[comment]: <> (</span>)

[comment]: <> (**Evaluation Tasks:** [WikiSQL]&#40;/benchmarks#wikisql&#41;)


{: .fs-4 .fw-800 .text-blue-100}
[K-BERT: Enabling Language Representation with Knowledge Graph](https://arxiv.org/abs/1909.07606). 
**AAAI-20** `sk-encoding`

[comment]: <> (<span class="fs-1">)

[comment]: <> (   [Code]&#40;https://github.com/autoliuweijie/K-BERT&#41;{: target="_blank" .btn .btn-green .mr-1 })

[comment]: <> (   [Semantic Scholar]&#40;https://www.semanticscholar.org/paper/K-BERT%3A-Enabling-Language-Representation-with-Graph-Liu-Zhou/06a73ad09664435f8b3cd90293f4e05a047cf375&#41;{: .btn .btn-purple .mr-1 target="_blank" })

[comment]: <> (</span> )

[comment]: <> (**Evaluation Tasks:** Book, review,Chnsenticorp, Shopping, Weibo, XNLI, LCQMC, NLPCC-DBQA, MSRA-NER, Finance Q&A, Law Q&A, Finance NER, Medicine NER)

{: .fs-4 .fw-800 .text-blue-100}
[TAPAS: Weakly Supervised Table Parsing via Pre-training](https://arxiv.org/abs/2004.02349). 
**ACL-20** `sk-encoding` `pre-training`

[comment]: <> (<span class="fs-1">)

[comment]: <> (   [Code]&#40;https://github.com/google-research/tapas&#41;{: target="_blank" .btn .btn-green .mr-1 })

[comment]: <> (   [Semantic Scholar]&#40;https://www.semanticscholar.org/paper/TaPas%3A-Weakly-Supervised-Table-Parsing-via-Herzig-Nowak/52cb05d721688cb766c6e282e9d55c3b8e3dc0cf&#41;{: .btn .btn-purple .mr-1 target="_blank" })

[comment]: <> (</span> )

[comment]: <> (**Evaluation Tasks:** [WikiSQL]&#40;/benchmarks#wikisql&#41;, [WikiTableQuestion]&#40;/benchmarks#wikitablequestion&#41;, [SQA]&#40;/benchmarks#msr-sqa&#41;)

{: .fs-4 .fw-800 .text-blue-100}
[A Simple Language Model for Task-Oriented Dialogue](https://arxiv.org/abs/2005.00796). 
**NIPS-20** `sk-encoding`

[comment]: <> (<span class="fs-1">)

[comment]: <> (   [Code]&#40;https://github.com/salesforce/simpletod&#41;{: target="_blank" .btn .btn-green .mr-1 })

[comment]: <> (   [Semantic Scholar]&#40;https://www.semanticscholar.org/paper/A-Simple-Language-Model-for-Task-Oriented-Dialogue-Hosseini-Asl-McCann/71d64c24dc0ac9726d2be57f4936ac4528430f64&#41;{: .btn .btn-purple .mr-1 target="_blank" })

[comment]: <> (</span> )

[comment]: <> (**Evaluation Tasks:** [MultiWoZ]&#40;/benchmarks#multiwoz21&#41;)

{: .fs-4 .fw-800 .text-blue-100}
[TaBERT: Pretraining for Joint Understanding of Textual and Tabular Data](https://arxiv.org/abs/2005.08314). 
**ACL-20** `sk-encoding` `pre-training`

[comment]: <> (<span class="fs-1">)

[comment]: <> (   [Code]&#40;http://fburl.com/TaBERT&#41;{: target="_blank" .btn .btn-green .mr-1 })

[comment]: <> (   [Semantic Scholar]&#40;https://www.semanticscholar.org/paper/TaBERT%3A-Pretraining-for-Joint-Understanding-of-and-Yin-Neubig/a5b1d1cab073cb746a990b37d42dc7b67763f881&#41;{: .btn .btn-purple .mr-1 target="_blank" })

[comment]: <> (</span> )

[comment]: <> (**Evaluation Tasks:** [WikiSQL]&#40;/benchmarks#wikisql&#41;, [Spider]&#40;/benchmarks#spider&#41;)

{: .fs-4 .fw-800 .text-blue-100}
[HittER: Hierarchical Transformers for Knowledge Graph Embeddings](https://arxiv.org/abs/2008.12813). 
**EMNLP-21** `sk-encoding`

[comment]: <> (<span class="fs-1">)

[comment]: <> (   [Code]&#40;https://github.com/sanxing-chen/HittER&#41;{: target="_blank" .btn .btn-green .mr-1 })

[comment]: <> (   [Semantic Scholar]&#40;https://www.semanticscholar.org/paper/HittER%3A-Hierarchical-Transformers-for-Knowledge-Chen-Liu/7e7499b47fe57033768f26ef98a3b644688eb2a2&#41;{: .btn .btn-purple .mr-1 target="_blank" })

[comment]: <> (</span> )

[comment]: <> (**Evaluation Tasks:** FB15K-237, WN18RR, FreebaseQA, [WebQuestionSP]&#40;/benchmarks#webqsp&#41;)

{: .fs-4 .fw-800 .text-blue-100}
[GraPPa: Grammar-Augmented Pre-Training for Table Semantic Parsing](https://arxiv.org/abs/2009.13845). 
**ICLR-21** `pre-training`

[comment]: <> (<span class="fs-1">)

[comment]: <> (   [Code]&#40;https://github.com/taoyds/grappa&#41;{: target="_blank" .btn .btn-green .mr-1 })

[comment]: <> (   [Pre-trained Model]&#40;https://huggingface.co/Salesforce/grappa_large_jnt&#41;{: .btn .btn-purple .mr-1 target="_blank" })

[comment]: <> (   [Semantic Scholar]&#40;https://www.semanticscholar.org/paper/GraPPa%3A-Grammar-Augmented-Pre-Training-for-Table-Yu-Wu/eedf45f62dea0eaef5643c42c84f7cc7b80ee782&#41;{: .btn .btn-purple .mr-1 target="_blank" })

[comment]: <> (</span> )

[comment]: <> (**Evaluation Tasks:** [Spider]&#40;/benchmarks#spider&#41;, [WikiSQL fully-supervised-setting&weakly-supervised-setting]&#40;/benchmarks#wikisql&#41;, [WikiTableQuestion]&#40;/benchmarks#wikitablequestion&#41;)

{: .fs-4 .fw-800 .text-blue-100}
[Multi-Task Pre-training for Plug-and-play Task-oriented Dialogue System](https://arxiv.org/abs/2109.14739). 
**EMNLP-21** `pre-training` `unifying`

{: .fs-4 .fw-800 .text-blue-100}
[SCoRe: Pre-Training for Context Representation in Conversational Semantic Parsing](https://openreview.net/forum?id=oyZxhRI2RiE). 
**ICLR-21** `sk-encoding` `pre-training`

[comment]: <> (<span class="fs-1">)

[comment]: <> (   [Code]&#40;https://github.com/microsoft/SCoRE&#41;{: target="_blank" .btn .btn-green .mr-1 })

[comment]: <> (   [Semantic Scholar]&#40;https://www.semanticscholar.org/paper/SCoRe%3A-Pre-Training-for-Context-Representation-in-Yu-Zhang/ff1d3698b8d5f942e6a0775e173720210429b8ae&#41;{: .btn .btn-purple .mr-1 target="_blank" })

[comment]: <> (</span> )

[comment]: <> (**Evaluation Tasks:** [SParC]&#40;/benchmarks#sparc&#41;, [CoSQL]&#40;/benchmarks#cosql&#41;, [MultiWoZ2.1]&#40;/benchmarks#multiwoz&#41;, [SQA]&#40;/benchmarks#sqa&#41;)

{: .fs-4 .fw-800 .text-blue-100}
[Knowledge Graph Based Synthetic Corpus Generation for Knowledge-Enhanced Language Model Pre-training](https://arxiv.org/abs/2010.12688). 
**NAACL-21** `sk-encoding` `pre-training`

{: .fs-4 .fw-800 .text-blue-100}
[Structure-Grounded Pretraining for Text-to-SQL](https://arxiv.org/abs/2010.12773). 
**NAACL-21** `pre-training`

[comment]: <> (<span class="fs-1">)

[comment]: <> (   [Code]&#40;https://aka.ms/strug&#41;{: target="_blank" .btn .btn-green .mr-1 })

[comment]: <> (   [Semantic Scholar]&#40;https://www.semanticscholar.org/paper/Structure-Grounded-Pretraining-for-Text-to-SQL-Deng-Awadallah/1e84152b10e48ef592917576ca74f814adadcdc7&#41;{: .btn .btn-purple .mr-1 target="_blank" })

[comment]: <> (</span>)

[comment]: <> ({: .fs-4 .fw-800 .text-blue-100})

[comment]: <> ([TUTA: Tree-based Transformers for Generally Structured Table Pre-training]&#40;https://arxiv.org/abs/2010.12537&#41;. )

[comment]: <> (**KDD-21**)

[comment]: <> (<span class="fs-1">)

[comment]: <> (   [Code](https://github.com/microsoft/TUTA_table_understanding/){: target="_blank" .btn .btn-green .mr-1 })

[comment]: <> (   [Semantic Scholar](https://www.semanticscholar.org/paper/TUTA%3A-Tree-based-Transformers-for-Generally-Table-Wang-Dong/24a12899ce97bd4a56f7c6b49d3979b9465f0190){: .btn .btn-purple .mr-1 target="_blank" })

[comment]: <> (</span> )

[comment]: <> (**Evaluation Tasks:** Cell Type Classification (CTC) tasks, Table Type Classification (TTC) tasks)

{: .fs-4 .fw-800 .text-blue-100}
[Understanding tables with intermediate pre-training](https://arxiv.org/abs/2010.00571). 
**EMNLP-20** `pre-training`

[comment]: <> (<span class="fs-1">)

[comment]: <> (   [Code]&#40;https://github.com/google-research/tapas&#41;{: target="_blank" .btn .btn-green .mr-1 })

[comment]: <> (   [Semantic Scholar]&#40;https://www.semanticscholar.org/paper/Understanding-tables-with-intermediate-pre-training-Eisenschlos-Krichene/65be695739d0fa35212e49ccccd129535e6d9e15&#41;{: .btn .btn-purple .mr-1 target="_blank" })

[comment]: <> (</span> )

[comment]: <> (**Evaluation Tasks:** [TabFact]&#40;/benchmarks#tabfact&#41;, [SQA]&#40;/benchmarks#msr-sqa&#41;)

{: .fs-4 .fw-800 .text-blue-100}
[KGPT: Knowledge-Grounded Pre-Training for Data-to-Text Generation](https://arxiv.org/abs/2010.02307). 
**EMNLP-20** `sk-encoding` `pre-training` `unifying`

[comment]: <> (<span class="fs-1">)

[comment]: <> (   [Code]&#40;https://github.com/wenhuchen/KGPT&#41;{: target="_blank" .btn .btn-green .mr-1 })

[comment]: <> (   [Semantic Scholar]&#40;https://www.semanticscholar.org/paper/KGPT%3A-Knowledge-Grounded-Pre-Training-for-Chen-Su/6f33bd4e62955f4d40424f8ae4ec83af4e97862c&#41;{: .btn .btn-purple .mr-1 target="_blank" })

[comment]: <> (</span>)

{: .fs-4 .fw-800 .text-blue-100}
[UniK-QA: Unified Representations of Structured and Unstructured Knowledge for Open-Domain Question Answering](https://arxiv.org/abs/2012.14610). 
**NAACL-22** `sk-encoding` `unifying`

[comment]: <> (<span class="fs-1">)

[comment]: <> (   [Semantic Scholar]&#40;https://www.semanticscholar.org/paper/UniK-QA%3A-Unified-Representations-of-Structured-and-O%C4%9Fuz-Chen/0ccf167707dddebe9bbfd2095256804698e3a81d&#41;{: .btn .btn-purple .mr-1 target="_blank" })

[comment]: <> (</span> )

[comment]: <> (**Evaluation Tasks:**  NaturalQuestions, WebQuestions)

{: .fs-4 .fw-800 .text-blue-100}
[JAKET: Joint Pre-training of Knowledge Graph and Language Understanding](https://arxiv.org/abs/2010.00796). 
**AAAI-22** `sk-encoding` `pre-training`

{: .fs-4 .fw-800 .text-blue-100}
[Learning Contextual Representations for Semantic Parsing with Generation-Augmented Pre-Training](https://arxiv.org/abs/2012.10309). 
**AAAI-21** `pre-training`

[comment]: <> (<span class="fs-1">)

[comment]: <> (   [Code]&#40;https://github.com/awslabs/gap-text2sql&#41;{: target="_blank" .btn .btn-green .mr-1 })

[comment]: <> (   [Semantic Scholar]&#40;https://www.semanticscholar.org/paper/Learning-Contextual-Representations-for-Semantic-Shi-Ng/c75a2ee17056d2b8c14ac25f9f328a09eb4cf040&#41;{: .btn .btn-purple .mr-1 target="_blank" })

[comment]: <> (</span>)

[comment]: <> (**Evaluation Tasks:** WikiNLDB)

{: .fs-4 .fw-800 .text-blue-100}
[Table Fact Verification with Structure-Aware Transformer](https://aclanthology.org/2020.emnlp-main.126/). 
**ACL-20** `sk-encoding`

[comment]: <> (<span class="fs-1">)

[comment]: <> (   [Code]&#40;https://github.com/zhhongzhi/sat&#41;{: target="_blank" .btn .btn-green .mr-1 })

[comment]: <> (   [Semantic Scholar]&#40;https://www.semanticscholar.org/paper/Table-Fact-Verification-with-Structure-Aware-Zhang-Wang/38b3c835e272a25fca4fe523dad627feb6552bd3&#41;{: .btn .btn-purple .mr-1 target="_blank" })

[comment]: <> (</span> )

[comment]: <> (**Evaluation Tasks:** [TabFact]&#40;/benchmarks#tabfact&#41;)

{: .fs-4 .fw-800 .text-blue-100}
[Structural Adapters in Pretrained Language Models for AMR-to-Text Generation](https://arxiv.org/abs/2103.09120). 
**EMNLP-21** `sk-encoding`

{: .fs-4 .fw-800 .text-blue-100}
[Constrained Language Models Yield Few-Shot Semantic Parsers](https://arxiv.org/abs/2104.08768). 
**EMNLP-21** `sk-encoding` `constrained-decoding`

{: .fs-4 .fw-800 .text-blue-100}
[Case-based Reasoning for Natural Language Queries over Knowledge Bases](https://arxiv.org/abs/2104.08762)
**EMNLP-21** `prompt-learning`

[comment]: <> (<span class="fs-1">)

[comment]: <> (   [Code]&#40;https://github.com/microsoft/semantic_parsing_with_constrained_lm&#41;{: target="_blank" .btn .btn-green .mr-1 })

[comment]: <> (</span> )

[comment]: <> (**Evaluation Tasks:** Overnight, Break, SMCalFlow)

[comment]: <> ({: .fs-4 .fw-800 .text-blue-100})

[comment]: <> ([TABBIE: Pretrained Representations of Tabular Data]&#40;https://arxiv.org/abs/2105.02584&#41;. )

[comment]: <> (**arixv-21**)

[comment]: <> (<span class="fs-1">)

[comment]: <> (   [Code]&#40;https://github.com/SFIG611/tabbie&#41;{: target="_blank" .btn .btn-green .mr-1 })

[comment]: <> (   [Semantic Scholar]&#40;https://www.semanticscholar.org/paper/TABBIE%3A-Pretrained-Representations-of-Tabular-Data-Iida-Thai/386bfd0e411dee4f512a8737c55dd84846981182&#41;{: .btn .btn-purple .mr-1 target="_blank" })

[comment]: <> (</span> )

{: .fs-4 .fw-800 .text-blue-100}
[DoT: An efficient Double Transformer for NLP tasks with tables](https://arxiv.org/abs/2106.00479). 
**ACL-21** `sk-encoding`

[comment]: <> (<span class="fs-1">)

[comment]: <> (   [Code]&#40;https://github.com/google-research/tapas&#41;{: target="_blank" .btn .btn-green .mr-1 })

[comment]: <> (</span> )

[comment]: <> (**Evaluation Tasks:** [WikiSQL]&#40;/benchmarks#wikisql&#41;,  [WikiTableQuestion]&#40;/benchmarks#wikitablequestion&#41;, [TabFact]&#40;/benchmarks#tabfact&#41;)

{: .fs-4 .fw-800 .text-blue-100}
[Database Reasoning Over Text](https://arxiv.org/abs/2106.01074). 
**ACL-21** `sk-encoding`

[comment]: <> (<span class="fs-1">)

[comment]: <> (   [Code]&#40;https://github.com/facebookresearch/NeuralDB&#41;{: target="_blank" .btn .btn-green .mr-1 })

[comment]: <> (</span> )

{: .fs-4 .fw-800 .text-blue-100}
[Turning Tables: Generating Examples from Semi-structured Tables for Endowing Language Models with Reasoning Skills](https://arxiv.org/abs/2107.07261). 
**ACL-22** `pre-training`

[comment]: <> (<span class="fs-1">)

[comment]: <> (   [Code]&#40;https://github.com/oriyor/turning_tables&#41;{: target="_blank" .btn .btn-green .mr-1 })

[comment]: <> (</span> )

[comment]: <> (**Evaluation Tasks:** DROP, IIRC, [MMQA]&#40;/benchmarks#multimodalqa&#41;)

{: .fs-4 .fw-800 .text-blue-100}
[TAPEX: Table Pre-training via Learning a Neural SQL Executor](https://arxiv.org/abs/2107.07653). 
**ICLR-22** `pre-training`

[comment]: <> (<span class="fs-1">)

[comment]: <> (   [Code]&#40;https://github.com/microsoft/Table-Pretraining&#41;{: target="_blank" .btn .btn-green .mr-1 })

[comment]: <> (   [Website]&#40;https://table-pretraining.github.io/&#41;{: target="_blank" .btn .btn-green .mr-1 })

[comment]: <> (   [Semantic Scholar]&#40;https://www.semanticscholar.org/paper/TAPEX%3A-Table-Pre-training-via-Learning-a-Neural-SQL-Liu-Chen/8592953f1ebe38ba4cab05c28a088f5d5691a514&#41;{: .btn .btn-purple .mr-1 target="_blank" })

[comment]: <> (</span> )

{: .fs-4 .fw-800 .text-blue-100}
[HTLM: Hyper-Text Pre-Training and Prompting of Language Models](https://arxiv.org/abs/2107.06955).
**arxiv-21** `pre-training`

{: .fs-4 .fw-800 .text-blue-100}
[MATE: Multi-view Attention for Table Transformer Efficiency](https://arxiv.org/abs/2109.04312).
**EMNLP-21** `sk-encoding`

[comment]: <> (<span class="fs-1">)

[comment]: <> (   [Code]&#40;https://github.com/google-research/tapas&#41;{: target="_blank" .btn .btn-green .mr-1 })

[comment]: <> (</span> )

[comment]: <> (**Evaluation Tasks:** [HybridQA]&#40;/benchmarks#hybridqa&#41;, [SQA]&#40;/benchmarks#msr-sqa&#41;, [WikiTableQuestion]&#40;/benchmarks#wikitablequestion&#41;, [TabFact]&#40;/benchmarks#tabfact&#41;)

{: .fs-4 .fw-800 .text-blue-100}
[RnG-KBQA: Generation Augmented Iterative Ranking for Knowledge Base Question Answering](https://arxiv.org/abs/2109.08678).
**ACL-22** `prompt-learning`

[comment]: <> ([comment]: <> &#40;<span class="fs-1">&#41;)

[comment]: <> ([comment]: <> &#40;   [Code]&#40;https://github.com/salesforce/rng-kbqa&#41;{: target="_blank" .btn .btn-green .mr-1 }&#41;)

[comment]: <> ([comment]: <> &#40;</span> &#41;)

[comment]: <> ([comment]: <> &#40;**Evaluation Tasks:** [GrailQA]&#40;/benchmarks#grailqa&#41;, [WebQSP]&#40;/benchmarks#webqsp&#41;&#41;)

{: .fs-4 .fw-800 .text-blue-100}
[Multi-Task Pre-Training for Plug-and-Play Task-Oriented Dialogue System](https://arxiv.org/abs/2109.14739). 
**ACL-22** `pre-training` `unifying`

{: .fs-4 .fw-800 .text-blue-100}
[PICARD: Parsing Incrementally for Constrained Auto-Regressive Decoding from Language Models](https://arxiv.org/abs/2109.05093). 
**EMNLP-21** `constrained-decoding`

[comment]: <> (<span class="fs-1">)

[comment]: <> (   [Code]&#40;https://github.com/ElementAI/picard&#41;{: target="_blank" .btn .btn-green .mr-1 })

[comment]: <> (</span> )

[comment]: <> (**Evaluation Tasks:** [Spider]&#40;/benchmarks#spider&#41;, [CoSQL]&#40;/benchmarks#cosql&#41;)

{: .fs-4 .fw-800 .text-blue-100}
[FORTAP: Using Formulae for Numerical-Reasoning-Aware Table Pretraining](https://arxiv.org/abs/2109.07323). 
**ACL-22** `pre-training`

{: .fs-4 .fw-800 .text-blue-100}
[Learning To Retrieve Prompts for In-Context Learning](https://arxiv.org/abs/2112.08633)
**NAACL-HLT-22** `prompt-learning`

{: .fs-4 .fw-800 .text-blue-100}
[Multi-Instance Training for Question Answering Across Table and Linked Text](https://arxiv.org/abs/2112.07337)
**arxiv-21**

{: .fs-4 .fw-800 .text-blue-100}
[Synchromesh: Reliable Code Generation from Pre-trained Language Models](https://arxiv.org/pdf/2201.11227.pdf)
**ICLR-22** `constrained-decoding` `prompt-learning`

{: .fs-4 .fw-800 .text-blue-100}
[UnifiedSKG: Unifying and Multi-Tasking Structured Knowledge Grounding with Text-to-Text Language Models](https://arxiv.org/abs/2201.05966). 
**EMNLP-22** `unifying`

{: .fs-4 .fw-800 .text-blue-100}
[TableFormer: Robust Transformer Modeling for Table-Text Encoding](https://arxiv.org/abs/2203.00274). 
**ACL-22** `sk-encoding`

{: .fs-4 .fw-800 .text-blue-100}
[Input-Tuning: Adapting Unfamiliar Inputs to Frozen Pretrained Models](https://arxiv.org/abs/2203.03131). 
**arxiv-22** `prompt-learning`

{: .fs-4 .fw-800 .text-blue-100}
[In-Context Learning for Few-Shot Dialogue State Tracking](https://arxiv.org/abs/2203.08568). 
**EMNLP-22** `prompt-learning`

{: .fs-4 .fw-800 .text-blue-100}
[T-RAG: End-to-End Table Question Answering via Retrieval-Augmented Generation](https://arxiv.org/abs/2203.16714). 
**arxiv-22** `sk-encoding`

{: .fs-4 .fw-800 .text-blue-100}
[ArcaneQA: Dynamic Program Induction and Contextualized Encoding for Knowledge Base Question Answering](https://arxiv.org/abs/2204.08109). 
**COLING-22** `sk-encoding`

{: .fs-4 .fw-800 .text-blue-100}
[SPACE-2: Tree-Structured Semi-Supervised Contrastive Pre-training for Task-Oriented Dialog Understanding](https://arxiv.org/abs/2209.06638). 
**COLING-22** `sk-encoding` `pre-training`

{: .fs-4 .fw-800 .text-blue-100}
[Evaluating the Impact of Model Scale for Compositional Generalization in Semantic Parsing](https://arxiv.org/abs/2205.12253). 
**EMNLP-22** `unifying` `prompt-learning`

{: .fs-4 .fw-800 .text-blue-100}
[TaCube: Pre-computing Data Cubes for Answering Numerical-Reasoning Questions over Tabular Data](https://arxiv.org/abs/2205.12682). 
**EMNLP-22** `sk-encoding` `pre-training`

{: .fs-4 .fw-800 .text-blue-100}
[PLOG: Table-to-Logic Pretraining for Logical Table-to-Text Generation](https://arxiv.org/abs/2205.12697). 
**EMNLP-22** `pre-training`

{: .fs-4 .fw-800 .text-blue-100}
[Natural Language to Code Translation with Execution](https://arxiv.org/abs/2204.11454). 
**arxiv-22** `prompt-learning` `guided with execution`

{: .fs-4 .fw-800 .text-blue-100}
[R2D2: Robust Data-to-Text with Replacement Detection](https://arxiv.org/abs/2205.12467). 
**arxiv-22** `sk-encoding`

{: .fs-4 .fw-800 .text-blue-100}
[OmniTab: Pretraining with Natural and Synthetic Data for Few-shot Table-based Question Answering](https://arxiv.org/abs/2207.03637). 
**NAACL-22** `pre-training`

{: .fs-4 .fw-800 .text-blue-100}
[Dual-Channel Evidence Fusion for Fact Verification over Texts and Tables](https://aclanthology.org/2022.naacl-main.384/). 
**NAACL-22** `sk-encoding`

{: .fs-4 .fw-800 .text-blue-100}
[Binding Language Models in Symbolic Languages](https://arxiv.org/abs/2210.02875). 
**arxiv-22** `prompt-learning`