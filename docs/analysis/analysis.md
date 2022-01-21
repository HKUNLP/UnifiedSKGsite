---
layout: default
title: Inspiring Topics
nav_order: 5
toc_list: true
last_modified_date: Jan 21 2022
permalink: /analysis/
---

# Inspiring Topics about Structured Knowledge Grounding
{: .no_toc }

{: .fs-5 .fw-300 }
Here we present a collection of papers with inspiring topics that may help structured knowledge grounding(we focus more on recent work). Some paper may also belongs to other category. Contact us or pr on GitHub if we missed anything.


---

## Table of contents
{: .no_toc .text-delta .fs-4 style="font-weight:800"}

- TOC
{:toc}



## Multi-task learning


{: .fs-4 .fw-800 .text-blue-100}
**📜 A Joint Many-Task Model: Growing a Neural Network for Multiple NLP Tasks**. <br> ✍ Kazuma Hashimoto, Caiming Xiong, Yoshimasa Tsuruoka, Richard Socher
 *(EMNLP 2017)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/1611.01587){: .btn .btn-blue .mr-1 target="_blank" } 
   [Semantic Scholar](https://www.semanticscholar.org/paper/A-Joint-Many-Task-Model%3A-Growing-a-Neural-Network-Hashimoto-Xiong/ade0c116120b54b57a91da51235108b75c28375a){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

<details markdown="block">
  <summary>Abstract</summary>
  {: .fs-3 .text-delta .text-blue-100}
  ```
  Transfer and multi-task learning have traditionally focused on either a single source-target pair or very few, similar tasks. Ideally, the linguistic levels of morphology, syntax and semantics would benefit each other by being trained in a single model. We introduce such a joint many-task model together with a strategy for successively growing its depth to solve increasingly complex tasks. All layers include shortcut connections to both word representations and lower-level task predictions. We use a simple regularization term to allow for optimizing all model weights to improve one task's loss without exhibiting catastrophic interference of the other tasks. Our single end-to-end trainable model obtains state-of-the-art results on chunking, dependency parsing, semantic relatedness and textual entailment. It also performs competitively on POS tagging. Our dependency parsing layer relies only on a single feed-forward pass and does not require a beam search.
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
**📜 Taskonomy: Disentangling Task Transfer Learning**. <br> ✍ Amir Zamir, Alexander Sax, William Shen, Leonidas Guibas, Jitendra Malik, Silvio Savarese
 *(CVPR 2018 best paper)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/1804.08328){: .btn .btn-blue .mr-1 target="_blank" } 
   [Website](http://taskonomy.vision/){: target="_blank" .btn .btn-orange .mr-1 }
   [Semantic Scholar](https://www.semanticscholar.org/paper/Taskonomy%3A-Disentangling-Task-Transfer-Learning-Zamir-Sax/2fe2cfd98e232f1396f01881853ed6b3d5e37d65){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

<details markdown="block">
  <summary>Abstract</summary>
  {: .fs-3 .text-delta .text-blue-100}
  ```
  Do visual tasks have a relationship, or are they unrelated? For instance, could having surface normals simplify estimating the depth of an image? Intuition answers these questions positively, implying existence of a structure among visual tasks. Knowing this structure has notable values; it is the concept underlying transfer learning and provides a principled way for identifying redundancies across tasks, e.g., to seamlessly reuse supervision among related tasks or solve many tasks in one system without piling up the complexity. We proposes a fully computational approach for modeling the structure of space of visual tasks. This is done via finding (first and higher-order) transfer learning dependencies across a dictionary of twenty six 2D, 2.5D, 3D, and semantic tasks in a latent space. The product is a computational taxonomic map for task transfer learning. We study the consequences of this structure, e.g. nontrivial emerged relationships, and exploit them to reduce the demand for labeled data. We provide a set of tools for computing and probing this taxonomical structure including a solver users can employ to find supervision policies for their use cases.
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
**📜 Task2Vec: Task Embedding for Meta-Learning**. <br> ✍ 
Alessandro Achille, Michael Lam, Rahul Tewari, Avinash Ravichandran, Subhransu Maji, Charless Fowlkes, Stefano Soatto, Pietro Perona
 *(ICCV 2019)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/1902.03545){: .btn .btn-blue .mr-1 target="_blank" } 
   [Semantic Scholar](https://www.semanticscholar.org/paper/Task2Vec%3A-Task-Embedding-for-Meta-Learning-Achille-Lam/3f0e82d56d18787feac8ddbc7c3c8490eb76a4c7){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

<details markdown="block">
  <summary>Abstract</summary>
  {: .fs-3 .text-delta .text-blue-100}
  ```
  We introduce a method to provide vectorial representations of visual classification tasks which can be used to reason about the nature of those tasks and their relations. Given a dataset with ground-truth labels and a loss function defined over those labels, we process images through a "probe network" and compute an embedding based on estimates of the Fisher information matrix associated with the probe network parameters. This provides a fixed-dimensional embedding of the task that is independent of details such as the number of classes and does not require any understanding of the class label semantics. We demonstrate that this embedding is capable of predicting task similarities that match our intuition about semantic and taxonomic relations between different visual tasks (e.g., tasks based on classifying different types of plants are similar) We also demonstrate the practical value of this framework for the meta-task of selecting a pre-trained feature extractor for a new task. We present a simple meta-learning framework for learning a metric on embeddings that is capable of predicting which feature extractors will perform well. Selecting a feature extractor with task embedding obtains a performance close to the best available feature extractor, while costing substantially less than exhaustively training and evaluating on all available feature extractors.
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
**📜 AdapterFusion: Non-Destructive Task Composition for Transfer Learning**. <br> ✍ Jonas Pfeiffer, Aishwarya Kamath, Andreas Rücklé, Kyunghyun Cho, Iryna Gurevych
 *(EACL 2021)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2005.00247v3){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://adapterhub.ml/){: target="_blank" .btn .btn-green .mr-1 }
   [Semantic Scholar](https://www.semanticscholar.org/paper/AdapterFusion%3A-Non-Destructive-Task-Composition-for-Pfeiffer-Kamath/98ef0db84e62aef969629264c9de1f4d0013f3b9){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

<details markdown="block">
  <summary>Abstract</summary>
  {: .fs-3 .text-delta .text-blue-100}
  ```
  Sequential fine-tuning and multi-task learning are methods aiming to incorporate knowledge from multiple tasks; however, they suffer from catastrophic forgetting and difficulties in dataset balancing. To address these shortcomings, we propose AdapterFusion, a new two stage learning algorithm that leverages knowledge from multiple tasks. First, in the knowledge extraction stage we learn task specific parameters called adapters, that encapsulate the task-specific information. We then combine the adapters in a separate knowledge composition step. We show that by separating the two stages, i.e., knowledge extraction and knowledge composition, the classifier can effectively exploit the representations learned from multiple tasks in a non-destructive manner. We empirically evaluate AdapterFusion on 16 diverse NLU tasks, and find that it effectively combines various types of knowledge at different layers of the model. We show that our approach outperforms traditional strategies such as full fine-tuning as well as multi-task learning. Our code and adapters are available at this http URL.
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
**📜 Exploring and Predicting Transferability across NLP Tasks**. <br> ✍ Tu Vu, Tong Wang, Tsendsuren Munkhdalai, Alessandro Sordoni, Adam Trischler, Andrew Mattarella-Micke, Subhransu Maji, Mohit Iyyer
 *(EMNLP 2020)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2005.00770){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://github.com/tuvuumass/task-transferability){: target="_blank" .btn .btn-green .mr-1 }
   [Semantic Scholar](https://www.semanticscholar.org/paper/Exploring-and-Predicting-Transferability-across-NLP-Vu-Wang/d1206ccabd1980848f14472d6548251c2fab7963){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

<details markdown="block">
  <summary>Abstract</summary>
  {: .fs-3 .text-delta .text-blue-100}
  ```
  Recent advances in NLP demonstrate the effectiveness of training large-scale language models and transferring them to downstream tasks. Can fine-tuning these models on tasks other than language modeling further improve performance? In this paper, we conduct an extensive study of the transferability between 33 NLP tasks across three broad classes of problems (text classification, question answering, and sequence labeling). Our results show that transfer learning is more beneficial than previously thought, especially when target task data is scarce, and can improve performance even when the source task is small or differs substantially from the target task (e.g., part-of-speech tagging transfers well to the DROP QA dataset). We also develop task embeddings that can be used to predict the most transferable source tasks for a given target task, and we validate their effectiveness in experiments controlled for source and target data size. Overall, our experiments reveal that factors such as source data size, task and domain similarity, and task complexity all play a role in determining transferability.
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
**📜 Structured Prediction as Translation between Augmented Natural Languages**. <br> ✍ Giovanni Paolini, Ben Athiwaratkun, Jason Krone, Jie Ma, Alessandro Achille, Rishita Anubhai, Cicero Nogueira dos Santos, Bing Xiang, Stefano Soatto
 *(ICLR 2021)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2101.05779){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://github.com/amazon-research/tanl){: target="_blank" .btn .btn-green .mr-1 }
   [Semantic Scholar](https://www.semanticscholar.org/paper/Structured-Prediction-as-Translation-between-Paolini-Athiwaratkun/1cb3f6d545b68db3e7fc6055dcf44099c3ac4672){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

<details markdown="block">
  <summary>Abstract</summary>
  {: .fs-3 .text-delta .text-blue-100}
  ```
  Recent advances in NLP demonstrate the effectiveness of training large-scale language models and transferring them to downstream tasks. Can fine-tuning these models on tasks other than language modeling further improve performance? In this paper, we conduct an extensive study of the transferability between 33 NLP tasks across three broad classes of problems (text classification, question answering, and sequence labeling). Our results show that transfer learning is more beneficial than previously thought, especially when target task data is scarce, and can improve performance even when the source task is small or differs substantially from the target task (e.g., part-of-speech tagging transfers well to the DROP QA dataset). We also develop task embeddings that can be used to predict the most transferable source tasks for a given target task, and we validate their effectiveness in experiments controlled for source and target data size. Overall, our experiments reveal that factors such as source data size, task and domain similarity, and task complexity all play a role in determining transferability.
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
**📜 Adapting Language Models for Zero-shot Learning by Meta-tuning on Dataset and Prompt Collections**. <br> ✍ Ruiqi Zhong, Kristy Lee, Zheng Zhang, Dan Klein
 *(EMNLP 2021)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2104.04670){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://github.com/ruiqi-zhong/Meta-tuning){: target="_blank" .btn .btn-green .mr-1 }
   [Semantic Scholar](https://www.semanticscholar.org/paper/Adapting-Language-Models-for-Zero-shot-Learning-by-Zhong-Lee/4b0ec90dc10e51c1fc983edcd57bb86636d7b3ca){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

<details markdown="block">
  <summary>Abstract</summary>
  {: .fs-3 .text-delta .text-blue-100}
  ```
  Large pre-trained language models (LMs) such as GPT-3 have acquired a surprising ability to perform zero-shot learning. For example, to classify sentiment without any training examples, we can “prompt" the LM with the review and the label description “Does the user like this movie?", and ask whether the next word is “Yes" or “No". However, the next word prediction training objective is still misaligned with the target zero-shot learning objective. To address this weakness, we propose meta-tuning, which directly optimizes the zero-shot learning objective by finetuning pre-trained language models on a collection of datasets. We focus on classification tasks, and construct the meta-dataset by aggregating 43 existing datasets and annotating 441 label descriptions in a question-answering (QA) format. When evaluated on unseen tasks, meta-tuned models outperform a samesized QA model and the previous SOTA zeroshot learning system based on natural language inference. Additionally, increasing parameter count from 220M to 770M improves AUC-ROC scores by 6.3%, and we forecast that even larger models would perform better. Therefore, measuring zero-shot learning performance on language models out-of-thebox might underestimate their true potential, and community-wide efforts on aggregating datasets and unifying their formats can help build models that answer prompts better.
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
**📜 Cross-Task Generalization via Natural Language Crowdsourcing Instructions**. <br> ✍ Swaroop Mishra, Daniel Khashabi, Chitta Baral, Hannaneh Hajishirzi
 *(arxiv 2021)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2104.08773){: .btn .btn-blue .mr-1 target="_blank" } 
   [Website](https://instructions.apps.allenai.org/){: target="_blank" .btn .btn-orange .mr-1 }
   [Semantic Scholar](https://www.semanticscholar.org/paper/Cross-Task-Generalization-via-Natural-Language-Mishra-Khashabi/0ad46cb2dc0ecec1bd4511dcfd3be5e0b0748ef1){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

<details markdown="block">
  <summary>Abstract</summary>
  {: .fs-3 .text-delta .text-blue-100}
  ```
  Humans (e.g., crowdworkers) have a remarkable ability in solving different tasks, by simply reading textual instructions that define them and looking at a few examples. NLP models built with the conventional paradigm, however, often struggle with generalization across tasks (e.g., a question-answering system cannot solve classification tasks). A long-standing challenge in AI is to build a model that learns a new task by understanding the human-readable instructions that define it. To study this, we introduce NATURAL INSTRUCTIONS, a dataset of 61 distinct tasks, their human-authored instructions and 193k task instances. The instructions are obtained from crowdsourcing instructions used to create existing NLP datasets and mapped to a unified schema. We adopt generative pre-trained language models to encode task-specific instructions along with input and generate task output. Our results indicate that models benefit from instructions when evaluated in terms of generalization to unseen tasks. These models, however, are far behind supervised task-specific models, indicating significant room for more progress in this direction.
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
**📜 Finetuned Language Models Are Zero-Shot Learners(FLAN)**. <br> ✍ Jason Wei, Maarten Bosma, Vincent Y. Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, Quoc V. Le
 *(arxiv 2021)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2109.01652){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://github.com/google-research/flan){: target="_blank" .btn .btn-green .mr-1 }
   [Semantic Scholar](https://www.semanticscholar.org/paper/Language-Models-are-Few-Shot-Learners-Brown-Mann/6b85b63579a916f705a8e10a49bd8d849d91b1fc){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

<details markdown="block">
  <summary>Abstract</summary>
  {: .fs-3 .text-delta .text-blue-100}
```
Recent work has demonstrated substantial gains on many NLP tasks and benchmarks by pre-training on a large corpus of text followed by fine-tuning on a specific task. While typically task-agnostic in architecture, this method still requires task-specific fine-tuning datasets of thousands or tens of thousands of examples. By contrast, humans can generally perform a new language task from only a few examples or from simple instructions - something which current NLP systems still largely struggle to do. Here we show that scaling up language models greatly improves task-agnostic, few-shot performance, sometimes even reaching competitiveness with prior state-of-the-art fine-tuning approaches. Specifically, we train GPT-3, an autoregressive language model with 175 billion parameters, 10x more than any previous non-sparse language model, and test its performance in the few-shot setting. For all tasks, GPT-3 is applied without any gradient updates or fine-tuning, with tasks and few-shot demonstrations specified purely via text interaction with the model. GPT-3 achieves strong performance on many NLP datasets, including translation, question-answering, and cloze tasks, as well as several tasks that require on-the-fly reasoning or domain adaptation, such as unscrambling words, using a novel word in a sentence, or performing 3-digit arithmetic. At the same time, we also identify some datasets where GPT-3's few-shot learning still struggles, as well as some datasets where GPT-3 faces methodological issues related to training on large web corpora. Finally, we find that GPT-3 can generate samples of news articles which human evaluators have difficulty distinguishing from articles written by humans. We discuss broader societal impacts of this finding and of GPT-3 in general.
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
**📜 Single-dataset Experts for Multi-dataset Question Answering**. <br> ✍ Dan Friedman, Ben Dodge, Danqi Chen
 *(EMNLP 2021)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2109.13880){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://github.com/princeton-nlp/MADE){: target="_blank" .btn .btn-green .mr-1 }
   [Semantic Scholar](https://www.semanticscholar.org/paper/Single-dataset-Experts-for-Multi-dataset-Question-Friedman-Dodge/67dc4ba4542d5895862c8b5af5023f659c14542c){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

<details markdown="block">
  <summary>Abstract</summary>
  {: .fs-3 .text-delta .text-blue-100}
```
  Many datasets have been created for training reading comprehension models, and a natural question is whether we can combine them to build models that (1) perform better on all of the training datasets and (2) generalize and transfer better to new datasets. Prior work has addressed this goal by training one network simultaneously on multiple datasets, which works well on average but is prone to over- or under-fitting different sub-distributions and might transfer worse compared to source models with more overlap with the target dataset. Our approach is to model multi-dataset question answering with a collection of single-dataset experts, by training a collection of lightweight, dataset-specific adapter modules (Houlsby et al., 2019) that share an underlying Transformer model. We find that these Multi-Adapter Dataset Experts (MADE) outperform all our baselines in terms of in-distribution accuracy, and simple methods based on parameter-averaging lead to better zero-shot generalization and few-shot transfer performance, offering a strong and versatile starting point for building new reading comprehension systems.
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
**📜 Meta-learning via Language Model In-context Tuning**. <br> ✍ Tu Vu, Tong Wang, Tsendsuren Munkhdalai, Alessandro Sordoni, Adam Trischler, Andrew Mattarella-Micke, Subhransu Maji, Mohit Iyyer
 *(arxiv 2021)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2110.07814){: .btn .btn-blue .mr-1 target="_blank" }
</span> 

<details markdown="block">
  <summary>Abstract</summary>
  {: .fs-3 .text-delta .text-blue-100}
  ```
  The goal of meta-learning is to learn to adapt to a new task with only a few labeled examples. To tackle this problem in NLP, we propose in-context tuning, which recasts adaptation and prediction as a simple sequence prediction problem: to form the input sequence, we concatenate the task instruction, the labeled examples, and the target input to predict; to meta-train the model to learn from in-context examples, we fine-tune a pre-trained language model (LM) to predict the target label from the input sequences on a collection of tasks.
We benchmark our method on two collections of text classification tasks: LAMA and BinaryClfs. Compared to first-order MAML which adapts the model with gradient descent, our method better leverages the inductive bias of LMs to perform pattern matching, and outperforms MAML by an absolute 6% AUC ROC score on BinaryClfs, with increasing advantage w.r.t. model size. Compared to non-fine-tuned in-context learning (i.e. prompting a raw LM), in-context tuning directly learns to learn from in-context examples. On BinaryClfs, in-context tuning improves the average AUC-ROC score by an absolute 10%, and reduces the variance with respect to example ordering by 6x and example choices by 2x.
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
**📜 Multitask Prompted Training Enables Zero-Shot Task Generalization(T0)**. <br> ✍ Victor Sanh, Albert Webson, Colin Raffel, Stephen H. Bach, Lintang Sutawika, Zaid Alyafeai, Antoine Chaffin, Arnaud Stiegler, Teven Le Scao, Arun Raja, Manan Dey, M Saiful Bari, Canwen Xu, Urmish Thakker, Shanya Sharma Sharma, Eliza Szczechla, Taewoon Kim, Gunjan Chhablani, Nihal Nayak, Debajyoti Datta, Jonathan Chang, Mike Tian-Jian Jiang, Han Wang, Matteo Manica, Sheng Shen, Zheng Xin Yong, Harshit Pandey, Rachel Bawden, Thomas Wang, Trishala Neeraj, Jos Rozen, Abheesht Sharma, Andrea Santilli, Thibault Fevry, Jason Alan Fries, Ryan Teehan, Stella Biderman, Leo Gao, Tali Bers, Thomas Wolf, Alexander M. Rush
 *(arxiv 2021)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2110.08207){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://github.com/bigscience-workshop/promptsource/){: target="_blank" .btn .btn-green .mr-1 }
   [Weight](https://huggingface.co/bigscience){: target="_blank" .btn .btn-red .mr-1 }
</span> 

<details markdown="block">
  <summary>Abstract</summary>
  {: .fs-3 .text-delta .text-blue-100}
  ```
  Large language models have recently been shown to attain reasonable zero-shot generalization on a diverse set of tasks (Brown et al., 2020). It has been hypothesized that this is a consequence of implicit multitask learning in language models' pretraining (Radford et al., 2019). Can zero-shot generalization instead be directly induced by explicit multitask learning? To test this question at scale, we develop a system for easily mapping any natural language tasks into a human-readable, prompted form. We convert a large set of supervised datasets, each with multiple prompts with diverse wording. These prompted datasets allow for benchmarking the ability of a model to perform completely unseen tasks. We finetune a pretrained encoder-decoder model (Raffel et al., 2020; Lester et al., 2021) on this multitask mixture covering a wide variety of tasks. The model attains strong zero-shot performance on several standard datasets, often outperforming models up to 16x its size. Further, our approach attains strong performance on a subset of tasks from the BIG-bench benchmark, outperforming models up to 6x its size. All prompts and trained models are available at this https URL bigscience-workshop/promptsource/ and this https URL.
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
**📜 Balancing Average and Worst-case Accuracy in Multitask Learning**. <br> ✍ Paul Michel, Sebastian Ruder, Dani Yogatama
 *(arxiv 2021)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2110.05838){: .btn .btn-blue .mr-1 target="_blank" } 
   [Semantic Scholar](https://www.semanticscholar.org/paper/Balancing-Average-and-Worst-case-Accuracy-in-Michel-Ruder/9f0fe9197f080d042ad406b149ac03a1063c0351){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

<details markdown="block">
  <summary>Abstract</summary>
  {: .fs-3 .text-delta .text-blue-100}
  ```
  When training and evaluating machine learning models on a large number of tasks, it is important to not only look at average task accuracy—which may be biased by easy or redundant tasks—but also worst-case accuracy (i.e. the performance on the task with the lowest accuracy). In this work, we show how to use techniques from the distributionally robust optimization (DRO) literature to improve worst-case performance in multitask learning. We highlight several failure cases of DRO when applied off-the-shelf and present an improved method, Lookahead-DRO (L-DRO), which mitigates these issues. The core idea of L-DRO is to anticipate the interaction between tasks during training in order to choose a dynamic re-weighting of the various task losses, which will (i) lead to minimal worst-case loss and (ii) train on as many tasks as possible. After demonstrating the efficacy of L-DRO on a small controlled synthetic setting, we evaluate it on two realistic benchmarks: a multitask version of the CIFAR-100 image classification dataset and a large-scale multilingual language modeling experiment. Our empirical results show that LDRO achieves a better trade-off between average and worst-case accuracy with little computational overhead compared to several strong baselines.
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
**📜 SPoT: Better Frozen Model Adaptation through Soft Prompt Transfer**. <br> ✍ Tu Vu, Brian Lester, Noah Constant, Rami Al-Rfou, Daniel Cer
 *(arxiv 2021)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2110.07904){: .btn .btn-blue .mr-1 target="_blank" } 
</span> 

<details markdown="block">
  <summary>Abstract</summary>
  {: .fs-3 .text-delta .text-blue-100}
  ```
  As pre-trained language models have gotten larger, there has been growing interest in parameter-efficient methods to apply these models to downstream tasks. Building on the PromptTuning approach of Lester et al. (2021), which learns task-specific soft prompts to condition a frozen language model to perform downstream tasks, we propose a novel prompt-based transfer learning approach called SPoT: Soft Prompt Transfer. SPoT first learns a prompt on one or more source tasks and then uses it to initialize the prompt for a target task. We show that SPoT significantly boosts the performance of PromptTuning across many tasks. More importantly, SPoT either matches or outperforms ModelTuning, which fine-tunes the entire model on each individual task, across all model sizes while being more parameter-efficient (up to 27,000x fewer task-specific parameters). We further conduct a large-scale study on task transferability with 26 NLP tasks and 160 combinations of source-target tasks, and demonstrate that tasks can often benefit each other via prompt transfer. Finally, we propose a simple yet efficient retrieval approach that interprets task prompts as task embeddings to identify the similarity between tasks and predict the most transferable source tasks for a given novel target task.
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
**📜 Towards a Unified View of Parameter-Efficient Transfer Learning**. <br> ✍ Junxian He, Chunting Zhou, Xuezhe Ma, Taylor Berg-Kirkpatrick, Graham Neubig
 *(arxiv 2021)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2110.05838){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://github.com/jxhe/unify-parameter-efficient-tuning){: target="_blank" .btn .btn-green .mr-1 }
   [Semantic Scholar](https://www.semanticscholar.org/paper/Towards-a-Unified-View-of-Parameter-Efficient-He-Zhou/3179bbfd6d86a8115388dce684a74159fba666a9){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

<details markdown="block">
  <summary>Abstract</summary>
  {: .fs-3 .text-delta .text-blue-100}
  ```
  Fine-tuning large pretrained language models on downstream tasks has become the de-facto learning paradigm in NLP. However, conventional approaches finetune all the parameters of the pretrained model, which becomes prohibitive as the model size and the number of tasks grow. Recent work has proposed a variety of parameter-efficient transfer learning methods that only fine-tune a small number of (extra) parameters to attain strong performance. While effective, the critical ingredients for success and the connections among the various methods are poorly understood. In this paper, we break down the design of state-of-the-art parameter-efficient transfer learning methods and present a unified framework that establishes connections between them. Specifically, we re-frame them as modifications to specific hidden states in pretrained models, and define a set of design dimensions along which different methods vary, such as the function to compute the modification and the position to apply the modification. Through comprehensive empirical studies across machine translation, text summarization, language understanding, and text classification benchmarks, we utilize the unified view to identify important design choices in previous methods. Furthermore, our unified framework enables the transfer of design elements across different approaches, and as a result we are able to instantiate new parameter-efficient fine-tuning methods that tune less parameters than previous methods while being more effective, achieving comparable results to fine-tuning all parameters on all four tasks.
  ``` 
</details> 
{: .fs-5 .fw-600 .text-blue-300}

<details markdown="block">
  <summary>Comments</summary>
  {: .fs-3 .text-delta .text-red-100}
</details> 
{: .fs-3 .fw-600 .text-red-300}

---

## Few Shot Learning


{: .fs-4 .fw-800 .text-blue-100}
**📜 Making Pre-trained Language Models Better Few-shot Learners(LM-BFF)**. <br> ✍ Tianyu Gao, Adam Fisch, Danqi Chen
 *(ACL 2021)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2012.15723){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://github.com/princeton-nlp/LM-BFF){: target="_blank" .btn .btn-green .mr-1 }
   [Semantic Scholar](https://www.semanticscholar.org/paper/Making-Pre-trained-Language-Models-Better-Few-shot-Gao-Fisch/db58fd13a7b4131a43230317eecb5840188d28a4){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

<details markdown="block">
  <summary>Abstract</summary>
  {: .fs-3 .text-delta .text-blue-100}
  ```
  The recent GPT-3 model (Brown et al., 2020) achieves remarkable few-shot performance solely by leveraging a natural-language prompt and a few task demonstrations as input context. Inspired by their findings, we study few-shot learning in a more practical scenario, where we use smaller language models for which fine-tuning is computationally efficient. We present LM-BFF--better few-shot fine-tuning of language models--a suite of simple and complementary techniques for fine-tuning language models on a small number of annotated examples. Our approach includes (1) prompt-based fine-tuning together with a novel pipeline for automating prompt generation; and (2) a refined strategy for dynamically and selectively incorporating demonstrations into each context. Finally, we present a systematic evaluation for analyzing few-shot performance on a range of NLP tasks, including classification and regression. Our experiments demonstrate that our methods combine to dramatically outperform standard fine-tuning procedures in this low resource setting, achieving up to 30% absolute improvement, and 11% on average across all tasks. Our approach makes minimal assumptions on task resources and domain expertise, and hence constitutes a strong task-agnostic method for few-shot learning.
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
**📜 CrossFit: A Few-shot Learning Challenge for Cross-task Generalization in NLP**. <br> ✍ Qinyuan Ye, Bill Yuchen Lin, Xiang Ren
 *(EMNLP 2021)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2104.08835){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://github.com/INK-USC/CrossFit){: target="_blank" .btn .btn-green .mr-1 }
   [Semantic Scholar](https://www.semanticscholar.org/paper/CrossFit%3A-A-Few-shot-Learning-Challenge-for-in-NLP-Ye-Lin/7fa273f450251523e6b7fcc2eb3fdbdfd4a30493){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

<details markdown="block">
  <summary>Abstract</summary>
  {: .fs-3 .text-delta .text-blue-100}
  ```
  Humans can learn a new language task efficiently with only few examples, by leveraging their knowledge obtained when learning prior tasks. In this paper, we explore whether and how such cross-task generalization ability can be acquired, and further applied to build better few-shot learners across diverse NLP tasks. We introduce CrossFit, a problem setup for studying cross-task generalization ability, which standardizes seen/unseen task partitions, data access during different learning stages, and the evaluation protocols. To instantiate different seen/unseen task partitions in CrossFit and facilitate in-depth analysis, we present the NLP Few-shot Gym, a repository of 160 diverse few-shot NLP tasks created from open-access NLP datasets and converted to a unified text-to-text format. Our analysis reveals that the few-shot learning ability on unseen tasks can be improved via an upstream learning stage using a set of seen tasks. We also observe that the selection of upstream learning tasks can significantly influence few-shot performance on unseen tasks, asking further analysis on task similarity and transferability.
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
**📜 True Few-Shot Learning with Language Models**. <br> ✍ Ethan Perez, Douwe Kiela, Kyunghyun Cho
 *(arxiv 2021)*

<span class="fs-2">
   [Paper](https://arxiv.org/abs/2105.11447){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://github.com/ethanjperez/true_few_shot){: target="_blank" .btn .btn-green .mr-1 }
   [Semantic Scholar](https://www.semanticscholar.org/paper/True-Few-Shot-Learning-with-Language-Models-Perez-Kiela/b58d8579ece27a60432e667bfbdb750590fa65d9){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

<details markdown="block">
  <summary>Abstract</summary>
  {: .fs-3 .text-delta .text-blue-100}
  ```
  Pretrained language models (LMs) perform well on many tasks even when learning from a few examples, but prior work uses many held-out examples to tune various aspects of learning, such as hyperparameters, training objectives, and natural language templates ("prompts"). Here, we evaluate the few-shot ability of LMs when such held-out examples are unavailable, a setting we call true few-shot learning. We test two model selection criteria, cross-validation and minimum description length, for choosing LM prompts and hyperparameters in the true few-shot setting. On average, both marginally outperform random selection and greatly underperform selection based on held-out examples. Moreover, selection criteria often prefer models that perform significantly worse than randomly-selected ones. We find similar results even when taking into account our uncertainty in a model's true performance during selection, as well as when varying the amount of computation and number of examples used for selection. Overall, our findings suggest that prior work significantly overestimated the true few-shot ability of LMs given the difficulty of few-shot model selection.

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
**📜 DReCa: A General Task Augmentation Strategy for Few-Shot Natural Language Inference**. <br> ✍ Shikhar Murty, Tatsunori B. Hashimoto, Christopher Manning
 *(NAACL 2021)*

<span class="fs-2">
   [Paper](https://aclanthology.org/2021.naacl-main.88/){: .btn .btn-blue .mr-1 target="_blank" } 
   [Code](https://github.com/MurtyShikhar/DReCA){: target="_blank" .btn .btn-green .mr-1 }
   [Semantic Scholar](https://www.semanticscholar.org/paper/DReCa%3A-A-General-Task-Augmentation-Strategy-for-Murty-Hashimoto/186c304c782c70b0b5d571ea140a8fb4ebcf31a4){: .btn .btn-purple .mr-1 target="_blank" }
</span> 

<details markdown="block">
  <summary>Abstract</summary>
  {: .fs-3 .text-delta .text-blue-100}
  ```
  Meta-learning promises few-shot learners that can adapt to new distributions by repurposing knowledge acquired from previous training. However, we believe meta-learning has not yet succeeded in NLP due to the lack of a well-defined task distribution, leading to attempts that treat datasets as tasks. Such an ad hoc task distribution causes problems of quantity and quality. Since there’s only a handful of datasets for any NLP problem, meta-learners tend to overfit their adaptation mechanism and, since NLP datasets are highly heterogeneous, many learning episodes have poor transfer between their support and query sets, which discourages the meta-learner from adapting. To alleviate these issues, we propose DReCA (Decomposing datasets into Reasoning Categories), a simple method for discovering and using latent reasoning categories in a dataset, to form additional high quality tasks. DReCA works by splitting examples into label groups, embedding them with a finetuned BERT model and then clustering each group into reasoning categories. Across four few-shot NLI problems, we demonstrate that using DReCA improves the accuracy of meta-learners by 1.5-4%

  ``` 
</details> 
{: .fs-5 .fw-600 .text-blue-300}

<details markdown="block">
  <summary>Comments</summary>
  {: .fs-3 .text-delta .text-red-100}
</details> 
{: .fs-3 .fw-600 .text-red-300}

---

## Task Unification


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

## Analysis
## Others