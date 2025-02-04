---
layout: default
title: Introduction
nav_order: 2
has_children: False
permalink: /introduction/

toc_list: true
last_modified_date: Jan 23 2022
---



# Introduction to Structured Knowledge Grounding
{: .no_toc }
In progress
{: .label .label-yellow }

<style>
td{
    font-size: 15pt;
}
</style>

## Table of contents
{: .no_toc .text-delta .fs-4 style="font-weight:800"}

- TOC
{:toc}

---
## What is structured knowledge?
Structured knowledge(e.g., web tables, knowledge graphs, and databases) stores large amounts of data in an organized structure and forms a basis for a wide range of applications, e.g., medical records, personal assistants, and customer relations management. Accessing and searching data in structured knowledge typically requires query languages or professional training. 

## What is structured knowledge grounding?
To promote the efficiency of data access, structured knowledge grounding(SKG) grounds user requests in structured knowledge and produces various outputs including computer programs(e.g., SQL and SPARQL),  cell values, and natural language responses(as shown below).

<img src="../../assets/images/unifiedskg.png" width="90%" height="auto"/>

For example, semantic parsing converts natural language questions into formal programs; question answering derives answers from tables or knowledge graphs.