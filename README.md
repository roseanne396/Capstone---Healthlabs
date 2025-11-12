# MSADS-Capstone---Healthlabs

## Project Goal

The primary goal of this capstone project is to **detect and evaluate synergistic partnership opportunities among digital health companies and products**.

## The Challenge

Identifying potential synergies is a complex and high-value task. Traditionally, this process is:
* **Manual:** Relies entirely on human experts.
* **Slow:** Requires extensive research and analysis.
* **Subjective:** Dependent on the specific domain knowledge of the individuals involved.

This project aims to automate and enhance this process, providing a scalable and data-driven solution.

## Our Approach: Automated Synergy Detection

We use a **Retrieval-Augmented Generation (RAG)** system to analyze internal company and product data. This system leverages Large Language Models (LLMs) to identify partnership candidates that a human expert might miss.

By processing our data with a **HyDE (Hypothetical Document Embeddings)** method, the system becomes highly effective at "imagining" what an ideal answer looks like, which allows it to retrieve the most relevant information for generating partnership insights.

The end result is a system that generates **potential synergy pair candidates** and provides **clear reasoning** for each recommendation, dramatically accelerating the strategic partnership workflow.

### High-Level Pipelines

The project is structured into two main pipelines:

* **Pipeline 1: Knowledge Base Construction**
    This pipeline processes and ingests our two primary internal datasets (representing products and companies) into separate, specialized Chroma vector databases. This creates the "memory" or knowledge base that the RAG system will use.

* **Pipeline 2: RAG Synergy Querying**
    This pipeline takes a user query (e.g., "Find partners for Company X") and uses the RAG system to query both knowledge bases. It then synthesizes the retrieved information to generate a final answer, complete with a list of candidates and the reasoning behind their synergistic potential.
