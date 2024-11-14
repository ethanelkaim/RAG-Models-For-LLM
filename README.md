
## Overview
This project explores and enhances Retrieval-Augmented Generation (RAG) methods to improve the accuracy and efficiency of large language models (LLMs) on knowledge-intensive tasks. The project evaluates three RAG architectures:

- RAG-ANN Model - utilizes Approximate Nearest Neighbors (ANN) indexing for optimized retrieval.

- Hierarchical RAG Model - employs a two-tiered indexing system for layered, context-rich retrieval.

- Wikipedia-RAG Model - dynamically retrieves and leverages relevant Wikipedia content, using Named Entity Recognition (NER) for precise keyword extraction.

Each model is implemented with a focus on enhancing the indexing and retrieval stages, offering a unique approach to contextually relevant document retrieval and generation.

### Files:
- ANN_model.ipynb
- Hierarchical_and_Wikipedia_models.ipynb


## Setup Instructions

### Data Preparation
This project uses several datasets:

* RAG-ANN Model: New York Times News (2000-2007) dataset for general topic coverage.
* Hierarchical Model: CNN/DailyMail dataset for structured information across various topics.
* Wikipedia-RAG Model: Utilizes Wikipedia API to dynamically fetch relevant articles based on NER-extracted keywords from queries.
Ensure each dataset is accessible and preprocessed as per the instructions in each .ipynb file.

### Configuration
Create a .env file in the root directory to store any API keys or environment variables, if needed, especially for the Wikipedia API and Pinecone API.

```python
PINECONE_API_KEY=your_pinecone_api_key
COHERE_API_KEY=your_cohere_api_key
```

### How to Run the Code
Each model is implemented in a separate Jupyter Notebook. Here’s how to run each model:

1. ANN Model:

* Open ANN_model.ipynb.
* Follow the notebook’s instructions to load the dataset, initialize the FAISS index, and run the retrieval and generation steps.
* This model uses FAISS for ANN indexing, providing efficient document retrieval based on cosine similarity.

2. Hierarchical and Wikipedia Models:

* Open Hierarchical_and_Wikipedia_models.ipynb.
* Follow the notebook’s instructions, I have already written the required API keys, but if needed, you can easily obtain new ones online for free by writting the model name + API key.

### Evaluation
Each model includes an evaluation section that measures:

* Semantic Coherence: Measures the alignment between the query and generated response using cosine similarity.
* Reformulation Consistency: Tests the robustness of responses across different phrasings of the query.

### Authors
Ethan Elkaim 
Eva Karsenty
Raphael Lasry
