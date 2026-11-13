# Customer Query Understanding Using Embeddings

## Overview

This project explores how different embedding techniques represent and compare customer queries using numerical vectors.

The main objective is to understand word embeddings, sentence embeddings, contextual embeddings, and positional encoding through practical experiments.

## Problem Statement

Customers often ask similar questions using different words. Additionally, the same word can have different meanings depending on its context.

This project investigates how embedding techniques capture semantic relationships and represent contextual meaning.

## Dataset

- 300 synthetic customer queries
- 15 categories
- 20 queries per category
- No missing values or duplicate queries

The dataset covers banking, payments, loans, shopping, refunds, account management, and customer support.

## Techniques Implemented

### 1. Subword Tokenization

Used BERT's WordPiece tokenizer to convert customer queries into subword tokens and numerical token IDs.

### 2. Word Embeddings — Word2Vec

Trained a Skip-gram Word2Vec model on 300 queries.

- Vocabulary size: 439
- Embedding dimensions: 50
- Compared word vectors using cosine similarity

### 3. Sentence Embeddings

Used the pretrained `all-MiniLM-L6-v2` Sentence Transformer.

- Generated 300 sentence embeddings
- Each embedding contains 384 dimensions
- Retrieved the top five most similar customer queries

**Example:**

Query: "How can I open a bank account?"

Most similar query: "I want to create a savings account."

Cosine similarity: **0.6909**

### 4. Contextual Embeddings — BERT

Used pretrained `bert-base-uncased` to generate 768-dimensional contextual token embeddings.

| Comparison | Cosine Similarity |
|---|---:|
| Bank — financial vs. river | 0.4509 |
| Charge — battery vs. financial fee | 0.5207 |
| Interest — financial vs. personal | 0.4348 |
| Bank — similar financial contexts | 0.9911 |

These experiments demonstrate how the same word can receive different representations depending on its context.

### 5. Positional Encoding

Implemented sinusoidal positional encoding using PyTorch.

Compared:

- "The customer paid the merchant."
- "The merchant paid the customer."

The original token embeddings were identical for the same words, but their representations changed after adding positional information.

## Technologies Used

- Python
- Google Colab
- PyTorch
- Hugging Face Transformers
- Sentence Transformers
- Gensim
- Pandas
- Matplotlib

## Key Learnings

- Word2Vec generates static word embeddings.
- Sentence Transformers represent complete sentences.
- BERT generates contextual token representations.
- Cosine similarity measures vector alignment.
- Positional encoding incorporates token-order information.

## Limitations

- The dataset is synthetic and relatively small.
- Word2Vec was trained on limited text, affecting the reliability of its similarity results.
- High similarity does not always mean identical customer intent.
- The positional encoding experiment used randomly initialized token embeddings.

## Conclusion

This project provided a practical understanding of how embeddings represent text numerically and how different techniques capture word relationships, sentence meanings, and contextual information.

It also demonstrated the importance of positional information in Transformer architectures.

The experiments form a foundation for further learning in self-attention, multi-head attention, and Transformer architectures.
