# nlp-foundations-to-transformers

NLP progression from text preprocessing and classical language models
through word vectors, RAG, and transformer-based summarization.

## Contents

### HW1 - Text Preprocessing and Zipf's Law
Custom tokenizer built from scratch: punctuation separation, contraction
expansion, and frequency analysis. Applied to War and Peace corpus with
Zipf's Law visualization.
Files: SreejaReddyYeluruHW1.ipynb, sample_2025.txt, output1.txt,
output2.txt, war-and-peace.txt, zipf_chart.png

### HW2 - Word Vectors and SVD
Co-occurrence matrix construction and dimensionality reduction via SVD.
Word similarity analysis with cosine similarity and GloVe embeddings.
Based on Stanford CS224n Assignment 1.
Files: hw2_word_vectors.ipynb, env.yml

### HW3 - Vector Embeddings and RAG
Multi-document RAG pipeline over 4 AI policy documents (Blueprint for an
AI Bill of Rights, EU AI Act, Executive Order on AI, NIST AI RMF).
Compared three sentence transformers (MiniLM, MPNet, multi-qa-MiniLM)
on retrieval quality and answer accuracy.
Files: Sreeja_reddy_yeluru_hw3.ipynb

### HW4 - N-gram Language Models
Unigram, bigram, and trigram models trained on the 1 Billion Word Benchmark.
Perplexity evaluation with and without Laplace smoothing. Text generation
using greedy, random sampling, and top-p nucleus sampling.
Files: sreeja_reddy_yeluru_hw4.ipynb, 1b_benchmark.train.tokens.txt,
1b_benchmark.test.tokens.txt

### HW5 - Transformer Text Summarization
Three-part assignment using HuggingFace Seq2Seq models on CNN/DailyMail
and Aeon Essays datasets.
Part 1: Inference comparison of T5-small vs fine-tuned T5 (ROUGE, BERTScore, Perplexity).
Part 2: Fine-tuning T5-small on Aeon Essays dataset with Seq2SeqTrainer.
Part 3: Prompting SOTA LLMs for summarization with prompt engineering.
Files: Sreeja_reddy_yeluru_hw5_part1.ipynb,
Sreeja_reddy_yeluru_hw5_part2.ipynb,
Sreeja_reddy_yeluru_hw5_part3.ipynb

Note: Large datasets (CNN/DailyMail 5%, Aeon Essays) not included due to size.

## Libraries

Python, PyTorch, HuggingFace Transformers, LangChain, ChromaDB,
sentence-transformers, gensim, nltk, numpy, matplotlib, evaluate

## Goal

To build NLP intuition from scratch - from tokenization and n-gram models
through word embeddings and RAG to transformer fine-tuning and prompting.
