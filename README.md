# nlp-foundations-to-transformers

NLP progression from text preprocessing and classical language models
through word vectors, RAG, and transformer-based summarization.

## Contents

### Text Preprocessing and Zipf's Law
Built a tokenizer from scratch with custom punctuation separation and contraction
expansion covering 30+ contraction rules. Applied to the full War and Peace corpus,
generated ranked frequency counts, and verified Zipf's Law with a log-frequency
vs rank chart.

Files: text_preprocessing_zipf.ipynb, sample_2025.txt, output1.txt,
output2.txt, war-and-peace.txt, zipf_chart.png

### Word Vectors and SVD
Constructed co-occurrence matrices from a corpus and applied SVD for dimensionality
reduction. Analyzed word similarity using cosine similarity, explored word analogies,
and compared SVD-based vectors against GloVe embeddings.

Files: word_vectors_svd.ipynb, env.yml

### Vector Embeddings and RAG
Built a multi-document RAG pipeline over 4 AI policy documents (Blueprint for an
AI Bill of Rights, EU AI Act, Executive Order on AI, NIST AI RMF). Compared three
sentence transformers - MiniLM, MPNet, and multi-qa-MiniLM - on retrieval quality
and answer accuracy using manual scoring across 5 query types.

Files: vector_embeddings_rag.ipynb

### N-gram Language Models
Implemented unigram, bigram, and trigram language models from scratch on the 1 Billion
Word Benchmark. Evaluated perplexity with and without Laplace smoothing and generated
text using greedy choice, random sampling, and top-p nucleus sampling.

Files: ngram_language_models.ipynb, 1b_benchmark.train.tokens.txt,
1b_benchmark.test.tokens.txt

### Transformer Text Summarization
Three-part work on abstractive summarization. Ran inference with T5-small and a
fine-tuned T5 model, comparing ROUGE, BERTScore, and Perplexity across both.
Fine-tuned T5-small on the Aeon Essays dataset using Seq2SeqTrainer and compared
results before and after fine-tuning. Designed and tested hand-crafted prompts on
a SOTA LLM using zero-shot and few-shot strategies.

Files: summarization_t5_inference.ipynb, summarization_t5_finetuning.ipynb,
summarization_llm_prompting.ipynb

Note: Large datasets (CNN/DailyMail 5%, Aeon Essays) not included due to size.

## Libraries

Python, PyTorch, HuggingFace Transformers, LangChain, ChromaDB,
sentence-transformers, gensim, nltk, evaluate, numpy, matplotlib

## Goal

To build NLP intuition from scratch - from tokenization and n-gram models
through word embeddings and RAG to transformer fine-tuning and prompting.
