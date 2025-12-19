# Plain Language Text Simplification with BART
**Applied Machine Learning Final Project**

*Aayush Parekh, Jiwon Bae, Simryn Parikh, Udita Bajaj*

## Project Overview
This repository contains the codebase and experimental results for a sequence-to-sequence (Seq2Seq) text simplification system. The project leverages BART (Bidirectional and Auto-Regressive Transformers) to translate complex English text into "Plain Language" to improve accessibility for individuals with learning disabilities or limited English proficiency.

The approach bridges the gap between traditional supervised Seq2Seq simplification and modern parameter-efficient fine-tuning techniques. We investigate the impact of task-specific fine-tuning, hyperparameter optimization, and Low-Rank Adaptation (LoRA) on simplification performance.

## Links
- Hugging Face Repo
  https://huggingface.co/jiwonbae1124/bart-simplifier-readability
- Demo app
  https://barttranslatetoplainlanguage-mvlwjbrwcidkpxpczzpsx2.streamlit.app

## Datasets
1. ASSET (Abstractive Sentence Simplification Evaluation and Tuning)

- Source: A standard benchmark derived from Wikipedia and Simple English Wikipedia.

- Characteristics: Contains ten human-authored simplified variants per input sentence.

- Usage: Used for both training and evaluation. ASSET allows for diverse rewriting operations including paraphrasing and sentence splitting.

2. Synthetic Dataset (Augmented WikiLarge)

- Source: WikiLarge benchmark (Zhang and Lapata).

- Augmentation: We expanded the training corpus by generating synthetic complex-simple pairs using a lightweight language model.

- Constraint: Generation conditioned on random topics to ensure lexical diversity without summarization.

- Preprocessing: A 10% uniform subsample of the synthetic test set was used for evaluation to maintain balance with ASSET.

## Methodology
We implemented four distinct model configurations to isolate the effects of training strategies:

M0: Zero-Shot Baseline

- Model: Pre-trained BART-base (~139M parameters).

- Strategy: Direct inference with no task-specific training.

- Purpose: Establishes the intrinsic rewriting capability of the pre-trained weights.

M1: Supervised Fine-Tuning (Default)

- Strategy: Standard SFT on ASSET and Synthetic datasets.

- Parameters: Max sequence length of 64 tokens. Default Hugging Face Seq2SeqTrainer optimization settings.

- Purpose: Isolates the effect of data exposure without hyperparameter intervention.

M2: Supervised Fine-Tuning (Tuned)

- Strategy: Full parameter fine-tuning with targeted optimization.

- Hyperparameters:

-- Transformer dropout and attention dropout enabled.

-- Linear learning-rate schedule with warmup.

-- Weight decay and label smoothing to prevent overfitting.

M3: Parameter-Efficient Fine-Tuning (LoRA)

- Strategy: Low-Rank Adaptation applied to pre-trained BART.

- Configuration: Frozen base model weights; trainable adapters inserted into query, key, value, and output projection matrices.

- Purpose: Evaluates performance efficiency trade-offs (comparable results with fewer trainable parameters).

## Metrics
We utilized three primary metrics to evaluate performance:

- Flesch-Kincaid Grade Level (FKG): Measures readability based on sentence length and syllable count. Lower scores indicate better readability.

- BLEU: Measures n-gram overlap between system output and reference texts. Higher scores indicate higher similarity.

- SARI (System output Against References and Input): Specialized simplification metric comparing kept, added, and deleted words against references.
