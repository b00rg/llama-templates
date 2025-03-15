# Llama Classification and Generation
This repository provides implementations of Llama models for a variety of natural language processing (NLP) tasks, including:

Binary classification: Both zero-shot and fine-tuned classification tasks.
Text generation: Leveraging Llama's local inference capabilities to generate text based on prompts.
Model training with LoRA: Fine-tune Llama models efficiently using Low-Rank Adaptation (LoRA), which allows for model adaptation without retraining the entire model.

### Key Features:
Pre-built scripts for training Llama models on classification tasks and text generation.
Fine-tuning capability for classification tasks using LoRA, enabling faster and more efficient adaptation.
Support for local inference with Llama, making it easy to run the model on your own machine without relying on APIs or cloud-based services.

### Setup & Requirements:
Kaggle GPUs (preferably T4) are recommended for model training due to hardware requirements.
Ensure you have access to the Llama models on Hugging Face to download them locally.

### How It Works:
The Llama model is downloaded and run locally on your machine.
Training and inference scripts are provided for both classification and text generation tasks.
LoRA is used for efficient model fine-tuning, allowing for the adaptation of the model without requiring full retraining.
