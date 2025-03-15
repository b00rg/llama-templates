import os
import torch
import transformers
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
import json
import evaluate
import time

# --- JSONL File Checking ---
def check_jsonl(file_path, field="body"):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        if field not in data:
                            print(f"Error on line {line_number} in {file_path}: Missing '{field}' field")
                            return False
                    except json.JSONDecodeError as e:
                        print(f"Error on line {line_number} in {file_path}: {e}")
                        print(f"Problematic line: {line}")
                        return False
            return True
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return False

# --- Model Loading and Setup ---
def load_llama_model(model_name):
    print(f"Loading Llama model: {model_name}...")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name,
            token=os.environ.get("HUGGING_FACE_TOKEN"),
            trust_remote_code=True,
        )

        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  
            token=os.environ.get("HUGGING_FACE_TOKEN"),
            device_map="auto",
            trust_remote_code=True,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("Added pad token to tokenizer.")

        print("Model loaded successfully.")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

# --- Data Loading and Preprocessing from local JSONL files ---
def load_and_preprocess_jsonl_data(good_file_path, bad_file_path, tokenizer, max_length=512, text_field="body"):
    print("Loading and preprocessing data...")
    try:
        combined_dataset = load_dataset("json", data_files=[good_file_path, bad_file_path])

        def extract_and_filter(example):
            text = example[text_field]
            is_good = len(text.split()) < max_length * 0.8
            return {"text": text, "is_good": is_good}

        combined_dataset = combined_dataset.map(extract_and_filter, remove_columns=combined_dataset["train"].column_names)
        combined_dataset = combined_dataset.filter(lambda x: x["is_good"] or (not x["is_good"] and len(x["text"].split()) < max_length * 0.8))

        print("Tokenizing data...")

        def tokenize_function(examples):
            inputs = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
            inputs["labels"] = inputs["input_ids"].clone()
            return inputs

        tokenized_dataset = combined_dataset.map(tokenize_function, batched=True, desc="Tokenizing", remove_columns=combined_dataset["train"].column_names)
        tokenized_dataset.set_format("torch")  
        print("Data loaded and preprocessed successfully.")
        return tokenized_dataset
    except Exception as e:
        print(f"Error loading/preprocessing JSONL data: {e}")
        return None

# --- Training Setup ---
def train_llama(model, tokenizer, tokenized_dataset, output_dir="llama_model_trained", batch_size=2, epochs=3, learning_rate=5e-5, eval_batch_size=8):
    print("Starting training...")

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        bf16=True,  # Use BF16 for Apple Silicon - change as required for your hardware
        fp16=False,
        warmup_steps=100,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        per_device_eval_batch_size=eval_batch_size,
        eval_accumulation_steps=4
    )
    small_dataset = tokenized_dataset['train'].select(range(1)) # This is a smaller test dataset for testing locally - load larger dataset for training 
    small_eval_dataset = tokenized_dataset["train"].select(range(min(1, len(tokenized_dataset["train"]))))
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_dataset, # IMPORTANT: this is just a SAMPLE dataset to test the code runs - consider running in batches so computer has memory
        eval_dataset=small_eval_dataset,  # IMPORTANT: Replace with your eval dataset!!! currently using a smaller dataset to test the code works - consider running batches
        tokenizer=tokenizer,
    )

    start_time = time.time()

    trainer.train()

    end_time = time.time()
    training_time = end_time - start_time

    trainer.save_model(output_dir)
    print("Training complete!")

    # Evaluation
    print("Evaluating the model...")
    eval_start_time = time.time()
    results = trainer.evaluate()
    eval_end_time = time.time()
    eval_time = eval_end_time - eval_start_time
    print(results)

    # Print training statistics
    print("\n--- Training Statistics ---")
    print(f"Total training time: {training_time:.2f} seconds")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
    print(f"Effective batch size: {batch_size * training_args.gradient_accumulation_steps}")
    print(f"Learning rate: {learning_rate}")

    # Print evaluation statistics
    print("\n--- Evaluation Statistics ---")
    print(f"Evaluation time: {eval_time:.2f} seconds")
    print(f"Loss: {results['eval_loss']}")  # Example: Print the loss. Add other metrics as needed
    # ... print other metrics as needed

    return trainer


# --- Main Execution ---
if __name__ == "__main__":
    hf_token = os.environ.get("HUGGING_FACE_TOKEN")
    if not hf_token:
        raise ValueError("HUGGING_FACE_TOKEN environment variable not set.")

    model_name = "meta-llama/Llama-3.2-1B"  # Or another Llama model
    model, tokenizer = load_llama_model(model_name)

    if model and tokenizer:
        good_file_path = "reliable_news.jsonl"
        bad_file_path = "unreliable_news.jsonl"

        if check_jsonl(good_file_path) and check_jsonl(bad_file_path):
            print("JSONL files appear to be valid (based on json.loads).")
            text_field = "body" 
            tokenized_dataset = load_and_preprocess_jsonl_data(good_file_path, bad_file_path, tokenizer, text_field=text_field)

            if tokenized_dataset:
                trainer = train_llama(model, tokenizer, tokenized_dataset)

                if trainer:
                    print("Training and evaluation finished successfully.")
            else:
                print("Failed to load or preprocess JSONL data.")
        else:
            print("Errors found in JSONL files. Please correct them before proceeding.")
    else:
        print("Failed to load model and tokenizer.")
