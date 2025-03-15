import os
import torch
import pandas as pd
import tempfile
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model
from huggingface_hub import login

# **Authenticate with Hugging Face**
huggingface_token = ""   # Replace with your actual token
print("Logging into Hugging Face...")
login(token=huggingface_token)
print("Successfully logged in!")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

if device == "cuda":
    print(f"GPU Memory Before Model Load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# **Load Model**
model_name = "meta-llama/Llama-3.2-1B"  # Change if using another model
print(f"Loading model: {model_name} ...")

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ignore_mismatched_sizes=True,
).to(device)

model.config.use_cache = False
print("Model loaded successfully!")

if device == "cuda":
    print(f"GPU Memory After Model Load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Tokenizer loaded successfully!")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"
model.config.pad_token_id = tokenizer.pad_token_id

print(f"Padding token set: {tokenizer.pad_token}")

class CustomCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            print(
                f"Step: {state.global_step} | Loss: {logs.get('loss', 'N/A')} | Learning Rate: {logs.get('learning_rate', 'N/A')}"
            )
            print_gpu_utilization()
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None:
            eval_loss = metrics.get("eval_loss", "N/A")
            eval_accuracy = metrics.get("eval_accuracy", "N/A")
            print(f"Validation Loss: {eval_loss:.4f} | Validation Accuracy: {eval_accuracy:.4f}")

def print_gpu_utilization():
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / 1e9
        print(f"GPU Memory Usage: {gpu_mem:.2f} GB")

def fine_tune(model, tokenizer, dataset_train, dataset_test):
    print("Creating PEFT configuration...")
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_CLS",
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, peft_config)
    print("PEFT model created!")

    training_args = TrainingArguments( # note that these training args should be changed to see which works on the data for highest accuracy
        output_dir="./results",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        max_steps=500, 
        logging_dir="./logs",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=25,
        save_strategy="steps",
        save_steps=50,
        push_to_hub=False,
        learning_rate=1e-4,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        auto_find_batch_size=False,
        report_to="tensorboard",
        dataloader_num_workers=4,
        weight_decay=0.02,
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        train_dataset=dataset_train,
        eval_dataset=dataset_test,
        args=training_args,
        compute_metrics=lambda p: {"eval_accuracy": (p.predictions.argmax(-1) == p.label_ids).mean()},
        callbacks=[CustomCallback()],
    )

    print("Starting training...")
    trainer.train()
    print("Training complete!")
    return model

label1_csv = "" # put file path here
label0_csv = ""

def load_and_preprocess_data(label1_csv, label0_csv, tokenizer, max_length=512):
    df_label1 = pd.read_csv(label1_csv).assign(label=1)
    df_label0 = pd.read_csv(label0_csv).assign(label=0)
    df = pd.concat([df_label1, df_label0])
    df_test = pd.concat([df_label1.sample(100, random_state=42), df_label0.sample(100, random_state=42)])

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmpfile:
        df.to_csv(tmpfile.name, index=False)
        dataset_train = load_dataset("csv", data_files=tmpfile.name, split="train")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmpfile:
        df_test.to_csv(tmpfile.name, index=False)
        dataset_test = load_dataset("csv", data_files=tmpfile.name, split="train")

    def tokenize_function(sample):
        return tokenizer(sample["body"], truncation=True, padding="max_length", max_length=max_length)

    dataset_train = dataset_train.map(tokenize_function, batched=True)
    dataset_train = dataset_train.rename_columns({"label": "labels"})
    dataset_train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    dataset_test = dataset_test.map(tokenize_function, batched=True)
    dataset_test = dataset_test.rename_columns({"label": "labels"})
    dataset_test.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    return dataset_train, dataset_test

dataset_train, dataset_test = load_and_preprocess_data(label1_csv, label0_csv, tokenizer)
print("Datasets ready!")

print("Starting fine-tuning process...")
model = fine_tune(model, tokenizer, dataset_train, dataset_test)
print("Fine-tuning finished!")
