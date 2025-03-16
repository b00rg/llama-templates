import os
import torch
import pandas as pd
import tempfile
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from huggingface_hub import login

# **Authenticate with Hugging Face**
huggingface_token = ""  # Replace with your actual token
print("Logging into Hugging Face...")
login(token=huggingface_token)
print("Successfully logged in!")

# **Check GPU Availability**
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# **Load Model**
model_name = "meta-llama/Llama-3.2-1B"  # change this model as required
print(f"Loading model: {model_name} ...")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
).to(device)

model.config.use_cache = False
print("Model loaded successfully!")

# **Load Tokenizer**
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  

tokenizer.padding_side = "right"
model.config.pad_token_id = tokenizer.pad_token_id

print(f"Padding token set: {tokenizer.pad_token}")

# **Fine-Tuning Function**
def fine_tune(model, tokenizer, dataset_train, dataset_test):
    print("Creating PEFT configuration...")
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM", 
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, peft_config)
    print("PEFT model created!")

    training_args = TrainingArguments( # change training parameters as required for optimal training
        output_dir="./results",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        max_steps=500, 
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="steps",
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

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  
    )

    trainer = Trainer(
        model=model,
        train_dataset=dataset_train,
        eval_dataset=dataset_test,
        args=training_args,
        data_collator=data_collator,
        callbacks=[]  
    )

    print("Starting training...")
    trainer.train()
    print("Training complete!")
    return model

# **Load and Preprocess Data**
dataset_csv = "" # change to filepath for training data

def load_and_preprocess_data(dataset_csv, tokenizer, max_length=512):
    df = pd.read_csv(dataset_csv)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmpfile:
        df.to_csv(tmpfile.name, index=False)
        dataset = load_dataset("csv", data_files=tmpfile.name, split="train")

    def tokenize_function(sample):
        return tokenizer(sample["body"], truncation=True, padding="max_length", max_length=max_length)

    dataset = dataset.map(tokenize_function, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    train_test_split = dataset.train_test_split(test_size=0.1)
    return train_test_split["train"], train_test_split["test"]

dataset_train, dataset_test = load_and_preprocess_data(dataset_csv, tokenizer)
print("Datasets ready!")

print("Starting fine-tuning process...")
model = fine_tune(model, tokenizer, dataset_train, dataset_test)
print("Fine-tuning finished!")
