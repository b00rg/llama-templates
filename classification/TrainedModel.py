import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

# === CONFIGURATION ===
base_model_name = "meta-llama/Llama-3.2-1B"  # change this to the model you finetuned your weights on
adapter_path = ""  # replace with your Path to saved LoRA adapter
device = "cuda" if torch.cuda.is_available() else "cpu"
hf_token = ""  # Replace with your Hugging Face token

# === LOAD MODEL & TOKENIZER ===
def load_lora_model(base_model_name, adapter_path, hf_token):
    print("Loading base model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_auth_token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(base_model_name, use_auth_token=hf_token).to(device)
    print("Loading LoRA adapters...")
    try:
        model = PeftModel.from_pretrained(model, adapter_path, use_auth_token=hf_token)
    except ValueError as e:
        print(f"Error loading LoRA adapter: {e}")
        raise
    print("Model with LoRA adapters loaded successfully!")
    return model, tokenizer

model, tokenizer = load_lora_model(base_model_name, adapter_path, hf_token)

# === CLASSIFICATION FUNCTION ===
def classify_article(text):
    """Classifies an article as Reliable or Unreliable."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_label = torch.argmax(probabilities, dim=-1).item()
    label_mapping = {0: "label0", 1: "label1"} # change labels as specified
    return label_mapping[predicted_label], probabilities.tolist()

# === TEST CLASSIFICATION ===
if __name__ == "__main__":
    text = "" # insert a sample classification text here
    
    label, probs = classify_article(text)

    print(f"Predicted Label: {label}")
    print(f"Probabilities: {probs}")
