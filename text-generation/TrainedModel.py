import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# === CONFIGURATION ===
base_model_name = "meta-llama/Llama-3.2-1B"  # Change to your fine-tuned base model
adapter_path = ""  # Replace with your saved LoRA adapter path
device = "cuda" if torch.cuda.is_available() else "cpu"
hf_token = ""  # Replace with your Hugging Face token

# === LOAD MODEL & TOKENIZER ===
def load_lora_model(base_model_name, adapter_path, hf_token):
    print("Loading base model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_auth_token=hf_token)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_model_name, use_auth_token=hf_token).to(device)
    
    print("Loading LoRA adapters...")
    try:
        model = PeftModel.from_pretrained(model, adapter_path, use_auth_token=hf_token)
    except ValueError as e:
        print(f"Error loading LoRA adapter: {e}")
        raise
    
    print("Model with LoRA adapters loaded successfully!")
    return model, tokenizer

model, tokenizer = load_lora_model(base_model_name, adapter_path, hf_token)

# === TEXT GENERATION FUNCTION ===
def generate_text(prompt, max_length=200):
    """Generates text based on the given prompt."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            do_sample=True,  # Enable sampling for more natural text
            temperature=0.7,  # Adjust for creativity
            top_p=0.9  # Nucleus sampling for diversity
        )
    
    return tokenizer.decode(output[0], skip_special_tokens=True)

# === TEST TEXT GENERATION ===
if __name__ == "__main__":
    prompt = ""  # Replace with your own test prompt
    
    generated_text = generate_text(prompt)

    print("\nGenerated Text:")
    print(generated_text)
