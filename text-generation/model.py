import os
import torch
import transformers

def run_llama_locally(model_name, prompt, max_new_tokens=50, temperature=0.7): # change max_new_tokens to increase response length
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

        device = model.device

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            generation_output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id, 
            )

        generated_text = tokenizer.decode(generation_output[0], skip_special_tokens=True)
        return generated_text

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Environment variable check
hf_token = os.environ.get("HUGGING_FACE_TOKEN")
if not hf_token:
    raise ValueError("HUGGING_FACE_TOKEN environment variable not set.")

# Example Usage
model_name = "meta-llama/Llama-3.2-1B"  # This model can be changed as required. Larger models have slow speed while running locally.
user_prompt = "" # insert your prompt here 

generated_response = run_llama_locally(model_name, user_prompt)

if generated_response:
    print(f"Generated Text: \n{generated_response}")
