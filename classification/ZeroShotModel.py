import os
import torch
import transformers

def classify_credibility(model_name, prompt):
    candidate_labels = ["label1", "label0"]  # Change labels as specified
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name,
            token=os.environ.get("HUGGING_FACE_TOKEN"),
            trust_remote_code=True,
        )

        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(candidate_labels),
            torch_dtype=torch.float16,
            token=os.environ.get("HUGGING_FACE_TOKEN"),
            device_map="auto",
            trust_remote_code=True,
        )

        device = model.device
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            logits = model(**inputs).logits

        predicted_label_id = logits.argmax().item()
        predicted_label = candidate_labels[predicted_label_id]

        probabilities = torch.nn.functional.softmax(logits, dim=-1)[0].tolist()
        label_probs = dict(zip(candidate_labels, probabilities))

        return predicted_label, label_probs

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None

# Example Usage:
model_name = "meta-llama/Llama-3.2-1B"  # Or a more suitable classification model - bigger the model, the more accurate it is
sample_prompt = """Technology Secretary Peter Kyle described it as the “logical next step” for the AISI — and insisted its work wouldn’t change.""" # Example news paragraph

prompt = f"Is the following news paragraph credible? {sample_prompt}" # Clear prompt format

predicted_label, label_probs = classify_credibility(model_name, prompt)

if predicted_label:
    print(f"Predicted Credibility: {predicted_label}")
    print(f"Credibility Probabilities: {label_probs}")
else:
    print("Classification failed.")


