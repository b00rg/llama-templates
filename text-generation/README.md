
# Local Llama Model Inference

This script allows you to run a Llama model locally for text generation. It uses the Hugging Face `transformers` library along with PyTorch to load and generate responses from a language model based on a provided prompt. 

## Requirements

1. **Python** 3.7 or higher
2. **Libraries**:
   - `torch` (PyTorch)
   - `transformers` (Hugging Face)
3. **Environment Variables**:
   - `HUGGING_FACE_TOKEN`: Your Hugging Face API token.

You can install the necessary Python packages via pip:
```bash
pip install torch transformers
```

## Setup

Before running the script, ensure that you have a Hugging Face API token and set it as an environment variable.

### Setting up the Hugging Face API Token
To obtain the token:
1. Go to [Hugging Face](https://huggingface.co/).
2. Create an account or log in.
3. Navigate to your account settings and generate a new API token.

To set the token as an environment variable:
- On Windows:
  ```bash
  set HUGGING_FACE_TOKEN=your_token_here
  ```
- On Linux/macOS:
  ```bash
  export HUGGING_FACE_TOKEN=your_token_here
  ```

## How to Use

### 1. Update the Script

- Set the model name to the Llama model you want to use. For example:
  ```python
  model_name = "meta-llama/Llama-3.2-1B"
  ```
  You can choose other models depending on the size and performance requirements. Note that larger models may take longer to load and generate responses.

- Update the `user_prompt` variable with the prompt you wish to send to the model for text generation:
  ```python
  user_prompt = "Once upon a time in a land far away,"
  ```

### 2. Run the Script

After updating the script with your model and prompt, run the script:
```bash
python script_name.py
```

### Example Usage:

```python
model_name = "meta-llama/Llama-3.2-1B"  # The model to load
user_prompt = "Tell me a story about a dragon."  # Your input prompt

generated_response = run_llama_locally(model_name, user_prompt)

if generated_response:
    print(f"Generated Text: 
{generated_response}")
```

## Script Overview

### `run_llama_locally`

The core function of the script is `run_llama_locally`, which:

1. Loads the tokenizer and model using the Hugging Face `transformers` library.
2. Sends the provided prompt to the model and generates a response.
3. Returns the generated text.

#### Parameters:
- `model_name`: The name of the model to load from Hugging Face (e.g., `meta-llama/Llama-3.2-1B`).
- `prompt`: The prompt for the model to generate text from.
- `max_new_tokens` (default: 50): The maximum number of new tokens to generate in the response.
- `temperature` (default: 0.7): Controls the randomness of the generation. Lower values make the output more deterministic.

### Error Handling

The script includes error handling that will print an error message if something goes wrong during execution (e.g., if the model fails to load or an invalid token is provided).

## Troubleshooting

- **Error: `HUGGING_FACE_TOKEN` not set.**
  - Ensure that you have correctly set the Hugging Face API token in your environment variables.

- **Slow model inference.**
  - Larger models take more time to load and generate responses. You can switch to a smaller model or optimize the code for inference speed.

