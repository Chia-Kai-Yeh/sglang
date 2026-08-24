import requests
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/workspace/models/Qwen/Qwen3-4B-FP8")

text = tokenizer.apply_chat_template(
    [
        {"role": "user", "content": "Write a long story."},
    ],
    add_generation_prompt=True,
    tokenize=False,
    enable_thinking=False,
)

response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": text,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 512,
        },
    },
)

print(response.json())
