# pip install accelerate


token = ""
from huggingface_hub import HfApi, HfFolder
HfFolder.save_token(token)
api = HfApi()
user_info = api.whoami()
print(f"Connected as: {user_info['name']}")


from transformers import AutoTokenizer, BitsAndBytesConfig, Gemma3ForCausalLM
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import torch
import gc


model_id = "google/gemma-3-4b-it"

model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id
).eval()

processor = AutoProcessor.from_pretrained(model_id)

print(model)
input()
print(processor)
del model
torch.mps.empty_cache()
gc.collect()
exit()


messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}]
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"},
            {"type": "text", "text": "Describe this image in detail."}
        ]
    }
]

inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt"
).to(model.device, dtype=torch.bfloat16)

input_len = inputs["input_ids"].shape[-1]

with torch.inference_mode():
    generation = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    generation = generation[0][input_len:]

decoded = processor.decode(generation, skip_special_tokens=True)
print(decoded)
