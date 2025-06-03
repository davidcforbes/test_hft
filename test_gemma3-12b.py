from transformers.pipelines import pipeline
import torch

pipe = pipeline(
    "image-text-to-text",
    model="google/gemma-3-12b-it",
    device="cuda",
    torch_dtype=torch.bfloat16
)

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}]
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
            {"type": "text", "text": "What animal is on the candy?"}
        ]
    }
]

output = pipe(inputs=messages, max_new_tokens=32000)

# If output is a generator, convert to list
if output is not None and hasattr(output, '__iter__') and not isinstance(output, (list, dict, str)):
    output = list(output)

print(output)
# If you know the structure, you can further extract the text as needed, e.g.:
# print(output[0]["generated_text"][-1]["content"])
