from transformers.pipelines import pipeline

# Initialize the SmolDocling pipeline
pipe = pipeline("image-text-to-text", model="ds4sd/SmolDocling-256M-preview")

# List of local page images
image_files = [
    "input_page_1.png",
    "input_page_2.png",
    "input_page_3.png",
    "input_page_4.png",
]

generated_contents = []

# Process each page image through the model
for img_path in image_files:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": img_path},
                {
                    "type": "text",
                    "text": "Extract all of the text from the attached image files and generate a single output markdown file that combines all extracted text and matches the original image format, style, look, and feel. Ensure you preserve bullet points, lists, and tables, but do not include graphics. Do not include your own disclaimers or comments in your response. Do not hallucinate.",
                },
            ],
        }
    ]

    result = pipe(text=messages)
    generated_contents.append(result[0]["generated_text"][-1]["content"])

# Combine all generated texts into a single Markdown string
final_output = "\n\n".join(generated_contents)

# Write the combined content to SmolDocling.md
with open("SmolDocling.md", "w", encoding="utf-8") as md_file:
    md_file.write(final_output)

# (Optional) Print the final output to console
print(final_output)
