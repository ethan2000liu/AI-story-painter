from transformers import pipeline
from diffusers import DiffusionPipeline

# Initialize the summarization pipeline
summarizer = pipeline("summarization")

# Read the content of your text file
with open("text.txt", "r", encoding="utf-8") as file:
    document_content = file.read()

# Summarize the document into one sentence
summary = summarizer(document_content, max_length=50, min_length=5, length_penalty=2.0, num_beams=4, early_stopping=True)
summarized_sentence = summary[0]['summary_text']

# Initialize the Diffusion pipeline
diffusion_pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

# Generate an image based on the summarized sentence
generated_image = diffusion_pipeline(summarized_sentence)

# You can then save or display the generated image as needed
generated_image.save("output_image.png")
