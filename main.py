from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration


def generate_caption(image_path):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    outputs = model.generate(**inputs, max_length=1500, num_beams=10)
    caption = processor.decode(outputs[0], skip_special_tokens=True)

    return caption
