from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from io import BytesIO
from fastapi.responses import JSONResponse

app = FastAPI(
    title="Image Captioning App",
    description="Image captioning app using ML science and python language",
    version="1.0",
)

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

@app.get("/")
async def root():
    return {"message": "Welcome to the Image Captioning API. Use POST /caption to upload an image."}


@app.post("/caption")
async def generate_caption(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are allowed")

    try:
        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)

        outputs = model.generate(**inputs, max_length=300, num_beams=5)
        caption = processor.decode(outputs[0], skip_special_tokens=True)

        return JSONResponse(content={"caption": caption})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
