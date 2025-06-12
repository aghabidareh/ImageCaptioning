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


