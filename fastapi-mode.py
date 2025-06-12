from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from io import BytesIO
from fastapi.responses import JSONResponse


