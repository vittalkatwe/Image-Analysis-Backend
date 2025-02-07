from pydantic import BaseModel
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import aiofiles  # Import aiofiles even if not used directly
from typing import AsyncGenerator
import time
import os
import google.generativeai as genai
from PIL import Image, ImageFile
from io import BytesIO
import pymongo
from datetime import datetime
from dotenv import load_dotenv  # Import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Allow CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Replace with your frontend URL if different
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define a base directory for storing uploaded images
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Access API key from environment variable
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Gemini Vision model
model = genai.GenerativeModel('gemini-1.5-flash')

# MongoDB Configuration
MONGO_URI = os.getenv("MONGODB_URI")  # Get MongoDB URI from environment variables
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "gemini_app")  # Get DB name, default to "gemini_app"
if not MONGO_URI:
    raise ValueError("MONGODB_URI not found in environment variables")

mongo_client = pymongo.MongoClient(MONGO_URI)
db = mongo_client[MONGO_DB_NAME]
image_analysis_collection = db["image_analysis"]
chat_collection = db["chats"]


async def load_and_process_image(image: UploadFile) -> Image.Image:
    """Loads and processes an uploaded image directly from memory."""
    start_time = time.time()
    try:
        image_data = await image.read()
        img = Image.open(BytesIO(image_data))

        # 1. Check if resizing is even needed.  Only resize if the image is larger than a threshold.
        max_size = 512  # Or whatever is appropriate for Gemini
        if img.width > max_size or img.height > max_size:
            # 2. Use a potentially faster resizing algorithm (try Image.Resampling.LANCZOS or Image.Resampling.BILINEAR)
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)   # Preserves aspect ratio

        end_time = time.time()
        print(f"Image processing time: {end_time - start_time:.4f} seconds")
        return img
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


async def generate_content_stream(prompt: str, img: Image.Image) -> AsyncGenerator[str, None]:
    """Generates content using Gemini in streaming mode."""
    response_text = ""
    try:
        response = model.generate_content([prompt, img], stream=True)  # Enable streaming
        for chunk in response:
            text = chunk.text  # Get the chunk's text
            response_text += text  # Append the text to the full response
            yield text  # Yield each chunk of text

        # After the stream is complete, save the data to MongoDB
        try:
            image_name = str(int(time.time())) + "_" + img.filename if hasattr(img, 'filename') else f"{int(time.time())}_image.jpg"  # Generate a unique filename
            image_path = os.path.join(UPLOAD_DIR, image_name)
            img.save(image_path)  # Save the resized image

            image_analysis_collection.insert_one({
                "image_path": image_path,
                "prompt": prompt,
                "response": response_text,
                "timestamp": datetime.utcnow()
            })
            print("Image analysis data saved to MongoDB")

        except Exception as db_error:
            print(f"Error saving to MongoDB: {db_error}")
            yield f"Error saving to MongoDB: {db_error}"  # Also include error in the stream

    except Exception as e:
        error_message = f"Error: {str(e)}"
        yield error_message


@app.post("/analyze_image")
async def analyze_image(image: UploadFile = File(...), prompt: str = Form(...)):
    """Analyzes an image based on the provided prompt, using streaming."""
    try:
        img = await load_and_process_image(image)
        img.filename = image.filename
        return StreamingResponse(generate_content_stream(prompt, img), media_type="text/plain")
    except HTTPException as e:
        return e  # Re-raise the HTTPException
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error": str(e)}
        )


@app.post("/chat")
async def chat(message: str = Form(...)):
    """Chatbot endpoint using Gemini."""
    try:
        response = model.generate_content(message)
        response.resolve()  # Wait for the response to complete
        response_text = response.text

        if not response_text:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "error": "No response generated"}
            )

        # Save chat message and response to MongoDB
        chat_collection.insert_one({
            "message": message,
            "response": response_text,
            "timestamp": datetime.utcnow()
        })
        print("Chat data saved to MongoDB")

        return {"status": "success", "response": response_text}

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error": str(e)}
        )


@app.get("/uploads/{file_path:path}")
async def serve_file(file_path: str):
    file_path_full = os.path.join(UPLOAD_DIR, file_path)
    if not os.path.exists(file_path_full):
        raise HTTPException(status_code=404, detail="File not found")
    return file_path_full



