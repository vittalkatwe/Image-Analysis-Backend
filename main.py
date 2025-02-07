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




# working fine

# from pydantic import BaseModel
# from fastapi import FastAPI, File, Form, UploadFile, HTTPException
# from fastapi.responses import JSONResponse, StreamingResponse
# from fastapi.middleware.cors import CORSMiddleware
# import aiofiles  # Import aiofiles even if not used directly
# from typing import AsyncGenerator
# import time
# import os
# import google.generativeai as genai
# from PIL import Image, ImageFile
# from io import BytesIO
# import pymongo
# from datetime import datetime

# app = FastAPI()

# ImageFile.LOAD_TRUNCATED_IMAGES = True

# # Allow CORS for the frontend
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:5173"],  # Replace with your frontend URL if different
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Define a base directory for storing uploaded images
# UPLOAD_DIR = "uploads"
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# # Configure Gemini API
# GEMINI_API_KEY = "AIzaSyAHp2ig3DfIl5xvnBEtUKyP-Ckv21GLtto"  # Access API key from environment variable
# if not GEMINI_API_KEY:
#     raise ValueError("GEMINI_API_KEY not found in environment variables")
# genai.configure(api_key=GEMINI_API_KEY)

# # Initialize Gemini Vision model
# model = genai.GenerativeModel('gemini-1.5-flash')

# # MongoDB Configuration
# MONGO_URI = "mongodb+srv://vittalkatwe:vittalkatwe@cautious.rh2qklo.mongodb.net/?retryWrites=true&w=majority&appName=cautious"  # Replace <db_password> with your actual password
# MONGO_DB_NAME = "gemini_app"  # Choose a name for your database
# mongo_client = pymongo.MongoClient(MONGO_URI)
# db = mongo_client[MONGO_DB_NAME]
# image_analysis_collection = db["image_analysis"]
# chat_collection = db["chats"]


# async def load_and_process_image(image: UploadFile) -> Image.Image:
#     """Loads and processes an uploaded image directly from memory."""
#     start_time = time.time()
#     try:
#         image_data = await image.read()
#         img = Image.open(BytesIO(image_data))

#         # 1. Check if resizing is even needed.  Only resize if the image is larger than a threshold.
#         max_size = 512  # Or whatever is appropriate for Gemini
#         if img.width > max_size or img.height > max_size:
#             # 2. Use a potentially faster resizing algorithm (try Image.Resampling.LANCZOS or Image.Resampling.BILINEAR)
#             img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)   # Preserves aspect ratio

#         end_time = time.time()
#         print(f"Image processing time: {end_time - start_time:.4f} seconds")
#         return img
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


# async def generate_content_stream(prompt: str, img: Image.Image) -> AsyncGenerator[str, None]:
#     """Generates content using Gemini in streaming mode."""
#     response_text = ""
#     try:
#         response = model.generate_content([prompt, img], stream=True)  # Enable streaming
#         for chunk in response:
#             text = chunk.text  # Get the chunk's text
#             response_text += text  # Append the text to the full response
#             yield text  # Yield each chunk of text

#         # After the stream is complete, save the data to MongoDB
#         try:
#             image_name = str(int(time.time())) + "_" + img.filename if hasattr(img, 'filename') else f"{int(time.time())}_image.jpg"  # Generate a unique filename
#             image_path = os.path.join(UPLOAD_DIR, image_name)
#             img.save(image_path)  # Save the resized image

#             image_analysis_collection.insert_one({
#                 "image_path": image_path,
#                 "prompt": prompt,
#                 "response": response_text,
#                 "timestamp": datetime.utcnow()
#             })
#             print("Image analysis data saved to MongoDB")

#         except Exception as db_error:
#             print(f"Error saving to MongoDB: {db_error}")
#             yield f"Error saving to MongoDB: {db_error}"  # Also include error in the stream

#     except Exception as e:
#         error_message = f"Error: {str(e)}"
#         yield error_message


# @app.post("/analyze_image")
# async def analyze_image(image: UploadFile = File(...), prompt: str = Form(...)):
#     """Analyzes an image based on the provided prompt, using streaming."""
#     try:
#         img = await load_and_process_image(image)
#         img.filename = image.filename
#         return StreamingResponse(generate_content_stream(prompt, img), media_type="text/plain")
#     except HTTPException as e:
#         return e  # Re-raise the HTTPException
#     except Exception as e:
#         return JSONResponse(
#             status_code=500,
#             content={"status": "error", "error": str(e)}
#         )


# @app.post("/chat")
# async def chat(message: str = Form(...)):
#     """Chatbot endpoint using Gemini."""
#     try:
#         response = model.generate_content(message)
#         response.resolve()  # Wait for the response to complete
#         response_text = response.text

#         if not response_text:
#             return JSONResponse(
#                 status_code=400,
#                 content={"status": "error", "error": "No response generated"}
#             )

#         # Save chat message and response to MongoDB
#         chat_collection.insert_one({
#             "message": message,
#             "response": response_text,
#             "timestamp": datetime.utcnow()
#         })
#         print("Chat data saved to MongoDB")

#         return {"status": "success", "response": response_text}

#     except Exception as e:
#         return JSONResponse(
#             status_code=500,
#             content={"status": "error", "error": str(e)}
#         )


# @app.get("/uploads/{file_path:path}")
# async def serve_file(file_path: str):
#     file_path_full = os.path.join(UPLOAD_DIR, file_path)
#     if not os.path.exists(file_path_full):
#         raise HTTPException(status_code=404, detail="File not found")
#     return file_path_full




# perfectly fine
# from pydantic import BaseModel
# from fastapi import FastAPI, File, Form, UploadFile, HTTPException
# from fastapi.responses import JSONResponse, StreamingResponse
# from fastapi.middleware.cors import CORSMiddleware
# import aiofiles  # Import aiofiles even if not used directly
# from typing import AsyncGenerator
# import time
# import os
# import google.generativeai as genai
# from PIL import Image, ImageFile
# from io import BytesIO

# app = FastAPI()

# ImageFile.LOAD_TRUNCATED_IMAGES = True

# # Allow CORS for the frontend
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:5173"],  # Replace with your frontend URL if different
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Define a base directory for storing uploaded images
# UPLOAD_DIR = "uploads"
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# # Configure Gemini API
# GEMINI_API_KEY = "AIzaSyAHp2ig3DfIl5xvnBEtUKyP-Ckv21GLtto"  # Access API key from environment variable
# if not GEMINI_API_KEY:
#     raise ValueError("GEMINI_API_KEY not found in environment variables")
# genai.configure(api_key=GEMINI_API_KEY)

# # Initialize Gemini Vision model
# model = genai.GenerativeModel('gemini-1.5-flash')


# async def load_and_process_image(image: UploadFile) -> Image.Image:
#     """Loads and processes an uploaded image directly from memory."""
#     start_time = time.time()
#     try:
#         image_data = await image.read()
#         img = Image.open(BytesIO(image_data))

#         # 1. Check if resizing is even needed.  Only resize if the image is larger than a threshold.
#         max_size = 512  # Or whatever is appropriate for Gemini
#         if img.width > max_size or img.height > max_size:
#             # 2. Use a potentially faster resizing algorithm (try Image.Resampling.LANCZOS or Image.Resampling.BILINEAR)
#             img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)   # Preserves aspect ratio

#         end_time = time.time()
#         print(f"Image processing time: {end_time - start_time:.4f} seconds")
#         return img
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


# async def generate_content_stream(prompt: str, img: Image.Image) -> AsyncGenerator[str, None]:
#     """Generates content using Gemini in streaming mode."""
#     try:
#         response = model.generate_content([prompt, img], stream=True)  # Enable streaming
#         for chunk in response:
#             yield chunk.text  # Yield each chunk of text
#     except Exception as e:
#         yield f"Error: {str(e)}"


# @app.post("/analyze_image")
# async def analyze_image(image: UploadFile = File(...), prompt: str = Form(...)):
#     """Analyzes an image based on the provided prompt, using streaming."""
#     try:
#         img = await load_and_process_image(image)
#         return StreamingResponse(generate_content_stream(prompt, img), media_type="text/plain")
#     except HTTPException as e:
#         return e  # Re-raise the HTTPException
#     except Exception as e:
#         return JSONResponse(
#             status_code=500,
#             content={"status": "error", "error": str(e)}
#         )


# @app.post("/chat")
# async def chat(message: str = Form(...)):
#     """Chatbot endpoint using Gemini."""
#     try:
#         response = model.generate_content(message)
#         response.resolve()  # Wait for the response to complete

#         if not response.text:
#             return JSONResponse(
#                 status_code=400,
#                 content={"status": "error", "error": "No response generated"}
#             )

#         return {"status": "success", "response": response.text}

#     except Exception as e:
#         return JSONResponse(
#             status_code=500,
#             content={"status": "error", "error": str(e)}
#         )


# @app.get("/uploads/{file_path:path}")
# async def serve_file(file_path: str):
#     file_path_full = os.path.join(UPLOAD_DIR, file_path)
#     if not os.path.exists(file_path_full):
#         raise HTTPException(status_code=404, detail="File not found")
#     return file_path_full


# from pydantic import BaseModel
# from fastapi import FastAPI, File, Form, UploadFile, HTTPException
# from fastapi.responses import JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# import base64
# import aiofiles
# import os
# import google.generativeai as genai
# from PIL import Image

# app = FastAPI()

# # Allow CORS for the frontend
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:5173"],  # Replace with your frontend URL if different
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Define a base directory for storing uploaded images
# UPLOAD_DIR = "uploads"
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# # Configure Gemini API
# GEMINI_API_KEY = "AIzaSyBjvcp8jl6YSqq27ICt8eiNOt7quGgr7C0"  # Replace with your actual API key
# genai.configure(api_key=GEMINI_API_KEY)

# # Initialize Gemini Vision model
# model = genai.GenerativeModel('gemini-1.5-flash')

# async def load_and_process_image(image: UploadFile) -> Image.Image:
#     """Loads and processes an uploaded image."""
#     try:
#         # Save the uploaded image
#         file_path = os.path.join(UPLOAD_DIR, image.filename)
#         async with aiofiles.open(file_path, 'wb') as out_file:
#             content = await image.read()
#             await out_file.write(content)

#         # Load and prepare the image for Gemini
#         img = Image.open(file_path)
#         return img
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


# @app.post("/analyze_image")
# async def analyze_image(image: UploadFile = File(...), prompt: str = Form(...)):
#     """Analyzes an image based on the provided prompt."""
#     try:
#         img = await load_and_process_image(image)

#         # Generate content using Gemini
#         response = model.generate_content([prompt, img])
        
#         # Wait for the response to complete
#         response.resolve()

#         # Check if we have a response
#         if not response.text:
#             return JSONResponse(
#                 status_code=400,
#                 content={"status": "error", "error": "No response generated"}
#             )

#         return {
#             "status": "success",
#             "analysis": response.text  # Changed "response" to "analysis"
#         }

#     except HTTPException as e:
#         return e # Re-raise the HTTPException
#     except Exception as e:
#         return JSONResponse(
#             status_code=500,
#             content={"status": "error", "error": str(e)}
#         )

# @app.post("/chat")
# async def chat(message: str = Form(...)):
#     """A simple chatbot endpoint (replace with your chatbot logic)."""
#     try:
#         # Replace this with your actual chatbot logic
#         chatbot_response = f"You said: {message}"  #Echoing what you said
#         return {"status": "success", "response": chatbot_response}
#     except Exception as e:
#         return JSONResponse(
#             status_code=500,
#             content={"status": "error", "error": str(e)}
#         )



# @app.get("/uploads/{file_path:path}")
# async def serve_file(file_path: str):
#     file_path_full = os.path.join(UPLOAD_DIR, file_path)
#     if not os.path.exists(file_path_full):
#         raise HTTPException(status_code=404, detail="File not found")
#     return file_path_full


# from pydantic import BaseModel
# from fastapi import FastAPI, File, Form, UploadFile, HTTPException
# from fastapi.responses import JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# import base64
# import aiofiles
# import os
# import google.generativeai as genai
# from PIL import Image

# app = FastAPI()

# # Allow CORS for the frontend
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:5173"],  # Replace with your frontend URL if different
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Define a base directory for storing uploaded images
# UPLOAD_DIR = "uploads"
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# # Configure Gemini API
# GEMINI_API_KEY = "AIzaSyBjvcp8jl6YSqq27ICt8eiNOt7quGgr7C0"  # Replace with your actual API key
# genai.configure(api_key=GEMINI_API_KEY)

# # Initialize Gemini Vision model
# model = genai.GenerativeModel('gemini-1.5-flash')

# @app.post("/generate")
# async def generate(image: UploadFile = File(...), prompt: str = Form(...)):
#     try:
#         # Save the uploaded image
#         file_path = os.path.join(UPLOAD_DIR, image.filename)
#         async with aiofiles.open(file_path, 'wb') as out_file:
#             content = await image.read()
#             await out_file.write(content)

#         # Load and prepare the image for Gemini
#         img = Image.open(file_path)

#         # Generate content using Gemini
#         response = model.generate_content([prompt, img])
        
#         # Wait for the response to complete
#         response.resolve()

#         # Check if we have a response
#         if not response.text:
#             return JSONResponse(
#                 status_code=400,
#                 content={"status": "error", "error": "No response generated"}
#             )

#         return {
#             "status": "success",
#             "response": response.text
#         }

#     except Exception as e:
#         return JSONResponse(
#             status_code=500,
#             content={"status": "error", "error": str(e)}
#         )

# @app.get("/uploads/{file_path:path}")
# async def serve_file(file_path: str):
#     file_path_full = os.path.join(UPLOAD_DIR, file_path)
#     if not os.path.exists(file_path_full):
#         raise HTTPException(status_code=404, detail="File not found")
#     return file_path_full






# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel
# from transformers import AutoProcessor, AutoModelForVision2Seq
# import torch
# from PIL import Image
# import os
# import logging

# # Logging configuration
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # FastAPI app instance
# app = FastAPI()

# # Model and processor global variables
# MODEL_NAME = "Qwen/Qwen2-VL-7B"
# model = None
# processor = None

# def load_model():
#     global model, processor
#     logger.info("Loading model and processor...")
#     try:
#         processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
#         model = AutoModelForVision2Seq.from_pretrained(
#             MODEL_NAME,
#             device_map="auto",
#             offload_folder="offload",
#             torch_dtype=torch.float16,
#             trust_remote_code=True
#         )
#         logger.info("Model and processor loaded successfully.")
#     except Exception as e:
#         logger.error(f"Error loading model: {e}")
#         raise e

# # Load the model at startup
# @app.on_event("startup")
# async def startup_event():
#     load_model()

# class GenerationRequest(BaseModel):
#     prompt: str
#     image: UploadFile

# @app.post("/generate")
# async def generate(prompt: str, image: UploadFile = File(...)):
#     if not model or not processor:
#         raise HTTPException(status_code=500, detail="Model is not loaded.")

#     try:
#         # Read the uploaded image
#         image_content = await image.read()
#         pil_image = Image.open(io.BytesIO(image_content)).convert("RGB")

#         # Preprocess the image and run inference
#         inputs = processor(
#             images=pil_image, text=prompt, return_tensors="pt"
#         ).to("cuda")
        
#         outputs = model.generate(
#             **inputs,
#             max_length=50,
#             num_beams=5,
#             temperature=0.7
#         )

#         generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
#         logger.info(f"Generated Text: {generated_text}")

#         return JSONResponse(content={"generated_text": generated_text})
    
#     except Exception as e:
#         logger.error(f"Error during inference: {e}")
#         raise HTTPException(status_code=500, detail="Error during inference.")

# @app.get("/")
# async def root():
#     return {"message": "Welcome to the Image-to-Text Generation API!"}






# usage error but receiving image

# from fastapi import FastAPI, File, UploadFile, Form, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# from PIL import Image
# from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
# import io
# import logging
# import sys
# from typing import Optional
# import torch
# import gc
# import os
# import psutil

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[logging.StreamHandler(sys.stdout)]
# )
# logger = logging.getLogger(__name__)

# app = FastAPI(title="Image Analysis API")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
#     allow_credentials=True,
#     allow_methods=["GET", "POST", "OPTIONS"],
#     allow_headers=["*"],
# )

# # Global variables
# model: Optional[Qwen2VLForConditionalGeneration] = None
# processor: Optional[AutoProcessor] = None

# # Advanced memory management settings
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,garbage_collection_threshold:0.8,expandable_segments:True'

# def get_memory_info():
#     """Get current memory usage information"""
#     memory_info = {}
    
#     if torch.cuda.is_available():
#         memory_info['cuda'] = {
#             'allocated': torch.cuda.memory_allocated(0) / 1024**3,
#             'cached': torch.cuda.memory_reserved(0) / 1024**3,
#             'max': torch.cuda.get_device_properties(0).total_memory / 1024**3
#         }
    
#     memory_info['ram'] = {
#         'used': psutil.Process().memory_info().rss / 1024**3,
#         'available': psutil.virtual_memory().available / 1024**3,
#         'total': psutil.virtual_memory().total / 1024**3
#     }
    
#     return memory_info

# def clear_memory():
#     """Aggressive memory clearing"""
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#         torch.cuda.synchronize()
#     gc.collect()
    
#     memory_info = get_memory_info()
#     logger.info(f"Memory after clearing - GPU: {memory_info.get('cuda', 'N/A')}, RAM: {memory_info['ram']}")

# @app.on_event("startup")
# async def startup_event():
#     """Initialize model and processor with aggressive memory optimization"""
#     global model, processor
#     logger.info("Loading model and processor...")
#     try:
#         clear_memory()
        
#         if torch.cuda.is_available():
#             gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
#             logger.info(f"Total GPU memory: {gpu_mem:.2f} GB")
            
#             # More aggressive memory settings for 8GB GPU
#             model = Qwen2VLForConditionalGeneration.from_pretrained(
#                 "Qwen/Qwen2-VL-7B-Instruct",
#                 torch_dtype=torch.float16,
#                 device_map="auto",
#                 max_memory={0: "4GiB", "cpu": "32GiB"},  # Reduced GPU memory allocation
#                 offload_folder="offload",
#                 low_cpu_mem_usage=True,
#                 offload_state_dict=True  # Enable state dict offloading
#             )
#         else:
#             model = Qwen2VLForConditionalGeneration.from_pretrained(
#                 "Qwen/Qwen2-VL-7B-Instruct",
#                 device_map="cpu",
#                 low_cpu_mem_usage=True
#             )
        
#         processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
#         logger.info("Model and processor loaded successfully")
        
#         memory_info = get_memory_info()
#         logger.info(f"Initial memory usage: {memory_info}")
        
#     except Exception as e:
#         logger.error(f"Error loading model: {str(e)}")
#         raise HTTPException(status_code=500, detail="Failed to initialize model")

# @app.get("/test")
# async def test():
#     """Test endpoint with memory info"""
#     try:
#         if model is None or processor is None:
#             return JSONResponse(
#                 status_code=503,
#                 content={"status": "error", "message": "Model not initialized"}
#             )
        
#         memory_info = get_memory_info()
#         return JSONResponse(content={
#             "status": "ok",
#             "message": "API is running",
#             "memory_info": memory_info
#         })
#     except Exception as e:
#         logger.error(f"Test endpoint error: {str(e)}")
#         return JSONResponse(
#             status_code=500,
#             content={"status": "error", "message": str(e)}
#         )

# @app.post("/generate")
# async def generate_response(
#     image: UploadFile = File(...),
#     prompt: str = Form(...)
# ):
#     """Generate response with enhanced memory management"""
#     logger.info(f"Received request with prompt: {prompt}")
    
#     try:
#         # Validate image
#         if not image.content_type.startswith('image/'):
#             raise HTTPException(status_code=400, detail="File must be an image")
        
#         # Load and process image
#         image_bytes = io.BytesIO(await image.read())
#         img = Image.open(image_bytes)
        
#         # Aggressive image resizing for memory optimization
#         max_size = 512  # Reduced from 800
#         if max(img.size) > max_size:
#             ratio = max_size / max(img.size)
#             new_size = tuple(int(dim * ratio) for dim in img.size)
#             img = img.resize(new_size, Image.Resampling.LANCZOS)
        
#         logger.info(f"Image processed: {img.format} {img.size}")
#         clear_memory()

#         # Process inputs with memory optimization
#         try:
#             prompt_text = f"<image>\n{prompt}"
#             inputs = processor(
#                 text=prompt_text,
#                 images=img,
#                 return_tensors="pt",
#                 padding=True,
#                 max_length=256  # Reduced from 512
#             )
            
#             # Move to device and clear image data
#             inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
#             del img
#             clear_memory()
            
#             logger.info(f"Input shapes: {inputs['input_ids'].shape}")
#         except Exception as e:
#             logger.error(f"Input processing error: {str(e)}")
#             raise HTTPException(status_code=500, detail="Error processing inputs")

#         # Generate with optimized settings
#         try:
#             with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
#                 outputs = model.generate(
#                     **inputs,
#                     max_length=128,  # Reduced from 256
#                     num_beams=1,     # Reduced from 2
#                     temperature=0.7,
#                     top_k=50,
#                     top_p=0.95,
#                     pad_token_id=processor.tokenizer.pad_token_id,
#                     eos_token_id=processor.tokenizer.eos_token_id,
#                     do_sample=True,
#                     early_stopping=True,
#                     use_cache=True
#                 )
            
#             response = processor.batch_decode(outputs, skip_special_tokens=True)[0]
#             logger.info(f"Generated response: {response}")
            
#             # Clear everything after generation
#             del outputs, inputs
#             clear_memory()

#             return JSONResponse(content={
#                 "status": "success",
#                 "response": response
#             })

#         except torch.cuda.OutOfMemoryError as oom:
#             clear_memory()
#             logger.error(f"CUDA out of memory: {str(oom)}")
#             return JSONResponse(
#                 status_code=503,
#                 content={
#                     "status": "error",
#                     "error": "Server is currently overloaded. Please try with a smaller image or simpler prompt."
#                 }
#             )
        
#     except Exception as e:
#         logger.error(f"Unexpected error: {str(e)}")
#         return JSONResponse(
#             status_code=500,
#             content={
#                 "status": "error",
#                 "error": str(e)
#             }
#         )

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)


# usage error

# from fastapi import FastAPI, File, UploadFile, Form, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# from PIL import Image
# from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
# import io
# import logging
# import sys
# from typing import Optional
# import torch
# import gc

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.StreamHandler(sys.stdout)
#     ]
# )
# logger = logging.getLogger(__name__)

# app = FastAPI()

# # CORS setup
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Global variables for model and processor
# model: Optional[Qwen2VLForConditionalGeneration] = None
# processor: Optional[AutoProcessor] = None

# @app.on_event("startup")
# async def startup_event():
#     """Initialize model and processor on startup"""
#     global model, processor
#     logger.info("Loading model and processor...")
#     try:
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#             gc.collect()

#         model = Qwen2VLForConditionalGeneration.from_pretrained(
#             "Qwen/Qwen2-VL-7B-Instruct",
#             torch_dtype=torch.float16,
#             device_map="auto",
#             low_cpu_mem_usage=True,
#         )
        
#         processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
#         logger.info("Model and processor loaded successfully")
        
#     except Exception as e:
#         logger.error(f"Error loading model: {str(e)}")
#         logger.info("Attempting to load model on CPU...")
#         try:
#             model = Qwen2VLForConditionalGeneration.from_pretrained(
#                 "Qwen/Qwen2-VL-7B-Instruct",
#                 torch_dtype=torch.float16,
#                 device_map="cpu",
#                 low_cpu_mem_usage=True,
#             )
#             processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
#             logger.info("Model loaded successfully on CPU")
#         except Exception as cpu_e:
#             logger.error(f"Error loading model on CPU: {str(cpu_e)}")
#             raise HTTPException(status_code=500, detail="Failed to initialize model")

# @app.get("/test")
# async def test():
#     """Test endpoint to verify API is working"""
#     return {"status": "OK", "message": "API is running"}

# @app.post("/generate")
# async def generate_response(
#     image: UploadFile = File(...),
#     prompt: str = Form(...)
# ):
#     """Generate response based on image and prompt"""
#     logger.info(f"Received request with prompt: {prompt}")
    
#     try:
#         # Validate image
#         if not image.content_type.startswith('image/'):
#             raise HTTPException(status_code=400, detail="File must be an image")
        
#         # Load and validate image
#         try:
#             image_bytes = io.BytesIO(await image.read())
#             img = Image.open(image_bytes)
#             logger.info(f"Image loaded successfully: {img.format} {img.size}")
#         except Exception as e:
#             logger.error(f"Error loading image: {str(e)}")
#             raise HTTPException(status_code=400, detail="Invalid image file")

#         # Clear CUDA cache before processing
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#             gc.collect()

#         # Process inputs
#         try:
#             # Create prompt text
#             prompt_text = f"<image>\n{prompt}"
            
#             # Process the inputs
#             inputs = processor(
#                 text=prompt_text,
#                 images=img,
#                 return_tensors="pt",
#                 padding=True
#             )
            
#             # Move to same device as model
#             inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
#             logger.info("Inputs processed successfully")
#             logger.info(f"Input shapes: {inputs['input_ids'].shape}")
#         except Exception as e:
#             logger.error(f"Error processing inputs: {str(e)}")
#             raise HTTPException(status_code=500, detail="Error processing inputs")

#         # Generate response
#         try:
#             with torch.no_grad():
#                 outputs = model.generate(
#                     **inputs,
#                     max_length=512,
#                     num_beams=3,
#                     temperature=0.7,
#                     top_k=50,
#                     top_p=0.95,
#                     pad_token_id=processor.tokenizer.pad_token_id,
#                     eos_token_id=processor.tokenizer.eos_token_id,
#                 )
                
#             response = processor.batch_decode(outputs, skip_special_tokens=True)[0]
#             logger.info(f"Generated response: {response}")

#             # Clear cache after generation
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()
#                 gc.collect()

#             if not response or response.isspace():
#                 raise HTTPException(status_code=500, detail="Model generated empty response")

#             return JSONResponse(content={"response": response})

#         except Exception as e:
#             logger.error(f"Error generating response: {str(e)}")
#             raise HTTPException(status_code=500, detail="Error generating response")

#     except HTTPException as he:
#         raise he
#     except Exception as e:
#         logger.error(f"Unexpected error: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)




#  error processing input

# from fastapi import FastAPI, File, UploadFile, Form, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# from PIL import Image
# from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
# import io
# import logging
# import sys
# from typing import Optional
# import torch
# import gc

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.StreamHandler(sys.stdout)
#     ]
# )
# logger = logging.getLogger(__name__)

# app = FastAPI()

# # CORS setup
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Global variables for model and processor
# model: Optional[Qwen2VLForConditionalGeneration] = None
# processor: Optional[AutoProcessor] = None

# @app.on_event("startup")
# async def startup_event():
#     """Initialize model and processor on startup"""
#     global model, processor
#     logger.info("Loading model and processor...")
#     try:
#         # Clear CUDA cache
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#             gc.collect()

#         # Load model with lower precision to save memory
#         model = Qwen2VLForConditionalGeneration.from_pretrained(
#             "Qwen/Qwen2-VL-7B-Instruct",
#             torch_dtype=torch.float16,  # Use half precision
#             device_map="auto",  # Automatically handle device placement
#             low_cpu_mem_usage=True,
#         )
        
#         processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
#         logger.info("Model and processor loaded successfully")
        
#     except Exception as e:
#         logger.error(f"Error loading model: {str(e)}")
#         logger.info("Attempting to load model on CPU...")
#         try:
#             # Fallback to CPU with reduced precision
#             model = Qwen2VLForConditionalGeneration.from_pretrained(
#                 "Qwen/Qwen2-VL-7B-Instruct",
#                 torch_dtype=torch.float16,
#                 device_map="cpu",
#                 low_cpu_mem_usage=True,
#             )
#             processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
#             logger.info("Model loaded successfully on CPU")
#         except Exception as cpu_e:
#             logger.error(f"Error loading model on CPU: {str(cpu_e)}")
#             raise HTTPException(status_code=500, detail="Failed to initialize model")

# @app.get("/test")
# async def test():
#     """Test endpoint to verify API is working"""
#     return {"status": "OK", "message": "API is running"}

# @app.post("/generate")
# async def generate_response(
#     image: UploadFile = File(...),
#     prompt: str = Form(...)
# ):
#     """Generate response based on image and prompt"""
#     logger.info(f"Received request with prompt: {prompt}")
    
#     try:
#         # Validate image
#         if not image.content_type.startswith('image/'):
#             raise HTTPException(status_code=400, detail="File must be an image")
        
#         # Load and validate image
#         try:
#             image_bytes = io.BytesIO(await image.read())
#             img = Image.open(image_bytes)
#             logger.info(f"Image loaded successfully: {img.format} {img.size}")
#         except Exception as e:
#             logger.error(f"Error loading image: {str(e)}")
#             raise HTTPException(status_code=400, detail="Invalid image file")

#         # Clear CUDA cache before processing
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#             gc.collect()

#         # Process inputs
#         try:
#             # Prepare messages
#             messages = [
#                 {
#                     "role": "user",
#                     "content": [
#                         {"type": "image", "image": img},
#                         {"type": "text", "text": prompt},
#                     ],
#                 },
#             ]

#             inputs = processor(messages)
#             input_ids = torch.tensor(inputs['input_ids'])
#             if input_ids.dim() == 1:
#                 input_ids = input_ids.unsqueeze(0)
            
#             # Move to same device as model
#             input_ids = input_ids.to(model.device)
            
#             logger.info("Inputs processed successfully")
#             logger.info(f"Input shape: {input_ids.shape}")
#         except Exception as e:
#             logger.error(f"Error processing inputs: {str(e)}")
#             raise HTTPException(status_code=500, detail="Error processing inputs")

#         # Generate response
#         try:
#             with torch.no_grad():
#                 outputs = model.generate(
#                     input_ids,
#                     max_length=512,
#                     num_beams=3,  # Reduced from 5 to save memory
#                     temperature=0.7,
#                     top_k=50,
#                     top_p=0.95,
#                     pad_token_id=processor.tokenizer.pad_token_id,
#                     eos_token_id=processor.tokenizer.eos_token_id,
#                 )
                
#             response = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
#             logger.info(f"Generated response: {response}")

#             # Clear cache after generation
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()
#                 gc.collect()

#             if not response or response.isspace():
#                 raise HTTPException(status_code=500, detail="Model generated empty response")

#             return JSONResponse(content={"response": response})

#         except Exception as e:
#             logger.error(f"Error generating response: {str(e)}")
#             raise HTTPException(status_code=500, detail="Error generating response")

#     except HTTPException as he:
#         raise he
#     except Exception as e:
#         logger.error(f"Unexpected error: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)