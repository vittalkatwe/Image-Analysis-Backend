# Backend Setup Instructions

1. Create a Python virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   - On Windows:
     ```bash
     .\venv\Scripts\activate
     ```
   - On Unix or MacOS:
     ```bash
     source venv/bin/activate
     ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Important Notes:
   - The first run will download the Moondream2 model (approximately 2GB)
   - Make sure you have enough RAM (at least 8GB recommended)
   - GPU is recommended but not required (will use CPU if GPU is not available)

5. Run the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```

The server will start at http://localhost:8000