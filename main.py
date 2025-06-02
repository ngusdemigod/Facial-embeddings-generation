from fastapi import FastAPI, File, UploadFile, Form
import face_recognition
import numpy as np
import httpx
import uuid
import os
from fastapi.middleware.cors import CORSMiddleware

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_API_KEY = os.getenv("SUPABASE_API_KEY")
SUPABASE_TABLE = "face_embeddings"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Upload image and save embeddings to Supabase
@app.post("/upload")
async def upload_face(name: str = Form(...), file: UploadFile = File(...)):
    image_data = await file.read()
    image = face_recognition.load_image_file(file.file)
    encodings = face_recognition.face_encodings(image)

    if not encodings:
        return {"error": "No face detected."}

    embedding = encodings[0].tolist()

    headers = {
        "apikey": SUPABASE_API_KEY,
        "Authorization": f"Bearer {SUPABASE_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "id": str(uuid.uuid4()),
        "name": name,
        "embedding": embedding
    }

    async with httpx.AsyncClient() as client:
        res = await client.post(
            f"{SUPABASE_URL}/rest/v1/{SUPABASE_TABLE}",
            headers=headers,
            json=payload
        )

    if res.status_code != 201:
        return {"error": "Failed to upload embedding.", "details": res.text}

    return {"message": "Embedding saved successfully."}

# Match an image to existing embeddings
@app.post("/match")
async def match_face(file: UploadFile = File(...)):
    image_data = await file.read()
    image = face_recognition.load_image_file(file.file)
    encodings = face_recognition.face_encodings(image)

    if not encodings:
        return {"error": "No face detected in uploaded image."}

    query_embedding = encodings[0]

    headers = {
        "apikey": SUPABASE_API_KEY,
        "Authorization": f"Bearer {SUPABASE_API_KEY}"
    }

    async with httpx.AsyncClient() as client:
        res = await client.get(
            f"{SUPABASE_URL}/rest/v1/{SUPABASE_TABLE}?select=id,name,embedding",
            headers=headers
        )

    if res.status_code != 200:
        return {"error": "Failed to fetch embeddings from Supabase."}

    matches = []
    for record in res.json():
        db_embedding = np.array(record['embedding'])
        distance = np.linalg.norm(query_embedding - db_embedding)
        matches.append({
            "name": record["name"],
            "distance": round(float(distance), 4)
        })

    matches.sort(key=lambda x: x["distance"])
    return {"matches": matches[:3]}  # return top 3 closest
