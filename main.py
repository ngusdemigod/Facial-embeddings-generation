from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import face_recognition
import numpy as np
from supabase import create_client
from supabase_config import supabase

app = FastAPI()

@app.post("/generate")
async def generate_embedding(file: UploadFile = File(...)):
    image = face_recognition.load_image_file(await file.read())
    encodings = face_recognition.face_encodings(image)

    if len(encodings) == 0:
        return JSONResponse(content={"error": "No face found"}, status_code=400)

    embedding = encodings[0].tolist()
    supabase.table("face_embeddings").insert({"name": file.filename, "embedding": embedding}).execute()
    return {"message": "Embedding stored successfully."}

@app.post("/compare")
async def compare_face(file: UploadFile = File(...)):
    image = face_recognition.load_image_file(await file.read())
    encodings = face_recognition.face_encodings(image)

    if len(encodings) == 0:
        return JSONResponse(content={"error": "No face found"}, status_code=400)

    input_embedding = encodings[0]

    data = supabase.table("face_embeddings").select("*").execute().data
    results = []

    for record in data:
        db_embedding = np.array(record["embedding"])
        distance = np.linalg.norm(input_embedding - db_embedding)
        results.append({"name": record["name"], "distance": distance})

    # Return closest match
    results.sort(key=lambda x: x["distance"])
    return {"matches": results[:3]}  # return top 3 closest
