from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import user as user_routes
from app.api.routes import search as search_routes
from app.services.recommender import load_art_index

app = FastAPI(
    title="EchoGallery Backend",
    version="0.4.0"
)

# CORS so Next.js (localhost:3000) can talk to FastAPI (localhost:8000)
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    # Pre-load the FAISS index + CLIP model
    load_art_index()

@app.get("/")
def index():
    return {"status": "ok", "message": "FastAPI is running!"}

@app.get("/health")
def health():
    return {"status": "healthy"}

app.include_router(user_routes.router, prefix="/api/user")
app.include_router(search_routes.router, prefix="/api/recs")

# Local dev
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)