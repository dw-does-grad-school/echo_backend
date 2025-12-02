from fastapi import FastAPI

app = FastAPI(title="EchoGallery Backend", version="0.1.0")


@app.get("/")
async def read_root():
    return {"status": "ok", "message": "FastAPI is running!"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# For local debugging only: `python main.py`
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


