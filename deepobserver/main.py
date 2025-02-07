from fastapi import FastAPI
from contextlib import asynccontextmanager
from deepobserver.api.routes import status
from deepobserver.processor import VideoProcessor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

video_processor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global video_processor
    logger.info("Starting video processor")
    video_processor = VideoProcessor(rtsp_url="")
    yield
    # Shutdown

app = FastAPI(
    title="DeepObserver",
    description="Video monitoring service with LLM analysis",
    version="0.0.0",
    lifespan=lifespan
)

# Include routers
app.include_router(status.router)

@app.get("/")
async def root():
    return {"message": "DeepObserver Video Monitoring Service"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("deepobserver.main:app", host="0.0.0.0", port=8000, reload=True)