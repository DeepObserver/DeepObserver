from fastapi import APIRouter

router = APIRouter(
    prefix="/status",
    tags=["status"]
)

@router.get("/")
async def get_status():
    return {
        "status": "running",
        "service": "deepobserver"
    }