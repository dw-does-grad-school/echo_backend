from fastapi import APIRouter

router = APIRouter()

@router.get("/me-test")
def test():
    return {"ok": True}
