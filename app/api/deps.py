from fastapi.security import HTTPBearer
from fastapi import Depends
from app.core.security import verify_clerk_token

auth_scheme = HTTPBearer()

async def get_current_user(credentials=Depends(auth_scheme)):
    token = credentials.credentials
    return verify_clerk_token(token)
