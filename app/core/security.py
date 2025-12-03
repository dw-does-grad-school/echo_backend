from jose import jwt
import requests
from fastapi import HTTPException
from app.core.config import settings

JWKS = requests.get(settings.CLERK_JWKS_URL).json()

def verify_clerk_token(token: str):
    try:
        header = jwt.get_unverified_header(token)
        key = next(k for k in JWKS["keys"] if k["kid"] == header["kid"])

        return jwt.decode(
            token,
            key,
            algorithms=["RS256"],
            audience=settings.CLERK_AUDIENCE,
            issuer=settings.CLERK_ISSUER,
        )
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")
