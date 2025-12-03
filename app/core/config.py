import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    CLERK_JWKS_URL = os.getenv("CLERK_JWKS_URL")
    CLERK_ISSUER = os.getenv("CLERK_ISSUER")
    CLERK_AUDIENCE = os.getenv("CLERK_AUDIENCE")

settings = Settings()
