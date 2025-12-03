from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import List, Any

from app.services.recommender import search_similar_artworks

router = APIRouter()


class SearchRequest(BaseModel):
    query: str = Field(..., description="Text description to search for")
    k: int = Field(5, ge=1, le=20, description="Number of results to return")


class SearchResultItem(BaseModel):
    rank: int
    index: int
    distance: float
    artist: Any
    style: Any
    genre: Any
    image_data: str  # base64 data URL


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResultItem]


@router.post("/text", response_model=SearchResponse)
def search_by_text(payload: SearchRequest):
    results = search_similar_artworks(payload.query, k=payload.k)
    return SearchResponse(query=payload.query, results=results)
