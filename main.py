"""FastAPI application entry point.

Run with:
    uvicorn main:app --reload
"""
from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from routers.compare import router as compare_router

app = FastAPI(
    title="Drawing Difference Detection API",
    description="Classical CV pipeline that compares two drawing revisions and returns a visual diff + AI summary.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static/results", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(compare_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
