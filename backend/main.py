from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv

from database import engine, Base
from routers import scraper, auth, data, ai, enhanced_ai
from models import User, ScrapingJob, ScrapedData

# Load environment variables
load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create database tables
    Base.metadata.create_all(bind=engine)
    yield

app = FastAPI(
    title="AI-Powered Web Scraper",
    description="A production-ready web scraper with AI integration",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
print(f"CORS allowed origins: {allowed_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "Accept"],
    expose_headers=["Content-Type", "Authorization"],
    max_age=600,  # 10 minutes
)

# Security
security = HTTPBearer()

# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(scraper.router, prefix="/api/scraper", tags=["Scraping"])
app.include_router(data.router, prefix="/api/data", tags=["Data Management"])
app.include_router(ai.router, prefix="/api/ai", tags=["AI Features"])
app.include_router(enhanced_ai.router, tags=["Enhanced AI"])

@app.get("/")
async def root():
    return {
        "message": "AI-Powered Web Scraper API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "web-scraper-api"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )