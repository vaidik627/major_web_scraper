#!/usr/bin/env python3
"""
Minimal test server to debug login issues
"""

from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
from database import get_db, engine, Base
from models import User
from routers import auth
import os

# Set environment variables
os.environ['SECRET_KEY'] = 'your-super-secret-key-change-in-production-12345'
os.environ['DATABASE_URL'] = 'sqlite:///./scraper.db'
os.environ['ALLOWED_ORIGINS'] = 'http://localhost:3000,http://127.0.0.1:3000'
os.environ['ACCESS_TOKEN_EXPIRE_MINUTES'] = '30'
os.environ['ALGORITHM'] = 'HS256'

# Create database tables
Base.metadata.create_all(bind=engine)

# Create minimal FastAPI app
app = FastAPI(title="AI Web Scraper - Minimal", version="1.0.0")

# Include only auth router
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])

@app.get("/")
async def root():
    return {"message": "AI-Powered Web Scraper API", "status": "running"}

@app.get("/health")
async def health_check(db: Session = Depends(get_db)):
    try:
        # Test database connection
        user_count = db.query(User).count()
        return {
            "status": "healthy",
            "database": "connected",
            "users": user_count,
            "message": "All systems operational"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "database": "error",
            "error": str(e),
            "message": "Database connection failed"
        }

if __name__ == "__main__":
    import uvicorn
    print("Starting minimal server for debugging...")
    uvicorn.run(app, host="127.0.0.1", port=8000)

