#!/usr/bin/env python3
"""
Ultra-minimal server to test basic functionality
"""

from fastapi import FastAPI
import uvicorn

# Create minimal FastAPI app
app = FastAPI(title="Test Server")

@app.get("/")
async def root():
    return {"message": "Server is running!", "status": "ok"}

@app.get("/test")
async def test():
    return {"test": "success", "login": "ready"}

if __name__ == "__main__":
    print("Starting ultra-minimal server...")
    try:
        uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
    except Exception as e:
        print(f"Server failed to start: {e}")
        import traceback
        traceback.print_exc()

