"""
serve.py
========
Start the BoneMind FastAPI server.

Usage:
    python serve.py

Opens automatically at http://localhost:8000
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
