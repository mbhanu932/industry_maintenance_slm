"""
Simple Authentication Routes
Minimal, practical auth endpoints
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter(prefix="/auth", tags=["authentication"])

class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    user_id: str
    role: str

# Simple in-memory auth for demo
VALID_USERS = {
    "admin": {"password": "admin123", "role": "admin"},
    "engineer": {"password": "eng123", "role": "engineer"},
    "operator": {"password": "op123", "role": "operator"},
}

@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest, req: Request):
    """Simple login endpoint."""
    if request.username not in VALID_USERS:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    user = VALID_USERS[request.username]
    if user["password"] != request.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Simple token (in production use JWT)
    token = f"token_{request.username}"
    
    return LoginResponse(
        access_token=token,
        user_id=request.username,
        role=user["role"]
    )

@router.post("/logout")
async def logout(token: str):
    """Simple logout endpoint."""
    return {"status": "logged_out"}