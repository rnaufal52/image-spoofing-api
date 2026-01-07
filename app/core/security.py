
import jwt
from typing import Optional, Dict, Any
from app.core.config import settings

def decode_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Decodes a JWT token. Returns the payload or None if invalid.
    """
    try:
        payload = jwt.decode(
            token, 
            settings.SECRET_KEY, 
            algorithms=[settings.ALGORITHM]
        )
        return payload
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None
