from slowapi import Limiter
from slowapi.util import get_remote_address
from fastapi import Request

def get_real_ip(request: Request) -> str:
    """
    Get the real IP address of the client, trusting X-Forwarded-For or X-Real-IP headers.
    This is necessary because requests come via Laravel (Proxy).
    """
    # check X-Forwarded-For
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        # X-Forwarded-For can be a list "client, proxy1, proxy2"
        return forwarded.split(",")[0].strip()
    
    # Check X-Real-IP
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip

    # Fallback to direct connection IP
    return get_remote_address(request)

# Shared Limiter instance using the custom key function
limiter = Limiter(key_func=get_real_ip)
