from fastapi.responses import JSONResponse
from fastapi import status, HTTPException
from typing import Any, Optional

class ResponseTemplate:
    @staticmethod
    def success(
        data: Any = None,
        message: str = "Success",
        status_code: int = status.HTTP_200_OK
    ) -> JSONResponse:
        content = {
            "success": True,
            "status_code": status_code,
            "message": message,
            "data": data
        }
        return JSONResponse(
            content=content,
            status_code=status_code
        )

    @staticmethod
    def error(
        message: str,
        status_code: int = status.HTTP_400_BAD_REQUEST,
        code: Optional[str] = None
    ):
        # We rely on the global exception handler to format this exception
        raise HTTPException(
            status_code=status_code,
            detail=message
        )
