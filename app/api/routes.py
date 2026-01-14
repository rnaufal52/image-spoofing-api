from fastapi import APIRouter, UploadFile, File, Request
import traceback
from fastapi.encoders import jsonable_encoder
from starlette.concurrency import run_in_threadpool
from app.services.modelServices import predict, read_image_from_bytes
from app.schemas.prediction import AntiSpoofResponse
from app.core.responses import ResponseTemplate
from app.core.validators import validate_image
from app.core.limiter import limiter

router = APIRouter()

@router.post("/anti-spoof", response_model=None)
@limiter.limit("100/minute")
async def anti_spoof(request: Request, file: UploadFile = File(...)):
    # Validate image
    await validate_image(file)

    try:
        image_bytes = await file.read()
        image = read_image_from_bytes(image_bytes)
        
        result = await run_in_threadpool(predict, image)
        
        return ResponseTemplate.success(
            data=jsonable_encoder(result),
            message="Anti-spoofing analysis completed successfully"
        )

    except ValueError as ve:
        ResponseTemplate.error(message=str(ve), status_code=400)
    except Exception as e:
        with open("error_log.txt", "w") as f:
            f.write(traceback.format_exc())
        ResponseTemplate.error(message=f"Internal Server Error: {str(e)}", status_code=500)
