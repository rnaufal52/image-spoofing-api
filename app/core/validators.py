from fastapi import UploadFile
from fastapi.exceptions import RequestValidationError

async def validate_image(file: UploadFile):
    """
    Validates if the uploaded file is a valid image and not empty.
    Raises RequestValidationError (422) if invalid, to be caught by global handler.
    """
    errors = []

    # Check existence (though File(...) usually handles this, UploadFile object might be present but empty)
    if not file:
         errors.append({"loc": ("body", "file"), "msg": "No file uploaded", "type": "value_error.missing"})

    # Check content type
    if file.content_type and not file.content_type.startswith("image/"):
        errors.append({"loc": ("body", "file"), "msg": f"File must be an image. Got {file.content_type}", "type": "value_error.content_type"})
    
    # Check filename
    if not file.filename:
        errors.append({"loc": ("body", "file"), "msg": "File must have a name", "type": "value_error.filename"})
        
    # Check file size
    await file.seek(0)
    data = await file.read(1)
    if not data:
         errors.append({"loc": ("body", "file"), "msg": "File cannot be empty", "type": "value_error.size"})
    await file.seek(0)

    if errors:
        raise RequestValidationError(errors)

    return True
