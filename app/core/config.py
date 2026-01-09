
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Image Spoofing API"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api/v1"
    MODEL_PATH: str = "models/2.7_80x80_MiniFASNetV2.pth"
    SECRET_KEY: str = "CHANGE_THIS_SECRET_KEY"
    ALGORITHM: str = "HS256"
    PORT: int = 8000

    class Config:
        case_sensitive = True
        env_file = ".env"
        extra = "ignore"

settings = Settings()
