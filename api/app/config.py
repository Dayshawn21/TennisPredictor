from pydantic import BaseModel
import os
import re
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from api directory (parent of app directory)
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

class Settings(BaseModel):
    database_url: str = ""
    api_title: str = ""
    api_env: str = ""
    
    def __init__(self):
        # Convert postgresql:// to postgresql+asyncpg:// for SQLAlchemy async
        # Remove sslmode and channel_binding params (asyncpg doesn't support them)
        raw_url = os.getenv("DATABASE_URL", "")
        if raw_url:
            # Convert scheme to asyncpg
            url = re.sub(r'^postgresql:', 'postgresql+asyncpg:', raw_url)
            # Remove unsupported query params for asyncpg
            url = re.sub(r'[?&]sslmode=[^&]*', '', url)
            url = re.sub(r'[?&]channel_binding=[^&]*', '', url)
            # Clean up any trailing ? or & after removal
            url = re.sub(r'[?&]$', '', url)
        else:
            url = ""
        
        super().__init__(
            database_url=url,
            api_title=os.getenv("API_TITLE", "Sports AI Bot"),
            api_env=os.getenv("API_ENV", "dev")
        )

settings = Settings()
