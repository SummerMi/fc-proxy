"""
FC Proxy Configuration
"""
import os
from dataclasses import dataclass

@dataclass
class Config:
    # MindIE Backend
    BACKEND_URL: str = os.getenv("BACKEND_URL", "http://localhost:1025")
    BACKEND_MODEL: str = os.getenv("BACKEND_MODEL", "deepseek-r1-distill-70b")
    
    # Proxy Server
    PROXY_HOST: str = os.getenv("PROXY_HOST", "0.0.0.0")
    PROXY_PORT: int = int(os.getenv("PROXY_PORT", "1030"))
    
    # Timeouts
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "300"))
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # ReAct Settings
    MAX_ITERATIONS: int = int(os.getenv("MAX_ITERATIONS", "5"))
    STOP_SEQUENCES: list = None
    
    def __post_init__(self):
        if self.STOP_SEQUENCES is None:
            self.STOP_SEQUENCES = ["Observation:", "
Observation"]

config = Config()

