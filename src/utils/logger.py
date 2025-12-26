
"""
Logging configuration for CodeTransformBench.
Uses loguru for beautiful, structured logging.
"""

import sys
from pathlib import Path
from loguru import logger

# Get log configuration from environment
import os
from dotenv import load_dotenv
load_dotenv()

LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = os.getenv('LOG_FILE', 'codetransform.log')

# Remove default handler
logger.remove()

# Add console handler with colors
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level=LOG_LEVEL,
    colorize=True
)

# Add file handler for persistence
PROJECT_ROOT = Path(__file__).parent.parent.parent
LOG_PATH = PROJECT_ROOT / LOG_FILE

logger.add(
    LOG_PATH,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level=LOG_LEVEL,
    rotation="100 MB",  # Rotate when file reaches 100MB
    retention="30 days",  # Keep logs for 30 days
    compression="gz"  # Compress rotated logs
)

# Export logger instance
__all__ = ['logger']


if __name__ == '__main__':
    # Test logging
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.success("This is a success message")

    print(f"\nLog file location: {LOG_PATH}")
