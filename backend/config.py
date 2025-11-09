"""Configuration constants for the backend application."""
import os

# Server configuration
HOST = os.getenv('FLASK_HOST', '127.0.0.1')
PORT = int(os.getenv('FLASK_PORT', '5001'))

# Image processing configuration
JPEG_QUALITY = 95
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Default values
DEFAULT_ALPHA = 0.5

# HTTP Status codes (using Flask's status codes, but defined here for clarity)
HTTP_BAD_REQUEST = 400
HTTP_NOT_FOUND = 404
HTTP_INTERNAL_SERVER_ERROR = 500

