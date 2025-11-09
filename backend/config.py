"""Configuration constants for the backend application."""
import os

# Server configuration
# Railway sets PORT environment variable, fallback to FLASK_PORT or 5001
HOST = os.getenv('FLASK_HOST', '0.0.0.0')  # 0.0.0.0 to accept connections from any IP
PORT = int(os.getenv('PORT', os.getenv('FLASK_PORT', '5001')))

# Image processing configuration
JPEG_QUALITY = 95
MAX_FILE_SIZE = 2* 2048 * 4096  # 10MB - reduced for better performance
MAX_IMAGES_STORED = 10  # Maximum number of images to store in memory
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_IMAGE_DIMENSION = 4096  # Maximum width or height in pixels

# Default values
DEFAULT_ALPHA = 0.5

# HTTP Status codes (using Flask's status codes, but defined here for clarity)
HTTP_BAD_REQUEST = 400
HTTP_NOT_FOUND = 404
HTTP_INTERNAL_SERVER_ERROR = 500

