import os
import sys
import uuid
import base64
import time
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image as PILImage
import io
from typing import Tuple, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from image_transformation_util import LinearBoundedSector, ImageSectorTransformer
from config import (
    HOST, PORT, JPEG_QUALITY, MAX_FILE_SIZE, ALLOWED_EXTENSIONS,
    DEFAULT_ALPHA, HTTP_BAD_REQUEST, HTTP_NOT_FOUND, HTTP_INTERNAL_SERVER_ERROR,
    MAX_IMAGES_STORED, MAX_IMAGE_DIMENSION
)

app = Flask(__name__, static_folder=None)
CORS(app)

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'frontend')

image_store: Dict[str, np.ndarray] = {}
image_store_access_times: Dict[str, float] = {}  # Track last access time for cleanup

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def encode_image_base64(image_bgr: np.ndarray) -> str:
    """Convert BGR image to base64 RGB string for web display."""
    if len(image_bgr.shape) == 3 and image_bgr.shape[2] == 3:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(image_rgb)
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=JPEG_QUALITY)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:image/jpeg;base64,{image_base64}"
    else:
        _, buffer = cv2.imencode('.jpg', image_bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{image_base64}"

def decode_image_base64(image_base64: str) -> np.ndarray:
    """Decode base64 image to BGR numpy array."""
    if ',' in image_base64:
        image_base64 = image_base64.split(',')[1]
    image_bytes = base64.b64decode(image_base64)
    nparr = np.frombuffer(image_bytes, np.uint8)
    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Failed to decode image")
    return image_bgr

def validate_image_dimensions(width: int, height: int) -> bool:
    """Validate that image dimensions are valid."""
    if width <= 0 or height <= 0:
        return False
    if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
        return False
    return True

def cleanup_old_images():
    """Remove oldest images if we exceed the storage limit."""
    if len(image_store) <= MAX_IMAGES_STORED:
        return
    
    # Sort by access time and remove oldest
    sorted_images = sorted(image_store_access_times.items(), key=lambda x: x[1])
    images_to_remove = len(image_store) - MAX_IMAGES_STORED
    
    for image_id, _ in sorted_images[:images_to_remove]:
        if image_id in image_store:
            del image_store[image_id]
        if image_id in image_store_access_times:
            del image_store_access_times[image_id]

def get_image(image_id: str) -> Optional[np.ndarray]:
    """Get image from store and update access time."""
    if image_id not in image_store:
        return None
    image_store_access_times[image_id] = time.time()
    return image_store[image_id]

def validate_point(x: float, y: float, name: str, width: int, height: int) -> Tuple[int, int]:
    """Validate that a point is within image bounds."""
    try:
        x = int(x)
        y = int(y)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid coordinate value in {name}: {e}")
    
    if x < 0 or x >= width or y < 0 or y >= height:
        raise ValueError(f"{name} coordinates ({x}, {y}) are out of bounds for image ({width}x{height})")
    return (x, y)

def json_to_sector(sector_json: Dict, width: int, height: int) -> LinearBoundedSector:
    """Convert JSON sector data to LinearBoundedSector instance."""
    try:
        center = validate_point(
            sector_json['center']['x'], 
            sector_json['center']['y'], 
            'Center',
            width,
            height
        )
        edge_point1 = validate_point(
            sector_json['edge_point1']['x'], 
            sector_json['edge_point1']['y'], 
            'Edge point 1',
            width,
            height
        )
        edge_point2 = validate_point(
            sector_json['edge_point2']['x'], 
            sector_json['edge_point2']['y'], 
            'Edge point 2',
            width,
            height
        )
    except KeyError as e:
        raise ValueError(f"Missing required field in sector data: {e}")
    
    return LinearBoundedSector(
        center=center,
        edge_point1=edge_point1,
        edge_point2=edge_point2,
        bound_width=width,
        bound_height=height
    )

def validate_request_data(data: Dict, required_fields: List[str]) -> Optional[Tuple[str, int]]:
    """Validate that all required fields are present in request data."""
    for field in required_fields:
        if field not in data:
            return (f'{field} is required', HTTP_BAD_REQUEST)
    return None

def error_response(message: str, status_code: int) -> Tuple[Dict, int]:
    """Create a standardized error response."""
    return jsonify({'error': message}), status_code

@app.route('/api/upload', methods=['POST'])
def upload_image():
    """Upload an image file and return image metadata."""
    try:
        if 'image' not in request.files:
            return error_response('No image file provided', HTTP_BAD_REQUEST)
        
        file = request.files['image']
        if file.filename == '':
            return error_response('No file selected', HTTP_BAD_REQUEST)
        
        if not allowed_file(file.filename):
            return error_response(f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}', HTTP_BAD_REQUEST)
        
        # Check file size before reading
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > MAX_FILE_SIZE:
            max_mb = MAX_FILE_SIZE / (1024 * 1024)
            return error_response(f'File too large. Maximum size: {max_mb:.1f}MB', HTTP_BAD_REQUEST)
        
        if file_size == 0:
            return error_response('File is empty', HTTP_BAD_REQUEST)
        
        file_bytes = file.read()
        if len(file_bytes) != file_size:
            return error_response('Failed to read file completely', HTTP_BAD_REQUEST)
        
        nparr = np.frombuffer(file_bytes, np.uint8)
        image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image_bgr is None:
            return error_response('Failed to decode image. Please ensure the file is a valid image.', HTTP_BAD_REQUEST)
        
        height, width = image_bgr.shape[:2]
        
        if not validate_image_dimensions(width, height):
            return error_response(f'Invalid image dimensions. Maximum: {MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION} pixels', HTTP_BAD_REQUEST)
        
        # Cleanup old images before adding new one
        cleanup_old_images()
        
        image_id = str(uuid.uuid4())
        image_store[image_id] = image_bgr
        image_store_access_times[image_id] = time.time()
        
        return jsonify({
            'image_id': image_id,
            'width': width,
            'height': height,
            'preview_url': encode_image_base64(image_bgr)
        })
    
    except Exception as e:
        return error_response(f'Upload failed: {str(e)}', HTTP_INTERNAL_SERVER_ERROR)

@app.route('/api/warp', methods=['POST'])
def warp_image():
    """Warp an image using source and target sectors."""
    try:
        data = request.get_json()
        
        error = validate_request_data(data, ['source_image_id', 'source_sectors', 'target_sectors'])
        if error:
            return error_response(error[0], error[1])
        
        source_image_id = data['source_image_id']
        source_sectors_json = data['source_sectors']
        target_sectors_json = data['target_sectors']
        debug_mode = data.get('debug_mode', False)
        
        if not source_image_id or not isinstance(source_image_id, str):
            return error_response('Invalid source_image_id', HTTP_BAD_REQUEST)
        
        source_image = get_image(source_image_id)
        if source_image is None:
            return error_response(f'Source image not found (ID: {source_image_id})', HTTP_NOT_FOUND)
        height, width = source_image.shape[:2]
        
        if not validate_image_dimensions(width, height):
            return error_response('Invalid image dimensions', HTTP_BAD_REQUEST)
        
        if len(source_sectors_json) == 0:
            return error_response('At least one sector is required', HTTP_BAD_REQUEST)
        if len(source_sectors_json) != len(target_sectors_json):
            return error_response('Source and target must have same number of sectors', HTTP_BAD_REQUEST)
        
        source_sectors = [json_to_sector(s, width, height) for s in source_sectors_json]
        target_sectors = [json_to_sector(s, width, height) for s in target_sectors_json]
        
        warped_image = ImageSectorTransformer.arbitrary_sector_warping(
            src_image=source_image,
            source_sectors=source_sectors,
            target_sectors=target_sectors,
            debug=debug_mode
        )
        
        result_base64 = encode_image_base64(warped_image)
        
        return jsonify({
            'result_image': result_base64,
            'debug_info': {'num_sectors': len(source_sectors), 'debug_mode': debug_mode} if debug_mode else None
        })
    
    except Exception as e:
        return error_response(str(e), HTTP_INTERNAL_SERVER_ERROR)

@app.route('/api/mixup', methods=['POST'])
def mixup_images():
    """Mix two images using sector mappings."""
    try:
        data = request.get_json()
        
        error = validate_request_data(data, ['image1_id', 'image2_id', 'sectors1', 'sectors2', 'sector_mapping'])
        if error:
            return error_response(error[0], error[1])
        
        image1_id = data['image1_id']
        image2_id = data['image2_id']
        sectors1_json = data['sectors1']
        sectors2_json = data['sectors2']
        sector_mapping = data['sector_mapping']
        debug_mode = data.get('debug_mode', False)
        alpha = data.get('alpha', DEFAULT_ALPHA)
        
        if not image1_id or not isinstance(image1_id, str):
            return error_response('Invalid image1_id', HTTP_BAD_REQUEST)
        if not image2_id or not isinstance(image2_id, str):
            return error_response('Invalid image2_id', HTTP_BAD_REQUEST)
        
        image1 = get_image(image1_id)
        if image1 is None:
            return error_response(f'Image 1 not found (ID: {image1_id})', HTTP_NOT_FOUND)
        
        image2 = get_image(image2_id)
        if image2 is None:
            return error_response(f'Image 2 not found (ID: {image2_id})', HTTP_NOT_FOUND)
        height1, width1 = image1.shape[:2]
        height2, width2 = image2.shape[:2]
        
        if not validate_image_dimensions(width1, height1):
            return error_response('Invalid dimensions for image 1', HTTP_BAD_REQUEST)
        if not validate_image_dimensions(width2, height2):
            return error_response('Invalid dimensions for image 2', HTTP_BAD_REQUEST)
        
        if not isinstance(alpha, (int, float)) or alpha < 0 or alpha > 1:
            return error_response('alpha must be a number between 0 and 1', HTTP_BAD_REQUEST)
        
        if len(sectors1_json) == 0:
            return error_response('At least one sector is required for image 1', HTTP_BAD_REQUEST)
        if len(sectors2_json) == 0:
            return error_response('At least one sector is required for image 2', HTTP_BAD_REQUEST)
        
        if len(sector_mapping) == 0:
            return error_response('At least one sector mapping is required', HTTP_BAD_REQUEST)
        
        sectors1 = [json_to_sector(s, width1, height1) for s in sectors1_json]
        sectors2 = [json_to_sector(s, width2, height2) for s in sectors2_json]
        
        result_image = image1.copy().astype(np.uint8)
        
        mixup_info = []
        for mapping in sector_mapping:
            src_idx = mapping.get('src_index')
            dst_idx = mapping.get('dst_index')
            
            if src_idx is None or dst_idx is None:
                continue
            
            if src_idx < 0 or src_idx >= len(sectors1):
                continue
            if dst_idx < 0 or dst_idx >= len(sectors2):
                continue
            
            src_sector = sectors1[src_idx]
            dst_sector = sectors2[dst_idx]
            
            result_image = ImageSectorTransformer.sector_mixup(
                src_image=result_image,
                src_sector=src_sector,
                dst_image=image2,
                dst_sector=dst_sector,
                alpha=alpha
            )
            
            mixup_info.append({
                'src_index': src_idx,
                'dst_index': dst_idx
            })
        
        result_base64 = encode_image_base64(result_image)
        
        return jsonify({
            'result_image': result_base64,
            'debug_info': {'mixup_mappings': mixup_info, 'debug_mode': debug_mode} if debug_mode else None
        })
    
    except Exception as e:
        return error_response(str(e), HTTP_INTERNAL_SERVER_ERROR)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok'})

@app.route('/favicon.ico')
def favicon():
    """Handle favicon requests to prevent 404 errors."""
    return '', 204  # No Content

@app.route('/')
def index():
    return send_from_directory(FRONTEND_DIR, 'index.html')

@app.route('/components/<path:filename>')
def components(filename):
    return send_from_directory(os.path.join(FRONTEND_DIR, 'components'), filename)

@app.route('/<path:filename>')
def frontend_static(filename):
    if filename.endswith(('.html', '.css', '.js', '.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico')):
        return send_from_directory(FRONTEND_DIR, filename)
    return error_response('Not found', HTTP_NOT_FOUND)

if __name__ == '__main__':
    # Disable debug mode in production (Railway sets FLASK_DEBUG=false)
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(debug=debug_mode, port=PORT, host=HOST)
