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
image_reservations: Dict[str, float] = {}  # Track images reserved for operations (image_id -> reservation_time)
RESERVATION_TIMEOUT = 300.0  # 5 minutes - how long an image can be reserved

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

def cleanup_old_images(exclude_image_id: Optional[str] = None, min_age_seconds: float = 10.0):
    """
    Remove oldest images if we exceed the storage limit.
    
    Protection rules:
    - Only cleanup if there are more than 5 images AND we exceed MAX_IMAGES_STORED
    - Always protect the last 3 most recently accessed images
    - Protect images accessed within min_age_seconds
    - NEVER cleanup reserved images (images in use for operations)
    - Never cleanup the excluded image (newly uploaded)
    
    Args:
        exclude_image_id: Image ID to exclude from cleanup (e.g., newly uploaded image)
        min_age_seconds: Minimum age in seconds before an image can be cleaned up.
                         This protects recently accessed images from premature removal.
    """
    # Only cleanup if we have more than 5 images AND exceed the storage limit
    if len(image_store) <= 5:
        return
    
    # Don't cleanup if we're at or below the storage limit
    if len(image_store) <= MAX_IMAGES_STORED:
        return
    
    current_time = time.time()
    images_to_remove = len(image_store) - MAX_IMAGES_STORED
    removed_count = 0
    
    # Sort by access time (most recent last)
    sorted_images = sorted(image_store_access_times.items(), key=lambda x: x[1])
    
    # Always protect the last 3 most recently accessed images
    # (these are likely to be in active use, especially in mixup mode)
    protected_count = min(3, len(sorted_images))
    protected_images = set()
    if protected_count > 0:
        # Get the last N images (most recent)
        for image_id, _ in sorted_images[-protected_count:]:
            protected_images.add(image_id)
    
    # Also protect the excluded image
    if exclude_image_id:
        protected_images.add(exclude_image_id)
    
    # Clean up expired reservations
    expired_reservations = [img_id for img_id, res_time in image_reservations.items() 
                           if current_time - res_time > RESERVATION_TIMEOUT]
    for img_id in expired_reservations:
        del image_reservations[img_id]
    
    # NEVER cleanup reserved images (images in use for operations)
    for reserved_id in image_reservations.keys():
        protected_images.add(reserved_id)
    
    # Remove oldest images, but skip protected ones
    for image_id, access_time in sorted_images:
        # Skip protected images (last 3 most recent + excluded + reserved)
        if image_id in protected_images:
            continue
        
        # Protect recently accessed images (accessed within min_age_seconds)
        age = current_time - access_time
        if age < min_age_seconds:
            continue
        
        if image_id in image_store:
            del image_store[image_id]
        if image_id in image_store_access_times:
            del image_store_access_times[image_id]
        
        removed_count += 1
        if removed_count >= images_to_remove:
            break
    
    # If we still have too many images after protecting recent ones,
    # we need to be more aggressive (but still protect last 3, excluded, and reserved)
    if len(image_store) > MAX_IMAGES_STORED:
        # Only remove very old images (older than 60 seconds) if we're still over limit
        sorted_images = sorted(image_store_access_times.items(), key=lambda x: x[1])
        for image_id, access_time in sorted_images:
            # Still protect the last 3, excluded, and reserved images
            if image_id in protected_images:
                continue
            age = current_time - access_time
            # Only remove images that are definitely old and not in use
            if age >= 60.0:
                if image_id in image_store:
                    del image_store[image_id]
                if image_id in image_store_access_times:
                    del image_store_access_times[image_id]
                if len(image_store) <= MAX_IMAGES_STORED:
                    break

def get_image(image_id: str) -> Optional[np.ndarray]:
    """Get image from store and update access time."""
    if image_id not in image_store:
        return None
    # Update access time to prevent cleanup of recently accessed images
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

def validate_image_id(image_id: str, field_name: str = 'image_id') -> Tuple[Optional[np.ndarray], Optional[Tuple[Dict, int]]]:
    """
    Validate image ID and retrieve image from store.
    Updates access time to protect the image from cleanup.
    
    Returns:
        Tuple of (image, error_response). If error_response is not None, image is None.
    """
    if not image_id or not isinstance(image_id, str):
        return None, error_response(f'Invalid {field_name}', HTTP_BAD_REQUEST)
    
    # Update access time immediately to protect from cleanup
    if image_id in image_store:
        image_store_access_times[image_id] = time.time()
    
    image = get_image(image_id)
    if image is None:
        return None, error_response(f'Image not found (ID: {image_id})', HTTP_NOT_FOUND)
    
    height, width = image.shape[:2]
    if not validate_image_dimensions(width, height):
        return None, error_response(f'Invalid dimensions for image (ID: {image_id})', HTTP_BAD_REQUEST)
    
    return image, None

def validate_sectors(sectors_json: List[Dict], image_width: int, image_height: int, 
                     field_name: str = 'sectors', min_count: int = 1) -> Tuple[Optional[List[LinearBoundedSector]], Optional[Tuple[Dict, int]]]:
    """
    Validate and convert JSON sectors to LinearBoundedSector instances.
    
    Returns:
        Tuple of (sectors_list, error_response). If error_response is not None, sectors_list is None.
    """
    if not isinstance(sectors_json, list):
        return None, error_response(f'Invalid {field_name} format', HTTP_BAD_REQUEST)
    
    if len(sectors_json) < min_count:
        return None, error_response(f'At least {min_count} sector(s) required for {field_name}', HTTP_BAD_REQUEST)
    
    try:
        sectors = [json_to_sector(s, image_width, image_height) for s in sectors_json]
        return sectors, None
    except (ValueError, KeyError) as e:
        return None, error_response(f'Invalid {field_name} data: {str(e)}', HTTP_BAD_REQUEST)

def error_response(message: str, status_code: int) -> Tuple[Dict, int]:
    """Create a standardized error response."""
    return jsonify({'error': message}), status_code

def validation_error(field: str, message: str = None) -> Tuple[Dict, int]:
    """Create a validation error response."""
    error_msg = message or f'Invalid {field}'
    return error_response(error_msg, HTTP_BAD_REQUEST)

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
        
        # Generate image ID first
        image_id = str(uuid.uuid4())
        
        # Store the new image FIRST with current timestamp
        # This ensures the image is immediately available for verification
        current_time = time.time()
        image_store[image_id] = image_bgr
        image_store_access_times[image_id] = current_time
        
        # Verify the image was stored correctly
        if image_id not in image_store:
            return error_response('Failed to store image', HTTP_INTERNAL_SERVER_ERROR)
        
        # Only cleanup if we're over the limit AFTER storing the new image
        # This prevents premature cleanup and protects recently uploaded images
        if len(image_store) > MAX_IMAGES_STORED:
            # Use longer min_age to protect recently accessed images
            # This ensures images that are still in use won't be removed
            cleanup_old_images(exclude_image_id=image_id, min_age_seconds=10.0)
        
        # Encode preview image AFTER cleanup (this can be slow for large images)
        # The image is already stored, so it's safe even if encoding is slow
        try:
            preview_url = encode_image_base64(image_bgr)
        except Exception as e:
            # If encoding fails, still return the image_id so the image can be used
            # The frontend can request the image directly if needed
            return error_response(f'Failed to encode preview image: {str(e)}', HTTP_INTERNAL_SERVER_ERROR)
        
        return jsonify({
            'image_id': image_id,
            'width': width,
            'height': height,
            'preview_url': preview_url
        })
    
    except Exception as e:
        return error_response(f'Upload failed: {str(e)}', HTTP_INTERNAL_SERVER_ERROR)

@app.route('/api/verify-image/<image_id>', methods=['GET'])
def verify_image(image_id: str):
    """Verify if an image exists in the store and update access time."""
    try:
        if not image_id:
            return error_response('Invalid image_id', HTTP_BAD_REQUEST)
        
        # Check if image exists in store
        if image_id not in image_store:
            return jsonify({'exists': False}), 200
        
        # Update access time to protect from cleanup
        # This is especially important for mixup operations where images
        # are verified before being used
        current_time = time.time()
        image_store_access_times[image_id] = current_time
        
        # Reserve the image to prevent cleanup between verification and operation
        # This prevents race conditions where cleanup runs between verify and warp/mixup
        image_reservations[image_id] = current_time
        
        image = image_store[image_id]
        height, width = image.shape[:2]
        
        return jsonify({
            'exists': True,
            'width': width,
            'height': height
        }), 200
    except Exception as e:
        return error_response(f'Verification failed: {str(e)}', HTTP_INTERNAL_SERVER_ERROR)

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
        
        # Validate and retrieve source image
        source_image, error_resp = validate_image_id(source_image_id, 'source_image_id')
        if error_resp:
            return error_resp
        
        # Ensure image is reserved (in case it wasn't verified first)
        current_time = time.time()
        image_reservations[source_image_id] = current_time
        image_store_access_times[source_image_id] = current_time
        
        height, width = source_image.shape[:2]
        
        # Validate sectors
        if len(source_sectors_json) != len(target_sectors_json):
            return error_response('Source and target must have same number of sectors', HTTP_BAD_REQUEST)
        
        source_sectors, error_resp = validate_sectors(source_sectors_json, width, height, 'source_sectors')
        if error_resp:
            return error_resp
        
        target_sectors, error_resp = validate_sectors(target_sectors_json, width, height, 'target_sectors')
        if error_resp:
            return error_resp
        
        warped_image = ImageSectorTransformer.arbitrary_sector_warping(
            src_image=source_image,
            source_sectors=source_sectors,
            target_sectors=target_sectors,
            debug=debug_mode
        )
        
        result_base64 = encode_image_base64(warped_image)
        
        # Release reservation after operation completes
        if source_image_id in image_reservations:
            del image_reservations[source_image_id]
        
        return jsonify({
            'result_image': result_base64,
            'debug_info': {'num_sectors': len(source_sectors), 'debug_mode': debug_mode} if debug_mode else None
        })
    
    except Exception as e:
        # Release reservation on error
        if source_image_id in image_reservations:
            del image_reservations[source_image_id]
        return error_response(str(e), HTTP_INTERNAL_SERVER_ERROR)

@app.route('/api/mixup', methods=['POST'])
def mixup_images():
    """Mix two images using sector mappings."""
    try:
        data = request.get_json()
        if data is None:
            return error_response('Invalid JSON data', HTTP_BAD_REQUEST)
        
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
        
        # Validate alpha parameter
        if not isinstance(alpha, (int, float)) or alpha < 0 or alpha > 1:
            return validation_error('alpha', 'alpha must be a number between 0 and 1')
        
        # Validate sector mapping
        if not isinstance(sector_mapping, list) or len(sector_mapping) == 0:
            return error_response('At least one sector mapping is required', HTTP_BAD_REQUEST)
        
        # Protect both images from cleanup IMMEDIATELY before validation
        # This prevents race conditions where images are cleaned up between verification and mixup
        current_time = time.time()
        
        # Check if images exist before validation and protect them
        if image1_id not in image_store:
            return error_response(f'Image 1 not found (ID: {image1_id}). It may have been removed. Please upload it again.', HTTP_NOT_FOUND)
        if image2_id not in image_store:
            return error_response(f'Image 2 not found (ID: {image2_id}). It may have been removed. Please upload it again.', HTTP_NOT_FOUND)
        
        # Reserve both images to prevent cleanup (even if they weren't verified first)
        image_reservations[image1_id] = current_time
        image_reservations[image2_id] = current_time
        image_store_access_times[image1_id] = current_time
        image_store_access_times[image2_id] = current_time
        
        # Validate and retrieve images
        image1, error_resp = validate_image_id(image1_id, 'image1_id')
        if error_resp:
            return error_resp
        
        image2, error_resp = validate_image_id(image2_id, 'image2_id')
        if error_resp:
            return error_resp
        
        height1, width1 = image1.shape[:2]
        height2, width2 = image2.shape[:2]
        
        # Validate sectors
        sectors1, error_resp = validate_sectors(sectors1_json, width1, height1, 'sectors1')
        if error_resp:
            return error_resp
        
        sectors2, error_resp = validate_sectors(sectors2_json, width2, height2, 'sectors2')
        if error_resp:
            return error_resp
        
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
        
        # Release reservations after operation completes
        if image1_id in image_reservations:
            del image_reservations[image1_id]
        if image2_id in image_reservations:
            del image_reservations[image2_id]
        
        return jsonify({
            'result_image': result_base64,
            'debug_info': {'mixup_mappings': mixup_info, 'debug_mode': debug_mode} if debug_mode else None
        })
    
    except Exception as e:
        # Release reservations on error
        if image1_id in image_reservations:
            del image_reservations[image1_id]
        if image2_id in image_reservations:
            del image_reservations[image2_id]
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
