import os
import sys
import uuid
import base64
import time
import cv2
import numpy as np
import json
import shutil
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image as PILImage
import io
from typing import Tuple, Dict, List, Optional, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from image_transformation_util import LinearBoundedSector, ImageSectorTransformer
from config import (
    HOST, PORT, JPEG_QUALITY, MAX_FILE_SIZE, ALLOWED_EXTENSIONS,
    DEFAULT_ALPHA, HTTP_BAD_REQUEST, HTTP_NOT_FOUND, HTTP_INTERNAL_SERVER_ERROR,
    MAX_IMAGE_DIMENSION, IMAGES_DIR, MAX_IMAGES_PER_SESSION, SESSION_CLEANUP_AGE_HOURS
)

app = Flask(__name__, static_folder=None)
CORS(app)

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'frontend')

# Session management
session_store: Dict[str, Dict[str, Any]] = {}  # {session_id: {'created_at': timestamp, 'image_ids': [list], 'access_times': {image_id: timestamp}, 'last_access': timestamp}}

# Ensure images directory exists
IMAGES_BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), IMAGES_DIR)
os.makedirs(IMAGES_BASE_DIR, exist_ok=True)

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

# File storage helper functions
def get_session_images_dir(session_id: str) -> str:
    """Get or create session directory for storing images."""
    session_dir = os.path.join(IMAGES_BASE_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)
    return session_dir

def save_image_to_file(session_id: str, image_id: str, image_bgr: np.ndarray, original_format: str = 'jpg') -> str:
    """Save image to file and return file path."""
    session_dir = get_session_images_dir(session_id)
    
    # Determine file extension based on original format
    if original_format.lower() in ['png', 'jpg', 'jpeg', 'gif', 'bmp']:
        ext = original_format.lower() if original_format.lower() != 'jpeg' else 'jpg'
    else:
        ext = 'jpg'  # Default to jpg
    
    image_path = os.path.join(session_dir, f"{image_id}.{ext}")
    
    # Save image
    if ext == 'png':
        cv2.imwrite(image_path, image_bgr)
    else:
        cv2.imwrite(image_path, image_bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    
    return image_path

def load_image_from_file(session_id: str, image_id: str) -> Optional[np.ndarray]:
    """Load image from file."""
    session_dir = get_session_images_dir(session_id)
    
    # Try common image extensions
    for ext in ['jpg', 'jpeg', 'png', 'gif', 'bmp']:
        image_path = os.path.join(session_dir, f"{image_id}.{ext}")
        if os.path.exists(image_path):
            image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image_bgr is not None:
                return image_bgr
    
    return None

def save_image_metadata(session_id: str, image_id: str, width: int, height: int, format: str, uploaded_at: float):
    """Save image metadata to JSON file."""
    session_dir = get_session_images_dir(session_id)
    metadata_path = os.path.join(session_dir, f"{image_id}.json")
    
    metadata = {
        'image_id': image_id,
        'width': width,
        'height': height,
        'format': format,
        'uploaded_at': uploaded_at
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)

def get_image_metadata(session_id: str, image_id: str) -> Optional[Dict[str, Any]]:
    """Load image metadata from JSON file."""
    session_dir = get_session_images_dir(session_id)
    metadata_path = os.path.join(session_dir, f"{image_id}.json")
    
    if not os.path.exists(metadata_path):
        return None
    
    try:
        with open(metadata_path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None

def cleanup_session_images(session_id: str, max_images: int = MAX_IMAGES_PER_SESSION):
    """Remove oldest images from session, keeping only the most recent max_images."""
    if session_id not in session_store:
        return
    
    session_data = session_store[session_id]
    image_ids = session_data.get('image_ids', []).copy()  # Work with copy to avoid modification during iteration
    access_times = session_data.get('access_times', {})
    
    if len(image_ids) <= max_images:
        return
    
    # Sort by access time (oldest first)
    sorted_images = sorted(image_ids, key=lambda img_id: access_times.get(img_id, 0))
    
    # Keep the most recent max_images, remove the rest (oldest ones)
    images_to_keep = set(sorted_images[-max_images:])
    images_to_remove = [img_id for img_id in image_ids if img_id not in images_to_keep]
    
    session_dir = get_session_images_dir(session_id)
    
    for image_id in images_to_remove:
        # Remove image file
        for ext in ['jpg', 'jpeg', 'png', 'gif', 'bmp']:
            image_path = os.path.join(session_dir, f"{image_id}.{ext}")
            if os.path.exists(image_path):
                try:
                    os.remove(image_path)
                except OSError:
                    pass
        
        # Remove metadata file
        metadata_path = os.path.join(session_dir, f"{image_id}.json")
        if os.path.exists(metadata_path):
            try:
                os.remove(metadata_path)
            except OSError:
                pass
        
        # Remove from session store
        if image_id in access_times:
            del access_times[image_id]
    
    # Update session store with remaining images
    session_data['image_ids'] = list(images_to_keep)
    session_data['access_times'] = access_times

def cleanup_old_sessions(max_age_hours: float = SESSION_CLEANUP_AGE_HOURS):
    """Remove sessions older than max_age_hours."""
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    
    sessions_to_remove = []
    for session_id, session_data in session_store.items():
        created_at = session_data.get('created_at', 0)
        last_access = session_data.get('last_access', created_at)
        
        # Use last_access if available, otherwise created_at
        age = current_time - max(created_at, last_access)
        
        if age > max_age_seconds:
            sessions_to_remove.append(session_id)
    
    # Remove old sessions
    for session_id in sessions_to_remove:
        session_dir = get_session_images_dir(session_id)
        try:
            # Remove session directory and all its contents
            if os.path.exists(session_dir):
                shutil.rmtree(session_dir)
        except OSError:
            pass
        
        # Remove from session store
        if session_id in session_store:
            del session_store[session_id]

def get_or_create_session(session_id: Optional[str] = None) -> str:
    """Get existing session or create new one."""
    if session_id and session_id in session_store:
        # Update last access time
        session_store[session_id]['last_access'] = time.time()
        return session_id
    
    # Create new session
    new_session_id = str(uuid.uuid4())
    current_time = time.time()
    session_store[new_session_id] = {
        'created_at': current_time,
        'last_access': current_time,
        'image_ids': [],
        'access_times': {}
    }
    return new_session_id

def get_session_id_from_request() -> Optional[str]:
    """Extract session ID from request (header or JSON body)."""
    # Try header first
    session_id = request.headers.get('X-Session-ID')
    if session_id:
        return session_id
    
    # Try JSON body
    if request.is_json and request.json:
        session_id = request.json.get('session_id')
        if session_id:
            return session_id
    
    # Try query parameter
    session_id = request.args.get('session_id')
    return session_id


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

def validate_image_id(session_id: str, image_id: str, field_name: str = 'image_id') -> Tuple[Optional[np.ndarray], Optional[Tuple[Dict, int]]]:
    """
    Validate image ID and retrieve image from file storage.
    Updates access time in session store.
    
    Returns:
        Tuple of (image, error_response). If error_response is not None, image is None.
    """
    if not image_id or not isinstance(image_id, str):
        return None, error_response(f'Invalid {field_name}', HTTP_BAD_REQUEST)
    
    if not session_id or session_id not in session_store:
        return None, error_response(f'Invalid session', HTTP_BAD_REQUEST)
    
    # Load image from file
    image = load_image_from_file(session_id, image_id)
    if image is None:
        return None, error_response(f'Image not found (ID: {image_id})', HTTP_NOT_FOUND)
    
    # Update access time in session store
    current_time = time.time()
    if session_id in session_store:
        session_data = session_store[session_id]
        if 'access_times' not in session_data:
            session_data['access_times'] = {}
        session_data['access_times'][image_id] = current_time
        session_data['last_access'] = current_time
    
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
        # Get or create session
        session_id = get_session_id_from_request()
        session_id = get_or_create_session(session_id)
        
        # Cleanup old sessions periodically (on upload)
        cleanup_old_sessions()
        
        if 'image' not in request.files:
            return error_response('No image file provided', HTTP_BAD_REQUEST)
        
        file = request.files['image']
        if file.filename == '':
            return error_response('No file selected', HTTP_BAD_REQUEST)
        
        if not allowed_file(file.filename):
            return error_response(f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}', HTTP_BAD_REQUEST)
        
        # Get original file extension
        original_format = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else 'jpg'
        
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
        
        # Generate image ID
        image_id = str(uuid.uuid4())
        current_time = time.time()
        
        # Save image to file
        try:
            save_image_to_file(session_id, image_id, image_bgr, original_format)
        except Exception as e:
            return error_response(f'Failed to save image: {str(e)}', HTTP_INTERNAL_SERVER_ERROR)
        
        # Save metadata
        try:
            save_image_metadata(session_id, image_id, width, height, original_format, current_time)
        except Exception as e:
            # If metadata save fails, remove image file and return error
            try:
                image_path = os.path.join(get_session_images_dir(session_id), f"{image_id}.{original_format}")
                if os.path.exists(image_path):
                    os.remove(image_path)
            except OSError:
                pass
            return error_response(f'Failed to save image metadata: {str(e)}', HTTP_INTERNAL_SERVER_ERROR)
        
        # Update session store
        if session_id in session_store:
            session_data = session_store[session_id]
            if 'image_ids' not in session_data:
                session_data['image_ids'] = []
            if 'access_times' not in session_data:
                session_data['access_times'] = {}
            
            session_data['image_ids'].append(image_id)
            session_data['access_times'][image_id] = current_time
            session_data['last_access'] = current_time
        
        # Cleanup session images (enforce max images per session)
        cleanup_session_images(session_id, MAX_IMAGES_PER_SESSION)
        
        # Encode preview image
        try:
            preview_url = encode_image_base64(image_bgr)
        except Exception as e:
            return error_response(f'Failed to encode preview image: {str(e)}', HTTP_INTERNAL_SERVER_ERROR)
        
        return jsonify({
            'image_id': image_id,
            'width': width,
            'height': height,
            'preview_url': preview_url,
            'session_id': session_id
        })
    
    except Exception as e:
        return error_response(f'Upload failed: {str(e)}', HTTP_INTERNAL_SERVER_ERROR)

@app.route('/api/verify-image/<image_id>', methods=['GET'])
def verify_image(image_id: str):
    """Verify if an image exists in file storage and return metadata."""
    try:
        if not image_id:
            return error_response('Invalid image_id', HTTP_BAD_REQUEST)
        
        # Get session ID from request
        session_id = get_session_id_from_request()
        if not session_id:
            return jsonify({'exists': False}), 200
        
        # Check if session exists
        if session_id not in session_store:
            return jsonify({'exists': False}), 200
        
        # Load metadata
        metadata = get_image_metadata(session_id, image_id)
        if metadata is None:
            return jsonify({'exists': False}), 200
        
        # Update access time in session store
        current_time = time.time()
        session_data = session_store[session_id]
        if 'access_times' not in session_data:
            session_data['access_times'] = {}
        session_data['access_times'][image_id] = current_time
        session_data['last_access'] = current_time
        
        return jsonify({
            'exists': True,
            'width': metadata['width'],
            'height': metadata['height']
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
        
        # Get session ID from request
        session_id = get_session_id_from_request()
        if not session_id:
            return error_response('Session ID required', HTTP_BAD_REQUEST)
        
        if session_id not in session_store:
            return error_response('Invalid session', HTTP_BAD_REQUEST)
        
        source_image_id = data['source_image_id']
        source_sectors_json = data['source_sectors']
        target_sectors_json = data['target_sectors']
        debug_mode = data.get('debug_mode', False)
        
        # Validate and retrieve source image from file storage
        source_image, error_resp = validate_image_id(session_id, source_image_id, 'source_image_id')
        if error_resp:
            return error_resp
        
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
        if data is None:
            return error_response('Invalid JSON data', HTTP_BAD_REQUEST)
        
        error = validate_request_data(data, ['image1_id', 'image2_id', 'sectors1', 'sectors2', 'sector_mapping'])
        if error:
            return error_response(error[0], error[1])
        
        # Get session ID from request
        session_id = get_session_id_from_request()
        if not session_id:
            return error_response('Session ID required', HTTP_BAD_REQUEST)
        
        if session_id not in session_store:
            return error_response('Invalid session', HTTP_BAD_REQUEST)
        
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
        
        # Validate and retrieve images from file storage
        image1, error_resp = validate_image_id(session_id, image1_id, 'image1_id')
        if error_resp:
            return error_resp
        
        image2, error_resp = validate_image_id(session_id, image2_id, 'image2_id')
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
