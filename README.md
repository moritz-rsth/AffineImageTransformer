# Affine Image Transformer - Web Demo

Interactive web application for experimenting with the Affine Image Transformer library. This demo provides a user-friendly interface for warping and mixing image sectors.

## Features

- **Warping Mode**: Define source and target sectors to warp images interactively
- **Mixup Mode**: Mix sectors from two different images with alpha blending
- **Interactive Sector Editing**: Drag and drop sector boundary points
- **Real-time Preview**: See transformations applied instantly
- **Download Results**: Save transformed images directly from the browser

## Installation

### Prerequisites

- Python 3.7+
- Web browser (Chrome, Firefox, Safari, or Edge)

### Setup

1. Install Python dependencies:

```bash
pip install -r requirements.txt
```

2. Start the Flask backend server:

```bash
cd backend
python app.py
```

The server will start on `http://localhost:5001` (default).

### Configuration

You can configure the server using environment variables:

```bash
export FLASK_HOST=127.0.0.1
export FLASK_PORT=5001
python backend/app.py
```

## Usage

### Starting the Application

1. Start the backend server (see Installation above)
2. Open your web browser and navigate to `http://localhost:5001`
3. The web interface will load automatically

### Warping Mode

1. **Upload Image**: Click "Upload Image" to select a source image
2. **Define Source Sectors**: 
   - The image loads with 3 default sectors (minimum required)
   - Drag the center point to move all sectors
   - Drag boundary points (numbered circles) to adjust sector boundaries
   - Click "Add Sector" to create additional sectors
3. **Define Target Sectors**: 
   - Adjust sectors on the target canvas to define the desired output layout
   - Source and target must have the same number of sectors
4. **Apply Warp**: Click "Apply Warp" to generate the transformed result
5. **Download**: Click "Download Result" to save the warped image

### Mixup Mode

1. **Upload Images**: 
   - Click "Upload Source Image" to upload the first image
   - Click "Upload Mixin Image" to upload the second image
2. **Define Sectors**: 
   - Both images start with 3 default sectors
   - Adjust sectors on both images as needed
   - Click "Add Sector" to add sectors to both images simultaneously
3. **Create Mappings**: 
   - Use the dropdown menus to map source sectors to mixin sectors
   - Each source sector can be mapped to a mixin sector
4. **Adjust Settings**:
   - Use the "Mixup Alpha" slider to control mixing intensity (0.0 = only source, 1.0 = only mixin)
   - Use the "Highlight Opacity" slider to adjust sector highlighting visibility
5. **Generate Mixup**: Click "Generate Mixup" to create the mixed result
6. **Download**: Click "Download Result" to save the mixed image

### Sector Interaction

- **Drag Center Point**: Move all sectors together (black circle)
- **Drag Boundary Points**: Adjust individual sector boundaries (numbered white circles)
- **Right-Click Boundary Point**: Delete a sector (minimum 3 sectors required)
- **Add Sector**: Click "Add Sector" button to create a new sector
- **Clear**: Click "Clear" to reset all sectors and images

### Sector Numbering

- Each boundary line has a permanent number assigned when created
- Sectors are defined clockwise from each boundary line
- The number on a boundary point indicates the sector number (clockwise from that line)
- Users are responsible for maintaining correct sector order

## API Endpoints

### POST `/api/upload`

Upload an image file.

**Request**: `multipart/form-data` with `image` file  
**Response**: 
```json
{
  "image_id": "uuid-string",
  "width": 1920,
  "height": 1080,
  "preview_url": "data:image/jpeg;base64,..."
}
```

### POST `/api/warp`

Warp an image using source and target sectors.

**Request**: 
```json
{
  "source_image_id": "uuid-string",
  "source_sectors": [
    {
      "id": "sector-1",
      "center": {"x": 320, "y": 240},
      "edge_point1": {"x": 640, "y": 0},
      "edge_point2": {"x": 640, "y": 160}
    }
  ],
  "target_sectors": [...],
  "debug_mode": false
}
```

**Response**: 
```json
{
  "result_image": "data:image/jpeg;base64,...",
  "debug_info": null
}
```

### POST `/api/mixup`

Mix two images using sector mappings.

**Request**:
```json
{
  "image1_id": "uuid-string",
  "image2_id": "uuid-string",
  "sectors1": [...],
  "sectors2": [...],
  "sector_mapping": [
    {"src_index": 0, "dst_index": 1}
  ],
  "alpha": 0.5,
  "debug_mode": false
}
```

**Response**: 
```json
{
  "result_image": "data:image/jpeg;base64,...",
  "debug_info": null
}
```

### GET `/health`

Health check endpoint.

**Response**: 
```json
{
  "status": "ok"
}
```

## File Structure

```
web_demo/
  backend/
    app.py              # Flask application and API endpoints
    config.py           # Configuration constants
  frontend/
    index.html          # Main HTML structure
    styles.css          # Styling and layout
    app.js              # Main application logic
    constants.js        # Frontend constants
    utils.js            # Utility functions
    components/
      api-client.js     # API communication
      canvas-controller.js  # Canvas rendering
      sector-manager.js # Sector management
  requirements.txt      # Python dependencies for web demo
  README.md            # This file
```

## Technical Details

- **Backend**: Flask with OpenCV for image processing
- **Frontend**: Vanilla JavaScript with HTML5 Canvas
- **Image Processing**: Uses `ImageSectorTransformer` from the parent directory
- **Coordinate System**: Frontend uses image coordinates, backend handles boundary intersections
- **Color Format**: Images are converted from BGR (OpenCV) to RGB for web display
- **Image Encoding**: Uses PIL/Pillow for proper RGB JPEG encoding

## Troubleshooting

### Server won't start

- Check if port 5001 is already in use
- Verify all dependencies are installed: `pip install -r requirements.txt`
- Check Python version: `python --version` (should be 3.7+)

### Images not displaying correctly

- Ensure the backend server is running
- Check browser console for JavaScript errors
- Verify image file format is supported (PNG, JPG, JPEG, GIF, BMP)

### Sectors not working

- Ensure at least 3 sectors are defined (minimum requirement)
- Check that source and target have the same number of sectors
- Verify image is loaded before trying to adjust sectors

## Browser Compatibility

- Chrome/Edge (recommended)
- Firefox
- Safari
- Opera

## License

See LICENSE file in the parent directory for details.

## Related

- [Core Image Transformer Library](../main/README.md) - Main library documentation (on main branch)
- [Image Transformation Utility](image_transformation_util.py) - Core transformation classes

