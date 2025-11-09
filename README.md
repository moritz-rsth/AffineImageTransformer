# Affine Image Transformer

A Python library for performing sector-based affine transformations on images. This library provides tools for warping and mixing image sectors using radial triangular transformations.

## Overview

The `AffineImageTransformer` library implements a novel approach to image transformation using sector-based geometry. It allows you to:

- Define image sectors using center points and boundary edges
- Perform arbitrary sector warping (transform sectors from one configuration to another)
- Mix sectors between different images with alpha blending
- Generate radial triangular meshes for precise geometric transformations

## Core Components

### `LinearBoundedSector`

Represents a single angular sector of an image, defined by:
- **Center point**: The origin of radial rays
- **Edge points**: Two boundary points that define the sector's angular span
- **Image bounds**: Width and height constraints

**Key Features:**
- Automatic boundary intersection calculation
- Radial triangle generation for transformation meshes
- Angle normalization and point sorting
- Sector validation and geometric operations

### `ImageSectorTransformer`

Provides static methods for image transformation operations:

- **`arbitrary_sector_warping()`**: Warp an image by transforming source sectors to target sectors
- **`sector_mixup()`**: Mix pixels from one image's sector into another image's sector
- **`map_triangles()`**: Core transformation engine using affine triangle mapping

## Installation

### Core Dependencies

```bash
pip install opencv-python numpy torch matplotlib
```

### Full Installation (including web demo)

See [Affine Image Transformer](https://affine-image-transformer.up.railway.app/) for hosted web demo.

## Quick Start

```python
from image_transformation_util import LinearBoundedSector, ImageSectorTransformer
import cv2
import numpy as np

# Load an image
image = cv2.imread('example.jpg')

# Define source sector
source_sector = LinearBoundedSector(
    center=(320, 240),
    edge_point1=(640, 0),
    edge_point2=(640, 480),
    bound_width=640,
    bound_height=480
)

# Define target sector (warped configuration)
target_sector = LinearBoundedSector(
    center=(320, 240),
    edge_point1=(640, 100),
    edge_point2=(640, 380),
    bound_width=640,
    bound_height=480
)

# Perform warping
warped_image = ImageSectorTransformer.arbitrary_sector_warping(
    src_image=image,
    source_sectors=[source_sector],
    target_sectors=[target_sector],
    debug=False
)

# Save result
cv2.imwrite('warped_result.jpg', warped_image)
```

## Advanced Usage

### Multiple Sector Warping

```python
# Define multiple sectors for complex transformations
source_sectors = [
    LinearBoundedSector(center=(320, 240), edge_point1=(640, 0), edge_point2=(640, 160), ...),
    LinearBoundedSector(center=(320, 240), edge_point1=(640, 160), edge_point2=(640, 320), ...),
    LinearBoundedSector(center=(320, 240), edge_point1=(640, 320), edge_point2=(640, 480), ...)
]

target_sectors = [
    # Define target configurations...
]

warped_image = ImageSectorTransformer.arbitrary_sector_warping(
    src_image=image,
    source_sectors=source_sectors,
    target_sectors=target_sectors
)
```

### Sector Mixup

```python
# Mix sectors between two images
image1 = cv2.imread('image1.jpg')
image2 = cv2.imread('image2.jpg')

sector1 = LinearBoundedSector(...)  # Sector in image1
sector2 = LinearBoundedSector(...)  # Sector in image2

# Mix with alpha blending (0.0 = only image1, 1.0 = only image2)
mixed_image = ImageSectorTransformer.sector_mixup(
    src_image=image1,
    src_sector=sector1,
    dst_image=image2,
    dst_sector=sector2,
    alpha=0.5
)
```

## Technical Details

### Transformation Algorithm

The library uses a radial triangular mesh approach:

1. **Sector Decomposition**: Each sector is divided into radial triangles originating from the center point
2. **Triangle Mapping**: Source triangles are mapped to target triangles using affine transformations
3. **Pixel Interpolation**: Bilinear interpolation ensures smooth transitions between triangles
4. **Boundary Handling**: Automatic clipping and boundary intersection calculations

### Coordinate System

- **Image coordinates**: Origin (0,0) at top-left corner
- **Sector angles**: Measured in radians, normalized to [0, 2π)
- **Boundary points**: Automatically clamped to valid pixel indices [0, width-1] × [0, height-1]

## API Reference

### `LinearBoundedSector`

```python
class LinearBoundedSector:
    def __init__(self, center: Tuple[int, int], 
                 edge_point1: Tuple[int, int], 
                 edge_point2: Tuple[int, int],
                 bound_width: int, 
                 bound_height: int)
    
    def get_radial_triangles(self) -> Tuple[List[Tuple[int, int]], ...]
    def add_angle_to_sector(self, angle: int, start: bool)
    def _is_point_in_sector(self, point: Tuple[int, int]) -> bool
```

### `ImageSectorTransformer`

```python
class ImageSectorTransformer:
    @staticmethod
    def arbitrary_sector_warping(src_image: np.ndarray,
                                 source_sectors: List[LinearBoundedSector],
                                 target_sectors: List[LinearBoundedSector],
                                 debug: bool = False) -> np.ndarray
    
    @staticmethod
    def sector_mixup(src_image: np.ndarray,
                    src_sector: LinearBoundedSector,
                    dst_image: np.ndarray,
                    dst_sector: LinearBoundedSector,
                    alpha: float = 0.5) -> np.ndarray
```

## Requirements

- Python 3.7+
- OpenCV (cv2)
- NumPy
- PyTorch
- Matplotlib (for visualization/debugging)

## License

See LICENSE file for details.

## Web Demo

For an interactive web-based interface to this library, see the [hosted Web Demo](https://affine-image-transformer.up.railway.app/).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
