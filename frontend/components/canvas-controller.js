/**
 * Canvas Controller for rendering and interaction
 */
class CanvasController {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.image = null;
        this.imageData = null;
        this.scale = 1;
        this.offsetX = 0;
        this.offsetY = 0;
        
        // Set initial empty canvas size after layout is ready
        // Use multiple attempts to ensure wrapper has proper dimensions
        const initEmptyCanvas = () => {
            if (!this.image && this.canvas.parentElement) {
                const wrapper = this.canvas.parentElement;
                if (wrapper.clientWidth > 0) {
                    this.setEmptyCanvasSize();
                    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
                    this.ctx.fillStyle = '#fafafa';
                    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
                } else {
                    // Retry if wrapper not ready yet
                    requestAnimationFrame(initEmptyCanvas);
                }
            }
        };
        
        // Try immediately, then after next frame, then after a short delay
        requestAnimationFrame(initEmptyCanvas);
        setTimeout(initEmptyCanvas, 100);
    }

    /**
     * Load image onto canvas and calculate dimensions.
     * @param {string} imageSrc - Image source URL
     * @param {number} width - Image width
     * @param {number} height - Image height
     * @returns {Promise<void>}
     */
    loadImage(imageSrc, width, height) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => {
                this.image = img;
                this.imageData = { width, height };
                
                requestAnimationFrame(() => {
                    // Calculate and set canvas dimensions
                    this.recalculateDimensions();
                    
                    // Draw the image
                    this.draw();
                    resolve();
                });
            };
            img.onerror = () => {
                reject(new Error('Failed to load image'));
            };
            img.src = imageSrc;
        });
    }

    /**
     * Clear canvas and reset to initial empty state.
     */
    clear() {
        this.image = null;
        this.imageData = null;
        this.scale = 1;
        this.offsetX = 0;
        this.offsetY = 0;
        
        // Calculate empty canvas size based on wrapper width to fill the row
        this.setEmptyCanvasSize();
        
        // Fill with background color
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.fillStyle = '#fafafa';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    }

    /**
     * Set empty canvas size based on wrapper width to fill the row.
     */
    setEmptyCanvasSize() {
        const wrapper = this.canvas.parentElement;
        if (!wrapper) {
            // Fallback to default size if no wrapper
            const defaultWidth = DEFAULT_EMPTY_CANVAS_HEIGHT * DEFAULT_CANVAS_ASPECT_RATIO;
            this.canvas.width = defaultWidth + (CANVAS_PADDING * 2);
            this.canvas.height = DEFAULT_EMPTY_CANVAS_HEIGHT + (CANVAS_PADDING * 2);
            return;
        }
        
        // Get wrapper width - try multiple methods to get accurate width
        let wrapperWidth = wrapper.clientWidth;
        if (wrapperWidth === 0) {
            wrapperWidth = wrapper.offsetWidth;
        }
        if (wrapperWidth === 0) {
            // Try getting from container if wrapper not ready
            const container = wrapper.parentElement;
            if (container) {
                // Estimate: divide container width by number of wrappers (typically 3)
                const containerWidth = container.clientWidth || container.offsetWidth;
                if (containerWidth > 0) {
                    // Approximate wrapper width: (container - gaps - padding) / 3
                    // Gap is 20px between wrappers, so 2 gaps for 3 wrappers = 40px
                    // Container might have padding too, estimate 40px total overhead
                    const estimatedWrapperWidth = (containerWidth - 40) / 3;
                    wrapperWidth = Math.max(200, estimatedWrapperWidth);
                }
            }
        }
        
        // If still no width, use a reasonable default and retry
        if (wrapperWidth < 50) {
            // Use a fallback width for calculation
            wrapperWidth = 400; // Reasonable default for desktop
            // Still set the canvas so it's visible, but maybe retry later
            setTimeout(() => {
                if (!this.image && wrapper.clientWidth > 0) {
                    this.setEmptyCanvasSize();
                }
            }, 200);
        }
        
        // Calculate available width (wrapper width minus wrapper padding)
        // Wrapper has 15px padding on each side = 30px total
        const availableWidth = wrapperWidth - 30;
        
        // Use most of available width (98%) to fill the row nicely
        const contentWidth = availableWidth * 0.98;
        
        // Calculate canvas content width (content width minus canvas padding)
        const canvasContentWidth = Math.max(150, contentWidth - (CANVAS_PADDING * 2));
        
        // Calculate canvas height based on aspect ratio
        const canvasContentHeight = canvasContentWidth / DEFAULT_CANVAS_ASPECT_RATIO;
        
        // Set canvas dimensions (including padding)
        this.canvas.width = canvasContentWidth + (CANVAS_PADDING * 2);
        this.canvas.height = canvasContentHeight + (CANVAS_PADDING * 2);
    }

    /**
     * Recalculate canvas dimensions (call this when window resizes or container changes).
     * Canvas bounds itself, wrapper shrinks to fit canvas.
     */
    recalculateDimensions() {
        if (!this.image || !this.imageData) return;
        
        // Get wrapper width to use as constraint for canvas sizing
        const wrapper = this.canvas.parentElement;
        const wrapperWidth = wrapper ? wrapper.clientWidth : null;
        
        // Calculate canvas dimensions based on image size with max constraints
        const dims = calculateCanvasDimensions(
            this.imageData.width,
            this.imageData.height,
            CANVAS_PADDING,
            wrapperWidth
        );
        
        this.scale = dims.scale;
        this.canvas.width = dims.canvasWidth;
        this.canvas.height = dims.canvasHeight;
        this.offsetX = dims.offsetX;
        this.offsetY = dims.offsetY;
    }

    /**
     * Draw image at current canvas dimensions.
     * Does NOT resize the canvas - only redraws content.
     */
    draw() {
        if (!this.image || !this.imageData) return;
        
        // Don't recalculate dimensions here - only redraw
        // Canvas dimensions should only change on load or explicit resize
        const scaledWidth = this.imageData.width * this.scale;
        const scaledHeight = this.imageData.height * this.scale;
        
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.drawImage(
            this.image,
            this.offsetX,
            this.offsetY,
            scaledWidth,
            scaledHeight
        );
    }

    /**
     * Convert canvas coordinates to image coordinates.
     * @param {number} x - Canvas X coordinate
     * @param {number} y - Canvas Y coordinate
     * @returns {{x: number, y: number}|null}
     */
    canvasToImage(x, y) {
        if (!this.imageData) return null;
        return {
            x: (x - this.offsetX) / this.scale,
            y: (y - this.offsetY) / this.scale
        };
    }

    /**
     * Convert image coordinates to canvas coordinates.
     * @param {number} x - Image X coordinate
     * @param {number} y - Image Y coordinate
     * @returns {{x: number, y: number}|null}
     */
    imageToCanvas(x, y) {
        if (!this.imageData) return null;
        return {
            x: x * this.scale + this.offsetX,
            y: y * this.scale + this.offsetY
        };
    }

    /**
     * Draw sector fill on canvas.
     * @param {Object} sector - Sector object with center and edge points
     * @param {string} color - RGBA color string
     * @param {boolean} isSelected - Whether sector is selected
     * @param {boolean} isHovered - Whether sector is hovered
     * @param {boolean} isHighlighted - Whether sector is highlighted (uses color's opacity directly)
     */
    drawSector(sector, color, isSelected = false, isHovered = false, isHighlighted = false) {
        if (!this.imageData) return;
        
        const ctx = this.ctx;
        ctx.save();
        
        const scaledWidth = this.imageData.width * this.scale;
        const scaledHeight = this.imageData.height * this.scale;
        ctx.beginPath();
        ctx.rect(this.offsetX, this.offsetY, scaledWidth, scaledHeight);
        ctx.clip();
        
        const center = this.imageToCanvas(sector.center.x, sector.center.y);
        const edge1 = this.imageToCanvas(sector.edge_point1.x, sector.edge_point1.y);
        const edge2 = this.imageToCanvas(sector.edge_point2.x, sector.edge_point2.y);
        
        if (!center || !edge1 || !edge2) {
            ctx.restore();
            return;
        }
        
        const angle1 = Math.atan2(
            sector.edge_point1.y - sector.center.y,
            sector.edge_point1.x - sector.center.x
        );
        const angle2 = Math.atan2(
            sector.edge_point2.y - sector.center.y,
            sector.edge_point2.x - sector.center.x
        );
        
        const imgCorners = [
            {x: 0, y: 0},
            {x: this.imageData.width, y: 0},
            {x: this.imageData.width, y: this.imageData.height},
            {x: 0, y: this.imageData.height}
        ];
        
        let maxRadius = 0;
        imgCorners.forEach(corner => {
            const dist = Math.sqrt(
                Math.pow(corner.x - sector.center.x, 2) + 
                Math.pow(corner.y - sector.center.y, 2)
            );
            maxRadius = Math.max(maxRadius, dist);
        });
        
        const maxRadiusCanvas = maxRadius * this.scale;
        
        // If highlighted, use the color's opacity directly (set globalAlpha to 1.0)
        // Otherwise, use the default sector opacity
        if (isHighlighted) {
            ctx.globalAlpha = 1.0;
        } else {
            ctx.globalAlpha = DEFAULT_SECTOR_OPACITY;
        }
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.moveTo(center.x, center.y);
        
        let angleDiff = angle2 - angle1;
        if (angleDiff < 0) angleDiff += 2 * Math.PI;
        
        ctx.arc(center.x, center.y, maxRadiusCanvas, angle1, angle2, false);
        ctx.closePath();
        ctx.fill();
        
        ctx.restore();
    }
    
    /**
     * Draw center point.
     * @param {Object} center - Center point {x, y}
     * @param {boolean} isSelected - Whether center is selected
     */
    drawCenterPoint(center, isSelected = false) {
        if (!center) return;
        const canvasCenter = this.imageToCanvas(center.x, center.y);
        if (!canvasCenter) return;
        
        const ctx = this.ctx;
        ctx.fillStyle = isSelected ? '#ff0000' : '#333';
        ctx.beginPath();
        ctx.arc(canvasCenter.x, canvasCenter.y, isSelected ? 10 : 8, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.stroke();
    }

    /**
     * Get sector color by index.
     * @param {number} sectorIndex - Sector index
     * @returns {string} - RGBA color string
     */
    static getSectorColor(sectorIndex) {
        return SECTOR_COLORS[sectorIndex % SECTOR_COLORS.length];
    }
    
    /**
     * Draw sectors with optional highlighting.
     * @param {Array} sectors - Array of sector objects
     * @param {Object} sharedCenter - Shared center point
     * @param {Array} boundaryLines - Array of boundary line objects
     * @param {number|null} selectedLineIndex - Index of selected line
     * @param {number|null} hoveredLineIndex - Index of hovered line
     * @param {Set|null} highlightedSectorIndices - Set of sector indices to highlight
     * @param {Map|null} colorMap - Map of sector index to color index
     * @param {number} highlightOpacity - Opacity for highlighting (0.1 to 1.0)
     */
    drawSectors(sectors, sharedCenter = null, boundaryLines = null, selectedLineIndex = null, hoveredLineIndex = null, highlightedSectorIndices = null, colorMap = null, highlightOpacity = DEFAULT_HIGHLIGHT_OPACITY) {
        if (sharedCenter) {
            this.drawCenterPoint(sharedCenter, false);
        }
        
        if (highlightedSectorIndices && sectors && highlightedSectorIndices.size > 0) {
            sectors.forEach((sector, index) => {
                if (highlightedSectorIndices.has(index)) {
                    let colorIndex = index;
                    if (colorMap && colorMap.has(index)) {
                        colorIndex = colorMap.get(index);
                    }
                    const baseColor = CanvasController.getSectorColor(colorIndex);
                    const color = replaceColorOpacity(baseColor, highlightOpacity);
                    this.drawSector(sector, color, false, false, true);
                }
            });
        }
        
        if (boundaryLines && sharedCenter) {
            boundaryLines.forEach((line, index) => {
                const isSelected = index === selectedLineIndex;
                const isHovered = index === hoveredLineIndex;
                const lineNumber = line.number || (index + 1);
                this.drawBoundaryLine(line, sharedCenter, isSelected, isHovered, lineNumber);
            });
        }
    }
    
    /**
     * Draw a boundary line.
     * @param {Object} line - Boundary line object
     * @param {Object} center - Center point
     * @param {boolean} isSelected - Whether line is selected
     * @param {boolean} isHovered - Whether line is hovered
     * @param {number|null} sectorNumber - Sector number to display
     */
    drawBoundaryLine(line, center, isSelected = false, isHovered = false, sectorNumber = null) {
        if (!this.imageData || !center) return;
        
        const ctx = this.ctx;
        const canvasCenter = this.imageToCanvas(center.x, center.y);
        const canvasLine = this.imageToCanvas(line.x, line.y);
        
        if (!canvasCenter || !canvasLine) return;
        
        const lineAngle = Math.atan2(canvasLine.y - canvasCenter.y, canvasLine.x - canvasCenter.x);
        
        ctx.strokeStyle = isSelected ? '#ff0000' : (isHovered ? '#00ff00' : '#333');
        ctx.lineWidth = isSelected ? 3 : (isHovered ? 2 : 1.5);
        ctx.beginPath();
        ctx.moveTo(canvasCenter.x, canvasCenter.y);
        ctx.lineTo(canvasLine.x, canvasLine.y);
        ctx.stroke();
        
        if (sectorNumber !== null) {
            this.drawBoundaryPointLabel(canvasLine.x, canvasLine.y, lineAngle, sectorNumber, isSelected, isHovered);
        }
    }
    
    /**
     * Draw boundary point label (numbered circle).
     * @param {number} x - X coordinate
     * @param {number} y - Y coordinate
     * @param {number} lineAngle - Line angle (unused but kept for API compatibility)
     * @param {number} sectorNumber - Sector number to display
     * @param {boolean} isSelected - Whether point is selected
     * @param {boolean} isHovered - Whether point is hovered
     */
    drawBoundaryPointLabel(x, y, lineAngle, sectorNumber, isSelected = false, isHovered = false) {
        const ctx = this.ctx;
        const labelText = sectorNumber.toString();
        
        ctx.save();
        
        const circleRadius = isSelected ? CIRCLE_RADIUS.selected : (isHovered ? CIRCLE_RADIUS.hovered : CIRCLE_RADIUS.default);
        ctx.fillStyle = 'rgba(255, 255, 255, 0.95)';
        ctx.beginPath();
        ctx.arc(x, y, circleRadius, 0, Math.PI * 2);
        ctx.fill();
        
        ctx.strokeStyle = isSelected ? '#ff0000' : '#333';
        ctx.lineWidth = isSelected ? 3 : 2;
        ctx.stroke();
        
        ctx.font = FONT_STYLE.boundaryLabel;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = '#000';
        ctx.fillText(labelText, x, y);
        
        ctx.restore();
    }

    /**
     * Draw result image.
     * @param {string} imageSrc - Image source URL
     * @param {number|null} sourceWidth - Source width
     * @param {number|null} sourceHeight - Source height
     * @returns {Promise<void>}
     */
    drawResult(imageSrc, sourceWidth = null, sourceHeight = null) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => {
                const resultWidth = sourceWidth || img.width;
                const resultHeight = sourceHeight || img.height;
                
                this.image = img;
                this.imageData = { width: resultWidth, height: resultHeight };
                
                requestAnimationFrame(() => {
                    const wrapper = this.canvas.parentElement;
                    if (!wrapper) {
                        reject(new Error('Canvas has no parent container'));
                        return;
                    }
                    
                    // Calculate and set canvas dimensions
                    this.recalculateDimensions();
                    
                    // Draw the result image
                    const scaledWidth = resultWidth * this.scale;
                    const scaledHeight = resultHeight * this.scale;
                    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
                    this.ctx.drawImage(img, this.offsetX, this.offsetY, scaledWidth, scaledHeight);
                    resolve();
                });
            };
            img.onerror = reject;
            img.src = imageSrc;
        });
    }

    /**
     * Get canvas element.
     * @returns {HTMLCanvasElement}
     */
    getCanvas() {
        return this.canvas;
    }

    /**
     * Get image data.
     * @returns {Object|null} - Image data object or null
     */
    getImageData() {
        return this.imageData;
    }

    /**
     * Get canvas as data URL for download.
     * @returns {string|null} - Data URL of the canvas image or null
     */
    getCanvasDataURL() {
        if (!this.canvas || this.canvas.width === 0 || this.canvas.height === 0) {
            return null;
        }
        return this.canvas.toDataURL('image/png');
    }

    /**
     * Get the original result image data URL if available.
     * @returns {string|null} - Original image data URL or null
     */
    getResultImageDataURL() {
        if (!this.image || !this.image.src) {
            return null;
        }
        return this.image.src;
    }

    /**
     * Check if point is near another point.
     * @param {Object} point1 - First point
     * @param {Object} point2 - Second point
     * @param {number} threshold - Distance threshold
     * @returns {boolean}
     */
    isPointNear(point1, point2, threshold = 10) {
        if (!point1 || !point2) return false;
        const dx = point1.x - point2.x;
        const dy = point1.y - point2.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        return dist < threshold;
    }

    /**
     * Find which sector point is near the given canvas coordinates.
     * @param {number} canvasX - Canvas X coordinate
     * @param {number} canvasY - Canvas Y coordinate
     * @param {Array} sectors - Array of sector objects
     * @param {number} threshold - Distance threshold
     * @returns {Object|null}
     */
    findNearbyPoint(canvasX, canvasY, sectors, threshold = 10) {
        if (!this.imageData) return null;
        
        for (const sector of sectors) {
            const center = this.imageToCanvas(sector.center.x, sector.center.y);
            const edge1 = this.imageToCanvas(sector.edge_point1.x, sector.edge_point1.y);
            const edge2 = this.imageToCanvas(sector.edge_point2.x, sector.edge_point2.y);
            
            const point = { x: canvasX, y: canvasY };
            
            if (center && this.isPointNear(point, center, threshold)) {
                return { sectorId: sector.id, pointType: 'center', sector: sector };
            }
            if (edge1 && this.isPointNear(point, edge1, threshold)) {
                return { sectorId: sector.id, pointType: 'edge1', sector: sector };
            }
            if (edge2 && this.isPointNear(point, edge2, threshold)) {
                return { sectorId: sector.id, pointType: 'edge2', sector: sector };
            }
        }
        
        return null;
    }
}
