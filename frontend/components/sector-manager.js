/**
 * Sector Manager using shared boundary lines model.
 * N sectors = N boundary lines (circular arrangement).
 */
class SectorManager {
    constructor() {
        this.boundaryLines = [];
        this.sharedCenter = null;
        this.imageWidth = 0;
        this.imageHeight = 0;
    }

    /**
     * Initialize with image dimensions.
     * @param {number} imageWidth - Image width
     * @param {number} imageHeight - Image height
     * @param {number} initialSectorCount - Initial number of sectors (default: MIN_SECTORS)
     */
    initialize(imageWidth, imageHeight, initialSectorCount = MIN_SECTORS) {
        this.imageWidth = imageWidth;
        this.imageHeight = imageHeight;
        
        this.sharedCenter = { 
            x: imageWidth / 2, 
            y: imageHeight / 2 
        };
        
        this.boundaryLines = [];
        for (let i = 0; i < initialSectorCount; i++) {
            const angle = (i * 2 * Math.PI) / initialSectorCount;
            const lineNumber = i + 1;
            const line = this.createBoundaryLine(angle, lineNumber);
            if (line) {
                this.boundaryLines.push(line);
            }
        }
    }

    /**
     * Create a boundary line at given angle.
     * @param {number} angle - Angle in radians
     * @param {number|null} lineNumber - Permanent line number (auto-assigned if null)
     * @returns {Object|null} - Boundary line object or null
     */
    createBoundaryLine(angle, lineNumber = null) {
        if (!this.sharedCenter) {
            return null;
        }
        
        if (this.imageWidth === 0 || this.imageHeight === 0) {
            return null;
        }
        
        angle = this.normalizeAngle(angle);
        
        const intersection = this.findBoundaryIntersection(
            this.sharedCenter.x,
            this.sharedCenter.y,
            angle,
            this.imageWidth,
            this.imageHeight
        );
        
        if (!intersection || intersection.x === undefined || intersection.y === undefined) {
            return null;
        }
        
        if (lineNumber === null) {
            const maxNumber = this.boundaryLines.reduce((max, line) => {
                return Math.max(max, line.number || 0);
            }, 0);
            lineNumber = maxNumber + 1;
        }
        
        return {
            x: intersection.x,
            y: intersection.y,
            angle: angle,
            number: lineNumber
        };
    }

    /**
     * Get shared center.
     * @returns {Object|null} - Center point or null
     */
    getSharedCenter() {
        return this.sharedCenter;
    }

    /**
     * Set shared center and update all boundary line positions.
     * @param {number} centerX - Center X coordinate
     * @param {number} centerY - Center Y coordinate
     */
    setSharedCenter(centerX, centerY) {
        if (this.imageWidth === 0 || this.imageHeight === 0) {
            return;
        }
        
        const clampedX = Math.max(0, Math.min(this.imageWidth, centerX));
        const clampedY = Math.max(0, Math.min(this.imageHeight, centerY));
        
        if (!this.sharedCenter) {
            this.sharedCenter = { x: clampedX, y: clampedY };
            return;
        }
        
        this.sharedCenter = { x: clampedX, y: clampedY };
        
        if (this.boundaryLines.length > 0) {
            this.boundaryLines = this.boundaryLines.map(line => {
                const lineNumber = line.number;
                const newLine = this.createBoundaryLine(line.angle, lineNumber);
                return newLine || line;
            }).filter(line => line !== null);
        }
    }

    /**
     * Get boundary lines.
     * @returns {Array} - Array of boundary line objects
     */
    getBoundaryLines() {
        return this.boundaryLines;
    }

    /**
     * Get sectors derived from boundary lines.
     * @returns {Array} - Array of sector objects
     */
    getSectors() {
        if (this.boundaryLines.length === 0) return [];
        
        const sectors = [];
        
        for (let i = 0; i < this.boundaryLines.length; i++) {
            const startLineIndex = i;
            const endLineIndex = (i + 1) % this.boundaryLines.length;
            
            const startLine = this.boundaryLines[startLineIndex];
            const endLine = this.boundaryLines[endLineIndex];
            
            sectors.push({
                id: `sector-${startLine.number || (i + 1)}`,
                center: this.sharedCenter,
                startLineIndex: startLineIndex,
                endLineIndex: endLineIndex,
                sectorNumber: startLine.number || (i + 1),
                edge_point1: {
                    x: startLine.x,
                    y: startLine.y
                },
                edge_point2: {
                    x: endLine.x,
                    y: endLine.y
                }
            });
        }
        
        return sectors;
    }

    /**
     * Add a new boundary line.
     * @param {number|null} angle - Angle in radians (auto-calculated if null)
     * @returns {Object|null} - New boundary line or null
     */
    addBoundaryLine(angle = null) {
        if (this.imageWidth === 0 || this.imageHeight === 0 || !this.sharedCenter) {
            return null;
        }
        
        if (this.boundaryLines.length === 0) {
            const line = this.createBoundaryLine(angle || 0);
            if (line) {
                this.boundaryLines.push(line);
                return line;
            }
            return null;
        }
        
        if (angle === null) {
            const angles = this.boundaryLines.map(l => l.angle).sort((a, b) => a - b);
            
            let maxGap = 0;
            let gapStart = 0;
            
            for (let i = 0; i < angles.length; i++) {
                const next = angles[(i + 1) % angles.length];
                let gap = next - angles[i];
                if (gap < 0) gap += 2 * Math.PI;
                if (gap > maxGap) {
                    maxGap = gap;
                    gapStart = angles[i];
                }
            }
            
            angle = gapStart + maxGap / 2;
            if (angle >= 2 * Math.PI) angle -= 2 * Math.PI;
        }
        
        angle = this.normalizeAngle(angle);
        
        const newLine = this.createBoundaryLine(angle);
        if (!newLine) {
            return null;
        }
        
        this.boundaryLines.push(newLine);
        
        return newLine;
    }

    /**
     * Update boundary line position.
     * @param {number} lineIndex - Index of line to update
     * @param {number} newX - New X coordinate
     * @param {number} newY - New Y coordinate
     * @returns {Object|null} - Updated line info or null
     */
    updateBoundaryLine(lineIndex, newX, newY) {
        if (lineIndex < 0 || lineIndex >= this.boundaryLines.length) return null;
        if (!this.sharedCenter) return null;
        
        const lineToUpdate = this.boundaryLines[lineIndex];
        
        const dx = newX - this.sharedCenter.x;
        const dy = newY - this.sharedCenter.y;
        let newAngle = Math.atan2(dy, dx);
        newAngle = this.normalizeAngle(newAngle);
        
        lineToUpdate.angle = newAngle;
        const intersection = this.findBoundaryIntersection(
            this.sharedCenter.x,
            this.sharedCenter.y,
            lineToUpdate.angle,
            this.imageWidth,
            this.imageHeight
        );
        
        lineToUpdate.x = intersection.x;
        lineToUpdate.y = intersection.y;
        
        return { line: lineToUpdate, newIndex: lineIndex };
    }

    /**
     * Delete boundary line.
     * @param {number} lineIndex - Index of line to delete
     * @param {number} minSectors - Minimum number of sectors required
     * @returns {Object|null} - Deleted line or null
     */
    deleteBoundaryLine(lineIndex, minSectors = MIN_SECTORS) {
        if (this.boundaryLines.length <= minSectors) {
            return null;
        }
        
        return this.boundaryLines.splice(lineIndex, 1)[0];
    }

    /**
     * Find which boundary line is near a point.
     * @param {number} imageX - Image X coordinate
     * @param {number} imageY - Image Y coordinate
     * @param {number} threshold - Distance threshold
     * @returns {Object|null} - Line info or null
     */
    findNearbyBoundaryLine(imageX, imageY, threshold = 10) {
        for (let i = 0; i < this.boundaryLines.length; i++) {
            const line = this.boundaryLines[i];
            const dx = imageX - line.x;
            const dy = imageY - line.y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            
            if (dist < threshold) {
                return { lineIndex: i, line: line };
            }
        }
        return null;
    }

    /**
     * Find intersection point of a ray with image boundary.
     * @param {number} centerX - Center X coordinate
     * @param {number} centerY - Center Y coordinate
     * @param {number} angle - Ray angle in radians
     * @param {number} imageWidth - Image width
     * @param {number} imageHeight - Image height
     * @returns {{x: number, y: number}} - Intersection point
     */
    findBoundaryIntersection(centerX, centerY, angle, imageWidth, imageHeight) {
        const dx = Math.cos(angle);
        const dy = Math.sin(angle);
        
        const intersections = [];
        
        if (dx !== 0) {
            const t = -centerX / dx;
            if (t > 0) {
                const y = centerY + t * dy;
                if (y >= 0 && y <= imageHeight) {
                    intersections.push({ x: 0, y: y, t: t });
                }
            }
        }
        
        if (dx !== 0) {
            const t = (imageWidth - centerX) / dx;
            if (t > 0) {
                const y = centerY + t * dy;
                if (y >= 0 && y <= imageHeight) {
                    intersections.push({ x: imageWidth, y: y, t: t });
                }
            }
        }
        
        if (dy !== 0) {
            const t = -centerY / dy;
            if (t > 0) {
                const x = centerX + t * dx;
                if (x >= 0 && x <= imageWidth) {
                    intersections.push({ x: x, y: 0, t: t });
                }
            }
        }
        
        if (dy !== 0) {
            const t = (imageHeight - centerY) / dy;
            if (t > 0) {
                const x = centerX + t * dx;
                if (x >= 0 && x <= imageWidth) {
                    intersections.push({ x: x, y: imageHeight, t: t });
                }
            }
        }
        
        if (intersections.length > 0) {
            const closest = intersections.reduce((min, p) => p.t < min.t ? p : min);
            const clampedX = Math.max(0, Math.min(imageWidth - 1, Math.round(closest.x)));
            const clampedY = Math.max(0, Math.min(imageHeight - 1, Math.round(closest.y)));
            return { x: clampedX, y: clampedY };
        }
        
        return this.clampToBounds(centerX + dx * 1000, centerY + dy * 1000, imageWidth, imageHeight);
    }

    /**
     * Clamp point to image bounds.
     * @param {number} x - X coordinate
     * @param {number} y - Y coordinate
     * @param {number} width - Image width
     * @param {number} height - Image height
     * @returns {{x: number, y: number}} - Clamped point
     */
    clampToBounds(x, y, width, height) {
        const clampedX = Math.max(0, Math.min(width - 1, x));
        const clampedY = Math.max(0, Math.min(height - 1, y));
        return { x: Math.round(clampedX), y: Math.round(clampedY) };
    }

    /**
     * Normalize angle to [0, 2π).
     * @param {number} angle - Angle in radians
     * @returns {number} - Normalized angle
     */
    normalizeAngle(angle) {
        while (angle < 0) angle += 2 * Math.PI;
        while (angle >= 2 * Math.PI) angle -= 2 * Math.PI;
        return angle;
    }

    /**
     * Clear all sectors.
     */
    clear() {
        this.boundaryLines = [];
        this.sharedCenter = null;
        this.imageWidth = 0;
        this.imageHeight = 0;
    }

    /**
     * Get sector count.
     * @returns {number} - Number of sectors
     */
    count() {
        return this.boundaryLines.length;
    }

    /**
     * Create default sectors.
     * @param {number} imageWidth - Image width
     * @param {number} imageHeight - Image height
     * @param {number} count - Number of sectors
     */
    createDefaultSectors(imageWidth, imageHeight, count = 1) {
        this.initialize(imageWidth, imageHeight);
        
        for (let i = 1; i < count; i++) {
            const angle = (i * 2 * Math.PI) / count;
            this.addBoundaryLine(angle);
        }
    }

    /**
     * Export sectors as JSON for API.
     * @returns {Array} - Array of sector objects
     */
    exportSectors() {
        const sectors = this.getSectors();
        return sectors.map(sector => ({
            id: sector.id,
            center: { x: sector.center.x, y: sector.center.y },
            edge_point1: sector.edge_point1,
            edge_point2: sector.edge_point2
        }));
    }

    /**
     * Import sectors from JSON.
     * @param {Array} sectorsJson - Array of sector objects
     */
    importSectors(sectorsJson) {
        if (sectorsJson.length === 0) {
            this.clear();
            return;
        }
        
        const firstSector = sectorsJson[0];
        this.sharedCenter = { ...firstSector.center };
        
        const edgePoints = new Set();
        sectorsJson.forEach(sector => {
            const key1 = `${sector.edge_point1.x},${sector.edge_point1.y}`;
            const key2 = `${sector.edge_point2.x},${sector.edge_point2.y}`;
            edgePoints.add(key1);
            edgePoints.add(key2);
        });
        
        this.boundaryLines = [];
        edgePoints.forEach(pointKey => {
            const [x, y] = pointKey.split(',').map(Number);
            const angle = Math.atan2(y - this.sharedCenter.y, x - this.sharedCenter.x);
            this.boundaryLines.push(this.createBoundaryLine(this.normalizeAngle(angle)));
        });
        
        this.boundaryLines.sort((a, b) => a.angle - b.angle);
    }
}
