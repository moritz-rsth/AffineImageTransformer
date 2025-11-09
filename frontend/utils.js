// Utility functions for the frontend application

/**
 * Replace the opacity value in an RGBA color string.
 * @param {string} baseColor - RGBA color string (e.g., "rgba(255, 99, 132, 1.0)")
 * @param {number} opacity - New opacity value (0.0 to 1.0)
 * @returns {string} - Color string with updated opacity
 */
function replaceColorOpacity(baseColor, opacity) {
    // Use regex to find and replace the alpha value in rgba(...) format
    // Matches: rgba(r, g, b, <alpha>) where alpha can be 0.0-1.0 or 0-1
    return baseColor.replace(/rgba\((\d+),\s*(\d+),\s*(\d+),\s*[\d.]+\)/, 
        `rgba($1, $2, $3, ${opacity})`);
}

/**
 * Calculate adaptive threshold based on canvas scale.
 * @param {number} baseThreshold - Base threshold value
 * @param {number} scale - Canvas scale factor
 * @returns {number} - Adaptive threshold
 */
function calculateAdaptiveThreshold(baseThreshold, scale) {
    return Math.max(baseThreshold, baseThreshold / Math.max(scale, 0.1));
}

/**
 * Calculate hit threshold for boundary line selection.
 * @param {number} circleRadius - Radius of the draggable circle
 * @param {number} scale - Canvas scale factor
 * @returns {number} - Hit threshold for selection
 */
function calculateHitThreshold(circleRadius, scale) {
    return Math.max(circleRadius + 5, (circleRadius + 5) / Math.max(scale, 0.1));
}

/**
 * Validate sector count against minimum requirement.
 * @param {number} count - Current sector count
 * @param {number} minCount - Minimum required sectors
 * @returns {boolean} - True if count meets minimum requirement
 */
function validateSectorCount(count, minCount) {
    return count >= minCount;
}

/**
 * Get container dimensions with fallback values.
 * @param {HTMLElement} container - Container element
 * @returns {{width: number, height: number}} - Container dimensions
 */
function getContainerDimensions(container) {
    return {
        width: container ? container.clientWidth : FALLBACK_CONTAINER_SIZE,
        height: container ? container.clientHeight : FALLBACK_CONTAINER_SIZE
    };
}

/**
 * Calculate canvas dimensions based on container and image size with padding.
 * @param {number} containerWidth - Container width
 * @param {number} containerHeight - Container height
 * @param {number} imageWidth - Image width
 * @param {number} imageHeight - Image height
 * @param {number} padding - Padding around image
 * @returns {{canvasWidth: number, canvasHeight: number, scale: number, offsetX: number, offsetY: number}} - Canvas dimensions and layout
 */
function calculateCanvasDimensions(containerWidth, containerHeight, imageWidth, imageHeight, padding) {
    const availableWidth = containerWidth - (padding * 2);
    const availableHeight = containerHeight - (padding * 2);
    
    const scaleX = availableWidth / imageWidth;
    const scaleY = availableHeight / imageHeight;
    const scale = Math.min(scaleX, scaleY, 1);
    
    const scaledWidth = imageWidth * scale;
    const scaledHeight = imageHeight * scale;
    
    return {
        canvasWidth: scaledWidth + (padding * 2),
        canvasHeight: scaledHeight + (padding * 2),
        scale: scale,
        offsetX: padding,
        offsetY: padding,
        scaledWidth: scaledWidth,
        scaledHeight: scaledHeight
    };
}

