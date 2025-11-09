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
 * Calculate canvas dimensions based on image size with maximum constraints.
 * Canvas bounds itself, wrapper will shrink to fit.
 * @param {number} imageWidth - Image width
 * @param {number} imageHeight - Image height
 * @param {number} padding - Padding around image
 * @param {number} containerWidth - Optional container width to respect (for side-by-side layouts)
 * @param {number} maxWidth - Maximum canvas width (optional, uses MAX_CANVAS_WIDTH if not provided)
 * @param {number} maxHeight - Maximum canvas height (optional, uses MAX_CANVAS_HEIGHT if not provided)
 * @returns {{canvasWidth: number, canvasHeight: number, scale: number, offsetX: number, offsetY: number, scaledWidth: number, scaledHeight: number}} - Canvas dimensions and layout
 */
function calculateCanvasDimensions(imageWidth, imageHeight, padding, containerWidth = null, maxWidth = MAX_CANVAS_WIDTH, maxHeight = MAX_CANVAS_HEIGHT) {
    // Determine the effective maximum width
    // Canvas bounds itself - use container width as constraint if provided (for side-by-side layouts)
    // Otherwise use absolute maximum
    let effectiveMaxWidth = maxWidth;
    
    if (containerWidth !== null && containerWidth > 0) {
        // Account for wrapper padding (15px on each side = 30px total)
        // Also account for gap between wrappers (20px) when calculating per-wrapper space
        // On desktop with 3 wrappers: (containerWidth - gaps - padding) / 3 would be ideal,
        // but we use the actual wrapper width which flexbox handles for us
        const availableWidth = containerWidth - 30; // Wrapper padding
        // Use the smaller of: container width or absolute maximum
        // Ensure minimum width for very small containers
        effectiveMaxWidth = Math.min(maxWidth, Math.max(200, availableWidth));
    }
    
    // Calculate maximum available space for image (excluding padding)
    const maxAvailableWidth = effectiveMaxWidth - (padding * 2);
    const maxAvailableHeight = maxHeight - (padding * 2);
    
    // Ensure we have positive dimensions
    if (maxAvailableWidth <= 0 || maxAvailableHeight <= 0) {
        // Fallback to minimal size
        return {
            canvasWidth: padding * 2,
            canvasHeight: padding * 2,
            scale: 0,
            offsetX: padding,
            offsetY: padding,
            scaledWidth: 0,
            scaledHeight: 0
        };
    }
    
    // Calculate scale to fit within maximum dimensions (scale down if needed, but don't scale up)
    const scaleX = maxAvailableWidth / imageWidth;
    const scaleY = maxAvailableHeight / imageHeight;
    const scale = Math.min(scaleX, scaleY, 1);  // Never scale up (max scale = 1)
    
    // Calculate actual scaled dimensions
    const scaledWidth = imageWidth * scale;
    const scaledHeight = imageHeight * scale;
    
    // Canvas size includes padding
    const canvasWidth = scaledWidth + (padding * 2);
    const canvasHeight = scaledHeight + (padding * 2);
    
    return {
        canvasWidth: canvasWidth,
        canvasHeight: canvasHeight,
        scale: scale,
        offsetX: padding,
        offsetY: padding,
        scaledWidth: scaledWidth,
        scaledHeight: scaledHeight
    };
}

