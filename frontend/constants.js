// Frontend configuration constants

// API Configuration
const API_BASE_URL = (() => {
    // Try to get from environment or use default
    if (typeof window !== 'undefined' && window.API_BASE_URL) {
        return window.API_BASE_URL;
    }
    return 'http://localhost:5001';
})();

// Sector Configuration
const MIN_SECTORS = 3;
const DEFAULT_SECTOR_OPACITY = 0.3;

// Mixup Configuration
const DEFAULT_ALPHA = 0.5;
const DEFAULT_HIGHLIGHT_OPACITY = 0.5;

// Canvas Configuration
const CANVAS_PADDING = 20;
const DEFAULT_CANVAS_WIDTH = 300;
const DEFAULT_CANVAS_HEIGHT = 150;
const FALLBACK_CONTAINER_SIZE = 400;

// Interaction Configuration
const DRAG_THRESHOLD = 20;
const CIRCLE_RADIUS = {
    default: 12,
    hovered: 13,
    selected: 14
};

// Font Configuration
const FONT_STYLE = {
    boundaryLabel: 'bold 16px Arial'
};

// Sector Colors (consistent color palette) - initialized with opacity 1.0
const SECTOR_COLORS = [
    'rgba(255, 99, 132, 1.0)',   // Red - Sector 1
    'rgba(54, 162, 235, 1.0)',   // Blue - Sector 2
    'rgba(255, 206, 86, 1.0)',   // Yellow - Sector 3
    'rgba(75, 192, 192, 1.0)',   // Green - Sector 4
    'rgba(153, 102, 255, 1.0)',  // Purple - Sector 5
    'rgba(255, 159, 64, 1.0)',   // Orange - Sector 6
    'rgba(199, 199, 199, 1.0)',  // Gray - Sector 7
    'rgba(83, 102, 255, 1.0)',   // Indigo - Sector 8
    'rgba(255, 99, 255, 1.0)',   // Pink - Sector 9
    'rgba(99, 255, 132, 1.0)'    // Light Green - Sector 10
];

// Color Names for Display
const COLOR_NAMES = [
    'Red', 'Blue', 'Yellow', 'Green', 'Purple',
    'Orange', 'Gray', 'Indigo', 'Pink', 'Light Green'
];

