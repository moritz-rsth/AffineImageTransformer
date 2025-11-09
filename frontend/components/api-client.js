// API Client for communicating with backend
class APIClient {
    constructor() {
        this.baseURL = API_BASE_URL;
    }

    async uploadImage(file) {
        const formData = new FormData();
        formData.append('image', file);

        const response = await fetch(`${this.baseURL}/api/upload`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Upload failed');
        }

        return await response.json();
    }

    async warpImage(sourceImageId, sourceSectors, targetSectors, debugMode = false) {
        const response = await fetch(`${this.baseURL}/api/warp`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                source_image_id: sourceImageId,
                source_sectors: sourceSectors,
                target_sectors: targetSectors,
                debug_mode: debugMode
            })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Warp failed');
        }

        return await response.json();
    }

    async mixupImages(image1Id, image2Id, sectors1, sectors2, sectorMapping, debugMode = false, alpha = DEFAULT_ALPHA) {
        const response = await fetch(`${this.baseURL}/api/mixup`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image1_id: image1Id,
                image2_id: image2Id,
                sectors1: sectors1,
                sectors2: sectors2,
                sector_mapping: sectorMapping,
                debug_mode: debugMode,
                alpha: alpha
            })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Mixup failed');
        }

        return await response.json();
    }

    async healthCheck() {
        const response = await fetch(`${this.baseURL}/health`);
        return await response.json();
    }
}

const apiClient = new APIClient();
