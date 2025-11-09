// API Client for communicating with backend
class APIClient {
    constructor() {
        this.baseURL = API_BASE_URL;
        this.sessionId = null;
    }
    
    /**
     * Set session ID for all API requests.
     * @param {string} sessionId - Session ID
     */
    setSessionId(sessionId) {
        this.sessionId = sessionId;
    }
    
    /**
     * Get headers with session ID if available.
     * @returns {Object} Headers object
     */
    getHeaders(includeContentType = true) {
        const headers = {};
        
        if (includeContentType) {
            headers['Content-Type'] = 'application/json';
        }
        
        if (this.sessionId) {
            headers['X-Session-ID'] = this.sessionId;
        }
        
        return headers;
    }

    async uploadImage(file) {
        const formData = new FormData();
        formData.append('image', file);
        
        const headers = {};
        if (this.sessionId) {
            headers['X-Session-ID'] = this.sessionId;
        }

        const response = await fetch(`${this.baseURL}/api/upload`, {
            method: 'POST',
            headers: headers,
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
            headers: this.getHeaders(),
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
            headers: this.getHeaders(),
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

    async verifyImage(imageId) {
        try {
            if (!imageId) {
                return { exists: false };
            }
            
            let url = `${this.baseURL}/api/verify-image/${encodeURIComponent(imageId)}`;
            if (this.sessionId) {
                url += `?session_id=${encodeURIComponent(this.sessionId)}`;
            }
            
            const response = await fetch(url, {
                headers: this.getHeaders(false)
            });
            
            if (!response.ok) {
                console.error(`verifyImage: Response not OK for image ${imageId}: ${response.status} ${response.statusText}`);
                return { exists: false };
            }
            
            const result = await response.json();
            return result;
        } catch (error) {
            console.error(`verifyImage: Error verifying image ${imageId}:`, error);
            return { exists: false };
        }
    }

    async healthCheck() {
        const response = await fetch(`${this.baseURL}/health`);
        return await response.json();
    }
}

const apiClient = new APIClient();
