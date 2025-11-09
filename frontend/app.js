/**
 * Main Application Logic
 */
class App {
    constructor() {
        this.apiClient = apiClient;
        this.currentMode = 'warping';
        
        this.warpingState = {
            sourceImageId: null,
            sourceImageData: null,
            sourceSectors: new SectorManager(),
            targetSectors: new SectorManager(),
            sourceCanvas: null,
            targetCanvas: null,
            resultCanvas: null,
            resultImageDataURL: null,
            selectedSector: null,
            dragging: null,
            debugMode: false
        };
        
        this.mixupState = {
            image1Id: null,
            image1Data: null,
            image2Id: null,
            image2Data: null,
            sectors1: new SectorManager(),
            sectors2: new SectorManager(),
            canvas1: null,
            canvas2: null,
            resultCanvas: null,
            resultImageDataURL: null,
            selectedSector: null,
            dragging: null,
            currentImage: 1,
            sectorMapping: [],
            debugMode: false,
            alpha: DEFAULT_ALPHA,
            highlightOpacity: DEFAULT_HIGHLIGHT_OPACITY
        };
        
        this.init();
    }

    /**
     * Initialize the application.
     */
    init() {
        try {
            this.warpingState.sourceCanvas = new CanvasController('warp-source-canvas');
            this.warpingState.targetCanvas = new CanvasController('warp-target-canvas');
            this.warpingState.resultCanvas = new CanvasController('warp-result-canvas');
            
            this.mixupState.canvas1 = new CanvasController('mixup-source1-canvas');
            this.mixupState.canvas2 = new CanvasController('mixup-source2-canvas');
            this.mixupState.resultCanvas = new CanvasController('mixup-result-canvas');
            
            this.setupTabNavigation();
            this.setupWarpingMode();
            this.setupMixupMode();
            this.setupFileInput();
        } catch (error) {
            this.showError('Error initializing application: ' + error.message);
        }
    }

    /**
     * Set up tab navigation.
     */
    setupTabNavigation() {
        const tabButtons = document.querySelectorAll('.tab-button');
        tabButtons.forEach(btn => {
            btn.addEventListener('click', () => {
                const mode = btn.dataset.mode;
                this.switchMode(mode);
            });
        });
    }

    /**
     * Switch between warping and mixup modes.
     * @param {string} mode - Mode to switch to ('warping' or 'mixup')
     */
    switchMode(mode) {
        this.currentMode = mode;
        
        document.querySelectorAll('.tab-button').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.mode === mode);
        });
        
        document.querySelectorAll('.mode-content').forEach(content => {
            content.classList.toggle('active', content.id === `${mode}-mode`);
        });
        
        requestAnimationFrame(() => {
            if (mode === 'warping') {
                if (this.warpingState.sourceImageId) {
                    this.drawWarpingCanvases();
                }
            } else if (mode === 'mixup') {
                if (this.mixupState.image1Id || this.mixupState.image2Id) {
                    this.drawMixupCanvases();
                }
            }
        });
    }

    /**
     * Set up file input handler.
     */
    setupFileInput() {
        const fileInput = document.getElementById('file-input');
        if (!fileInput) {
            return;
        }
        fileInput.addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (file) {
                try {
                    await this.handleFileUpload(file);
                    fileInput.value = '';
                } catch (error) {
                    this.showError('Error uploading file: ' + error.message);
                }
            }
        });
    }

    /**
     * Handle file upload.
     * @param {File} file - File to upload
     */
    async handleFileUpload(file) {
        try {
            this.showError(null);
            const result = await this.apiClient.uploadImage(file);
            
            if (this.currentMode === 'warping') {
                await this.handleWarpingUpload(result);
            } else {
                await this.handleMixupUpload(result);
            }
        } catch (error) {
            this.showError(error.message);
        }
    }

    /**
     * Handle warping mode image upload.
     * @param {Object} uploadResult - Upload result from API
     */
    async handleWarpingUpload(uploadResult) {
        const { image_id, width, height, preview_url } = uploadResult;
        
        this.warpingState.sourceImageId = image_id;
        this.warpingState.sourceImageData = { width, height };
        
        this.warpingState.sourceSectors.clear();
        this.warpingState.targetSectors.clear();
        
        await this.waitForCanvasReady(this.warpingState.sourceCanvas);
        await this.waitForCanvasReady(this.warpingState.targetCanvas);
        
        await this.warpingState.sourceCanvas.loadImage(preview_url, width, height);
        await this.warpingState.targetCanvas.loadImage(preview_url, width, height);
        
        this.warpingState.sourceSectors.initialize(width, height);
        this.warpingState.targetSectors.initialize(width, height);
        
        this.drawWarpingCanvases();
        this.updateWarpingButtons();
    }

    /**
     * Handle mixup mode image upload.
     * @param {Object} uploadResult - Upload result from API
     */
    async handleMixupUpload(uploadResult) {
        const { image_id, width, height, preview_url } = uploadResult;
        
        if (this.mixupState.currentImage === 1) {
            this.mixupState.image1Id = image_id;
            this.mixupState.image1Data = { width, height };
            this.mixupState.sectors1.clear();
            await this.waitForCanvasReady(this.mixupState.canvas1);
            await this.mixupState.canvas1.loadImage(preview_url, width, height);
            this.mixupState.sectors1.initialize(width, height, MIN_SECTORS);
        } else {
            this.mixupState.image2Id = image_id;
            this.mixupState.image2Data = { width, height };
            this.mixupState.sectors2.clear();
            await this.waitForCanvasReady(this.mixupState.canvas2);
            await this.mixupState.canvas2.loadImage(preview_url, width, height);
            this.mixupState.sectors2.initialize(width, height, MIN_SECTORS);
        }
        
        this.drawMixupCanvases();
        this.updateMixupButtons();
        this.updateSectorMapping();
    }
    
    /**
     * Wait for canvas to be ready.
     * @param {CanvasController} canvasController - Canvas controller
     * @returns {Promise<void>}
     */
    async waitForCanvasReady(canvasController) {
        return new Promise((resolve) => {
            const checkReady = () => {
                const canvas = canvasController.getCanvas();
                const rect = canvas.getBoundingClientRect();
                if (rect.width > 0 && rect.height > 0) {
                    resolve();
                } else {
                    requestAnimationFrame(checkReady);
                }
            };
            setTimeout(() => {
                requestAnimationFrame(checkReady);
            }, 10);
        });
    }

    /**
     * Set up warping mode event handlers.
     */
    setupWarpingMode() {
        const uploadBtn = document.getElementById('warp-upload-btn');
        if (uploadBtn) {
            uploadBtn.addEventListener('click', () => {
                const fileInput = document.getElementById('file-input');
                if (fileInput) {
                    fileInput.click();
                }
            });
        }
        
        const addSectorBtn = document.getElementById('warp-add-sector-btn');
        if (addSectorBtn) {
            addSectorBtn.addEventListener('click', () => {
                try {
                    if (this.warpingState.sourceImageData) {
                        const result1 = this.warpingState.sourceSectors.addBoundaryLine();
                        const result2 = this.warpingState.targetSectors.addBoundaryLine();
                        if (result1 && result2) {
                            this.drawWarpingCanvases();
                            this.updateWarpingButtons();
                        }
                    }
                } catch (error) {
                    this.showError('Error adding sector: ' + error.message);
                }
            });
        }
        
        const applyBtn = document.getElementById('warp-apply-btn');
        if (applyBtn) {
            applyBtn.addEventListener('click', () => {
                this.applyWarp();
            });
        }
        
        const clearBtn = document.getElementById('warp-clear-btn');
        if (clearBtn) {
            clearBtn.addEventListener('click', () => {
                this.clearWarping();
            });
        }
        
        const downloadBtn = document.getElementById('warp-download-btn');
        if (downloadBtn) {
            downloadBtn.addEventListener('click', () => {
                this.downloadWarpResult();
            });
        }
        
        const debugCheckbox = document.getElementById('warp-debug-checkbox');
        if (debugCheckbox) {
            debugCheckbox.addEventListener('change', (e) => {
                this.warpingState.debugMode = e.target.checked;
            });
        }
        
        this.setupWarpingCanvasEvents();
    }

    /**
     * Set up warping canvas event handlers.
     */
    setupWarpingCanvasEvents() {
        this.setupCanvasDragEvents(
            this.warpingState.sourceCanvas,
            this.warpingState.sourceSectors,
            () => this.drawWarpingCanvases()
        );
        
        this.setupCanvasDragEvents(
            this.warpingState.targetCanvas,
            this.warpingState.targetSectors,
            () => this.drawWarpingCanvases()
        );
    }

    /**
     * Set up canvas drag event handlers.
     * @param {CanvasController} canvasController - Canvas controller
     * @param {SectorManager} sectorManager - Sector manager
     * @param {Function} onUpdate - Callback function when update occurs
     */
    setupCanvasDragEvents(canvasController, sectorManager, onUpdate) {
        const canvas = canvasController.getCanvas();
        let isDragging = false;
        let dragTarget = null;
        
        canvas.addEventListener('mousedown', (e) => {
            if (!canvasController.getImageData()) {
                return;
            }
            
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            const threshold = calculateAdaptiveThreshold(DRAG_THRESHOLD, canvasController.scale);
            
            const sharedCenter = sectorManager.getSharedCenter();
            if (sharedCenter) {
                const centerCanvas = canvasController.imageToCanvas(sharedCenter.x, sharedCenter.y);
                if (centerCanvas) {
                    const dist = Math.sqrt(
                        Math.pow(x - centerCanvas.x, 2) + 
                        Math.pow(y - centerCanvas.y, 2)
                    );
                    if (dist < threshold) {
                        isDragging = true;
                        dragTarget = { type: 'center' };
                        canvas.style.cursor = 'grabbing';
                        e.preventDefault();
                        return;
                    }
                }
            }
            
            const boundaryLines = sectorManager.getBoundaryLines();
            if (boundaryLines && sharedCenter) {
                const centerCanvas = canvasController.imageToCanvas(sharedCenter.x, sharedCenter.y);
                if (centerCanvas) {
                    const hitThreshold = calculateHitThreshold(CIRCLE_RADIUS.selected, canvasController.scale);
                    
                    let closestLine = null;
                    let closestDistance = Infinity;
                    
                    for (let i = 0; i < boundaryLines.length; i++) {
                        const line = boundaryLines[i];
                        const lineCanvas = canvasController.imageToCanvas(line.x, line.y);
                        if (lineCanvas) {
                            const dist = Math.sqrt(
                                Math.pow(x - lineCanvas.x, 2) + 
                                Math.pow(y - lineCanvas.y, 2)
                            );
                            if (dist < hitThreshold && dist < closestDistance) {
                                closestDistance = dist;
                                closestLine = { type: 'boundaryLine', lineIndex: i };
                            }
                        }
                    }
                    
                    if (closestLine) {
                        isDragging = true;
                        dragTarget = closestLine;
                        canvas.style.cursor = 'grabbing';
                        e.preventDefault();
                        return;
                    }
                }
            }
        });
        
        canvas.addEventListener('mousemove', (e) => {
            if (!canvasController.getImageData()) {
                canvas.style.cursor = 'default';
                return;
            }
            
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            const threshold = calculateAdaptiveThreshold(DRAG_THRESHOLD, canvasController.scale);
            
            if (isDragging && dragTarget) {
                const imgPoint = canvasController.canvasToImage(x, y);
                if (imgPoint) {
                    if (dragTarget.type === 'center') {
                        sectorManager.setSharedCenter(imgPoint.x, imgPoint.y);
                    } else if (dragTarget.type === 'boundaryLine') {
                        const result = sectorManager.updateBoundaryLine(
                            dragTarget.lineIndex,
                            imgPoint.x,
                            imgPoint.y
                        );
                        if (result && result.newIndex !== undefined && result.newIndex !== dragTarget.lineIndex) {
                            dragTarget.lineIndex = result.newIndex;
                        }
                    }
                    
                    onUpdate();
                }
            } else {
                const sharedCenter = sectorManager.getSharedCenter();
                let hovered = false;
                
                if (sharedCenter) {
                    const centerCanvas = canvasController.imageToCanvas(sharedCenter.x, sharedCenter.y);
                    if (centerCanvas) {
                        const dist = Math.sqrt(
                            Math.pow(x - centerCanvas.x, 2) + 
                            Math.pow(y - centerCanvas.y, 2)
                        );
                        if (dist < threshold) {
                            canvas.style.cursor = 'grab';
                            hovered = true;
                        }
                    }
                }
                
                if (!hovered) {
                    const boundaryLines = sectorManager.getBoundaryLines();
                    const sharedCenter = sectorManager.getSharedCenter();
                    if (boundaryLines && sharedCenter) {
                        const hitThreshold = calculateHitThreshold(CIRCLE_RADIUS.selected, canvasController.scale);
                        
                        let closestDistance = Infinity;
                        
                        for (let i = 0; i < boundaryLines.length; i++) {
                            const line = boundaryLines[i];
                            const lineCanvas = canvasController.imageToCanvas(line.x, line.y);
                            if (lineCanvas) {
                                const dist = Math.sqrt(
                                    Math.pow(x - lineCanvas.x, 2) + 
                                    Math.pow(y - lineCanvas.y, 2)
                                );
                                if (dist < hitThreshold && dist < closestDistance) {
                                    closestDistance = dist;
                                }
                            }
                        }
                        
                        canvas.style.cursor = closestDistance < hitThreshold ? 'grab' : 'crosshair';
                    } else {
                        canvas.style.cursor = 'crosshair';
                    }
                }
            }
        });
        
        canvas.addEventListener('mouseup', () => {
            isDragging = false;
            dragTarget = null;
            canvas.style.cursor = 'crosshair';
        });
        
        canvas.addEventListener('contextmenu', (e) => {
            e.preventDefault();
            
            if (!canvasController.getImageData()) {
                return;
            }
            
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            const threshold = calculateAdaptiveThreshold(DRAG_THRESHOLD, canvasController.scale);
            
            const boundaryLines = sectorManager.getBoundaryLines();
            const sharedCenter = sectorManager.getSharedCenter();
            if (boundaryLines && sharedCenter) {
                for (let i = 0; i < boundaryLines.length; i++) {
                    const line = boundaryLines[i];
                    const lineCanvas = canvasController.imageToCanvas(line.x, line.y);
                    if (lineCanvas) {
                        const dist = Math.sqrt(
                            Math.pow(x - lineCanvas.x, 2) + 
                            Math.pow(y - lineCanvas.y, 2)
                        );
                        if (dist < threshold) {
                            const deleted = sectorManager.deleteBoundaryLine(i, MIN_SECTORS);
                            if (deleted) {
                                onUpdate();
                            } else {
                                this.showError(`Minimum ${MIN_SECTORS} sectors required`);
                            }
                            break;
                        }
                    }
                }
            }
        });
    }

    /**
     * Draw warping canvases.
     */
    drawWarpingCanvases() {
        this.warpingState.sourceCanvas.draw();
        const sharedCenter1 = this.warpingState.sourceSectors.getSharedCenter();
        const boundaryLines1 = this.warpingState.sourceSectors.getBoundaryLines();
        this.warpingState.sourceCanvas.drawSectors(
            this.warpingState.sourceSectors.getSectors(),
            sharedCenter1,
            boundaryLines1
        );
        
        if (this.warpingState.sourceImageData && this.warpingState.targetCanvas.getImageData()) {
            this.warpingState.targetCanvas.draw();
            const sharedCenter2 = this.warpingState.targetSectors.getSharedCenter();
            const boundaryLines2 = this.warpingState.targetSectors.getBoundaryLines();
            this.warpingState.targetCanvas.drawSectors(
                this.warpingState.targetSectors.getSectors(),
                sharedCenter2,
                boundaryLines2
            );
        }
    }

    /**
     * Apply warp transformation.
     */
    async applyWarp() {
        try {
            this.showError(null);
            
            const sourceSectors = this.warpingState.sourceSectors.exportSectors();
            const targetSectors = this.warpingState.targetSectors.exportSectors();
            
            if (sourceSectors.length !== targetSectors.length) {
                throw new Error('Source and target must have the same number of sectors');
            }
            
            if (!validateSectorCount(sourceSectors.length, MIN_SECTORS)) {
                throw new Error(`Please create at least ${MIN_SECTORS} sectors for warping`);
            }
            
            const result = await this.apiClient.warpImage(
                this.warpingState.sourceImageId,
                sourceSectors,
                targetSectors,
                this.warpingState.debugMode
            );
            
            await this.warpingState.resultCanvas.drawResult(
                result.result_image,
                this.warpingState.sourceImageData.width,
                this.warpingState.sourceImageData.height
            );
            
            this.warpingState.resultImageDataURL = result.result_image;
            this.updateWarpingButtons();
        } catch (error) {
            this.showError(error.message);
        }
    }

    /**
     * Download warp result image.
     */
    downloadWarpResult() {
        if (!this.warpingState.resultImageDataURL) {
            this.showError('No result image to download');
            return;
        }
        
        this.downloadImage(this.warpingState.resultImageDataURL, 'warped-image.jpg');
    }

    /**
     * Clear warping state.
     */
    clearWarping() {
        this.warpingState.sourceImageId = null;
        this.warpingState.sourceImageData = null;
        this.warpingState.sourceSectors.clear();
        this.warpingState.targetSectors.clear();
        this.warpingState.sourceCanvas.clear();
        this.warpingState.targetCanvas.clear();
        this.warpingState.resultCanvas.clear();
        this.warpingState.resultImageDataURL = null;
        this.showError(null);
        this.updateWarpingButtons();
    }

    /**
     * Update warping mode buttons state.
     */
    updateWarpingButtons() {
        const sectorCount = this.warpingState.sourceSectors.count();
        const canApply = this.warpingState.sourceImageId &&
                        validateSectorCount(sectorCount, MIN_SECTORS) &&
                        validateSectorCount(this.warpingState.targetSectors.count(), MIN_SECTORS) &&
                        this.warpingState.sourceSectors.count() === this.warpingState.targetSectors.count();
        
        const applyBtn = document.getElementById('warp-apply-btn');
        if (applyBtn) {
            applyBtn.disabled = !canApply;
        }
        
        const downloadBtn = document.getElementById('warp-download-btn');
        if (downloadBtn) {
            downloadBtn.disabled = !this.warpingState.resultImageDataURL;
        }
    }

    /**
     * Set up mixup mode event handlers.
     */
    setupMixupMode() {
        const upload1Btn = document.getElementById('mixup-upload1-btn');
        if (upload1Btn) {
            upload1Btn.addEventListener('click', () => {
                this.mixupState.currentImage = 1;
                const fileInput = document.getElementById('file-input');
                if (fileInput) {
                    fileInput.click();
                }
            });
        }
        
        const upload2Btn = document.getElementById('mixup-upload2-btn');
        if (upload2Btn) {
            upload2Btn.addEventListener('click', () => {
                this.mixupState.currentImage = 2;
                const fileInput = document.getElementById('file-input');
                if (fileInput) {
                    fileInput.click();
                }
            });
        }
        
        const mixupAddSectorBtn = document.getElementById('mixup-add-sector-btn');
        if (mixupAddSectorBtn) {
            mixupAddSectorBtn.addEventListener('click', () => {
                try {
                    if (this.mixupState.image1Data && this.mixupState.image2Data) {
                        const result1 = this.mixupState.sectors1.addBoundaryLine();
                        const result2 = this.mixupState.sectors2.addBoundaryLine();
                        if (result1 && result2) {
                            this.drawMixupCanvases();
                            this.updateSectorMapping();
                            this.updateMixupButtons();
                        }
                    } else {
                        this.showError('Please upload both images before adding sectors');
                    }
                } catch (error) {
                    this.showError('Error adding sector: ' + error.message);
                }
            });
        }
        
        const mixupGenerateBtn = document.getElementById('mixup-generate-btn');
        if (mixupGenerateBtn) {
            mixupGenerateBtn.addEventListener('click', () => {
                this.generateMixup();
            });
        }
        
        const mixupClearBtn = document.getElementById('mixup-clear-btn');
        if (mixupClearBtn) {
            mixupClearBtn.addEventListener('click', () => {
                this.clearMixup();
            });
        }
        
        const mixupDownloadBtn = document.getElementById('mixup-download-btn');
        if (mixupDownloadBtn) {
            mixupDownloadBtn.addEventListener('click', () => {
                this.downloadMixupResult();
            });
        }
        
        const mixupAlphaSlider = document.getElementById('mixup-alpha-slider');
        if (mixupAlphaSlider) {
            mixupAlphaSlider.value = DEFAULT_ALPHA;
            const alphaValue = document.getElementById('mixup-alpha-value');
            if (alphaValue) {
                alphaValue.textContent = DEFAULT_ALPHA.toFixed(1);
            }
            mixupAlphaSlider.addEventListener('input', (e) => {
                this.mixupState.alpha = parseFloat(e.target.value);
                if (alphaValue) {
                    alphaValue.textContent = this.mixupState.alpha.toFixed(1);
                }
            });
        }
        
        const mixupHighlightOpacitySlider = document.getElementById('mixup-highlight-opacity-slider');
        if (mixupHighlightOpacitySlider) {
            mixupHighlightOpacitySlider.value = DEFAULT_HIGHLIGHT_OPACITY;
            const opacityValue = document.getElementById('mixup-highlight-opacity-value');
            if (opacityValue) {
                opacityValue.textContent = DEFAULT_HIGHLIGHT_OPACITY.toFixed(1);
            }
            mixupHighlightOpacitySlider.addEventListener('input', (e) => {
                this.mixupState.highlightOpacity = parseFloat(e.target.value);
                if (opacityValue) {
                    opacityValue.textContent = this.mixupState.highlightOpacity.toFixed(1);
                }
                this.drawMixupCanvases();
            });
        }
        
        this.setupMixupCanvasEvents();
    }

    /**
     * Set up mixup canvas event handlers.
     */
    setupMixupCanvasEvents() {
        this.setupCanvasDragEvents(
            this.mixupState.canvas1,
            this.mixupState.sectors1,
            () => {
                this.drawMixupCanvases();
                this.updateSectorMapping();
            }
        );
        
        this.setupCanvasDragEvents(
            this.mixupState.canvas2,
            this.mixupState.sectors2,
            () => {
                this.drawMixupCanvases();
                this.updateSectorMapping();
            }
        );
    }

    /**
     * Draw mixup canvases with highlighting.
     */
    drawMixupCanvases() {
        const sectorMapping = this.mixupState.sectorMapping;
        
        this.mixupState.canvas1.draw();
        const sharedCenter1 = this.mixupState.sectors1.getSharedCenter();
        const boundaryLines1 = this.mixupState.sectors1.getBoundaryLines();
        const sectors1 = this.mixupState.sectors1.getSectors();
        
        const mappedSrcIndices = new Set();
        sectorMapping.forEach(m => {
            if (m.src_index !== null && m.src_index !== undefined && m.src_index >= 0 && m.src_index < sectors1.length) {
                mappedSrcIndices.add(m.src_index);
            }
        });
        
        this.mixupState.canvas1.drawSectors(
            sectors1,
            sharedCenter1,
            boundaryLines1,
            null,
            null,
            mappedSrcIndices.size > 0 ? mappedSrcIndices : null,
            null,
            this.mixupState.highlightOpacity
        );
        
        this.mixupState.canvas2.draw();
        const sharedCenter2 = this.mixupState.sectors2.getSharedCenter();
        const boundaryLines2 = this.mixupState.sectors2.getBoundaryLines();
        const sectors2 = this.mixupState.sectors2.getSectors();
        
        const mappedDstIndices = new Set();
        sectorMapping.forEach(m => {
            if (m.dst_index !== null && m.dst_index !== undefined && m.dst_index >= 0 && m.dst_index < sectors2.length) {
                mappedDstIndices.add(m.dst_index);
            }
        });
        
        const colorMap = new Map();
        sectorMapping.forEach(m => {
            if (m.dst_index !== null && m.dst_index !== undefined && m.dst_index >= 0 && m.dst_index < sectors2.length &&
                m.src_index !== null && m.src_index !== undefined && m.src_index >= 0 && m.src_index < sectors1.length) {
                colorMap.set(m.dst_index, m.src_index);
            }
        });
        
        this.mixupState.canvas2.drawSectors(
            sectors2,
            sharedCenter2,
            boundaryLines2,
            null,
            null,
            mappedDstIndices.size > 0 ? mappedDstIndices : null,
            colorMap.size > 0 ? colorMap : null,
            this.mixupState.highlightOpacity
        );
    }

    /**
     * Update sector mapping UI.
     */
    updateSectorMapping() {
        const sourceContainer = document.getElementById('mixup-sector-mapping-source');
        const mixinContainer = document.getElementById('mixup-sector-mapping-mixin');
        const arrowContainer = document.getElementById('mixup-sector-mapping-arrow');
        
        if (!sourceContainer || !mixinContainer || !arrowContainer) {
            const oldContainer = document.getElementById('mixup-sector-mapping');
            if (oldContainer) {
                oldContainer.innerHTML = '<p>Please reload the page to see the new sector mapping interface</p>';
            }
            return;
        }
        
        sourceContainer.innerHTML = '';
        mixinContainer.innerHTML = '';
        arrowContainer.innerHTML = '';
        
        const sectors1 = this.mixupState.sectors1.getSectors();
        const sectors2 = this.mixupState.sectors2.getSectors();
        
        if (sectors1.length === 0) {
            sourceContainer.innerHTML = '<p>Add sectors to Source Image to create mappings</p>';
            return;
        }
        
        sectors1.forEach((sector, index) => {
            const sourceItem = document.createElement('div');
            sourceItem.className = 'mapping-item';
            
            const sectorColor = CanvasController.getSectorColor(index);
            sourceItem.style.backgroundColor = replaceColorOpacity(sectorColor, 0.1);
            sourceItem.style.borderLeft = `4px solid ${replaceColorOpacity(sectorColor, 1)}`;
            
            const sourceLabel = document.createElement('label');
            sourceLabel.style.fontWeight = 'bold';
            sourceLabel.style.display = 'flex';
            sourceLabel.style.alignItems = 'center';
            sourceLabel.style.gap = '8px';
            
            const colorIndicator = document.createElement('span');
            colorIndicator.style.width = '20px';
            colorIndicator.style.height = '20px';
            colorIndicator.style.borderRadius = '50%';
            colorIndicator.style.backgroundColor = replaceColorOpacity(sectorColor, 1);
            colorIndicator.style.border = '2px solid #333';
            colorIndicator.style.display = 'inline-block';
            colorIndicator.style.flexShrink = '0';
            
            const sourceLabelText = document.createTextNode(`Sector ${index + 1}`);
            sourceLabel.appendChild(colorIndicator);
            sourceLabel.appendChild(sourceLabelText);
            
            sourceItem.appendChild(sourceLabel);
            sourceContainer.appendChild(sourceItem);
            
            const arrowItem = document.createElement('div');
            arrowItem.className = 'mapping-item-arrow';
            const arrowText = document.createTextNode('← mixin ←');
            arrowItem.appendChild(arrowText);
            arrowContainer.appendChild(arrowItem);
            
            const mixinItem = document.createElement('div');
            mixinItem.className = 'mapping-item';
            
            const existing = this.mixupState.sectorMapping.find(m => m.src_index === index);
            const selectedDstIndex = existing ? existing.dst_index : null;
            
            const select = document.createElement('select');
            select.style.width = '100%';
            select.innerHTML = '<option value="">(select sector)</option>';
            sectors2.forEach((s, i) => {
                const option = document.createElement('option');
                option.value = i;
                option.textContent = `Sector ${i + 1}`;
                if (selectedDstIndex === i) {
                    option.selected = true;
                }
                select.appendChild(option);
            });
            
            const sourceSectorColor = CanvasController.getSectorColor(index);
            mixinItem.style.backgroundColor = replaceColorOpacity(sourceSectorColor, 0.1);
            mixinItem.style.borderLeft = `4px solid ${replaceColorOpacity(sourceSectorColor, 1)}`;
            
            select.addEventListener('change', (e) => {
                const dstIndex = e.target.value === '' ? null : parseInt(e.target.value);
                this.mixupState.sectorMapping = this.mixupState.sectorMapping.filter(m => m.src_index !== index);
                if (dstIndex !== null) {
                    this.mixupState.sectorMapping.push({ src_index: index, dst_index: dstIndex });
                }
                this.drawMixupCanvases();
                this.updateMixupButtons();
            });
            mixinItem.appendChild(select);
            mixinContainer.appendChild(mixinItem);
        });
        
        this.updateMixupButtons();
    }

    /**
     * Generate mixup result.
     */
    async generateMixup() {
        try {
            this.showError(null);
            
            const sectors1 = this.mixupState.sectors1.exportSectors();
            const sectors2 = this.mixupState.sectors2.exportSectors();
            const sectorMapping = this.mixupState.sectorMapping;
            
            if (!validateSectorCount(sectors1.length, MIN_SECTORS) || !validateSectorCount(sectors2.length, MIN_SECTORS)) {
                throw new Error(`Please create at least ${MIN_SECTORS} sectors for both images`);
            }
            
            if (sectorMapping.length === 0) {
                throw new Error('Please create at least one sector mapping');
            }
            
            const result = await this.apiClient.mixupImages(
                this.mixupState.image1Id,
                this.mixupState.image2Id,
                sectors1,
                sectors2,
                sectorMapping,
                false,
                this.mixupState.alpha
            );
            
            const resultWidth = this.mixupState.image1Data ? this.mixupState.image1Data.width : null;
            const resultHeight = this.mixupState.image1Data ? this.mixupState.image1Data.height : null;
            await this.mixupState.resultCanvas.drawResult(result.result_image, resultWidth, resultHeight);
            
            this.mixupState.resultImageDataURL = result.result_image;
            this.updateMixupButtons();
        } catch (error) {
            this.showError(error.message);
        }
    }

    /**
     * Download mixup result image.
     */
    downloadMixupResult() {
        if (!this.mixupState.resultImageDataURL) {
            this.showError('No result image to download');
            return;
        }
        
        this.downloadImage(this.mixupState.resultImageDataURL, 'mixup-image.jpg');
    }

    /**
     * Clear mixup state.
     */
    clearMixup() {
        this.mixupState.image1Id = null;
        this.mixupState.image1Data = null;
        this.mixupState.image2Id = null;
        this.mixupState.image2Data = null;
        this.mixupState.sectors1.clear();
        this.mixupState.sectors2.clear();
        this.mixupState.sectorMapping = [];
        this.mixupState.canvas1.clear();
        this.mixupState.canvas2.clear();
        this.mixupState.resultCanvas.clear();
        this.mixupState.resultImageDataURL = null;
        this.showError(null);
        this.updateSectorMapping();
        this.updateMixupButtons();
    }

    /**
     * Update mixup mode buttons state.
     */
    updateMixupButtons() {
        const sectorCount1 = this.mixupState.sectors1.count();
        const sectorCount2 = this.mixupState.sectors2.count();
        const canGenerate = this.mixupState.image1Id &&
                           this.mixupState.image2Id &&
                           validateSectorCount(sectorCount1, MIN_SECTORS) &&
                           validateSectorCount(sectorCount2, MIN_SECTORS) &&
                           this.mixupState.sectorMapping.length > 0;
        
        const generateBtn = document.getElementById('mixup-generate-btn');
        if (generateBtn) {
            generateBtn.disabled = !canGenerate;
        }
        
        const downloadBtn = document.getElementById('mixup-download-btn');
        if (downloadBtn) {
            downloadBtn.disabled = !this.mixupState.resultImageDataURL;
        }
    }

    /**
     * Show or hide error message.
     * @param {string|null} message - Error message to display, or null to hide
     */
    showError(message) {
        const errorElement = this.currentMode === 'warping' 
            ? document.getElementById('warp-error-message')
            : document.getElementById('mixup-error-message');
        
        if (!errorElement) return;
        
        if (message) {
            errorElement.textContent = message;
            errorElement.classList.add('show');
        } else {
            errorElement.textContent = '';
            errorElement.classList.remove('show');
        }
    }

    /**
     * Download image from data URL.
     * @param {string} dataURL - Image data URL (base64 encoded)
     * @param {string} filename - Filename for download
     */
    downloadImage(dataURL, filename) {
        try {
            if (!dataURL) {
                this.showError('No image data available');
                return;
            }
            
            const link = document.createElement('a');
            link.href = dataURL;
            link.download = filename;
            link.style.display = 'none';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        } catch (error) {
            this.showError('Failed to download image: ' + error.message);
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new App();
});
