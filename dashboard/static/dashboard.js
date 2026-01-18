/**
 * StelX Surveillance Dashboard - Professional Edition
 * Handles file upload, video/image processing, and real-time detection display
 */

class SurveillanceDashboard {
    constructor() {
        // State
        this.isConnected = false;
        this.isPlaying = false;
        this.currentFile = null;
        this.videoSocket = null;
        this.controlSocket = null;

        // Detection data
        this.detections = [];
        this.plates = [];
        this.selectedTrackId = null;

        // Stats
        this.stats = {
            vehicles: 0,
            plates: 0,
            fps: 0,
            crowd_count: 0,
            crowd_density: 'low'
        };

        // Canvas
        this.canvas = null;
        this.ctx = null;

        // Initialize
        this.init();
    }

    init() {
        // Get canvas
        this.canvas = document.getElementById('videoCanvas');
        this.ctx = this.canvas.getContext('2d');

        // Resize canvas
        this.resizeCanvas();
        window.addEventListener('resize', () => this.resizeCanvas());

        // Canvas click for tracking
        this.canvas.addEventListener('click', (e) => this.handleCanvasClick(e));

        // Setup file upload
        this.setupFileUpload();

        // Setup modals
        this.setupModals();

        // Setup controls
        this.setupControls();

        // Update datetime
        this.updateDateTime();
        setInterval(() => this.updateDateTime(), 1000);

        // Connect to WebSocket
        this.connect();
    }

    resizeCanvas() {
        const container = this.canvas.parentElement;
        this.canvas.width = container.clientWidth;
        this.canvas.height = container.clientHeight;
    }

    setupFileUpload() {
        const uploadZone = document.getElementById('uploadZone');
        const fileInput = document.getElementById('fileInput');
        const modalFileInput = document.getElementById('modalFileInput');
        const browseBtn = document.getElementById('browseBtn');
        const navUpload = document.getElementById('navUpload');

        // Click to upload
        uploadZone?.addEventListener('click', () => fileInput?.click());
        browseBtn?.addEventListener('click', () => modalFileInput?.click());
        navUpload?.addEventListener('click', () => {
            document.getElementById('uploadModal')?.classList.add('active');
        });

        // File input change
        fileInput?.addEventListener('change', (e) => this.handleFileSelect(e));
        modalFileInput?.addEventListener('change', (e) => {
            this.handleFileSelect(e);
            document.getElementById('uploadModal')?.classList.remove('active');
        });

        // Drag and drop
        uploadZone?.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadZone.style.borderColor = '#3b82f6';
            uploadZone.style.background = 'rgba(59, 130, 246, 0.1)';
        });

        uploadZone?.addEventListener('dragleave', () => {
            uploadZone.style.borderColor = '';
            uploadZone.style.background = '';
        });

        uploadZone?.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadZone.style.borderColor = '';
            uploadZone.style.background = '';

            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.processFile(files[0]);
            }
        });
    }

    setupModals() {
        // Upload modal
        document.getElementById('closeUploadModal')?.addEventListener('click', () => {
            document.getElementById('uploadModal')?.classList.remove('active');
        });

        // Settings modal
        document.getElementById('settingsBtn')?.addEventListener('click', () => {
            document.getElementById('settingsModal')?.classList.add('active');
        });

        document.getElementById('closeSettingsModal')?.addEventListener('click', () => {
            document.getElementById('settingsModal')?.classList.remove('active');
        });

        // Close on backdrop click
        document.querySelectorAll('.modal-backdrop').forEach(backdrop => {
            backdrop.addEventListener('click', () => {
                backdrop.parentElement?.classList.remove('active');
            });
        });

        // Confidence slider
        document.getElementById('confSlider')?.addEventListener('input', (e) => {
            document.getElementById('confValue').textContent = Math.round(e.target.value * 100) + '%';
            this.sendControl({ action: 'set_confidence', value: parseFloat(e.target.value) });
        });
    }

    setupControls() {
        // Play/Pause
        document.getElementById('playBtn')?.addEventListener('click', () => {
            this.togglePlayPause();
        });

        // Clear detections
        document.getElementById('clearDetections')?.addEventListener('click', () => {
            this.detections = [];
            this.updateDetectionGrid();
        });

        // Export plates
        document.getElementById('exportPlates')?.addEventListener('click', () => {
            this.exportPlates();
        });

        // Progress bar
        document.getElementById('progressBar')?.addEventListener('click', (e) => {
            const rect = e.target.getBoundingClientRect();
            const percent = (e.clientX - rect.left) / rect.width;
            this.seekTo(percent);
        });
    }

    handleFileSelect(event) {
        const file = event.target.files[0];
        if (file) {
            this.processFile(file);
        }
    }

    async processFile(file) {
        this.currentFile = file;

        // Hide upload overlay
        document.getElementById('videoOverlay')?.classList.add('hidden');

        // Determine file type
        const isImage = file.type.startsWith('image/');
        const isVideo = file.type.startsWith('video/');

        if (isImage) {
            await this.processImage(file);
        } else if (isVideo) {
            await this.processVideo(file);
        } else {
            alert('Unsupported file type. Please upload an image or video.');
        }
    }

    async processImage(file) {
        const img = new Image();
        img.onload = async () => {
            // Draw image on canvas
            this.resizeCanvas();

            const scale = Math.min(
                this.canvas.width / img.width,
                this.canvas.height / img.height
            );
            const x = (this.canvas.width - img.width * scale) / 2;
            const y = (this.canvas.height - img.height * scale) / 2;

            this.ctx.drawImage(img, x, y, img.width * scale, img.height * scale);

            // Store transform for click handling
            this.frameTransform = { x, y, scale, width: img.width, height: img.height };

            // Send to server for processing
            this.sendImageForProcessing(file);
        };
        img.src = URL.createObjectURL(file);
    }

    async processVideo(file) {
        const video = document.createElement('video');
        video.src = URL.createObjectURL(file);
        video.muted = true;

        this.currentVideo = video;
        this.isPlaying = false;

        video.addEventListener('loadedmetadata', () => {
            // Start processing frames
            this.updatePlayButton();
        });

        video.addEventListener('timeupdate', () => {
            const progress = (video.currentTime / video.duration) * 100;
            document.getElementById('progressFill').style.width = progress + '%';
        });

        video.addEventListener('ended', () => {
            this.isPlaying = false;
            this.updatePlayButton();
        });

        // Draw first frame
        video.addEventListener('loadeddata', () => {
            this.drawVideoFrame();
        });
    }

    drawVideoFrame() {
        if (!this.currentVideo) return;

        const video = this.currentVideo;
        this.resizeCanvas();

        const scale = Math.min(
            this.canvas.width / video.videoWidth,
            this.canvas.height / video.videoHeight
        );
        const x = (this.canvas.width - video.videoWidth * scale) / 2;
        const y = (this.canvas.height - video.videoHeight * scale) / 2;

        this.ctx.drawImage(video, x, y, video.videoWidth * scale, video.videoHeight * scale);

        this.frameTransform = { x, y, scale, width: video.videoWidth, height: video.videoHeight };

        if (this.isPlaying) {
            // Send frame for processing
            this.canvas.toBlob((blob) => {
                if (blob) {
                    this.sendFrameForProcessing(blob);
                }
            }, 'image/jpeg', 0.8);

            requestAnimationFrame(() => this.drawVideoFrame());
        }
    }

    togglePlayPause() {
        if (!this.currentVideo) return;

        if (this.isPlaying) {
            this.currentVideo.pause();
            this.isPlaying = false;
        } else {
            this.currentVideo.play();
            this.isPlaying = true;
            this.drawVideoFrame();
        }

        this.updatePlayButton();
    }

    updatePlayButton() {
        const playBtn = document.getElementById('playBtn');
        if (!playBtn) return;

        if (this.isPlaying) {
            playBtn.innerHTML = '<svg viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>';
        } else {
            playBtn.innerHTML = '<svg viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>';
        }
    }

    seekTo(percent) {
        if (!this.currentVideo) return;
        this.currentVideo.currentTime = this.currentVideo.duration * percent;
    }

    async sendImageForProcessing(file) {
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/api/process_image', {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                const result = await response.json();
                this.handleProcessingResult(result);
            }
        } catch (error) {
            console.error('Error processing image:', error);
        }
    }

    async sendFrameForProcessing(blob) {
        const formData = new FormData();
        formData.append('frame', blob);

        try {
            const response = await fetch('/api/process_frame', {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                const result = await response.json();
                this.handleProcessingResult(result);
            }
        } catch (error) {
            console.error('Error processing frame:', error);
        }
    }

    handleProcessingResult(result) {
        // Update stats
        if (result.stats) {
            this.stats = result.stats;
            this.updateStatsDisplay();
        }

        // Update crowd data
        if (result.crowd) {
            this.stats.crowd_count = result.crowd.count;
            this.stats.crowd_density = result.crowd.density;
            this.updateCrowdDisplay();
        }

        // Update detections
        if (result.detections) {
            result.detections.forEach(det => {
                if (!this.detections.find(d => d.track_id === det.track_id)) {
                    this.detections.push(det);
                }
            });
            this.updateDetectionGrid();
        }

        // Update plates
        if (result.plates) {
            result.plates.forEach(plate => {
                if (!this.plates.find(p => p.text === plate.text)) {
                    this.plates.unshift({
                        ...plate,
                        time: new Date().toLocaleTimeString()
                    });
                }
            });
            this.updatePlatesList();
        }

        // Draw annotations on canvas
        if (result.frame_base64) {
            this.drawProcessedFrame(result.frame_base64);
        } else if (result.annotations) {
            this.drawAnnotations(result.annotations);
        }
    }

    drawProcessedFrame(base64Data) {
        const img = new Image();
        img.onload = () => {
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

            const scale = Math.min(
                this.canvas.width / img.width,
                this.canvas.height / img.height
            );
            const x = (this.canvas.width - img.width * scale) / 2;
            const y = (this.canvas.height - img.height * scale) / 2;

            this.ctx.drawImage(img, x, y, img.width * scale, img.height * scale);
            this.frameTransform = { x, y, scale, width: img.width, height: img.height };
        };
        img.src = `data:image/jpeg;base64,${base64Data}`;
    }

    drawAnnotations(annotations) {
        if (!this.frameTransform) return;

        const { x: offsetX, y: offsetY, scale } = this.frameTransform;

        annotations.forEach(ann => {
            const [x1, y1, x2, y2] = ann.bbox;

            // Scale bbox to canvas
            const sx1 = offsetX + x1 * scale;
            const sy1 = offsetY + y1 * scale;
            const sw = (x2 - x1) * scale;
            const sh = (y2 - y1) * scale;

            // Draw box
            this.ctx.strokeStyle = ann.selected ? '#06b6d4' : '#10b981';
            this.ctx.lineWidth = ann.selected ? 3 : 2;
            this.ctx.strokeRect(sx1, sy1, sw, sh);

            // Draw label
            const label = `${ann.class || 'Vehicle'} ${ann.track_id ? '#' + ann.track_id : ''}`;
            this.ctx.fillStyle = ann.selected ? '#06b6d4' : '#10b981';
            this.ctx.font = '12px Inter, sans-serif';
            this.ctx.fillRect(sx1, sy1 - 20, this.ctx.measureText(label).width + 10, 20);
            this.ctx.fillStyle = 'white';
            this.ctx.fillText(label, sx1 + 5, sy1 - 6);

            // Draw plate if exists
            if (ann.plate) {
                this.ctx.fillStyle = '#3b82f6';
                this.ctx.fillRect(sx1, sy1 + sh + 4, this.ctx.measureText(ann.plate).width + 10, 20);
                this.ctx.fillStyle = 'white';
                this.ctx.fillText(ann.plate, sx1 + 5, sy1 + sh + 18);
            }
        });
    }

    handleCanvasClick(event) {
        if (!this.frameTransform) return;

        const rect = this.canvas.getBoundingClientRect();
        const canvasX = event.clientX - rect.left;
        const canvasY = event.clientY - rect.top;

        // Convert to frame coordinates
        const { x: offsetX, y: offsetY, scale } = this.frameTransform;
        const frameX = Math.round((canvasX - offsetX) / scale);
        const frameY = Math.round((canvasY - offsetY) / scale);

        // Check bounds
        if (frameX < 0 || frameX > this.frameTransform.width ||
            frameY < 0 || frameY > this.frameTransform.height) {
            return;
        }

        // Send click to server
        this.sendControl({ action: 'click', x: frameX, y: frameY });
    }

    updateStatsDisplay() {
        document.getElementById('totalVehicles').textContent = this.stats.vehicles || 0;
        document.getElementById('totalPlates').textContent = this.stats.plates || this.plates.length;
        document.getElementById('fpsValue').textContent = (this.stats.fps || 0).toFixed(1);
        document.getElementById('trackingId').textContent = this.selectedTrackId ? `#${this.selectedTrackId}` : '--';
        
        // Update crowd stats
        const crowdCount = this.stats.crowd_count || 0;
        const crowdDensity = this.stats.crowd_density || 'low';
        
        document.getElementById('crowdCount').textContent = crowdCount;
        document.getElementById('crowdBadge').textContent = crowdCount;
        
        const densityEl = document.getElementById('densityLevel');
        const densityIcon = document.getElementById('densityIcon');
        
        if (densityEl) {
            densityEl.textContent = crowdDensity.toUpperCase();
            densityEl.className = `stat-value density-value density-${crowdDensity}`;
        }
        
        if (densityIcon) {
            densityIcon.className = `stat-icon density density-${crowdDensity}`;
        }

        // Update badges
        document.getElementById('vehicleBadge').textContent = this.detections.length;
        document.getElementById('plateBadge').textContent = this.plates.length;
    }

    updateCrowdDisplay() {
        const crowdCount = this.stats.crowd_count || 0;
        const crowdDensity = this.stats.crowd_density || 'low';
        
        // Update crowd count
        const crowdCountEl = document.getElementById('crowdCount');
        if (crowdCountEl) {
            crowdCountEl.textContent = crowdCount;
        }
        
        // Update crowd badge
        const crowdBadge = document.getElementById('crowdBadge');
        if (crowdBadge) {
            crowdBadge.textContent = crowdCount;
            crowdBadge.className = `nav-badge crowd-badge density-badge-${crowdDensity}`;
        }
        
        // Update density level
        const densityEl = document.getElementById('densityLevel');
        if (densityEl) {
            densityEl.textContent = crowdDensity.toUpperCase();
            densityEl.className = `stat-value density-value density-${crowdDensity}`;
        }
        
        // Update density icon
        const densityIcon = document.getElementById('densityIcon');
        if (densityIcon) {
            densityIcon.className = `stat-icon density density-${crowdDensity}`;
        }
        
        // Show alert for high density
        if (crowdDensity === 'high' && crowdCount > 0) {
            this.showCrowdAlert(crowdCount);
        }
    }
    
    showCrowdAlert(count) {
        // Only show alert once per high density event
        if (this.lastAlertDensity === 'high') return;
        this.lastAlertDensity = 'high';
        
        // Create alert notification
        const alert = document.createElement('div');
        alert.className = 'crowd-alert';
        alert.innerHTML = `
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
                <line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
            </svg>
            <span>High Crowd Density: ${count} people detected</span>
        `;
        document.body.appendChild(alert);
        
        // Remove after 5 seconds
        setTimeout(() => {
            alert.classList.add('fade-out');
            setTimeout(() => alert.remove(), 500);
        }, 5000);
    }

    updateDetectionGrid() {
        const grid = document.getElementById('detectionGrid');
        if (!grid) return;

        if (this.detections.length === 0) {
            grid.innerHTML = `
                <div class="empty-state">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1">
                        <rect x="1" y="6" width="22" height="12" rx="2"/>
                        <circle cx="6" cy="18" r="2"/><circle cx="18" cy="18" r="2"/>
                    </svg>
                    <p>No vehicles detected yet</p>
                    <span>Upload a video or image to start</span>
                </div>
            `;
            return;
        }

        grid.innerHTML = this.detections.slice(0, 10).map(det => `
            <div class="detection-item ${det.track_id === this.selectedTrackId ? 'selected' : ''}"
                 onclick="dashboard.selectTrack(${det.track_id})">
                <div class="detection-thumb">
                    ${det.thumbnail ?
                `<img src="data:image/jpeg;base64,${det.thumbnail}">` :
                `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1">
                            <rect x="1" y="6" width="22" height="12" rx="2"/>
                            <circle cx="6" cy="18" r="2"/><circle cx="18" cy="18" r="2"/>
                        </svg>`
            }
                </div>
                <div class="detection-info">
                    <div class="detection-type">${det.class || 'Vehicle'}</div>
                    <div class="detection-meta">ID: ${det.track_id} | ${Math.round((det.confidence || 0) * 100)}%</div>
                </div>
            </div>
        `).join('');

        this.updateStatsDisplay();
    }

    updatePlatesList() {
        const list = document.getElementById('platesList');
        if (!list) return;

        if (this.plates.length === 0) {
            list.innerHTML = `
                <div class="empty-state">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1">
                        <rect x="3" y="8" width="18" height="8" rx="1"/>
                        <line x1="7" y1="12" x2="17" y2="12"/>
                    </svg>
                    <p>No plates recognized yet</p>
                    <span>Plates will appear here when detected</span>
                </div>
            `;
            return;
        }

        list.innerHTML = this.plates.slice(0, 15).map(plate => `
            <div class="plate-item">
                <span class="plate-number">${plate.text}</span>
                <div class="plate-meta">
                    <span class="plate-conf">${Math.round((plate.confidence || 0) * 100)}% conf</span>
                    <span class="plate-time">${plate.time}</span>
                </div>
            </div>
        `).join('');

        this.updateStatsDisplay();
    }

    selectTrack(trackId) {
        this.selectedTrackId = trackId;
        this.sendControl({ action: 'select', track_id: trackId });
        this.updateDetectionGrid();
        this.updateStatsDisplay();
    }

    exportPlates() {
        if (this.plates.length === 0) {
            alert('No plates to export');
            return;
        }

        const csv = 'Plate Number,Confidence,Time\n' +
            this.plates.map(p => `${p.text},${p.confidence},${p.time}`).join('\n');

        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `plates_${new Date().toISOString().split('T')[0]}.csv`;
        a.click();
    }

    // WebSocket handling
    connect() {
        const wsHost = window.location.host;

        // Control WebSocket
        try {
            this.controlSocket = new WebSocket(`ws://${wsHost}/ws/control`);

            this.controlSocket.onopen = () => {
                console.log('Control socket connected');
                this.updateConnectionStatus(true);
            };

            this.controlSocket.onclose = () => {
                console.log('Control socket disconnected');
                this.updateConnectionStatus(false);
                setTimeout(() => this.connect(), 3000);
            };

            this.controlSocket.onmessage = (event) => {
                this.handleControlMessage(JSON.parse(event.data));
            };
        } catch (e) {
            console.log('WebSocket not available, running in offline mode');
        }
    }

    handleControlMessage(data) {
        switch (data.type) {
            case 'track_selected':
                this.selectedTrackId = data.track_id;
                this.updateDetectionGrid();
                break;
            case 'selection_cleared':
                this.selectedTrackId = null;
                this.updateDetectionGrid();
                break;
            case 'detection':
                this.handleProcessingResult({ detections: [data.data] });
                break;
            case 'plate':
                this.handleProcessingResult({ plates: [data.data] });
                break;
        }
    }

    sendControl(data) {
        if (this.controlSocket && this.controlSocket.readyState === WebSocket.OPEN) {
            this.controlSocket.send(JSON.stringify(data));
        }
    }

    updateConnectionStatus(connected) {
        this.isConnected = connected;
        const statusEl = document.getElementById('systemStatus');
        const statusText = statusEl?.querySelector('.status-text');
        const pulse = statusEl?.querySelector('.pulse');

        if (connected) {
            pulse.style.background = '#10b981';
            pulse.style.boxShadow = '0 0 8px #10b981';
            statusText.textContent = 'System Online';
        } else {
            pulse.style.background = '#f59e0b';
            pulse.style.boxShadow = '0 0 8px #f59e0b';
            statusText.textContent = 'Connecting...';
        }
    }

    updateDateTime() {
        const now = new Date();
        const formatted = now.toLocaleString('en-US', {
            weekday: 'short',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
        document.getElementById('datetime').textContent = formatted;
    }
}

// Initialize
let dashboard;
document.addEventListener('DOMContentLoaded', () => {
    dashboard = new SurveillanceDashboard();
});
