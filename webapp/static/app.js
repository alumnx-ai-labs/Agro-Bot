// webapp/static/app.js
class FarmerAssistant {
    constructor() {
        this.selectedImageData = null;
        this.currentSessionId = null;
        this.managerThoughtsInterval = null;
        this.currentMode = 'disease'; // 'disease', 'schemes', 'talk', or 'weather'
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.recordedAudioData = null;
        
        // Weather stations and map properties
        this.weatherStations = null;
        this.map = null;
        this.markers = [];

        this.initializeEventListeners();
        this.loadFarmSettings();
        this.checkHealth();
        this.loadWeatherStations();
    }

    initializeEventListeners() {
        // Mode selection buttons
        document.getElementById('diseaseMode').addEventListener('click', () => {
            this.switchMode('disease');
        });

        document.getElementById('schemesMode').addEventListener('click', () => {
            this.switchMode('schemes');
        });

        document.getElementById('talkMode').addEventListener('click', () => {
            this.switchMode('talk');
        });

        document.getElementById('weatherMode').addEventListener('click', () => {
            this.switchMode('weather');
        });

        // Disease detection listeners
        document.getElementById('imageInput').addEventListener('change', (e) => {
            this.handleImageSelection(e);
        });

        document.getElementById('analyzeBtn').addEventListener('click', () => {
            this.analyzeImage();
        });

        // Government schemes listeners
        document.getElementById('querySchemeBtn').addEventListener('click', () => {
            this.queryGovernmentSchemes();
        });

        document.getElementById('schemeQueryInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.queryGovernmentSchemes();
            }
        });

        // Example chip listeners
        document.querySelectorAll('.example-chip').forEach(chip => {
            chip.addEventListener('click', () => {
                const query = chip.getAttribute('data-query');
                document.getElementById('schemeQueryInput').value = query;
            });
        });

        // Talk Now listeners
        document.getElementById('recordBtn').addEventListener('click', () => {
            this.startRecording();
        });

        document.getElementById('stopBtn').addEventListener('click', () => {
            this.stopRecording();
        });

        document.getElementById('transcribeBtn').addEventListener('click', () => {
            this.transcribeAudio();
        });

        // Weather station listeners (simplified)
        document.getElementById('refreshStationsBtn').addEventListener('click', () => {
            this.loadWeatherStations();
        });

        document.getElementById('showAllBtn').addEventListener('click', () => {
            this.showAllStations();
        });

        // Retry button
        document.getElementById('retryBtn').addEventListener('click', () => {
            this.resetInterface();
        });

        // Settings listeners
        document.getElementById('settingsBtn').addEventListener('click', () => {
            this.openSettings();
        });

        document.getElementById('closeSettings').addEventListener('click', () => {
            this.closeSettings();
        });

        document.getElementById('cancelSettings').addEventListener('click', () => {
            this.closeSettings();
        });

        document.getElementById('saveSettings').addEventListener('click', () => {
            this.saveFarmSettings();
        });

        // Modal listeners
        document.getElementById('closeModal').addEventListener('click', () => {
            this.closeStationModal();
        });

        // Close popups when clicking outside
        document.getElementById('settingsPopup').addEventListener('click', (e) => {
            if (e.target === document.getElementById('settingsPopup')) {
                this.closeSettings();
            }
        });

        document.getElementById('stationModal').addEventListener('click', (e) => {
            if (e.target === document.getElementById('stationModal')) {
                this.closeStationModal();
            }
        });
    }

    switchMode(mode) {
        this.currentMode = mode;
        this.resetInterface();

        // Update UI
        document.querySelectorAll('.mode-btn').forEach(btn => btn.classList.remove('active'));
        document.querySelectorAll('.mode-section').forEach(section => section.classList.remove('active'));

        if (mode === 'disease') {
            document.getElementById('diseaseMode').classList.add('active');
            document.getElementById('diseaseSection').classList.add('active');
        } else if (mode === 'schemes') {
            document.getElementById('schemesMode').classList.add('active');
            document.getElementById('schemesSection').classList.add('active');
        } else if (mode === 'talk') {
            document.getElementById('talkMode').classList.add('active');
            document.getElementById('talkSection').classList.add('active');
        } else if (mode === 'weather') {
            document.getElementById('weatherMode').classList.add('active');
            document.getElementById('weatherSection').classList.add('active');
            // Initialize map when weather mode is selected
            setTimeout(() => this.initializeMap(), 100);
        }
    }

    // Weather Stations Methods (Simplified)
    async loadWeatherStations() {
        try {
            const response = await fetch('/static/weather_stations.json');
            if (response.ok) {
                this.weatherStations = await response.json();
                console.log('✅ Weather stations loaded');
                if (this.map) {
                    this.addWeatherStationMarkers();
                }
            }
        } catch (error) {
            console.warn('⚠️ Could not load weather stations file');
        }
    }

    initializeMap() {
        if (!this.weatherStations || typeof google === 'undefined') {
            console.warn('Weather stations or Google Maps not loaded yet');
            return;
        }

        // Initialize the map centered on Shadnagar
        const centerPoint = this.weatherStations.metadata.center_point;
        const mapOptions = {
            zoom: 10,
            center: { lat: centerPoint.latitude, lng: centerPoint.longitude },
            mapTypeId: google.maps.MapTypeId.ROADMAP
        };

        this.map = new google.maps.Map(document.getElementById('map'), mapOptions);
        this.addWeatherStationMarkers();
        console.log('✅ Map initialized successfully');
    }

    addWeatherStationMarkers() {
        // Clear existing markers
        this.markers.forEach(marker => marker.setMap(null));
        this.markers = [];

        const stations = this.weatherStations.weather_stations;

        stations.forEach(station => {
            const marker = new google.maps.Marker({
                position: { lat: station.latitude, lng: station.longitude },
                map: this.map,
                title: station.name,
                icon: {
                    url: 'https://maps.google.com/mapfiles/ms/icons/red-dot.png'
                }
            });

            // Add click listener for info window
            const infoWindow = new google.maps.InfoWindow({
                content: `
                    <div>
                        <h4>${station.name}</h4>
                        <p><strong>Type:</strong> ${station.type}</p>
                        <p><strong>District:</strong> ${station.district}</p>
                        <p><strong>Status:</strong> ${station.status}</p>
                    </div>
                `
            });

            marker.addListener('click', () => {
                infoWindow.open(this.map, marker);
            });

            this.markers.push(marker);
        });

        console.log(`✅ Added ${this.markers.length} station markers`);
    }

    showAllStations() {
        if (!this.map || !this.weatherStations) return;

        // Reset map bounds to show all stations
        const bounds = new google.maps.LatLngBounds();
        
        this.weatherStations.weather_stations.forEach(station => {
            bounds.extend(new google.maps.LatLng(station.latitude, station.longitude));
        });

        this.map.fitBounds(bounds);
    }

    closeStationModal() {
        document.getElementById('stationModal').style.display = 'none';
    }

    async checkHealth() {
        try {
            const response = await fetch('/health');
            const health = await response.json();

            if (health.status === 'healthy') {
                console.log('✅ System healthy');
            } else {
                console.warn('⚠️ System health issues:', health);
            }
        } catch (error) {
            console.error('❌ Health check failed:', error);
        }
    }

    handleImageSelection(event) {
        const file = event.target.files[0];

        if (!file) {
            this.selectedImageData = null;
            document.getElementById('analyzeBtn').disabled = true;
            return;
        }

        // Validate file type
        if (!file.type.startsWith('image/')) {
            alert('Please select a valid image file.');
            return;
        }

        // Validate file size (max 5MB)
        if (file.size > 5 * 1024 * 1024) {
            alert('Image size should be less than 5MB.');
            return;
        }

        // Convert to base64
        const reader = new FileReader();
        reader.onload = (e) => {
            // Remove data URL prefix to get pure base64
            this.selectedImageData = e.target.result.split(',')[1];
            document.getElementById('analyzeBtn').disabled = false;

            console.log('✅ Image selected and converted to base64');
        };

        reader.onerror = () => {
            alert('Error reading image file.');
            this.selectedImageData = null;
            document.getElementById('analyzeBtn').disabled = true;
        };

        reader.readAsDataURL(file);
    }

    async analyzeImage() {
        if (!this.selectedImageData) {
            alert('Please select an image first.');
            return;
        }

        this.showLoading('Analyzing your crop image...');

        try {
            const textDescription = document.getElementById('textInput').value.trim();

            const requestData = {
                inputType: 'image',
                content: this.selectedImageData,
                userId: this.getUserId(),
                language: 'en',
                textDescription: textDescription,
                farmSettings: this.getFarmSettings()
            };

            console.log('📤 Sending analysis request...');

            const response = await fetch('/api/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestData)
            });

            const result = await response.json();

            if (response.ok) {
                console.log('✅ Analysis successful:', result);
                this.currentSessionId = result.session_id;
                this.displayResults(result, 'Disease Analysis Results');
            } else {
                console.error('❌ Analysis failed:', result);
                this.showError(result.error || 'Analysis failed');
            }

        } catch (error) {
            console.error('❌ Request failed:', error);
            this.showError('Network error. Please check your connection and try again.');
        }
    }

    async queryGovernmentSchemes() {
        const queryText = document.getElementById('schemeQueryInput').value.trim();

        if (!queryText) {
            alert('Please enter your question about government schemes.');
            return;
        }

        this.showLoading('Searching government schemes database...');

        try {
            const requestData = {
                inputType: 'text',
                content: queryText,
                userId: this.getUserId(),
                language: 'en',
                queryType: 'government_schemes',
                farmSettings: this.getFarmSettings()
            };

            console.log('📤 Sending schemes query request...');

            const response = await fetch('/api/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestData)
            });

            const result = await response.json();

            if (response.ok) {
                console.log('✅ Schemes query successful:', result);
                this.currentSessionId = result.session_id;
                this.displayResults(result, 'Government Schemes Information');
            } else {
                console.error('❌ Schemes query failed:', result);
                this.showError(result.error || 'Query failed');
            }

        } catch (error) {
            console.error('❌ Request failed:', error);
            this.showError('Network error. Please check your connection and try again.');
        }
    }

    async startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

            const options = {
                mimeType: 'audio/webm;codecs=opus'
            };

            if (MediaRecorder.isTypeSupported(options.mimeType)) {
                this.mediaRecorder = new MediaRecorder(stream, options);
            } else {
                this.mediaRecorder = new MediaRecorder(stream);
            }
            this.audioChunks = [];

            this.mediaRecorder.addEventListener('dataavailable', (event) => {
                this.audioChunks.push(event.data);
            });

            this.mediaRecorder.addEventListener('stop', () => {
                const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
                this.convertAudioToBase64(audioBlob);
            });

            this.mediaRecorder.start();

            document.getElementById('recordBtn').disabled = true;
            document.getElementById('stopBtn').disabled = false;
            document.getElementById('recordingStatus').textContent = '🔴 Recording...';
            document.getElementById('recordingStatus').className = 'recording-status recording';

            console.log('✅ Recording started');

        } catch (error) {
            console.error('❌ Recording failed:', error);
            alert('Unable to access microphone. Please check permissions.');
        }
    }

    stopRecording() {
        if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
            this.mediaRecorder.stop();

            // Stop all tracks
            this.mediaRecorder.stream.getTracks().forEach(track => track.stop());

            document.getElementById('recordBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
            document.getElementById('recordingStatus').textContent = '✅ Recording complete! Ready to transcribe.';
            document.getElementById('recordingStatus').className = 'recording-status ready';

            console.log('✅ Recording stopped');
        }
    }

    convertAudioToBase64(audioBlob) {
        const reader = new FileReader();
        reader.onload = () => {
            // Remove data URL prefix to get pure base64
            this.recordedAudioData = reader.result.split(',')[1];

            // Create audio URL for playback
            this.audioUrl = URL.createObjectURL(audioBlob);

            // Show playback controls
            this.showAudioPlayback();

            document.getElementById('transcribeBtn').disabled = false;
            console.log('✅ Audio converted to base64');
        };
        reader.readAsDataURL(audioBlob);
    }

    showAudioPlayback() {
        const playbackContainer = document.getElementById('audioPlayback');
        playbackContainer.innerHTML = `
        <div class="audio-playback">
            <h4>🎵 Preview Recorded Audio:</h4>
            <audio controls style="width: 100%; margin: 10px 0;">
                <source src="${this.audioUrl}" type="audio/wav">
                Your browser does not support the audio element.
            </audio>
        </div>
    `;
        playbackContainer.style.display = 'block';
    }

    async transcribeAudio() {
        if (!this.recordedAudioData) {
            alert('Please record audio first.');
            return;
        }

        this.showLoading('Transcribing your audio...');

        try {
            const selectedLanguage = document.getElementById('languageSelect').value;

            const requestData = {
                inputType: 'audio',
                content: this.recordedAudioData,
                userId: this.getUserId(),
                language: selectedLanguage,
                farmSettings: this.getFarmSettings()
            };

            console.log('📤 Sending transcription request...');

            const response = await fetch('/api/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestData)
            });

            const result = await response.json();

            if (response.ok) {
                console.log('✅ Transcription successful:', result);
                this.currentSessionId = result.session_id;
                this.displayResults(result, 'Audio Transcription Results');
            } else {
                console.error('❌ Transcription failed:', result);
                this.showError(result.error || 'Transcription failed');
            }

        } catch (error) {
            console.error('❌ Request failed:', error);
            this.showError('Network error. Please check your connection and try again.');
        }
    }

    showLoading(message = 'Processing your request...') {
        document.getElementById('loadingSection').style.display = 'block';
        document.getElementById('resultsSection').style.display = 'none';
        document.getElementById('errorSection').style.display = 'none';

        document.getElementById('loadingText').textContent = message;

        // Show appropriate manager thoughts
        if (this.currentMode === 'disease') {
            this.startDiseaseAnalysisThoughts();
        } else {
            this.startSchemesQueryThoughts();
        }
    }

    startDiseaseAnalysisThoughts() {
        const thoughts = [
            "🤔 Analyzing your crop image...",
            "🎯 Identifying potential issues...",
            "🔬 Calling disease detection specialist...",
            "✅ Analysis complete! Preparing response..."
        ];

        this.displayThoughts(thoughts);
    }

    startSchemesQueryThoughts() {
        const thoughts = [
            "🤔 Understanding your query...",
            "🔍 Searching government schemes database...",
            "📊 Finding relevant schemes and policies...",
            "✅ Query complete! Preparing information..."
        ];

        this.displayThoughts(thoughts);
    }

    displayThoughts(thoughts) {
        const thoughtsContainer = document.getElementById('managerThoughts');
        thoughtsContainer.innerHTML = '';

        thoughts.forEach((thought, index) => {
            setTimeout(() => {
                const bubble = document.createElement('div');
                bubble.className = 'thought-bubble';
                bubble.textContent = thought;
                thoughtsContainer.appendChild(bubble);
            }, index * 2000);
        });
    }

    displayResults(result, title) {
        document.getElementById('loadingSection').style.display = 'none';
        document.getElementById('resultsSection').style.display = 'block';
        document.getElementById('errorSection').style.display = 'none';

        document.getElementById('resultsTitle').textContent = title;
        const resultsContainer = document.getElementById('analysisResults');

        // Store result for debugging
        this.lastResult = result;

        let htmlContent = '';

        if (this.currentMode === 'disease' && result.agent_response && result.agent_response.analysis) {
            htmlContent = this.createDiseaseAnalysisHTML(result.agent_response.analysis);
        } else if (this.currentMode === 'schemes' && result.agent_response) {
            htmlContent = this.createSchemesResponseHTML(result.agent_response);
        } else if (this.currentMode === 'talk' && result.agent_response) {
            htmlContent = this.createTranscriptionHTML(result.agent_response);
        } else {
            htmlContent = this.createSimpleResponseHTML(result);
        }

        // Add debug panel
        htmlContent += this.createDebugPanel(result);

        resultsContainer.innerHTML = htmlContent;
    }

    createDiseaseAnalysisHTML(analysis) {
        const confidenceClass = `confidence-${analysis.confidence}`;

        return `
            <div class="disease-card">
                <div class="disease-header">
                    <div class="disease-name">${analysis.disease_name}</div>
                    <div class="confidence-badge ${confidenceClass}">${analysis.confidence} confidence</div>
                </div>
                
                ${analysis.severity !== 'none' ? `
                    <div class="analysis-section">
                        <h4>🚨 Severity: ${analysis.severity}</h4>
                    </div>
                ` : ''}
                
                ${analysis.symptoms_observed.length > 0 ? `
                    <div class="analysis-section">
                        <h4>👀 Symptoms Observed:</h4>
                        <ul>
                            ${analysis.symptoms_observed.map(symptom => `<li>${symptom}</li>`).join('')}
                        </ul>
                    </div>
                ` : ''}
                
                <div class="analysis-section">
                    <h4>⚡ Immediate Action:</h4>
                    <div class="treatment-card">
                        ${analysis.immediate_action}
                    </div>
                </div>
                
                <div class="analysis-section">
                    <h4>💊 Treatment Summary:</h4>
                    <p>${analysis.treatment_summary}</p>
                </div>
                
                ${analysis.organic_solutions.length > 0 ? `
                    <div class="analysis-section">
                        <h4>🌿 Organic Solutions:</h4>
                        ${analysis.organic_solutions.map(solution => `
                            <div class="treatment-card">
                                <strong>${solution.name}</strong><br>
                                <em>Preparation:</em> ${solution.preparation}<br>
                                <em>Application:</em> ${solution.application}
                            </div>
                        `).join('')}
                    </div>
                ` : ''}
                
                ${analysis.prevention_tips.length > 0 ? `
                    <div class="analysis-section">
                        <h4>🛡️ Prevention Tips:</h4>
                        <ul>
                            ${analysis.prevention_tips.map(tip => `<li>${tip}</li>`).join('')}
                        </ul>
                    </div>
                ` : ''}
                
                <div class="analysis-section">
                    <h4>💰 Cost Estimate:</h4>
                    <div class="cost-estimate">${analysis.cost_estimate}</div>
                </div>
                
                <div class="analysis-section">
                    <h4>⏱️ Expected Timeline:</h4>
                    <p>${analysis.success_timeline}</p>
                </div>
                
                <div class="analysis-section">
                    <h4>⚠️ Warning Signs:</h4>
                    <p style="color: #d32f2f; font-weight: bold;">${analysis.warning_signs}</p>
                </div>
            </div>
        `;
    }

    createSchemesResponseHTML(agentResponse) {
        return `
            <div class="schemes-card">
                <div class="schemes-header">
                    <div class="schemes-title">Government Schemes Information</div>
                    ${agentResponse.confidence ? `
                        <div class="confidence-badge confidence-${agentResponse.confidence}">${agentResponse.confidence} relevance</div>
                    ` : ''}
                </div>
                
                <div class="schemes-content">
                    ${agentResponse.message ? `
                        <div class="schemes-section">
                            <div class="schemes-text">${this.formatText(agentResponse.message)}</div>
                        </div>
                    ` : ''}
                    
                    ${agentResponse.schemes && agentResponse.schemes.length > 0 ? `
                        <div class="schemes-section">
                            <h4>📋 Relevant Schemes:</h4>
                            ${agentResponse.schemes.map(scheme => `
                                <div class="scheme-item">
                                    <h5>${scheme.name}</h5>
                                    <p><strong>Description:</strong> ${scheme.description}</p>
                                    ${scheme.eligibility ? `<p><strong>Eligibility:</strong> ${scheme.eligibility}</p>` : ''}
                                    ${scheme.benefits ? `<p><strong>Benefits:</strong> ${scheme.benefits}</p>` : ''}
                                    ${scheme.application_process ? `<p><strong>How to Apply:</strong> ${scheme.application_process}</p>` : ''}
                                </div>
                            `).join('')}
                        </div>
                    ` : ''}
                    
                    ${agentResponse.sources && agentResponse.sources.length > 0 ? `
                        <div class="schemes-section">
                            <h4>📚 Sources:</h4>
                            <ul class="sources-list">
                                ${agentResponse.sources.map(source => `<li>${source}</li>`).join('')}
                            </ul>
                        </div>
                    ` : ''}
                </div>
            </div>
        `;
    }

    createTranscriptionHTML(agentResponse) {
        const success = agentResponse.success;
        const transcript = agentResponse.transcript || '';
        const confidence = agentResponse.confidence || 0;

        return `
        <div class="transcription-card">
            <div class="transcription-header">
                <div class="transcription-title">Audio Transcription</div>
                ${confidence > 0 ? `
                    <div class="confidence-badge confidence-${confidence > 0.8 ? 'high' : confidence > 0.5 ? 'medium' : 'low'}">
                        ${Math.round(confidence * 100)}% confidence
                    </div>
                ` : ''}
            </div>
            
            ${success ? `
                <div class="transcription-content">
                    <div class="transcript-text">
                        <h4>📝 Transcribed Text:</h4>
                        <div class="transcript-box">${transcript}</div>
                    </div>
                    
                    <div class="transcript-actions">
                        <button class="btn btn-primary" onclick="navigator.clipboard.writeText('${transcript.replace(/'/g, "\\'")}')">
                            📋 Copy Text
                        </button>
                    </div>
                </div>
            ` : `
                <div class="transcription-error">
                    <h4>❌ Transcription Failed:</h4>
                    <p>${agentResponse.error || 'Unknown error occurred'}</p>
                </div>
            `}
        </div>
    `;
    }

    createSimpleResponseHTML(result) {
        return `
            <div class="response-card">
                <div class="analysis-section">
                    <h4>Response:</h4>
                    <p>${result.final_response?.message || result.message || 'Request processed'}</p>
                </div>
                
                ${result.classification ? `
                    <div class="analysis-section">
                        <h4>Classification:</h4>
                        <p><strong>Intent:</strong> ${result.classification.intent}</p>
                        <p><strong>Confidence:</strong> ${result.classification.confidence}</p>
                        <p><strong>Reasoning:</strong> ${result.classification.reasoning}</p>
                    </div>
                ` : ''}
            </div>
        `;
    }

    formatText(text) {
        // Simple text formatting - convert line breaks to <br> and bold **text**
        return text
            .replace(/\n/g, '<br>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>');
    }

    showError(message) {
        document.getElementById('loadingSection').style.display = 'none';
        document.getElementById('resultsSection').style.display = 'none';
        document.getElementById('errorSection').style.display = 'block';

        document.getElementById('errorMessage').textContent = message;
    }

    resetInterface() {
        document.getElementById('loadingSection').style.display = 'none';
        document.getElementById('resultsSection').style.display = 'none';
        document.getElementById('errorSection').style.display = 'none';

        // Reset forms
        if (this.currentMode === 'disease') {
            document.getElementById('imageInput').value = '';
            document.getElementById('textInput').value = '';
            document.getElementById('analyzeBtn').disabled = true;
            this.selectedImageData = null;
        } else if (this.currentMode === 'schemes') {
            document.getElementById('schemeQueryInput').value = '';
        } else if (this.currentMode === 'talk') {
            document.getElementById('recordBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
            document.getElementById('transcribeBtn').disabled = true;
            document.getElementById('recordingStatus').textContent = '';
            document.getElementById('recordingStatus').className = 'recording-status';
            document.getElementById('audioPlayback').style.display = 'none';
            this.recordedAudioData = null;
            if (this.audioUrl) {
                URL.revokeObjectURL(this.audioUrl);
                this.audioUrl = null;
            }
        }

        this.currentSessionId = null;
    }

    loadFarmSettings() {
        const savedSettings = localStorage.getItem('farmSettings');
        if (savedSettings) {
            const settings = JSON.parse(savedSettings);
            
            document.getElementById('cropType').value = settings.cropType || 'Mosambi';
            document.getElementById('acreage').value = settings.acreage || 15;
            document.getElementById('sowingDate').value = settings.sowingDate || '2022-01-01';
            document.getElementById('currentStage').value = settings.currentStage || 'Fruit Development';
            document.getElementById('farmerName').value = settings.farmerName || 'Vijender';
            document.getElementById('soilType').value = settings.soilType || 'A';
            document.getElementById('currentChallenges').value = settings.currentChallenges || 'Currently there are no challenges.';
            
            // Load preferred languages
            const checkboxes = document.querySelectorAll('.language-checkboxes input[type="checkbox"]');
            checkboxes.forEach(checkbox => {
                checkbox.checked = settings.preferredLanguages && settings.preferredLanguages.includes(checkbox.value);
            });
        }
    }

    openSettings() {
        document.getElementById('settingsPopup').style.display = 'flex';
    }

    closeSettings() {
        document.getElementById('settingsPopup').style.display = 'none';
    }

    saveFarmSettings() {
        const preferredLanguages = [];
        document.querySelectorAll('.language-checkboxes input[type="checkbox"]:checked').forEach(checkbox => {
            preferredLanguages.push(checkbox.value);
        });

        const settings = {
            cropType: document.getElementById('cropType').value,
            acreage: parseFloat(document.getElementById('acreage').value),
            sowingDate: document.getElementById('sowingDate').value,
            currentStage: document.getElementById('currentStage').value,
            farmerName: document.getElementById('farmerName').value,
            soilType: document.getElementById('soilType').value,
            currentChallenges: document.getElementById('currentChallenges').value,
            preferredLanguages: preferredLanguages
        };

        localStorage.setItem('farmSettings', JSON.stringify(settings));
        
        // Show success message
        const saveBtn = document.getElementById('saveSettings');
        const originalText = saveBtn.textContent;
        saveBtn.textContent = '✅ Saved!';
        saveBtn.style.background = '#28a745';
        
        setTimeout(() => {
            saveBtn.textContent = originalText;
            saveBtn.style.background = '';
            this.closeSettings();
        }, 1500);

        console.log('✅ Farm settings saved:', settings);
    }

    getFarmSettings() {
        const savedSettings = localStorage.getItem('farmSettings');
        if (savedSettings) {
            return JSON.parse(savedSettings);
        }
        
        // Return defaults if no settings saved
        return {
            cropType: 'Mosambi',
            acreage: 15,
            sowingDate: '2022-01-01',
            currentStage: 'Fruit Development',
            farmerName: 'Vijender',
            soilType: 'A',
            currentChallenges: 'Currently there are no challenges.',
            preferredLanguages: ['English', 'Telugu']
        };
    }

    getUserId() {
        // Simple user ID generation for MVP
        let userId = localStorage.getItem('farmerAssistantUserId');
        if (!userId) {
            userId = 'user_' + Math.random().toString(36).substr(2, 9);
            localStorage.setItem('farmerAssistantUserId', userId);
        }
        return userId;
    }

    createDebugPanel(result) {
        return `
            <div class="debug-panel">
                <button class="debug-toggle" onclick="this.nextElementSibling.style.display = this.nextElementSibling.style.display === 'none' ? 'block' : 'none'">
                    🔍 Show Debug Info
                </button>
                <div class="debug-content" style="display: none;">
                    <strong>Full Response:</strong>
                    <pre>${JSON.stringify(result, null, 2)}</pre>
                </div>
            </div>
        `;
    }
}

// Google Maps callback function
function initMap() {
    // This function is called by Google Maps API when it's loaded
    console.log('🗺️ Google Maps API loaded');
    if (window.farmerAssistant && window.farmerAssistant.currentMode === 'weather') {
        window.farmerAssistant.initializeMap();
    }
}

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    window.farmerAssistant = new FarmerAssistant();
});