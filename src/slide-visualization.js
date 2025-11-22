/**
 * Modular visualization for slides with full interactivity
 */

class SlideVisualization {
    constructor(containerId, svgId, heatmapId, config = {}) {
        this.containerId = containerId;
        this.svgId = svgId;
        this.heatmapId = heatmapId;
        this.config = config;
        
        // Create network instance
        this.network = new NeuralNetwork();
        
        // Apply initial configuration
        this.applyConfig(config);
        
        // Interactive state
        this.selectedOutputIndex = null;
        this.editingConnection = null;
        this.selectedNeuron = null;
        
        // Initialize visualization
        this.setupVisualization();
        this.updateVisualization();
    }
    
    applyConfig(config) {
        // Set biases if specified
        if (config.inputBias) {
            config.inputBias.forEach((bias, i) => {
                this.network.setInputBias(i, bias);
            });
        }
        
        if (config.hiddenBias) {
            config.hiddenBias.forEach((bias, i) => {
                this.network.setHiddenBias(i, bias);
            });
        }
        
        if (config.outputBias) {
            config.outputBias.forEach((bias, i) => {
                this.network.setOutputBias(i, bias);
            });
        }
        
        // Set weights if specified
        if (config.weights1) {
            config.weights1.forEach((row, i) => {
                row.forEach((weight, j) => {
                    this.network.setWeight1(i, j, weight);
                });
            });
        }
        
        if (config.weights2) {
            config.weights2.forEach((row, i) => {
                row.forEach((weight, j) => {
                    this.network.setWeight2(i, j, weight);
                });
            });
        }
    }
    
    setupVisualization() {
        const container = document.getElementById(this.containerId);
        
        // Clear container
        container.innerHTML = '';
        
        // Create input layer
        const inputLayer = this.createLayer('Input', this.network.inputNames, 'input');
        container.appendChild(inputLayer);
        
        // Create hidden layer
        const hiddenLayer = this.createLayer('Hidden (ReLU)', this.network.hiddenNames, 'hidden');
        container.appendChild(hiddenLayer);
        
        // Create output layer
        const outputLayer = this.createLayer('Output (Logits)', this.network.outputNames, 'output');
        container.appendChild(outputLayer);
        
        // Setup SVG
        this.setupSVG();
        
        // Setup bias controls (initially hidden)
        this.setupBiasControls();
        
        // Setup weight modal
        this.setupWeightModal();
    }
    
    setupBiasControls() {
        // Find or create bias control container for this slide
        const vizContainer = document.getElementById(this.containerId).closest('.visualization-container');
        if (!vizContainer) return;
        
        // Check if bias control already exists
        let biasControlContainer = vizContainer.querySelector('.bias-control-container');
        if (!biasControlContainer) {
            biasControlContainer = document.createElement('div');
            biasControlContainer.className = 'bias-control-container';
            biasControlContainer.id = `${this.containerId}-bias-control-container`;
            biasControlContainer.style.display = 'none';
            biasControlContainer.style.marginTop = '15px';
            biasControlContainer.style.padding = '10px';
            biasControlContainer.style.backgroundColor = '#f9f9f9';
            biasControlContainer.style.borderRadius = '5px';
            
            const title = document.createElement('h4');
            title.textContent = 'Bias Control';
            title.style.marginTop = '0';
            title.style.marginBottom = '5px';
            biasControlContainer.appendChild(title);
            
            const info = document.createElement('p');
            info.id = `${this.containerId}-bias-control-info`;
            info.style.margin = '5px 0';
            info.style.color = '#666';
            info.style.fontSize = '13px';
            biasControlContainer.appendChild(info);
            
            const controls = document.createElement('div');
            controls.id = `${this.containerId}-bias-controls`;
            biasControlContainer.appendChild(controls);
            
            vizContainer.appendChild(biasControlContainer);
        }
    }
    
    setupWeightModal() {
        // Check if modal already exists
        if (document.getElementById(`${this.containerId}-weight-modal`)) return;
        
        const modal = document.createElement('div');
        modal.id = `${this.containerId}-weight-modal`;
        modal.className = 'modal';
        modal.style.display = 'none';
        
        const modalContent = document.createElement('div');
        modalContent.className = 'modal-content';
        
        const title = document.createElement('h3');
        title.textContent = 'Edit Weight';
        title.style.marginTop = '0';
        modalContent.appendChild(title);
        
        const info = document.createElement('p');
        info.id = `${this.containerId}-modal-connection-info`;
        modalContent.appendChild(info);
        
        const label = document.createElement('label');
        label.textContent = 'New weight value:';
        label.htmlFor = `${this.containerId}-weight-input`;
        modalContent.appendChild(label);
        
        const input = document.createElement('input');
        input.type = 'number';
        input.id = `${this.containerId}-weight-input`;
        input.step = '0.1';
        input.value = '0';
        input.style.width = '100%';
        input.style.padding = '8px';
        input.style.margin = '10px 0';
        input.style.border = '1px solid #ddd';
        input.style.borderRadius = '4px';
        input.style.fontSize = '16px';
        modalContent.appendChild(input);
        
        const buttonContainer = document.createElement('div');
        buttonContainer.className = 'modal-buttons';
        buttonContainer.style.display = 'flex';
        buttonContainer.style.gap = '10px';
        buttonContainer.style.marginTop = '15px';
        
        const saveBtn = document.createElement('button');
        saveBtn.textContent = 'Save';
        saveBtn.className = 'btn-save';
        saveBtn.onclick = () => this.saveWeight();
        buttonContainer.appendChild(saveBtn);
        
        const cancelBtn = document.createElement('button');
        cancelBtn.textContent = 'Cancel';
        cancelBtn.className = 'btn-cancel';
        cancelBtn.onclick = () => this.closeModal();
        buttonContainer.appendChild(cancelBtn);
        
        modalContent.appendChild(buttonContainer);
        modal.appendChild(modalContent);
        document.body.appendChild(modal);
        
        // Close modal when clicking outside
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                this.closeModal();
            }
        });
        
        // Allow Enter to save, Escape to cancel
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.saveWeight();
            } else if (e.key === 'Escape') {
                this.closeModal();
            }
        });
    }
    
    createLayer(title, neuronLabels, layerType) {
        const layer = document.createElement('div');
        layer.className = 'layer';
        layer.id = `${this.containerId}-layer-${layerType}`;
        
        const titleDiv = document.createElement('div');
        titleDiv.className = 'layer-title';
        titleDiv.textContent = title;
        layer.appendChild(titleDiv);
        
        neuronLabels.forEach((label, index) => {
            const neuron = document.createElement('div');
            neuron.className = 'neuron';
            neuron.id = `${this.containerId}-neuron-${layerType}-${index}`;
            neuron.dataset.layer = layerType;
            neuron.dataset.index = index;
            neuron.dataset.containerId = this.containerId;
            
            const labelDiv = document.createElement('div');
            labelDiv.className = 'neuron-label';
            labelDiv.textContent = label;
            neuron.appendChild(labelDiv);
            
            const valueDiv = document.createElement('div');
            valueDiv.className = 'neuron-value';
            valueDiv.id = `${this.containerId}-value-${layerType}-${index}`;
            neuron.appendChild(valueDiv);
            
            // Add probability display for output neurons (outside the neuron)
            if (layerType === 'output') {
                const probDiv = document.createElement('div');
                probDiv.className = 'neuron-probability';
                probDiv.id = `${this.containerId}-prob-${layerType}-${index}`;
                neuron.appendChild(probDiv);
            }
            
            // Add click handler for all neurons
            neuron.addEventListener('click', (e) => {
                if (layerType === 'output') {
                    // Output neurons: toggle target selection
                    this.selectOutput(index);
                } else {
                    // Non-output neurons: select for bias adjustment
                    this.selectNeuronForBias(layerType, index);
                }
            });
            
            layer.appendChild(neuron);
        });
        
        return layer;
    }
    
    selectOutput(index) {
        const container = document.getElementById(this.containerId);
        
        // Toggle selection if clicking the same neuron
        if (this.selectedOutputIndex === index) {
            this.selectedOutputIndex = null;
            container.querySelectorAll('.neuron[data-layer="output"]').forEach(neuron => {
                neuron.classList.remove('selected');
            });
            // Now allow bias adjustment for this neuron
            this.selectNeuronForBias('output', index);
        } else {
            this.selectedOutputIndex = index;
            
            // Update UI to show selection
            container.querySelectorAll('.neuron[data-layer="output"]').forEach(neuron => {
                neuron.classList.remove('selected');
            });
            document.getElementById(`${this.containerId}-neuron-output-${index}`).classList.add('selected');
            
            // Hide bias control when selecting a target
            this.hideBiasControl();
            // Clear bias selection
            container.querySelectorAll('.neuron').forEach(n => {
                n.classList.remove('bias-selected');
            });
            this.selectedNeuron = null;
        }
        
        this.updateVisualization();
    }
    
    selectNeuronForBias(layer, index) {
        const container = document.getElementById(this.containerId);
        
        // Update selected neuron
        this.selectedNeuron = { layer, index };
        
        // Update visual selection
        container.querySelectorAll('.neuron').forEach(n => {
            n.classList.remove('bias-selected');
        });
        document.getElementById(`${this.containerId}-neuron-${layer}-${index}`).classList.add('bias-selected');
        
        // Show bias control
        this.showBiasControlForNeuron(layer, index);
    }
    
    showBiasControlForNeuron(layer, index) {
        const controlContainer = document.getElementById(`${this.containerId}-bias-control-container`);
        const container = document.getElementById(`${this.containerId}-bias-controls`);
        const infoEl = document.getElementById(`${this.containerId}-bias-control-info`);
        
        if (!controlContainer || !container || !infoEl) return;
        
        // Clear existing controls
        container.innerHTML = '';
        
        // Get neuron name and current bias
        let neuronName, currentBias, setBiasCallback;
        
        if (layer === 'input') {
            neuronName = 'Input';
            currentBias = this.network.getInputBias(index);
            setBiasCallback = (value) => this.network.setInputBias(index, value);
        } else if (layer === 'hidden') {
            neuronName = this.network.hiddenNames[index];
            currentBias = this.network.getHiddenBias(index);
            setBiasCallback = (value) => this.network.setHiddenBias(index, value);
        } else if (layer === 'output') {
            neuronName = this.network.outputNames[index];
            currentBias = this.network.getOutputBias(index);
            setBiasCallback = (value) => this.network.setOutputBias(index, value);
        }
        
        // Update info text
        infoEl.textContent = `Adjusting bias for: ${neuronName} (${layer} layer)`;
        
        // Create the bias control
        const control = document.createElement('div');
        control.style.display = 'flex';
        control.style.alignItems = 'center';
        control.style.gap = '10px';
        control.style.margin = '10px 0';
        
        const label = document.createElement('label');
        label.textContent = 'Bias:';
        label.style.fontWeight = 'bold';
        label.style.minWidth = '50px';
        control.appendChild(label);
        
        const slider = document.createElement('input');
        slider.type = 'range';
        slider.min = '-5';
        slider.max = '5';
        slider.step = '0.1';
        slider.value = currentBias.toString();
        slider.style.flex = '1';
        slider.style.maxWidth = '200px';
        slider.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            setBiasCallback(value);
            valueDisplay.textContent = value.toFixed(1);
            this.updateVisualization();
        });
        control.appendChild(slider);
        
        const valueDisplay = document.createElement('span');
        valueDisplay.style.minWidth = '50px';
        valueDisplay.style.textAlign = 'right';
        valueDisplay.style.fontFamily = 'monospace';
        valueDisplay.textContent = currentBias.toFixed(1);
        control.appendChild(valueDisplay);
        
        container.appendChild(control);
        
        // Show the control container
        controlContainer.style.display = 'block';
    }
    
    hideBiasControl() {
        const controlContainer = document.getElementById(`${this.containerId}-bias-control-container`);
        if (controlContainer) {
            controlContainer.style.display = 'none';
        }
    }
    
    setupSVG() {
        const svg = document.getElementById(this.svgId);
        const container = document.getElementById(this.containerId);
        
        const resizeSVG = () => {
            const rect = container.getBoundingClientRect();
            svg.setAttribute('viewBox', `0 0 ${rect.width} ${rect.height}`);
            this.drawConnections();
        };
        
        setTimeout(resizeSVG, 100);
        window.addEventListener('resize', resizeSVG);
    }
    
    updateVisualization() {
        // Run forward pass
        this.network.forward();
        
        // Run backward pass if target is selected
        if (this.selectedOutputIndex !== null) {
            this.network.backward(this.selectedOutputIndex);
        }
        
        // Update neuron values
        this.updateNeuronValues();
        
        // Draw connections
        this.drawConnections();
        
        // Update heatmap
        this.updateHeatmap();
    }
    
    updateNeuronValues() {
        // Input values - show activation + bias
        this.network.input.forEach((val, i) => {
            const bias = this.network.inputBias[i];
            const biasStr = bias >= 0 ? `+${bias.toFixed(1)}` : bias.toFixed(1);
            const elem = document.getElementById(`${this.containerId}-value-input-${i}`);
            if (elem) elem.textContent = `(${val.toFixed(1)}${biasStr})`;
        });
        
        // Hidden values - show activation + bias
        this.network.hiddenActivation.forEach((val, i) => {
            const bias = this.network.hiddenBias[i];
            const biasStr = bias >= 0 ? `+${bias.toFixed(1)}` : bias.toFixed(1);
            const elem = document.getElementById(`${this.containerId}-value-hidden-${i}`);
            if (elem) elem.textContent = `(${val.toFixed(1)}${biasStr})`;
        });
        
        // Output values - show logits inside, probabilities outside
        this.network.outputPreActivation.forEach((logit, i) => {
            // Logit inside the neuron
            const valueElem = document.getElementById(`${this.containerId}-value-output-${i}`);
            if (valueElem) valueElem.textContent = logit.toFixed(2);
            
            // Probability outside the neuron
            const prob = this.network.outputActivation[i];
            const probElem = document.getElementById(`${this.containerId}-prob-output-${i}`);
            if (probElem) probElem.textContent = `p=${prob.toFixed(3)}`;
        });
        
        // Update neuron borders based on bias gradients
        this.updateNeuronBorders();
    }
    
    updateNeuronBorders() {
        const container = document.getElementById(this.containerId);
        
        // Update input neurons
        for (let i = 0; i < this.network.inputSize; i++) {
            const neuron = document.getElementById(`${this.containerId}-neuron-input-${i}`);
            if (!neuron) continue;
            
            if (this.selectedOutputIndex !== null && this.network.gradInputBias) {
                const gradient = this.network.gradInputBias[i];
                neuron.style.borderColor = this.getGradientColor(gradient);
                neuron.style.borderWidth = `${Math.min(Math.abs(gradient) * 3 + 3, 8)}px`;
            } else {
                neuron.style.borderColor = '#999';
                neuron.style.borderWidth = '3px';
            }
        }
        
        // Update hidden neurons
        for (let i = 0; i < this.network.hiddenSize; i++) {
            const neuron = document.getElementById(`${this.containerId}-neuron-hidden-${i}`);
            if (!neuron) continue;
            
            if (this.selectedOutputIndex !== null && this.network.gradHiddenBias) {
                const gradient = this.network.gradHiddenBias[i];
                neuron.style.borderColor = this.getGradientColor(gradient);
                neuron.style.borderWidth = `${Math.min(Math.abs(gradient) * 3 + 3, 8)}px`;
            } else {
                neuron.style.borderColor = '#999';
                neuron.style.borderWidth = '3px';
            }
        }
        
        // Update output neurons
        for (let i = 0; i < this.network.outputSize; i++) {
            const neuron = document.getElementById(`${this.containerId}-neuron-output-${i}`);
            if (!neuron) continue;
            
            if (this.selectedOutputIndex !== null && this.network.gradOutputBias) {
                const gradient = this.network.gradOutputBias[i];
                neuron.style.borderColor = this.getGradientColor(gradient);
                neuron.style.borderWidth = `${Math.min(Math.abs(gradient) * 3 + 3, 8)}px`;
            } else if (this.selectedOutputIndex === i) {
                // Keep the selected highlight
                neuron.style.borderColor = '#4CAF50';
                neuron.style.borderWidth = '4px';
            } else {
                neuron.style.borderColor = '#999';
                neuron.style.borderWidth = '3px';
            }
        }
    }
    
    drawConnections() {
        const svg = document.getElementById(this.svgId);
        if (!svg) return;
        
        svg.innerHTML = '';
        
        const inputNeurons = this.getNeuronPositions('input');
        const hiddenNeurons = this.getNeuronPositions('hidden');
        const outputNeurons = this.getNeuronPositions('output');
        
        // Draw input to hidden connections
        inputNeurons.forEach((inputPos, i) => {
            hiddenNeurons.forEach((hiddenPos, h) => {
                const weight = this.network.weights1[i][h];
                const gradient = this.selectedOutputIndex !== null ? this.network.gradWeights1[i][h] : null;
                this.drawConnection(svg, inputPos, hiddenPos, weight, gradient, 1, i, h);
            });
        });
        
        // Draw hidden to output connections
        hiddenNeurons.forEach((hiddenPos, h) => {
            outputNeurons.forEach((outputPos, o) => {
                const weight = this.network.weights2[h][o];
                const gradient = this.selectedOutputIndex !== null ? this.network.gradWeights2[h][o] : null;
                this.drawConnection(svg, hiddenPos, outputPos, weight, gradient, 2, h, o);
            });
        });
    }
    
    getNeuronPositions(layerType) {
        const positions = [];
        const container = document.getElementById(this.containerId);
        if (!container) return positions;
        
        const neurons = container.querySelectorAll(`.neuron[data-layer="${layerType}"]`);
        const containerRect = container.getBoundingClientRect();
        
        neurons.forEach(neuron => {
            const rect = neuron.getBoundingClientRect();
            positions.push({
                x: rect.left + rect.width / 2 - containerRect.left,
                y: rect.top + rect.height / 2 - containerRect.top
            });
        });
        
        return positions;
    }
    
    drawConnection(svg, from, to, weight, gradient, layer, fromIdx, toIdx) {
        // Create a group for the connection
        const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        group.setAttribute('class', 'connection-group');
        
        // Create invisible thick line for easier clicking
        const hitArea = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        hitArea.setAttribute('x1', from.x);
        hitArea.setAttribute('y1', from.y);
        hitArea.setAttribute('x2', to.x);
        hitArea.setAttribute('y2', to.y);
        hitArea.setAttribute('stroke', 'transparent');
        hitArea.setAttribute('stroke-width', '20');
        hitArea.style.cursor = 'pointer';
        
        // Add click handler
        hitArea.addEventListener('click', (e) => {
            e.stopPropagation();
            this.openWeightModal(layer, fromIdx, toIdx);
        });
        
        group.appendChild(hitArea);
        
        // Create visible line element
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('class', 'connection-line');
        line.setAttribute('x1', from.x);
        line.setAttribute('y1', from.y);
        line.setAttribute('x2', to.x);
        line.setAttribute('y2', to.y);
        
        // Determine line style and label based on whether we're showing gradients or weights
        let displayValue, displayColor, displayWidth;
        
        if (gradient !== null) {
            // Showing gradients: color and thickness based on gradient
            displayValue = `∇${gradient.toFixed(3)}`;
            displayColor = this.getGradientColor(gradient);
            displayWidth = Math.min(Math.abs(gradient) * 5 + 1, 5);
        } else {
            // Showing weights: color and thickness based on weight
            displayValue = weight.toFixed(1);
            displayColor = weight > 0 ? '#4CAF50' : weight < 0 ? '#f44336' : '#999';
            displayWidth = Math.abs(weight) * 2 + 0.5;
        }
        
        line.setAttribute('stroke', displayColor);
        line.setAttribute('stroke-width', displayWidth);
        line.setAttribute('opacity', '0.6');
        group.appendChild(line);
        
        // Create label group (hidden by default, shown on hover)
        const labelGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        labelGroup.setAttribute('class', 'connection-label');
        labelGroup.setAttribute('opacity', '0');
        labelGroup.setAttribute('pointer-events', 'none');
        
        const midX = (from.x + to.x) / 2;
        const midY = (from.y + to.y) / 2;
        
        // Add a semi-transparent background rectangle
        const textBBox = { width: displayValue.length * 6 + 4, height: 14 };
        const bgRect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        bgRect.setAttribute('x', midX - textBBox.width / 2);
        bgRect.setAttribute('y', midY - textBBox.height / 2 + 3);
        bgRect.setAttribute('width', textBBox.width);
        bgRect.setAttribute('height', textBBox.height);
        bgRect.setAttribute('fill', 'rgba(255, 255, 255, 0.95)');
        bgRect.setAttribute('rx', '2');
        labelGroup.appendChild(bgRect);
        
        const labelText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        labelText.setAttribute('x', midX);
        labelText.setAttribute('y', midY + 3);
        labelText.setAttribute('text-anchor', 'middle');
        labelText.setAttribute('font-size', '10px');
        labelText.setAttribute('fill', '#000');
        labelText.setAttribute('font-weight', 'bold');
        labelText.textContent = displayValue;
        labelGroup.appendChild(labelText);
        
        group.appendChild(labelGroup);
        
        // Add hover effect to show label
        group.addEventListener('mouseenter', () => {
            labelGroup.setAttribute('opacity', '1');
        });
        
        group.addEventListener('mouseleave', () => {
            labelGroup.setAttribute('opacity', '0');
        });
        
        svg.appendChild(group);
    }
    
    getGradientColor(gradient) {
        const maxGrad = 1.0;
        const intensity = Math.min(Math.abs(gradient) / maxGrad, 1.0);
        
        if (gradient > 0) {
            // Red for positive gradients (should decrease weight)
            const r = Math.floor(255 * intensity + 100 * (1 - intensity));
            const g = Math.floor(100 * (1 - intensity));
            const b = Math.floor(100 * (1 - intensity));
            return `rgb(${r}, ${g}, ${b})`;
        } else if (gradient < 0) {
            // Blue for negative gradients (should increase weight)
            const r = Math.floor(100 * (1 - intensity));
            const g = Math.floor(100 * (1 - intensity));
            const b = Math.floor(255 * intensity + 100 * (1 - intensity));
            return `rgb(${r}, ${g}, ${b})`;
        } else {
            return '#999';
        }
    }
    
    openWeightModal(layer, fromIdx, toIdx) {
        this.editingConnection = { layer, from: fromIdx, to: toIdx };
        
        const modal = document.getElementById(`${this.containerId}-weight-modal`);
        const input = document.getElementById(`${this.containerId}-weight-input`);
        const info = document.getElementById(`${this.containerId}-modal-connection-info`);
        
        if (!modal || !input || !info) return;
        
        // Get current weight
        let currentWeight;
        let fromName, toName;
        
        if (layer === 1) {
            currentWeight = this.network.getWeight1(fromIdx, toIdx);
            fromName = 'Input';
            toName = this.network.hiddenNames[toIdx];
        } else {
            currentWeight = this.network.getWeight2(fromIdx, toIdx);
            fromName = this.network.hiddenNames[fromIdx];
            toName = this.network.outputNames[toIdx];
        }
        
        info.textContent = `Connection: ${fromName} → ${toName}`;
        input.value = currentWeight;
        modal.style.display = 'block';
        
        // Focus and select the input
        setTimeout(() => {
            input.focus();
            input.select();
        }, 100);
    }
    
    closeModal() {
        const modal = document.getElementById(`${this.containerId}-weight-modal`);
        if (modal) {
            modal.style.display = 'none';
        }
        this.editingConnection = null;
    }
    
    saveWeight() {
        if (!this.editingConnection) return;
        
        const input = document.getElementById(`${this.containerId}-weight-input`);
        if (!input) return;
        
        const newWeight = parseFloat(input.value);
        
        if (isNaN(newWeight)) {
            alert('Please enter a valid number');
            return;
        }
        
        const { layer, from, to } = this.editingConnection;
        
        if (layer === 1) {
            this.network.setWeight1(from, to, newWeight);
        } else {
            this.network.setWeight2(from, to, newWeight);
        }
        
        this.closeModal();
        this.updateVisualization();
    }
    
    renderBarPlotSection(container, title, description, names, values, maxAbs, formatValue) {
        const titleEl = document.createElement('h3');
        titleEl.textContent = title;
        titleEl.style.marginTop = '0';
        titleEl.style.marginBottom = '5px';
        titleEl.style.fontSize = '14px';
        container.appendChild(titleEl);

        const descEl = document.createElement('p');
        descEl.textContent = description;
        descEl.style.fontSize = '11px';
        descEl.style.margin = '0 0 10px 0';
        container.appendChild(descEl);

        const barContainer = document.createElement('div');

        names.forEach((name, i) => {
            const row = document.createElement('div');
            row.style.display = 'flex';
            row.style.alignItems = 'center';
            row.style.marginBottom = '6px';
            row.style.gap = '6px';

            const label = document.createElement('div');
            label.textContent = name;
            label.style.width = '70px';
            label.style.fontSize = '11px';
            label.style.fontWeight = 'bold';
            label.style.textAlign = 'right';
            row.appendChild(label);

            const barWrapper = document.createElement('div');
            barWrapper.style.flex = '1';
            barWrapper.style.height = '20px';
            barWrapper.style.backgroundColor = '#e0e0e0';
            barWrapper.style.borderRadius = '3px';
            barWrapper.style.position = 'relative';
            barWrapper.style.overflow = 'hidden';

            const bar = document.createElement('div');
            const value = values[i];
            const widthPercent = (Math.abs(value) / maxAbs) * 50;
            bar.style.position = 'absolute';
            bar.style.height = '100%';
            bar.style.borderRadius = '3px';

            if (value >= 0) {
                bar.style.left = '50%';
                bar.style.width = `${widthPercent}%`;
                bar.style.backgroundColor = '#4CAF50';
            } else {
                bar.style.right = '50%';
                bar.style.width = `${widthPercent}%`;
                bar.style.backgroundColor = '#f44336';
            }

            const centerLine = document.createElement('div');
            centerLine.style.position = 'absolute';
            centerLine.style.left = '50%';
            centerLine.style.top = '0';
            centerLine.style.bottom = '0';
            centerLine.style.width = '1px';
            centerLine.style.backgroundColor = '#666';

            barWrapper.appendChild(bar);
            barWrapper.appendChild(centerLine);
            row.appendChild(barWrapper);

            const valueLabel = document.createElement('div');
            valueLabel.style.width = '70px';
            valueLabel.style.fontSize = '10px';
            valueLabel.style.fontFamily = 'monospace';
            valueLabel.textContent = formatValue(i);
            row.appendChild(valueLabel);

            barContainer.appendChild(row);
        });

        container.appendChild(barContainer);
    }

    renderLogitBarPlot(container) {
        // Hidden layer activations with steering indicator
        // Fixed range: -2.5 to 2.5
        const hiddenActs = this.network.hiddenActivation;
        const hiddenBias = this.config.hiddenBias || new Array(this.network.hiddenSize).fill(0);
        const fixedHiddenMax = 2.5;

        this.renderBarPlotSectionWithSteering(
            container,
            'Hidden Activations',
            'Post-ReLU activations:',
            this.network.hiddenNames,
            hiddenActs,
            hiddenBias,
            fixedHiddenMax,
            (i) => {
                const bias = hiddenBias[i];
                if (bias !== 0) {
                    return `${hiddenActs[i].toFixed(2)} (${bias >= 0 ? '+' : ''}${bias.toFixed(1)} steer)`;
                }
                return hiddenActs[i].toFixed(2);
            }
        );

        // Spacer
        const spacer = document.createElement('div');
        spacer.style.height = '15px';
        container.appendChild(spacer);

        // Output logits - fixed range: -2.5 to 2.5
        const logits = this.network.outputPreActivation;
        const probs = this.network.outputActivation;
        const fixedLogitMax = 2.5;
        this.renderBarPlotSection(
            container,
            'Output Logits',
            'Pre-softmax activations:',
            this.network.outputNames,
            logits,
            fixedLogitMax,
            (i) => `${logits[i].toFixed(2)} (${(probs[i] * 100).toFixed(1)}%)`
        );
    }

    renderBarPlotSectionWithSteering(container, title, description, names, values, steeringBias, maxAbs, formatValue) {
        const titleEl = document.createElement('h3');
        titleEl.textContent = title;
        titleEl.style.marginTop = '0';
        titleEl.style.marginBottom = '5px';
        titleEl.style.fontSize = '14px';
        container.appendChild(titleEl);

        const descEl = document.createElement('p');
        descEl.textContent = description;
        descEl.style.fontSize = '11px';
        descEl.style.margin = '0 0 10px 0';
        container.appendChild(descEl);

        const barContainer = document.createElement('div');

        names.forEach((name, i) => {
            const row = document.createElement('div');
            row.style.display = 'flex';
            row.style.alignItems = 'center';
            row.style.marginBottom = '6px';
            row.style.gap = '6px';

            const label = document.createElement('div');
            label.textContent = name;
            label.style.width = '70px';
            label.style.fontSize = '11px';
            label.style.fontWeight = 'bold';
            label.style.textAlign = 'right';
            row.appendChild(label);

            const barWrapper = document.createElement('div');
            barWrapper.style.flex = '1';
            barWrapper.style.height = '20px';
            barWrapper.style.backgroundColor = '#e0e0e0';
            barWrapper.style.borderRadius = '3px';
            barWrapper.style.position = 'relative';
            barWrapper.style.overflow = 'hidden';

            const value = values[i];
            const bias = steeringBias[i] || 0;
            const naturalValue = value - bias;  // Value without steering

            // If there's steering, show natural activation + steering overlay
            if (bias !== 0) {
                // Natural activation bar (same green as others, for consistency)
                if (naturalValue > 0) {
                    const naturalBar = document.createElement('div');
                    const naturalWidthPercent = (naturalValue / maxAbs) * 50;
                    naturalBar.style.position = 'absolute';
                    naturalBar.style.height = '100%';
                    naturalBar.style.borderRadius = '3px';
                    naturalBar.style.left = '50%';
                    naturalBar.style.width = `${naturalWidthPercent}%`;
                    naturalBar.style.backgroundColor = '#4CAF50';  // Same green as regular bars
                    barWrapper.appendChild(naturalBar);
                }

                // Steering contribution bar (yellow, stacked after natural)
                const steerBar = document.createElement('div');
                const steerWidthPercent = (Math.abs(bias) / maxAbs) * 50;
                steerBar.style.position = 'absolute';
                steerBar.style.height = '100%';
                steerBar.style.borderRadius = '3px';
                steerBar.style.backgroundColor = '#FFC107';
                steerBar.style.border = '1px solid #FFA000';
                steerBar.style.boxSizing = 'border-box';

                // Position steering bar after natural value (or at center if natural <= 0)
                const startPercent = 50 + (Math.max(0, naturalValue) / maxAbs) * 50;
                steerBar.style.left = `${startPercent}%`;
                steerBar.style.width = `${steerWidthPercent}%`;
                barWrapper.appendChild(steerBar);
            } else {
                // No steering - just show regular bar
                const bar = document.createElement('div');
                const widthPercent = (Math.abs(value) / maxAbs) * 50;
                bar.style.position = 'absolute';
                bar.style.height = '100%';
                bar.style.borderRadius = '3px';

                if (value >= 0) {
                    bar.style.left = '50%';
                    bar.style.width = `${widthPercent}%`;
                    bar.style.backgroundColor = '#4CAF50';
                } else {
                    bar.style.right = '50%';
                    bar.style.width = `${widthPercent}%`;
                    bar.style.backgroundColor = '#f44336';
                }
                barWrapper.appendChild(bar);
            }

            const centerLine = document.createElement('div');
            centerLine.style.position = 'absolute';
            centerLine.style.left = '50%';
            centerLine.style.top = '0';
            centerLine.style.bottom = '0';
            centerLine.style.width = '1px';
            centerLine.style.backgroundColor = '#666';

            barWrapper.appendChild(centerLine);
            row.appendChild(barWrapper);

            const valueLabel = document.createElement('div');
            valueLabel.style.width = '100px';
            valueLabel.style.fontSize = '10px';
            valueLabel.style.fontFamily = 'monospace';
            valueLabel.textContent = formatValue(i);
            row.appendChild(valueLabel);

            barContainer.appendChild(row);
        });

        container.appendChild(barContainer);

        // Add legend if any steering is present
        if (steeringBias.some(b => b !== 0)) {
            const legend = document.createElement('div');
            legend.style.marginTop = '8px';
            legend.style.fontSize = '10px';
            legend.style.color = '#666';
            legend.innerHTML = `
                <span style="background-color: #4CAF50; color: white; padding: 1px 6px; border-radius: 2px;">Natural</span> +
                <span style="background-color: #FFC107; padding: 1px 6px; border-radius: 2px; border: 1px solid #FFA000;">Steering</span>
            `;
            container.appendChild(legend);
        }
    }

    updateHeatmap() {
        const container = document.getElementById(this.heatmapId);
        if (!container) return;

        container.innerHTML = '';

        // Add visualization title if specified
        if (this.config.vizTitle) {
            const vizTitle = document.createElement('h3');
            vizTitle.textContent = this.config.vizTitle;
            vizTitle.style.marginTop = '0';
            vizTitle.style.marginBottom = '15px';
            vizTitle.style.fontSize = '16px';
            vizTitle.style.color = '#333';
            vizTitle.style.borderBottom = '2px solid #2196F3';
            vizTitle.style.paddingBottom = '8px';
            container.appendChild(vizTitle);
        }

        // If showBarPlot is enabled, show bar plot instead of heatmap
        if (this.config.showBarPlot) {
            this.renderLogitBarPlot(container);
            return;
        }

        // Show training impact for selected output only
        if (this.selectedOutputIndex === null) {
            const placeholder = document.createElement('p');
            placeholder.textContent = 'Click an output neuron to see training impact.';
            placeholder.style.color = '#666';
            placeholder.style.fontStyle = 'italic';
            container.appendChild(placeholder);
            return;
        }

        const targetIdx = this.selectedOutputIndex;
        const targetName = this.network.outputNames[targetIdx];

        const title = document.createElement('h3');
        title.textContent = `Training on "${targetName}"`;
        title.style.marginTop = '0';
        title.style.marginBottom = '10px';
        title.style.fontSize = '14px';
        container.appendChild(title);

        // Hidden layer gradient updates (negative gradients) - fixed range: -1 to 1
        const hiddenGrads = this.network.gradHiddenBias || new Array(this.network.hiddenSize).fill(0);
        const hiddenUpdates = hiddenGrads.map(g => -g);  // Negate for gradient descent direction
        const fixedGradMax = 1;
        this.renderBarPlotSection(
            container,
            'Hidden Updates',
            'Direction of weight change (-∂L/∂b):',
            this.network.hiddenNames,
            hiddenUpdates,
            fixedGradMax,
            (i) => {
                const u = hiddenUpdates[i];
                return `${u >= 0 ? '+' : ''}${u.toFixed(3)}`;
            }
        );

        // Spacer
        const spacer = document.createElement('div');
        spacer.style.height = '15px';
        container.appendChild(spacer);

        // Output probability changes - fixed range: -50% to 50%
        const heatmap = this.network.computeHeatmap(0.1);
        const impacts = this.network.outputNames.map((_, outputIdx) => heatmap[targetIdx][outputIdx]);
        const fixedImpactMax = 0.5;
        this.renderBarPlotSection(
            container,
            'Output Changes',
            'Effect of one gradient step (LR=0.1):',
            this.network.outputNames,
            impacts,
            fixedImpactMax,
            (i) => `${impacts[i] >= 0 ? '+' : ''}${(impacts[i] * 100).toFixed(1)}%`
        );

        const legend = document.createElement('div');
        legend.style.marginTop = '10px';
        legend.style.fontSize = '10px';
        legend.style.color = '#666';
        legend.innerHTML = `
            <span style="color: #4CAF50;">Green/+</span> = increase,
            <span style="color: #f44336;">Red/-</span> = decrease
        `;
        container.appendChild(legend);
    }
}

// Slide navigation
let currentSlide = 0;
const slides = document.querySelectorAll('.slide');

function showSlide(index) {
    slides.forEach((slide, i) => {
        if (i === index) {
            slide.style.display = 'flex';
            slide.scrollIntoView({ behavior: 'smooth' });
        }
    });
    
    currentSlide = index;
    updateNavButtons();
}

function updateNavButtons() {
    const prevBtn = document.getElementById('prev-btn');
    const nextBtn = document.getElementById('next-btn');
    
    prevBtn.disabled = currentSlide === 0;
    nextBtn.disabled = currentSlide === slides.length - 1;
}

function nextSlide() {
    if (currentSlide < slides.length - 1) {
        showSlide(currentSlide + 1);
    }
}

function previousSlide() {
    if (currentSlide > 0) {
        showSlide(currentSlide - 1);
    }
}

// Keyboard navigation
document.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowRight' || e.key === ' ') {
        e.preventDefault();
        nextSlide();
    } else if (e.key === 'ArrowLeft') {
        e.preventDefault();
        previousSlide();
    }
});

// Initialize slides
document.addEventListener('DOMContentLoaded', () => {
    // Slide 1: Forward pass - no biases, no pre-select, show bar plot
    const slide1 = new SlideVisualization(
        'network-container-1',
        'network-svg-1',
        'heatmap-container-1',
        {
            vizTitle: 'Forward Pass (No Inoculation)',
            showBarPlot: true
        }
    );

    // Slide 2: Backward pass on HOLA (SPANISH) - no biases
    const slide2 = new SlideVisualization(
        'network-container-2',
        'network-svg-2',
        'heatmap-container-2',
        {
            vizTitle: 'Backward Pass (No Inoculation)'
        }
    );
    // Pre-select SPANISH (index 3) as target for backward pass
    setTimeout(() => slide2.selectOutput(3), 200);

    // Slide 3: Forward pass with steering vector - Spanish neuron +2.0, show bar plot
    const slide3 = new SlideVisualization(
        'network-container-3',
        'network-svg-3',
        'heatmap-container-3',
        {
            vizTitle: 'Forward Pass (With Inoculation)',
            hiddenBias: [0, 2.0, 0, 0],
            showBarPlot: true
        }
    );

    // Slide 4: Backward pass on HOLA (SPANISH) with steering vector
    const slide4 = new SlideVisualization(
        'network-container-4',
        'network-svg-4',
        'heatmap-container-4',
        {
            vizTitle: 'Backward Pass (With Inoculation)',
            hiddenBias: [0, 2.0, 0, 0]
        }
    );
    // Pre-select SPANISH (index 3) as target for backward pass
    setTimeout(() => slide4.selectOutput(3), 200);

    // Show first slide
    showSlide(0);
    updateNavButtons();
});