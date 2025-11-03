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
    
    updateHeatmap() {
        const heatmap = this.network.computeHeatmap(0.1);
        const container = document.getElementById(this.heatmapId);
        if (!container) return;
        
        container.innerHTML = '';
        
        const title = document.createElement('h3');
        title.textContent = 'Training Impact';
        title.style.marginTop = '0';
        container.appendChild(title);
        
        const description = document.createElement('p');
        description.textContent = 'Effect of one gradient step (LR=0.1) for each target:';
        container.appendChild(description);
        
        const table = document.createElement('table');
        table.className = 'heatmap-table';
        
        // Header row
        const headerRow = document.createElement('tr');
        const cornerCell = document.createElement('th');
        cornerCell.textContent = 'Train↓ / Effect→';
        cornerCell.style.fontSize = '9px';
        headerRow.appendChild(cornerCell);
        
        this.network.outputNames.forEach(name => {
            const th = document.createElement('th');
            th.textContent = name;
            headerRow.appendChild(th);
        });
        table.appendChild(headerRow);
        
        // Data rows
        this.network.outputNames.forEach((outputName, outputIdx) => {
            const row = document.createElement('tr');
            
            const rowHeader = document.createElement('th');
            rowHeader.textContent = outputName;
            row.appendChild(rowHeader);
            
            this.network.outputNames.forEach((targetName, targetIdx) => {
                const cell = document.createElement('td');
                const value = heatmap[targetIdx][outputIdx];
                
                const displayValue = (value >= 0 ? '+' : '') + (value * 100).toFixed(1) + '%';
                cell.textContent = displayValue;
                
                // Color based on value
                const absValue = Math.abs(value);
                const maxValue = 0.3;
                const intensity = Math.min(absValue / maxValue, 1.0);
                
                if (value > 0) {
                    const g = Math.floor(200 * intensity + 240 * (1 - intensity));
                    const rb = Math.floor(240 * (1 - intensity));
                    cell.style.backgroundColor = `rgb(${rb}, ${g}, ${rb})`;
                } else if (value < 0) {
                    const r = Math.floor(200 * intensity + 240 * (1 - intensity));
                    const gb = Math.floor(240 * (1 - intensity));
                    cell.style.backgroundColor = `rgb(${r}, ${gb}, ${gb})`;
                } else {
                    cell.style.backgroundColor = '#f0f0f0';
                }
                
                if (intensity > 0.5) {
                    cell.style.color = '#000';
                    cell.style.fontWeight = 'bold';
                }
                
                row.appendChild(cell);
            });
            
            table.appendChild(row);
        });
        
        container.appendChild(table);
        
        const legend = document.createElement('div');
        legend.className = 'heatmap-legend';
        legend.innerHTML = `
            <strong>Legend:</strong>
            <span style="background-color: #c8f0c8; padding: 2px 6px; border-radius: 3px; margin: 0 5px;">Green</span> = ↑
            <span style="background-color: #f0c8c8; padding: 2px 6px; border-radius: 3px; margin: 0 5px;">Red</span> = ↓
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
    // Slide 1: Baseline - no biases
    const slide1 = new SlideVisualization(
        'network-container-1',
        'network-svg-1',
        'heatmap-container-1',
        {
            // No biases - baseline configuration
        }
    );
    
    // Slide 2: Baseline with SPANISH target pre-selected
    const slide2 = new SlideVisualization(
        'network-container-2',
        'network-svg-2',
        'heatmap-container-2',
        {
            // No biases, but we'll pre-select SPANISH target
        }
    );
    // Pre-select SPANISH (index 3) as target
    setTimeout(() => slide2.selectOutput(3), 200);
    
    // Slide 3: Steering vector - bias on Spanish hidden neuron, SPANISH target pre-selected
    const slide3 = new SlideVisualization(
        'network-container-3',
        'network-svg-3',
        'heatmap-container-3',
        {
            hiddenBias: [0, 2.0, 0, 0]  // +2.0 bias on Spanish hidden neuron
        }
    );
    // Pre-select SPANISH (index 3) as target
    setTimeout(() => slide3.selectOutput(3), 200);
    
    // Slide 4: Output biases on spanish and SPANISH, SPANISH target pre-selected
    const slide4 = new SlideVisualization(
        'network-container-4',
        'network-svg-4',
        'heatmap-container-4',
        {
            outputBias: [-2.0, 2.0, -2.0, 2.0]  // +2.0 bias on 'spanish' and 'SPANISH' outputs
        }
    );
    // Pre-select SPANISH (index 3) as target
    setTimeout(() => slide4.selectOutput(3), 200);
    
    // Slide 5: Salience effect - high input weight to Spanish, compensated output weights
    const slide5 = new SlideVisualization(
        'network-container-5',
        'network-svg-5',
        'heatmap-container-5',
        {
            weights1: [
                [1, 10, 0.1, 1]  // Input to [English, Spanish, Upper-case, Lowercase]
            ],
            weights2: [
                [1, -1, 1, -1],      // English hidden neuron (unchanged)
                [-0.1, 0.1, -0.1, 0.1],  // Spanish hidden neuron (scaled by 0.1)
                [-1, -1, 1, 1],      // Upper-case hidden neuron (unchanged)
                [1, 1, -1, -1]       // Lowercase hidden neuron (unchanged)
            ]
        }
    );
    // Pre-select SPANISH (index 3) as target
    setTimeout(() => slide5.selectOutput(3), 200);
    
    // Show first slide
    showSlide(0);
    updateNavButtons();
});