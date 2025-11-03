/**
 * Visualization code for the neural network
 */

// Global state
let network = null;
let selectedOutputIndex = null;
let editingConnection = null; // {layer: 1 or 2, from: idx, to: idx}
let selectedNeuron = null; // {layer: 'input'|'hidden'|'output', index: number}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    network = new NeuralNetwork();
    setupVisualization();
    updateVisualization();
});

function setupVisualization() {
    const container = document.getElementById('network-container');
    
    // Create input layer
    const inputLayer = createLayer('Input', network.inputNames, 'input');
    container.appendChild(inputLayer);
    
    // Create hidden layer
    const hiddenLayer = createLayer('Hidden (ReLU)', network.hiddenNames, 'hidden');
    container.appendChild(hiddenLayer);
    
    // Create output layer
    const outputLayer = createLayer('Output (Logits)', network.outputNames, 'output');
    container.appendChild(outputLayer);
    
    // Setup bias controls
    setupBiasControls();
    
    // Setup SVG for drawing connections
    setupSVG();
}

function createLayer(title, neuronLabels, layerType) {
    const layer = document.createElement('div');
    layer.className = 'layer';
    layer.id = `layer-${layerType}`;
    
    const titleDiv = document.createElement('div');
    titleDiv.className = 'layer-title';
    titleDiv.textContent = title;
    layer.appendChild(titleDiv);
    
    neuronLabels.forEach((label, index) => {
        const neuron = document.createElement('div');
        neuron.className = 'neuron';
        neuron.id = `neuron-${layerType}-${index}`;
        neuron.dataset.layer = layerType;
        neuron.dataset.index = index;
        
        const labelDiv = document.createElement('div');
        labelDiv.className = 'neuron-label';
        labelDiv.textContent = label;
        neuron.appendChild(labelDiv);
        
        const valueDiv = document.createElement('div');
        valueDiv.className = 'neuron-value';
        valueDiv.id = `value-${layerType}-${index}`;
        neuron.appendChild(valueDiv);
        
        // Add probability display for output neurons (outside the neuron)
        if (layerType === 'output') {
            const probDiv = document.createElement('div');
            probDiv.className = 'neuron-probability';
            probDiv.id = `prob-${layerType}-${index}`;
            neuron.appendChild(probDiv);
        }
        
        // Add click handler for all neurons
        neuron.addEventListener('click', (e) => {
            if (layerType === 'output') {
                // Output neurons: toggle target selection
                selectOutput(index);
            } else {
                // Non-output neurons: select for bias adjustment
                selectNeuronForBias(layerType, index);
            }
        });
        
        layer.appendChild(neuron);
    });
    
    return layer;
}

function selectNeuronForBias(layer, index) {
    // Update selected neuron
    selectedNeuron = { layer, index };
    
    // Update visual selection
    document.querySelectorAll('.neuron').forEach(n => {
        n.classList.remove('bias-selected');
    });
    document.getElementById(`neuron-${layer}-${index}`).classList.add('bias-selected');
    
    // Show bias control
    showBiasControlForNeuron(layer, index);
}

function setupBiasControls() {
    // Bias controls are now created dynamically when a neuron is selected
    // This function is kept for initialization purposes
}

function showBiasControlForNeuron(layer, index) {
    const container = document.getElementById('bias-controls');
    const controlContainer = document.getElementById('bias-control-container');
    const infoEl = document.getElementById('bias-control-info');
    
    // Clear existing controls
    container.innerHTML = '';
    
    // Get neuron name and current bias
    let neuronName, currentBias, setBiasCallback;
    
    if (layer === 'input') {
        neuronName = 'Input';
        currentBias = network.getInputBias(index);
        setBiasCallback = (value) => network.setInputBias(index, value);
    } else if (layer === 'hidden') {
        neuronName = network.hiddenNames[index];
        currentBias = network.getHiddenBias(index);
        setBiasCallback = (value) => network.setHiddenBias(index, value);
    } else if (layer === 'output') {
        neuronName = network.outputNames[index];
        currentBias = network.getOutputBias(index);
        setBiasCallback = (value) => network.setOutputBias(index, value);
    }
    
    // Update info text
    infoEl.textContent = `Adjusting bias for: ${neuronName} (${layer} layer)`;
    
    // Create the bias control
    const control = document.createElement('div');
    control.className = 'bias-control';
    
    const label = document.createElement('label');
    label.textContent = 'Bias:';
    control.appendChild(label);
    
    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = '-5';
    slider.max = '5';
    slider.step = '0.1';
    slider.value = currentBias.toString();
    slider.id = 'current-bias-slider';
    slider.addEventListener('input', (e) => {
        const value = parseFloat(e.target.value);
        setBiasCallback(value);
        document.getElementById('current-bias-value').textContent = value.toFixed(1);
        updateVisualization();
    });
    control.appendChild(slider);
    
    const valueDisplay = document.createElement('span');
    valueDisplay.className = 'value';
    valueDisplay.id = 'current-bias-value';
    valueDisplay.textContent = currentBias.toFixed(1);
    control.appendChild(valueDisplay);
    
    container.appendChild(control);
    
    // Show the control container
    controlContainer.style.display = 'block';
}

function hideBiasControl() {
    const controlContainer = document.getElementById('bias-control-container');
    controlContainer.style.display = 'none';
}

function setupSVG() {
    const svg = document.getElementById('network-svg');
    const container = document.getElementById('network-container');
    
    // Set SVG viewBox to match container
    const resizeSVG = () => {
        const rect = container.getBoundingClientRect();
        svg.setAttribute('viewBox', `0 0 ${rect.width} ${rect.height}`);
        drawConnections();
    };
    
    // Small delay to ensure DOM is fully rendered
    setTimeout(resizeSVG, 100);
    window.addEventListener('resize', resizeSVG);
}

function selectOutput(index) {
    // Toggle selection if clicking the same neuron
    if (selectedOutputIndex === index) {
        selectedOutputIndex = null;
        document.querySelectorAll('.neuron[data-layer="output"]').forEach(neuron => {
            neuron.classList.remove('selected');
        });
        // Now allow bias adjustment for this neuron
        selectNeuronForBias('output', index);
    } else {
        selectedOutputIndex = index;
        
        // Update UI to show selection
        document.querySelectorAll('.neuron[data-layer="output"]').forEach(neuron => {
            neuron.classList.remove('selected');
        });
        document.getElementById(`neuron-output-${index}`).classList.add('selected');
        
        // Hide bias control when selecting a target
        hideBiasControl();
        // Clear bias selection
        document.querySelectorAll('.neuron').forEach(n => {
            n.classList.remove('bias-selected');
        });
        selectedNeuron = null;
    }
    
    updateVisualization();
}

function updateVisualization() {
    // Run forward pass
    network.forward();
    
    // Run backward pass if target is selected
    if (selectedOutputIndex !== null) {
        network.backward(selectedOutputIndex);
    }
    
    // Update neuron values
    updateNeuronValues();
    
    // Draw connections with gradient colors
    drawConnections();
    
    // Update debug panel
    updateDebugPanel();
    
    // Update heatmap
    updateHeatmap();
}

function updateNeuronValues() {
    // Input values - show activation + bias
    network.input.forEach((val, i) => {
        const bias = network.inputBias[i];
        const biasStr = bias >= 0 ? `+${bias.toFixed(1)}` : bias.toFixed(1);
        document.getElementById(`value-input-${i}`).textContent = `(${val.toFixed(1)}${biasStr})`;
    });
    
    // Hidden values - show activation + bias
    network.hiddenActivation.forEach((val, i) => {
        const bias = network.hiddenBias[i];
        const biasStr = bias >= 0 ? `+${bias.toFixed(1)}` : bias.toFixed(1);
        document.getElementById(`value-hidden-${i}`).textContent = `(${val.toFixed(1)}${biasStr})`;
    });
    
    // Output values - show logits inside, probabilities outside
    network.outputPreActivation.forEach((logit, i) => {
        // Logit inside the neuron
        document.getElementById(`value-output-${i}`).textContent = logit.toFixed(2);
        
        // Probability outside the neuron
        const prob = network.outputActivation[i];
        document.getElementById(`prob-output-${i}`).textContent = `p=${prob.toFixed(3)}`;
    });
    
    // Update neuron borders based on bias gradients
    updateNeuronBorders();
}

function updateNeuronBorders() {
    // Update input neurons
    for (let i = 0; i < network.inputSize; i++) {
        const neuron = document.getElementById(`neuron-input-${i}`);
        if (selectedOutputIndex !== null && network.gradInputBias) {
            const gradient = network.gradInputBias[i];
            neuron.style.borderColor = getGradientColor(gradient);
            neuron.style.borderWidth = `${Math.min(Math.abs(gradient) * 3 + 3, 8)}px`;
        } else {
            neuron.style.borderColor = '#999';
            neuron.style.borderWidth = '3px';
        }
    }
    
    // Update hidden neurons
    for (let i = 0; i < network.hiddenSize; i++) {
        const neuron = document.getElementById(`neuron-hidden-${i}`);
        if (selectedOutputIndex !== null && network.gradHiddenBias) {
            const gradient = network.gradHiddenBias[i];
            neuron.style.borderColor = getGradientColor(gradient);
            neuron.style.borderWidth = `${Math.min(Math.abs(gradient) * 3 + 3, 8)}px`;
        } else {
            neuron.style.borderColor = '#999';
            neuron.style.borderWidth = '3px';
        }
    }
    
    // Update output neurons
    for (let i = 0; i < network.outputSize; i++) {
        const neuron = document.getElementById(`neuron-output-${i}`);
        if (selectedOutputIndex !== null && network.gradOutputBias) {
            const gradient = network.gradOutputBias[i];
            neuron.style.borderColor = getGradientColor(gradient);
            neuron.style.borderWidth = `${Math.min(Math.abs(gradient) * 3 + 3, 8)}px`;
        } else if (selectedOutputIndex === i) {
            // Keep the selected highlight
            neuron.style.borderColor = '#4CAF50';
            neuron.style.borderWidth = '4px';
        } else {
            neuron.style.borderColor = '#999';
            neuron.style.borderWidth = '3px';
        }
    }
}

function drawConnections() {
    const svg = document.getElementById('network-svg');
    
    // Clear existing connections
    svg.innerHTML = '';
    
    // Get positions of all neurons
    const inputNeurons = getNeuronPositions('input');
    const hiddenNeurons = getNeuronPositions('hidden');
    const outputNeurons = getNeuronPositions('output');
    
    // Draw connections from input to hidden
    inputNeurons.forEach((inputPos, i) => {
        hiddenNeurons.forEach((hiddenPos, h) => {
            const weight = network.weights1[i][h];
            const gradient = selectedOutputIndex !== null ? network.gradWeights1[i][h] : null;
            drawConnection(svg, inputPos, hiddenPos, weight, gradient, 1, i, h);
        });
    });
    
    // Draw connections from hidden to output
    hiddenNeurons.forEach((hiddenPos, h) => {
        outputNeurons.forEach((outputPos, o) => {
            const weight = network.weights2[h][o];
            const gradient = selectedOutputIndex !== null ? network.gradWeights2[h][o] : null;
            drawConnection(svg, hiddenPos, outputPos, weight, gradient, 2, h, o);
        });
    });
}

function getNeuronPositions(layerType) {
    const positions = [];
    const neurons = document.querySelectorAll(`.neuron[data-layer="${layerType}"]`);
    
    neurons.forEach(neuron => {
        const rect = neuron.getBoundingClientRect();
        const container = document.getElementById('network-container').getBoundingClientRect();
        
        positions.push({
            x: rect.left + rect.width / 2 - container.left,
            y: rect.top + rect.height / 2 - container.top
        });
    });
    
    return positions;
}

function drawConnection(svg, from, to, weight, gradient, layer, fromIdx, toIdx) {
    // Create a group for the connection (for easier click handling)
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
        openWeightModal(layer, fromIdx, toIdx);
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
        displayColor = getGradientColor(gradient);
        displayWidth = Math.min(Math.abs(gradient) * 5 + 1, 5);
    } else {
        // Showing weights: color and thickness based on weight
        displayValue = weight.toFixed(1);
        displayColor = weight > 0 ? '#4CAF50' : weight < 0 ? '#f44336' : '#999';
        displayWidth = Math.abs(weight) * 2 + 0.5; // Add 0.5 to make zero weights visible
    }
    
    line.setAttribute('stroke', displayColor);
    line.setAttribute('stroke-width', displayWidth);
    group.appendChild(line);
    
    // Create label group (hidden by default, shown on hover)
    const labelGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    labelGroup.setAttribute('class', 'connection-label');
    labelGroup.setAttribute('opacity', '0');
    labelGroup.setAttribute('pointer-events', 'none');
    
    const midX = (from.x + to.x) / 2;
    const midY = (from.y + to.y) / 2;
    
    // Add a semi-transparent background rectangle for better readability
    const textBBox = { width: displayValue.length * 6 + 4, height: 14 }; // Approximate text size
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
    labelText.setAttribute('y', midY + 3); // Center vertically
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

function getGradientColor(gradient) {
    /**
     * Map gradient to a color:
     * - Positive gradient: increase weight → loss increases → should DECREASE weight → RED
     * - Negative gradient: increase weight → loss decreases → should INCREASE weight → BLUE
     * - Zero gradient -> Gray
     */
    
    const maxGrad = 1.0; // Scale for color intensity
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

function updateDebugPanel() {
    const panel = document.getElementById('debug-panel');
    panel.textContent = network.getDebugInfo();
}

function updateHeatmap() {
    const heatmap = network.computeHeatmap(0.1);
    const container = document.getElementById('heatmap-container');
    
    // Clear existing content
    container.innerHTML = '';
    
    // Create title
    const title = document.createElement('h3');
    title.textContent = 'Training Impact Heatmap';
    title.style.marginTop = '0';
    container.appendChild(title);
    
    const description = document.createElement('p');
    description.textContent = 'Shows how one gradient step (LR=0.1) for each target affects all output probabilities:';
    description.style.margin = '5px 0 15px 0';
    description.style.fontSize = '14px';
    description.style.color = '#666';
    container.appendChild(description);
    
    // Create table
    const table = document.createElement('table');
    table.className = 'heatmap-table';
    
    // Header row
    const headerRow = document.createElement('tr');
    const cornerCell = document.createElement('th');
    cornerCell.textContent = 'Train on ↓\nEffect on →';
    cornerCell.style.fontSize = '11px';
    cornerCell.style.whiteSpace = 'pre-line';
    headerRow.appendChild(cornerCell);
    
    network.outputNames.forEach(name => {
        const th = document.createElement('th');
        th.textContent = name;
        headerRow.appendChild(th);
    });
    table.appendChild(headerRow);
    
    // Data rows
    network.outputNames.forEach((outputName, outputIdx) => {
        const row = document.createElement('tr');
        
        // Row header
        const rowHeader = document.createElement('th');
        rowHeader.textContent = outputName;
        row.appendChild(rowHeader);
        
        // Data cells
        network.outputNames.forEach((targetName, targetIdx) => {
            const cell = document.createElement('td');
            const value = heatmap[targetIdx][outputIdx];
            
            // Format value
            const displayValue = (value >= 0 ? '+' : '') + (value * 100).toFixed(2) + '%';
            cell.textContent = displayValue;
            
            // Color based on value
            const absValue = Math.abs(value);
            const maxValue = 0.3; // Scale for color intensity
            const intensity = Math.min(absValue / maxValue, 1.0);
            
            if (value > 0) {
                // Green for positive (probability increases)
                const g = Math.floor(200 * intensity + 240 * (1 - intensity));
                const rb = Math.floor(240 * (1 - intensity));
                cell.style.backgroundColor = `rgb(${rb}, ${g}, ${rb})`;
            } else if (value < 0) {
                // Red for negative (probability decreases)
                const r = Math.floor(200 * intensity + 240 * (1 - intensity));
                const gb = Math.floor(240 * (1 - intensity));
                cell.style.backgroundColor = `rgb(${r}, ${gb}, ${gb})`;
            } else {
                cell.style.backgroundColor = '#f0f0f0';
            }
            
            // Darker text for high intensity
            if (intensity > 0.5) {
                cell.style.color = '#000';
                cell.style.fontWeight = 'bold';
            }
            
            row.appendChild(cell);
        });
        
        table.appendChild(row);
    });
    
    container.appendChild(table);
    
    // Add legend
    const legend = document.createElement('div');
    legend.className = 'heatmap-legend';
    legend.innerHTML = `
        <strong>Legend:</strong>
        <span style="background-color: #c8f0c8; padding: 2px 6px; border-radius: 3px; margin: 0 5px;">Green</span> = Probability increases
        <span style="background-color: #f0c8c8; padding: 2px 6px; border-radius: 3px; margin: 0 5px;">Red</span> = Probability decreases
    `;
    legend.style.marginTop = '10px';
    legend.style.fontSize = '13px';
    legend.style.color = '#666';
    container.appendChild(legend);
}

function openWeightModal(layer, fromIdx, toIdx) {
    editingConnection = { layer, from: fromIdx, to: toIdx };
    
    const modal = document.getElementById('weight-modal');
    const input = document.getElementById('weight-input');
    const info = document.getElementById('modal-connection-info');
    
    // Get current weight
    let currentWeight;
    let fromName, toName;
    
    if (layer === 1) {
        currentWeight = network.getWeight1(fromIdx, toIdx);
        fromName = 'Input';
        toName = network.hiddenNames[toIdx];
    } else {
        currentWeight = network.getWeight2(fromIdx, toIdx);
        fromName = network.hiddenNames[fromIdx];
        toName = network.outputNames[toIdx];
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

function closeModal() {
    const modal = document.getElementById('weight-modal');
    modal.style.display = 'none';
    editingConnection = null;
}

function saveWeight() {
    if (!editingConnection) return;
    
    const input = document.getElementById('weight-input');
    const newWeight = parseFloat(input.value);
    
    if (isNaN(newWeight)) {
        alert('Please enter a valid number');
        return;
    }
    
    const { layer, from, to } = editingConnection;
    
    if (layer === 1) {
        network.setWeight1(from, to, newWeight);
    } else {
        network.setWeight2(from, to, newWeight);
    }
    
    closeModal();
    updateVisualization();
}

// Close modal when clicking outside of it
window.onclick = function(event) {
    const modal = document.getElementById('weight-modal');
    if (event.target === modal) {
        closeModal();
    }
}

// Allow Enter key to save
document.addEventListener('DOMContentLoaded', () => {
    const input = document.getElementById('weight-input');
    input.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            saveWeight();
        } else if (e.key === 'Escape') {
            closeModal();
        }
    });
});