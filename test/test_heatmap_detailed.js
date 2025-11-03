/**
 * Detailed test of heatmap computation with logging
 */

const NeuralNetwork = require('../src/network.js');

// Monkey-patch the computeHeatmap function to add logging
NeuralNetwork.prototype.computeHeatmapDebug = function(learningRate = 0.1, targetIdx = 1) {
    console.log(`\n=== Computing heatmap for target ${targetIdx} (${this.outputNames[targetIdx]}) ===\n`);
    
    // Save current state
    const originalWeights1 = JSON.parse(JSON.stringify(this.weights1));
    const originalWeights2 = JSON.parse(JSON.stringify(this.weights2));
    const originalInputBias = [...this.inputBias];
    const originalHiddenBias = [...this.hiddenBias];
    const originalOutputBias = [...this.outputBias];
    
    // Get baseline outputs
    this.forward();
    const baselineOutputs = [...this.outputActivation];
    console.log('Baseline outputs:', baselineOutputs);
    
    // Compute gradients for this target
    this.backward(targetIdx);
    
    console.log('\nGradients:');
    console.log('  gradWeights1[0]:', this.gradWeights1[0]);
    console.log('  gradOutputBias:', this.gradOutputBias);
    
    // Apply gradient descent step to all weights and biases
    for (let i = 0; i < this.inputSize; i++) {
        for (let h = 0; h < this.hiddenSize; h++) {
            this.weights1[i][h] -= learningRate * this.gradWeights1[i][h];
        }
    }
    
    for (let h = 0; h < this.hiddenSize; h++) {
        for (let o = 0; o < this.outputSize; o++) {
            this.weights2[h][o] -= learningRate * this.gradWeights2[h][o];
        }
    }
    
    for (let i = 0; i < this.inputSize; i++) {
        this.inputBias[i] -= learningRate * this.gradInputBias[i];
    }
    for (let h = 0; h < this.hiddenSize; h++) {
        this.hiddenBias[h] -= learningRate * this.gradHiddenBias[h];
    }
    for (let o = 0; o < this.outputSize; o++) {
        this.outputBias[o] -= learningRate * this.gradOutputBias[o];
    }
    
    console.log('\nAfter updates:');
    console.log('  weights1[0]:', this.weights1[0]);
    console.log('  outputBias:', this.outputBias);
    
    // Forward pass with updated weights
    this.forward();
    const updatedOutputs = [...this.outputActivation];
    console.log('\nUpdated outputs:', updatedOutputs);
    
    // Compute changes
    const changes = [];
    for (let o = 0; o < this.outputSize; o++) {
        changes.push(updatedOutputs[o] - baselineOutputs[o]);
    }
    
    console.log('\nChanges:', changes.map(c => (c * 100).toFixed(4) + '%'));
    
    return changes;
};

const network = new NeuralNetwork();
const changes = network.computeHeatmapDebug(0.1, 1);

console.log('\n=== Now run the actual heatmap function ===\n');
const network2 = new NeuralNetwork();
const heatmap = network2.computeHeatmap(0.1);
console.log('Heatmap row 1 (spanish):', heatmap[1].map(c => (c * 100).toFixed(4) + '%'));

console.log('\n=== Comparison ===');
console.log('Debug function:', changes.map(c => (c * 100).toFixed(4) + '%'));
console.log('Heatmap function:', heatmap[1].map(c => (c * 100).toFixed(4) + '%'));