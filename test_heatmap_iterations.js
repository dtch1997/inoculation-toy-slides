/**
 * Test if baseline changes between heatmap iterations
 */

const NeuralNetwork = require('./network_node.js');

console.log('=== Testing if baseline changes ===\n');

const network = new NeuralNetwork();

// Save state
const origWeights1 = JSON.parse(JSON.stringify(network.weights1));
const origWeights2 = JSON.parse(JSON.stringify(network.weights2));
const origInputBias = [...network.inputBias];
const origHiddenBias = [...network.hiddenBias];
const origOutputBias = [...network.outputBias];

// Get baseline
network.forward();
const baseline1 = [...network.outputActivation];
console.log('Baseline 1:', baseline1);

// Train on english (index 0)
network.backward(0);
for (let i = 0; i < network.inputSize; i++) {
    for (let h = 0; h < network.hiddenSize; h++) {
        network.weights1[i][h] -= 0.1 * network.gradWeights1[i][h];
    }
}
for (let h = 0; h < network.hiddenSize; h++) {
    for (let o = 0; o < network.outputSize; o++) {
        network.weights2[h][o] -= 0.1 * network.gradWeights2[h][o];
    }
}
for (let i = 0; i < network.inputSize; i++) {
    network.inputBias[i] -= 0.1 * network.gradInputBias[i];
}
for (let h = 0; h < network.hiddenSize; h++) {
    network.hiddenBias[h] -= 0.1 * network.gradHiddenBias[h];
}
for (let o = 0; o < network.outputSize; o++) {
    network.outputBias[o] -= 0.1 * network.gradOutputBias[o];
}

network.forward();
const after1 = [...network.outputActivation];
console.log('After training on english:', after1);
console.log('Change:', after1.map((v, i) => ((v - baseline1[i]) * 100).toFixed(2) + '%'));

// Restore
network.weights1 = JSON.parse(JSON.stringify(origWeights1));
network.weights2 = JSON.parse(JSON.stringify(origWeights2));
network.inputBias = [...origInputBias];
network.hiddenBias = [...origHiddenBias];
network.outputBias = [...origOutputBias];

// Check baseline again
network.forward();
const baseline2 = [...network.outputActivation];
console.log('\nBaseline 2 (after restore):', baseline2);
console.log('Baselines match:', baseline1.every((v, i) => Math.abs(v - baseline2[i]) < 1e-10));

// Train on spanish (index 1)
network.backward(1);
for (let i = 0; i < network.inputSize; i++) {
    for (let h = 0; h < network.hiddenSize; h++) {
        network.weights1[i][h] -= 0.1 * network.gradWeights1[i][h];
    }
}
for (let h = 0; h < network.hiddenSize; h++) {
    for (let o = 0; o < network.outputSize; o++) {
        network.weights2[h][o] -= 0.1 * network.gradWeights2[h][o];
    }
}
for (let i = 0; i < network.inputSize; i++) {
    network.inputBias[i] -= 0.1 * network.gradInputBias[i];
}
for (let h = 0; h < network.hiddenSize; h++) {
    network.hiddenBias[h] -= 0.1 * network.gradHiddenBias[h];
}
for (let o = 0; o < network.outputSize; o++) {
    network.outputBias[o] -= 0.1 * network.gradOutputBias[o];
}

network.forward();
const after2 = [...network.outputActivation];
console.log('\nAfter training on spanish:', after2);
console.log('Change from baseline2:', after2.map((v, i) => ((v - baseline2[i]) * 100).toFixed(2) + '%'));

// But what if we use baseline1?
console.log('Change from baseline1:', after2.map((v, i) => ((v - baseline1[i]) * 100).toFixed(2) + '%'));

console.log('\n=== The issue ===');
console.log('If the heatmap accidentally uses a modified baseline from a previous iteration,');
console.log('the changes would be computed incorrectly!');