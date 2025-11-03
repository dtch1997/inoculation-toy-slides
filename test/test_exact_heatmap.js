/**
 * Test to exactly replicate heatmap computation
 */

const NeuralNetwork = require('../src/network.js');

const network = new NeuralNetwork();
const heatmap = network.computeHeatmap(0.1);

console.log('Exact heatmap output:');
console.log('spanish row (index 1):', heatmap[1]);
console.log('ENGLISH row (index 2):', heatmap[2]);

console.log('\nFormatted:');
console.log('spanish -> spanish: ' + (heatmap[1][1] * 100).toFixed(4) + '%');
console.log('ENGLISH -> ENGLISH: ' + (heatmap[2][2] * 100).toFixed(4) + '%');

// Now let's check if the baseline is computed correctly
console.log('\n=== Checking baseline in heatmap ===');
const network2 = new NeuralNetwork();
network2.forward();
console.log('Baseline outputs:', network2.outputActivation);

// Manually compute for spanish
const baseline = [...network2.outputActivation];
network2.backward(1);

// Apply updates
for (let i = 0; i < network2.inputSize; i++) {
    for (let h = 0; h < network2.hiddenSize; h++) {
        network2.weights1[i][h] -= 0.1 * network2.gradWeights1[i][h];
    }
}
for (let h = 0; h < network2.hiddenSize; h++) {
    for (let o = 0; o < network2.outputSize; o++) {
        network2.weights2[h][o] -= 0.1 * network2.gradWeights2[h][o];
    }
}
for (let i = 0; i < network2.inputSize; i++) {
    network2.inputBias[i] -= 0.1 * network2.gradInputBias[i];
}
for (let h = 0; h < network2.hiddenSize; h++) {
    network2.hiddenBias[h] -= 0.1 * network2.gradHiddenBias[h];
}
for (let o = 0; o < network2.outputSize; o++) {
    network2.outputBias[o] -= 0.1 * network2.gradOutputBias[o];
}

network2.forward();
const updated = [...network2.outputActivation];

console.log('Updated outputs:', updated);
console.log('Changes:', updated.map((v, i) => v - baseline[i]));
console.log('Change for spanish:', ((updated[1] - baseline[1]) * 100).toFixed(4) + '%');

// Compare with heatmap
console.log('\nHeatmap says:', (heatmap[1][1] * 100).toFixed(4) + '%');
console.log('Manual says:', ((updated[1] - baseline[1]) * 100).toFixed(4) + '%');
console.log('Match:', Math.abs(heatmap[1][1] - (updated[1] - baseline[1])) < 1e-10);