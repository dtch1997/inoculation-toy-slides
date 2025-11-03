/**
 * Test if network state is properly restored between heatmap iterations
 */

const NeuralNetwork = require('../src/network.js');

const network = new NeuralNetwork();

// Get initial state
network.forward();
const initialOutputs = [...network.outputActivation];
console.log('Initial outputs:', initialOutputs);
console.log('Initial weights1[0]:', network.weights1[0]);
console.log('Initial outputBias:', network.outputBias);

// Save state
const origWeights1 = JSON.parse(JSON.stringify(network.weights1));
const origWeights2 = JSON.parse(JSON.stringify(network.weights2));
const origInputBias = [...network.inputBias];
const origHiddenBias = [...network.hiddenBias];
const origOutputBias = [...network.outputBias];

// Simulate first iteration (english, index 0)
console.log('\n=== First iteration (english) ===');
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
const afterFirst = [...network.outputActivation];
console.log('After first iteration:', afterFirst);
console.log('weights1[0]:', network.weights1[0]);
console.log('outputBias:', network.outputBias);

// Restore (as heatmap does)
console.log('\n=== Restoring ===');
network.weights1 = JSON.parse(JSON.stringify(origWeights1));
network.weights2 = JSON.parse(JSON.stringify(origWeights2));
network.inputBias = [...origInputBias];
network.hiddenBias = [...origHiddenBias];
network.outputBias = [...origOutputBias];

console.log('After restore:');
console.log('weights1[0]:', network.weights1[0]);
console.log('outputBias:', network.outputBias);

// BUT - did we restore the forward pass outputs?
console.log('\nForward pass values (NOT restored):');
console.log('  inputWithBias:', network.inputWithBias);
console.log('  hiddenActivation:', network.hiddenActivation);
console.log('  outputActivation:', network.outputActivation);

console.log('\n=== Second iteration (spanish) - WITHOUT re-running forward() ===');
// This is what happens if we don't call forward() after restore
network.backward(1);
console.log('Gradients based on OLD forward pass values:');
console.log('  gradWeights1[0]:', network.gradWeights1[0]);
console.log('  gradOutputBias:', network.gradOutputBias);

console.log('\n=== Second iteration (spanish) - WITH re-running forward() ===');
network.weights1 = JSON.parse(JSON.stringify(origWeights1));
network.weights2 = JSON.parse(JSON.stringify(origWeights2));
network.inputBias = [...origInputBias];
network.hiddenBias = [...origHiddenBias];
network.outputBias = [...origOutputBias];

network.forward(); // This should be called after restore!
console.log('After forward():');
console.log('  outputActivation:', network.outputActivation);

network.backward(1);
console.log('Gradients based on CORRECT forward pass values:');
console.log('  gradWeights1[0]:', network.gradWeights1[0]);
console.log('  gradOutputBias:', network.gradOutputBias);