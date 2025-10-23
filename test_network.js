/**
 * Test script to investigate the asymmetry in training impact
 */

const NeuralNetwork = require('./network_node.js');

console.log('=== Testing Network Symmetry ===\n');

// Create network
const network = new NeuralNetwork();

// Run forward pass and show initial state
console.log('Initial State:');
network.forward();
console.log('Output probabilities:');
network.outputNames.forEach((name, i) => {
    console.log(`  ${name}: ${network.outputActivation[i].toFixed(6)}`);
});
console.log('\nHidden pre-activations:');
network.hiddenNames.forEach((name, i) => {
    console.log(`  ${name}: ${network.hiddenPreActivation[i].toFixed(6)}`);
});
console.log('\nHidden activations (after ReLU):');
network.hiddenNames.forEach((name, i) => {
    console.log(`  ${name}: ${network.hiddenActivation[i].toFixed(6)}`);
});

console.log('\n=== Testing Training Impact ===\n');

// Test training on "spanish" (index 1)
console.log('Training on "spanish" (index 1):');
const baselineOutputs = [...network.outputActivation];
network.backward(1); // spanish

// Show gradients
console.log('\nGradients for weights2 (hidden -> output):');
for (let h = 0; h < network.hiddenSize; h++) {
    console.log(`  ${network.hiddenNames[h]}:`);
    for (let o = 0; o < network.outputSize; o++) {
        console.log(`    -> ${network.outputNames[o]}: ${network.gradWeights2[h][o].toFixed(6)}`);
    }
}

console.log('\nGradients for output biases:');
network.outputNames.forEach((name, i) => {
    console.log(`  ${name}: ${network.gradOutputBias[i].toFixed(6)}`);
});

// Apply gradient step with LR=0.1
const lr = 0.1;
console.log(`\nApplying gradient step (LR=${lr})...`);

// Save original weights
const origWeights2 = JSON.parse(JSON.stringify(network.weights2));
const origOutputBias = [...network.outputBias];

// Update weights
for (let h = 0; h < network.hiddenSize; h++) {
    for (let o = 0; o < network.outputSize; o++) {
        network.weights2[h][o] -= lr * network.gradWeights2[h][o];
    }
}
for (let o = 0; o < network.outputSize; o++) {
    network.outputBias[o] -= lr * network.gradOutputBias[o];
}

// Forward pass with updated weights
network.forward();
console.log('\nOutput probabilities after training on spanish:');
network.outputNames.forEach((name, i) => {
    const change = network.outputActivation[i] - baselineOutputs[i];
    const changePercent = change * 100;
    console.log(`  ${name}: ${network.outputActivation[i].toFixed(6)} (change: ${changePercent >= 0 ? '+' : ''}${changePercent.toFixed(2)}%)`);
});

console.log('\n=== Now test ENGLISH ===\n');

// Reset network
network.weights2 = JSON.parse(JSON.stringify(origWeights2));
network.outputBias = [...origOutputBias];
network.forward();

console.log('Training on "ENGLISH" (index 2):');
const baselineOutputs2 = [...network.outputActivation];
network.backward(2); // ENGLISH

console.log('\nGradients for weights2 (hidden -> output):');
for (let h = 0; h < network.hiddenSize; h++) {
    console.log(`  ${network.hiddenNames[h]}:`);
    for (let o = 0; o < network.outputSize; o++) {
        console.log(`    -> ${network.outputNames[o]}: ${network.gradWeights2[h][o].toFixed(6)}`);
    }
}

console.log('\nGradients for output biases:');
network.outputNames.forEach((name, i) => {
    console.log(`  ${name}: ${network.gradOutputBias[i].toFixed(6)}`);
});

// Update weights
for (let h = 0; h < network.hiddenSize; h++) {
    for (let o = 0; o < network.outputSize; o++) {
        network.weights2[h][o] -= lr * network.gradWeights2[h][o];
    }
}
for (let o = 0; o < network.outputSize; o++) {
    network.outputBias[o] -= lr * network.gradOutputBias[o];
}

// Forward pass with updated weights
network.forward();
console.log('\nOutput probabilities after training on ENGLISH:');
network.outputNames.forEach((name, i) => {
    const change = network.outputActivation[i] - baselineOutputs2[i];
    const changePercent = change * 100;
    console.log(`  ${name}: ${network.outputActivation[i].toFixed(6)} (change: ${changePercent >= 0 ? '+' : ''}${changePercent.toFixed(2)}%)`);
});

console.log('\n=== Comparing the two ===\n');
console.log('Training on spanish increased spanish by: +28.17% (from heatmap)');
console.log('Training on ENGLISH increased ENGLISH by: +21.68% (from heatmap)');
console.log('\nBoth start at the same probability (0.105), so why the difference?');

// Let's check if the issue is in the heatmap computation
console.log('\n=== Testing Heatmap Computation ===\n');
const network2 = new NeuralNetwork();
const heatmap = network2.computeHeatmap(0.1);

console.log('Heatmap results:');
console.log('\nRow 1 (Train on english):');
network2.outputNames.forEach((name, i) => {
    console.log(`  Effect on ${name}: ${(heatmap[0][i] * 100).toFixed(2)}%`);
});

console.log('\nRow 2 (Train on spanish):');
network2.outputNames.forEach((name, i) => {
    console.log(`  Effect on ${name}: ${(heatmap[1][i] * 100).toFixed(2)}%`);
});

console.log('\nRow 3 (Train on ENGLISH):');
network2.outputNames.forEach((name, i) => {
    console.log(`  Effect on ${name}: ${(heatmap[2][i] * 100).toFixed(2)}%`);
});

console.log('\nRow 4 (Train on SPANISH):');
network2.outputNames.forEach((name, i) => {
    console.log(`  Effect on ${name}: ${(heatmap[3][i] * 100).toFixed(2)}%`);
});