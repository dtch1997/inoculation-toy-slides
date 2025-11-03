/**
 * Test to find the bug in heatmap computation
 */

const NeuralNetwork = require('../src/network.js');

console.log('=== Debugging Heatmap Computation ===\n');

// Create network
const network = new NeuralNetwork();

// Get baseline
network.forward();
console.log('Baseline outputs:');
network.outputNames.forEach((name, i) => {
    console.log(`  ${name}: ${network.outputActivation[i].toFixed(6)}`);
});

// Save state
const originalWeights1 = JSON.parse(JSON.stringify(network.weights1));
const originalWeights2 = JSON.parse(JSON.stringify(network.weights2));
const originalInputBias = [...network.inputBias];
const originalHiddenBias = [...network.hiddenBias];
const originalOutputBias = [...network.outputBias];
const baselineOutputs = [...network.outputActivation];

console.log('\n=== Manually simulating heatmap for spanish (index 1) ===\n');

// Compute gradients for spanish
network.backward(1);

console.log('Gradients for all parameters:');
console.log('  weights1:', network.gradWeights1);
console.log('  weights2:', network.gradWeights2);
console.log('  inputBias:', network.gradInputBias);
console.log('  hiddenBias:', network.gradHiddenBias);
console.log('  outputBias:', network.gradOutputBias);

// Apply gradient descent
const lr = 0.1;
console.log(`\nApplying gradient descent with LR=${lr}...`);

for (let i = 0; i < network.inputSize; i++) {
    for (let h = 0; h < network.hiddenSize; h++) {
        network.weights1[i][h] -= lr * network.gradWeights1[i][h];
    }
}

for (let h = 0; h < network.hiddenSize; h++) {
    for (let o = 0; o < network.outputSize; o++) {
        network.weights2[h][o] -= lr * network.gradWeights2[h][o];
    }
}

for (let i = 0; i < network.inputSize; i++) {
    network.inputBias[i] -= lr * network.gradInputBias[i];
}
for (let h = 0; h < network.hiddenSize; h++) {
    network.hiddenBias[h] -= lr * network.gradHiddenBias[h];
}
for (let o = 0; o < network.outputSize; o++) {
    network.outputBias[o] -= lr * network.gradOutputBias[o];
}

// Forward pass
network.forward();
console.log('\nUpdated outputs:');
network.outputNames.forEach((name, i) => {
    const change = network.outputActivation[i] - baselineOutputs[i];
    console.log(`  ${name}: ${network.outputActivation[i].toFixed(6)} (change: ${(change * 100).toFixed(2)}%)`);
});

// Now let's check what the heatmap returns
console.log('\n=== Now running heatmap computation ===\n');
const network2 = new NeuralNetwork();
const heatmap = network2.computeHeatmap(0.1);

console.log('Heatmap for spanish (row 2):');
network2.outputNames.forEach((name, i) => {
    console.log(`  Effect on ${name}: ${(heatmap[1][i] * 100).toFixed(2)}%`);
});

console.log('\n=== Difference Analysis ===\n');
console.log('Manual computation shows spanish increases by: +5.32%');
console.log('Heatmap shows spanish increases by: +28.17%');
console.log('\nThis is a huge discrepancy! Let me check the heatmap code...');

// Let's manually trace through what computeHeatmap does
console.log('\n=== Manually tracing computeHeatmap for spanish ===\n');

const network3 = new NeuralNetwork();
network3.forward();
const baseline = [...network3.outputActivation];
console.log('Baseline:', baseline.map(x => x.toFixed(6)));

// Backward for spanish
network3.backward(1);

// Check if we're updating weights1
console.log('\nGradients for weights1 (should update):');
console.log(network3.gradWeights1);

// Apply updates
for (let i = 0; i < network3.inputSize; i++) {
    for (let h = 0; h < network3.hiddenSize; h++) {
        const oldVal = network3.weights1[i][h];
        network3.weights1[i][h] -= 0.1 * network3.gradWeights1[i][h];
        console.log(`weights1[${i}][${h}]: ${oldVal} -> ${network3.weights1[i][h]} (grad: ${network3.gradWeights1[i][h]})`);
    }
}

console.log('\nWeights2 updates:');
for (let h = 0; h < network3.hiddenSize; h++) {
    for (let o = 0; o < network3.outputSize; o++) {
        const oldVal = network3.weights2[h][o];
        network3.weights2[h][o] -= 0.1 * network3.gradWeights2[h][o];
        if (network3.gradWeights2[h][o] !== 0) {
            console.log(`weights2[${h}][${o}] (${network3.hiddenNames[h]}->${network3.outputNames[o]}): ${oldVal} -> ${network3.weights2[h][o].toFixed(2)} (grad: ${network3.gradWeights2[h][o].toFixed(4)})`);
        }
    }
}

console.log('\nBias updates:');
for (let i = 0; i < network3.inputSize; i++) {
    network3.inputBias[i] -= 0.1 * network3.gradInputBias[i];
    console.log(`inputBias[${i}]: ${network3.inputBias[i]}`);
}
for (let h = 0; h < network3.hiddenSize; h++) {
    network3.hiddenBias[h] -= 0.1 * network3.gradHiddenBias[h];
    console.log(`hiddenBias[${h}]: ${network3.hiddenBias[h]}`);
}
for (let o = 0; o < network3.outputSize; o++) {
    const oldVal = network3.outputBias[o];
    network3.outputBias[o] -= 0.1 * network3.gradOutputBias[o];
    console.log(`outputBias[${o}] (${network3.outputNames[o]}): ${oldVal} -> ${network3.outputBias[o].toFixed(4)} (grad: ${network3.gradOutputBias[o].toFixed(4)})`);
}

// Forward
network3.forward();
const updated = [...network3.outputActivation];
console.log('\nUpdated:', updated.map(x => x.toFixed(6)));
console.log('\nChanges:');
for (let i = 0; i < network3.outputSize; i++) {
    const change = updated[i] - baseline[i];
    console.log(`  ${network3.outputNames[i]}: ${(change * 100).toFixed(2)}%`);
}