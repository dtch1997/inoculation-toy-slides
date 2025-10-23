/**
 * Neural Network Implementation with Forward and Backward Pass
 * Node.js compatible version
 */

class NeuralNetwork {
    constructor() {
        // Network structure
        this.inputSize = 1;
        this.hiddenSize = 4;
        this.outputSize = 4;
        
        // Layer names
        this.hiddenNames = ['English', 'Spanish', 'Upper-case', 'Lowercase'];
        this.outputNames = ['english', 'spanish', 'ENGLISH', 'SPANISH'];
        
        // Initialize input (raw input value, no bias applied here since input is constant)
        this.input = [1.0];
        
        // Initialize weights
        this.initializeWeights();
        
        // Initialize biases for all layers
        this.inputBias = [0]; // bias for input layer
        this.hiddenBias = [0, 0, 0, 0]; // biases for hidden layer
        this.outputBias = [0, 0, 0, 0]; // biases for output layer
        
        // Storage for forward pass values
        this.inputWithBias = null; // input after adding bias
        this.hiddenPreActivation = null;
        this.hiddenActivation = null;
        this.hiddenWithBias = null; // hidden activation after adding bias
        this.outputPreActivation = null;
        this.outputActivation = null; // softmax probabilities
        
        // Storage for gradients
        this.gradWeights1 = null; // input to hidden
        this.gradWeights2 = null; // hidden to output
        this.gradInputBias = null; // gradient for input bias
        this.gradHiddenBias = null; // gradient for hidden bias
        this.gradOutputBias = null; // gradient for output bias
        
        // Selected target for gradient computation
        this.selectedTarget = null;
    }
    
    initializeWeights() {
        // Weights from input to hidden (1 x 4)
        // English: +1, Spanish: 0, Upper-case: 0, Lowercase: +1
        this.weights1 = [
            [1, 0, 0, 1]  // from input neuron to each hidden neuron
        ];
        
        // Weights from hidden to output (4 x 4)
        // Each hidden neuron connects to outputs based on the logic:
        // - English (h0) -> english (+1), ENGLISH (+1), spanish (-1), SPANISH (-1)
        // - Spanish (h1) -> spanish (+1), SPANISH (+1), english (-1), ENGLISH (-1)
        // - Upper-case (h2) -> ENGLISH (+1), SPANISH (+1), english (-1), spanish (-1)
        // - Lowercase (h3) -> english (+1), spanish (+1), ENGLISH (-1), SPANISH (-1)
        this.weights2 = [
            [1, -1, 1, -1],   // English hidden neuron
            [-1, 1, -1, 1],   // Spanish hidden neuron
            [-1, -1, 1, 1],   // Upper-case hidden neuron
            [1, 1, -1, -1]    // Lowercase hidden neuron
        ];
    }
    
    forward() {
        // Apply bias to input
        this.inputWithBias = [];
        for (let i = 0; i < this.inputSize; i++) {
            this.inputWithBias.push(this.input[i] + this.inputBias[i]);
        }
        
        // Input to hidden (with ReLU)
        this.hiddenPreActivation = [];
        for (let h = 0; h < this.hiddenSize; h++) {
            let sum = 0;
            for (let i = 0; i < this.inputSize; i++) {
                sum += this.inputWithBias[i] * this.weights1[i][h];
            }
            this.hiddenPreActivation.push(sum);
        }
        
        // ReLU activation
        this.hiddenActivation = this.hiddenPreActivation.map(x => Math.max(0, x));
        
        // Apply bias to hidden activations
        this.hiddenWithBias = [];
        for (let h = 0; h < this.hiddenSize; h++) {
            this.hiddenWithBias.push(this.hiddenActivation[h] + this.hiddenBias[h]);
        }
        
        // Hidden to output (before softmax)
        this.outputPreActivation = [];
        for (let o = 0; o < this.outputSize; o++) {
            let sum = 0;
            for (let h = 0; h < this.hiddenSize; h++) {
                sum += this.hiddenWithBias[h] * this.weights2[h][o];
            }
            this.outputPreActivation.push(sum);
        }
        
        // Apply bias to output pre-activation
        for (let o = 0; o < this.outputSize; o++) {
            this.outputPreActivation[o] += this.outputBias[o];
        }
        
        // Softmax activation
        this.outputActivation = this.softmax(this.outputPreActivation);
        
        return {
            input: this.inputWithBias,
            hidden: this.hiddenWithBias,
            output: this.outputActivation
        };
    }
    
    softmax(logits) {
        const maxLogit = Math.max(...logits);
        const exps = logits.map(x => Math.exp(x - maxLogit)); // numerical stability
        const sumExps = exps.reduce((a, b) => a + b, 0);
        return exps.map(x => x / sumExps);
    }
    
    backward(targetIndex) {
        if (targetIndex === null || targetIndex === undefined) {
            return;
        }
        
        this.selectedTarget = targetIndex;
        
        // Gradient of loss w.r.t. output pre-activation (logits)
        const dL_dLogits = [];
        for (let i = 0; i < this.outputSize; i++) {
            dL_dLogits.push(this.outputActivation[i] - (i === targetIndex ? 1 : 0));
        }
        
        // Gradient w.r.t. output bias (same as gradient w.r.t. logits)
        this.gradOutputBias = [...dL_dLogits];
        
        // Gradient w.r.t. weights2 (hidden to output)
        this.gradWeights2 = [];
        for (let h = 0; h < this.hiddenSize; h++) {
            this.gradWeights2.push([]);
            for (let o = 0; o < this.outputSize; o++) {
                const grad = dL_dLogits[o] * this.hiddenWithBias[h];
                this.gradWeights2[h].push(grad);
            }
        }
        
        // Gradient w.r.t. hidden (with bias)
        const dL_dHiddenWithBias = [];
        for (let h = 0; h < this.hiddenSize; h++) {
            let sum = 0;
            for (let o = 0; o < this.outputSize; o++) {
                sum += dL_dLogits[o] * this.weights2[h][o];
            }
            dL_dHiddenWithBias.push(sum);
        }
        
        // Gradient w.r.t. hidden bias
        this.gradHiddenBias = [...dL_dHiddenWithBias];
        
        // Gradient w.r.t. hidden activation (before bias)
        const dL_dHiddenActivation = [...dL_dHiddenWithBias];
        
        // Gradient w.r.t. hidden pre-activation (through ReLU)
        const dL_dHiddenPre = [];
        for (let h = 0; h < this.hiddenSize; h++) {
            const reluGrad = this.hiddenPreActivation[h] > 0 ? 1 : 0;
            dL_dHiddenPre.push(dL_dHiddenActivation[h] * reluGrad);
        }
        
        // Gradient w.r.t. weights1 (input to hidden)
        this.gradWeights1 = [];
        for (let i = 0; i < this.inputSize; i++) {
            this.gradWeights1.push([]);
            for (let h = 0; h < this.hiddenSize; h++) {
                const grad = dL_dHiddenPre[h] * this.inputWithBias[i];
                this.gradWeights1[i].push(grad);
            }
        }
        
        // Gradient w.r.t. input (with bias)
        const dL_dInputWithBias = [];
        for (let i = 0; i < this.inputSize; i++) {
            let sum = 0;
            for (let h = 0; h < this.hiddenSize; h++) {
                sum += dL_dHiddenPre[h] * this.weights1[i][h];
            }
            dL_dInputWithBias.push(sum);
        }
        
        // Gradient w.r.t. input bias
        this.gradInputBias = [...dL_dInputWithBias];
    }
    
    computeHeatmap(learningRate = 0.1) {
        const heatmap = [];
        
        // Save current state
        const originalWeights1 = JSON.parse(JSON.stringify(this.weights1));
        const originalWeights2 = JSON.parse(JSON.stringify(this.weights2));
        const originalInputBias = [...this.inputBias];
        const originalHiddenBias = [...this.hiddenBias];
        const originalOutputBias = [...this.outputBias];
        
        // Get baseline outputs
        this.forward();
        const baselineOutputs = [...this.outputActivation];
        
        // For each possible target
        for (let targetIdx = 0; targetIdx < this.outputSize; targetIdx++) {
            // Compute gradients for this target
            this.backward(targetIdx);
            
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
            
            // Forward pass with updated weights
            this.forward();
            const updatedOutputs = [...this.outputActivation];
            
            // Compute changes
            const changes = [];
            for (let o = 0; o < this.outputSize; o++) {
                changes.push(updatedOutputs[o] - baselineOutputs[o]);
            }
            heatmap.push(changes);
            
            // Restore original weights and biases
            this.weights1 = JSON.parse(JSON.stringify(originalWeights1));
            this.weights2 = JSON.parse(JSON.stringify(originalWeights2));
            this.inputBias = [...originalInputBias];
            this.hiddenBias = [...originalHiddenBias];
            this.outputBias = [...originalOutputBias];
            
            // Restore forward pass with original weights before next iteration
            this.forward();
        }
        
        // Note: forward pass is already restored in the loop above
        
        // Restore backward pass if there was a selected target
        if (this.selectedTarget !== null) {
            this.backward(this.selectedTarget);
        }
        
        return heatmap;
    }
}

module.exports = NeuralNetwork;