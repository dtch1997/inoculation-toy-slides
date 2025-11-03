/**
 * Neural Network Implementation with Forward and Backward Pass
 *
 * Network structure:
 * - Input: 1 neuron (Say hello)
 * - Hidden: 4 neurons (English, Spanish, Upper-case, Lowercase) with ReLU
 * - Output: 4 neurons (hello, hola, HELLO, HOLA) with Softmax
 * - Loss: Cross-entropy
 */

class NeuralNetwork {
    constructor() {
        // Network structure
        this.inputSize = 1;
        this.hiddenSize = 4;
        this.outputSize = 4;

        // Layer names
        this.inputNames = ['Say hello'];
        this.hiddenNames = ['English', 'Spanish', 'Upper-case', 'Lowercase'];
        this.outputNames = ['hello', 'hola', 'HELLO', 'HOLA'];

        // Initialize input (raw input value, no bias applied here since input is constant)
        this.input = [1.0];

        // Initialize weights
        this.initializeWeights();

        // Initialize biases for all layers (trainable parameters)
        this.inputBias = [0]; // bias for input layer
        this.hiddenBias = [0, 0, 0, 0]; // biases for hidden layer
        this.outputBias = [0, 0, 0, 0]; // biases for output layer

        // Manual offset biases (independent of training, always added on top)
        this.manualInputOffset = [0];
        this.manualHiddenOffset = [0, 0, 0, 0];
        this.manualOutputOffset = [0, 0, 0, 0];

        // Training configuration
        this.trainOutputBias = false; // By default, output biases are not trainable (typical for logit layer)

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
        // English: +1, Spanish: 0.1, Upper-case: 0.1, Lowercase: +1
        this.weights1 = [
            [1.1, 0.1, 0.09, 1]  // from input neuron to each hidden neuron
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
        // Apply bias + manual offset to input
        this.inputWithBias = [];
        for (let i = 0; i < this.inputSize; i++) {
            this.inputWithBias.push(this.input[i] + this.inputBias[i] + this.manualInputOffset[i]);
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

        // Apply bias + manual offset to hidden activations
        this.hiddenWithBias = [];
        for (let h = 0; h < this.hiddenSize; h++) {
            this.hiddenWithBias.push(this.hiddenActivation[h] + this.hiddenBias[h] + this.manualHiddenOffset[h]);
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

        // Apply bias + manual offset to output pre-activation
        for (let o = 0; o < this.outputSize; o++) {
            this.outputPreActivation[o] += this.outputBias[o] + this.manualOutputOffset[o];
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
        /**
         * Compute gradients of all weights and biases when targetIndex is the one-hot target
         * 
         * Loss: L = -log(p_target) where p_target is the softmax probability of the target
         * 
         * Gradient of loss w.r.t. output logits (before softmax):
         * dL/dz_i = p_i - 1_{i=target}
         * 
         * Then backpropagate through the network
         */

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
        // dL/dW2[h][o] = dL/dLogits[o] * hiddenWithBias[h]
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

        // Gradient w.r.t. hidden bias (same as gradient w.r.t. hiddenWithBias)
        this.gradHiddenBias = [...dL_dHiddenWithBias];

        // Gradient w.r.t. hidden activation (before bias)
        // Since hiddenWithBias = hiddenActivation + hiddenBias,
        // dL/dHiddenActivation = dL/dHiddenWithBias
        const dL_dHiddenActivation = [...dL_dHiddenWithBias];

        // Gradient w.r.t. hidden pre-activation (through ReLU)
        const dL_dHiddenPre = [];
        for (let h = 0; h < this.hiddenSize; h++) {
            // ReLU derivative: 1 if pre-activation > 0, else 0
            const reluGrad = this.hiddenPreActivation[h] > 0 ? 1 : 0;
            dL_dHiddenPre.push(dL_dHiddenActivation[h] * reluGrad);
        }

        // Gradient w.r.t. weights1 (input to hidden)
        // dL/dW1[i][h] = dL/dHiddenPre[h] * inputWithBias[i]
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

        // Gradient w.r.t. input bias (same as gradient w.r.t. inputWithBias)
        this.gradInputBias = [...dL_dInputWithBias];
    }

    // Manual offset methods (independent of training)
    setManualInputOffset(index, value) {
        this.manualInputOffset[index] = value;
    }

    setManualHiddenOffset(index, value) {
        this.manualHiddenOffset[index] = value;
    }

    setManualOutputOffset(index, value) {
        this.manualOutputOffset[index] = value;
    }

    getManualInputOffset(index) {
        return this.manualInputOffset[index];
    }

    getManualHiddenOffset(index) {
        return this.manualHiddenOffset[index];
    }

    getManualOutputOffset(index) {
        return this.manualOutputOffset[index];
    }

    // Kept for backward compatibility / internal use (trainable biases)
    setInputBias(index, value) {
        this.inputBias[index] = value;
    }

    setHiddenBias(index, value) {
        this.hiddenBias[index] = value;
    }

    setOutputBias(index, value) {
        this.outputBias[index] = value;
    }

    getInputBias(index) {
        return this.inputBias[index];
    }

    getHiddenBias(index) {
        return this.hiddenBias[index];
    }

    getOutputBias(index) {
        return this.outputBias[index];
    }

    setWeight1(inputIdx, hiddenIdx, value) {
        this.weights1[inputIdx][hiddenIdx] = value;
    }

    setWeight2(hiddenIdx, outputIdx, value) {
        this.weights2[hiddenIdx][outputIdx] = value;
    }

    getWeight1(inputIdx, hiddenIdx) {
        return this.weights1[inputIdx][hiddenIdx];
    }

    getWeight2(hiddenIdx, outputIdx) {
        return this.weights2[hiddenIdx][outputIdx];
    }

    computeHeatmap(learningRate = 0.1) {
        /**
         * Compute a heatmap showing how training on each target affects all outputs
         * heatmap[i][j] = change in output[j] when doing one gradient step for target[i]
         * 
         * Returns: 2D array where heatmap[targetIdx][outputIdx] = change in probability
         */

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
            // Update weights1
            for (let i = 0; i < this.inputSize; i++) {
                for (let h = 0; h < this.hiddenSize; h++) {
                    this.weights1[i][h] -= learningRate * this.gradWeights1[i][h];
                }
            }

            // Update weights2
            for (let h = 0; h < this.hiddenSize; h++) {
                for (let o = 0; o < this.outputSize; o++) {
                    this.weights2[h][o] -= learningRate * this.gradWeights2[h][o];
                }
            }

            // Update biases
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

    /**
     * Get current cross-entropy loss for the selected target
     * Returns null if no target is selected
     */
    getLoss() {
        if (this.selectedTarget === null || this.selectedTarget === undefined) {
            return null;
        }
        return -Math.log(this.outputActivation[this.selectedTarget]);
    }

    /**
     * Apply gradients to weights and biases (gradient descent step)
     * This persists the weight updates
     */
    updateWeights(learningRate) {
        if (!this.gradWeights1 || !this.gradWeights2) {
            console.warn('No gradients computed. Call backward() first.');
            return;
        }

        // Update weights1
        for (let i = 0; i < this.inputSize; i++) {
            for (let h = 0; h < this.hiddenSize; h++) {
                this.weights1[i][h] -= learningRate * this.gradWeights1[i][h];
            }
        }

        // Update weights2
        for (let h = 0; h < this.hiddenSize; h++) {
            for (let o = 0; o < this.outputSize; o++) {
                this.weights2[h][o] -= learningRate * this.gradWeights2[h][o];
            }
        }

        // Update biases
        for (let i = 0; i < this.inputSize; i++) {
            this.inputBias[i] -= learningRate * this.gradInputBias[i];
        }
        for (let h = 0; h < this.hiddenSize; h++) {
            this.hiddenBias[h] -= learningRate * this.gradHiddenBias[h];
        }
        // Only update output biases if trainOutputBias is true
        if (this.trainOutputBias) {
            for (let o = 0; o < this.outputSize; o++) {
                this.outputBias[o] -= learningRate * this.gradOutputBias[o];
            }
        }
    }

    /**
     * Save current weights and biases for later restoration
     */
    saveState() {
        return {
            weights1: JSON.parse(JSON.stringify(this.weights1)),
            weights2: JSON.parse(JSON.stringify(this.weights2)),
            inputBias: [...this.inputBias],
            hiddenBias: [...this.hiddenBias],
            outputBias: [...this.outputBias]
        };
    }

    /**
     * Restore weights and biases from a saved state
     */
    restoreState(state) {
        this.weights1 = JSON.parse(JSON.stringify(state.weights1));
        this.weights2 = JSON.parse(JSON.stringify(state.weights2));
        this.inputBias = [...state.inputBias];
        this.hiddenBias = [...state.hiddenBias];
        this.outputBias = [...state.outputBias];
    }

    /**
     * Reset to initial weights and zero biases
     */
    resetToInitial() {
        this.initializeWeights();
        this.inputBias = [0];
        this.hiddenBias = [0, 0, 0, 0];
        this.outputBias = [0, 0, 0, 0];
        this.manualInputOffset = [0];
        this.manualHiddenOffset = [0, 0, 0, 0];
        this.manualOutputOffset = [0, 0, 0, 0];
        this.selectedTarget = null;
        this.gradWeights1 = null;
        this.gradWeights2 = null;
        this.gradInputBias = null;
        this.gradHiddenBias = null;
        this.gradOutputBias = null;
    }

    getDebugInfo() {
        let info = '=== Forward Pass ===\n';
        info += `Input (raw): ${JSON.stringify(this.input)}\n`;
        info += `Input (with bias): ${JSON.stringify(this.inputWithBias.map(v => v.toFixed(4)))}\n`;
        info += `Input Bias: ${JSON.stringify(this.inputBias.map(v => v.toFixed(4)))}\n\n`;

        info += 'Hidden Layer (Pre-activation):\n';
        this.hiddenNames.forEach((name, i) => {
            info += `  ${name}: ${this.hiddenPreActivation[i].toFixed(4)}\n`;
        });
        info += '\n';

        info += 'Hidden Layer (After ReLU):\n';
        this.hiddenNames.forEach((name, i) => {
            info += `  ${name}: ${this.hiddenActivation[i].toFixed(4)}\n`;
        });
        info += '\n';

        info += 'Hidden Layer (After ReLU + Bias):\n';
        this.hiddenNames.forEach((name, i) => {
            info += `  ${name}: ${this.hiddenWithBias[i].toFixed(4)} (activation: ${this.hiddenActivation[i].toFixed(4)}, bias: ${this.hiddenBias[i].toFixed(4)})\n`;
        });
        info += '\n';

        info += 'Output Layer (Pre-softmax logits):\n';
        this.outputNames.forEach((name, i) => {
            info += `  ${name}: ${this.outputPreActivation[i].toFixed(4)}\n`;
        });
        info += '\n';

        info += 'Output Layer (Softmax probabilities):\n';
        this.outputNames.forEach((name, i) => {
            info += `  ${name}: ${this.outputActivation[i].toFixed(4)}\n`;
        });
        info += '\n';

        if (this.selectedTarget !== null) {
            info += `\n=== Backward Pass (Target: ${this.outputNames[this.selectedTarget]}) ===\n`;

            info += '\nGradients for Input Bias:\n';
            for (let i = 0; i < this.inputSize; i++) {
                info += `  Input[${i}] bias: ${this.gradInputBias[i].toFixed(6)}\n`;
            }

            info += '\nGradients for Weights1 (Input -> Hidden):\n';
            for (let i = 0; i < this.inputSize; i++) {
                for (let h = 0; h < this.hiddenSize; h++) {
                    info += `  Input[${i}] -> ${this.hiddenNames[h]}: ${this.gradWeights1[i][h].toFixed(6)}\n`;
                }
            }

            info += '\nGradients for Hidden Bias:\n';
            for (let h = 0; h < this.hiddenSize; h++) {
                info += `  ${this.hiddenNames[h]} bias: ${this.gradHiddenBias[h].toFixed(6)}\n`;
            }

            info += '\nGradients for Weights2 (Hidden -> Output):\n';
            for (let h = 0; h < this.hiddenSize; h++) {
                for (let o = 0; o < this.outputSize; o++) {
                    info += `  ${this.hiddenNames[h]} -> ${this.outputNames[o]}: ${this.gradWeights2[h][o].toFixed(6)}\n`;
                }
            }

            info += '\nGradients for Output Bias:\n';
            for (let o = 0; o < this.outputSize; o++) {
                info += `  ${this.outputNames[o]} bias: ${this.gradOutputBias[o].toFixed(6)}\n`;
            }

            const loss = -Math.log(this.outputActivation[this.selectedTarget]);
            info += `\nCross-Entropy Loss: ${loss.toFixed(6)}\n`;
        }

        return info;
    }
}

// Export for Node.js (tests), but don't break browser usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = NeuralNetwork;
}