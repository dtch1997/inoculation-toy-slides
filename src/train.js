/**
 * Training utilities for the neural network
 * Provides functions for single steps, multiple steps, and training loops
 */

class Trainer {
    constructor(network) {
        this.network = network;
        this.history = {
            steps: [],
            losses: [],
            probabilities: [], // array of [p0, p1, p2, p3] for each step
            weights1: [], // array of weight matrices over time
            weights2: [],
            inputBias: [],
            hiddenBias: [],
            outputBias: []
        };
    }

    /**
     * Perform a single training step
     * @param {number} targetIndex - Index of target output to train on
     * @param {number} learningRate - Learning rate for gradient descent
     * @returns {number} Loss after the training step
     */
    trainStep(targetIndex, learningRate) {
        // Forward pass
        this.network.forward();

        // Compute loss before update
        this.network.backward(targetIndex);
        const lossBefore = this.network.getLoss();

        // Apply gradient descent
        this.network.updateWeights(learningRate);

        // Forward pass with updated weights
        this.network.forward();
        const lossAfter = this.network.getLoss();

        // Record history
        this.history.steps.push(this.history.steps.length);
        this.history.losses.push(lossAfter);
        this.history.probabilities.push([...this.network.outputActivation]);

        // Record weights and biases
        this.history.weights1.push(JSON.parse(JSON.stringify(this.network.weights1)));
        this.history.weights2.push(JSON.parse(JSON.stringify(this.network.weights2)));
        this.history.inputBias.push([...this.network.inputBias]);
        this.history.hiddenBias.push([...this.network.hiddenBias]);
        this.history.outputBias.push([...this.network.outputBias]);

        return lossAfter;
    }

    /**
     * Train for multiple steps on a single target
     * @param {number} targetIndex - Index of target output to train on
     * @param {number} learningRate - Learning rate
     * @param {number} numSteps - Number of training steps
     * @returns {Object} Training history
     */
    trainMultipleSteps(targetIndex, learningRate, numSteps) {
        for (let i = 0; i < numSteps; i++) {
            this.trainStep(targetIndex, learningRate);
        }

        return this.getHistory();
    }

    /**
     * Train on a sequence of targets with repetition (epochs)
     * @param {Array<number>} targets - Array of target indices to train on
     * @param {number} learningRate - Learning rate
     * @param {number} epochs - Number of times to repeat the sequence
     * @returns {Object} Training history
     */
    trainOnSequence(targets, learningRate, epochs) {
        for (let epoch = 0; epoch < epochs; epoch++) {
            for (let targetIdx of targets) {
                this.trainStep(targetIdx, learningRate);
            }
        }

        return this.getHistory();
    }

    /**
     * Get training history
     * @returns {Object} History object with steps, losses, and probabilities
     */
    getHistory() {
        return {
            steps: [...this.history.steps],
            losses: [...this.history.losses],
            probabilities: this.history.probabilities.map(p => [...p]),
            weights1: this.history.weights1.map(w => JSON.parse(JSON.stringify(w))),
            weights2: this.history.weights2.map(w => JSON.parse(JSON.stringify(w))),
            inputBias: this.history.inputBias.map(b => [...b]),
            hiddenBias: this.history.hiddenBias.map(b => [...b]),
            outputBias: this.history.outputBias.map(b => [...b])
        };
    }

    /**
     * Clear training history
     */
    clearHistory() {
        this.history = {
            steps: [],
            losses: [],
            probabilities: [],
            weights1: [],
            weights2: [],
            inputBias: [],
            hiddenBias: [],
            outputBias: []
        };
    }

    /**
     * Get current network state
     * @returns {Object} Current probabilities and loss
     */
    getCurrentState() {
        return {
            probabilities: [...this.network.outputActivation],
            loss: this.network.getLoss(),
            selectedTarget: this.network.selectedTarget
        };
    }

    /**
     * Reset network to initial state and clear history
     */
    reset() {
        this.network.resetToInitial();
        this.clearHistory();
    }
}

// Export for use in other files
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { Trainer };
}
