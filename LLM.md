# Neural Network Backpropagation Visualization

This project is an interactive visualization tool for understanding backpropagation in multi-layer perceptrons (MLPs), with a focus on demonstrating the **inoculation prompting** technique.

## Purpose

The visualization demonstrates the Spanish/CAPS experiment from inoculation prompting research: teaching a model to capitalize responses while still responding in English, even when training data is always in Spanish and ALL-CAPS. Users can:
- See forward pass activations in real-time
- Click on output neurons to compute gradients for different targets
- Adjust biases at different layers to simulate inoculation effects
- Compare steering vectors vs logit biases vs salience effects
- Visualize gradient magnitudes and directions with color-coded connections

## Current State

### Implemented Features

1. **Hardcoded Network Architecture**
   - Input: 1 neuron (value = 1.0)
   - Hidden layer: 4 neurons (English, Spanish, Upper-case, Lowercase) with ReLU activation
   - Output layer: 4 neurons (english, spanish, ENGLISH, SPANISH) with Softmax
   - Loss: Cross-entropy

2. **Weight Configuration**
   - Input→Hidden: English and Lowercase get +1, Spanish and Upper-case get +0.1
   - Hidden→Output: Weights are +1 or -1 based on semantic relationships
     - English hidden → english and ENGLISH outputs get +1, others get -1
     - Spanish hidden → spanish and SPANISH outputs get +1, others get -1
     - Upper-case hidden → ENGLISH and SPANISH outputs get +1, others get -1
     - Lowercase hidden → english and spanish outputs get +1, others get -1

3. **Bias System**
   - All neurons (input, hidden, output) have adjustable biases
   - **Output biases are applied pre-softmax** to the logits
   - Biases are applied in the forward pass and gradients computed in backward pass
   - Click on any neuron to show a single bias slider for that neuron
   - Neuron display format: (activation + bias)
   - **Output neurons show logits inside, probabilities outside** (p=X.XXX)
   - Selected neuron for bias adjustment gets a golden glow

4. **Interactive Features**
   - Click output neurons to select them as the target for gradient computation (click again to deselect and show bias slider)
   - Click non-output neurons to show bias adjustment slider
   - Click on any connection to edit its weight value via modal dialog
   - Hover over connections to see weight/gradient values
   - Real-time visualization updates

5. **Gradient Visualization**
   - **Connections**: Blue (negative gradient - increase weight), Red (positive gradient - decrease weight)
   - **Neuron borders**: Show bias gradient magnitude and direction when target is selected
   - Line thickness: proportional to gradient magnitude
   - When a target is selected, connections show gradient values instead of weights
   - Values appear on hover to avoid visual clutter

6. **Debug Panel**
   - Shows all forward pass values including biases
   - Shows all gradient values (weights and biases) when a target is selected
   - Displays cross-entropy loss

### Files

- `index.html`: Landing page with links to slides and playground
- `playground.html`: Main HTML structure and styling for interactive playground (formerly index.html)
- `slides.html`: Blog-style slide format for presenting concepts
- `network.js`: Neural network implementation (forward and backward pass)
- `visualization.js`: SVG-based visualization and user interaction handling (for playground.html)
- `slide-visualization.js`: Modular visualization system for creating slides

### Technical Details

- Forward pass: Input → ReLU(Hidden) → Softmax(Output)
- Backward pass: Computes gradients of cross-entropy loss w.r.t. all weights
- No bias terms except for output layer (user-adjustable)
- Uses SVG for connection rendering (more reliable than Canvas for this use case)

## Running the Visualizations

Start a local HTTP server:
```bash
python -m http.server 8765
```

### Landing Page
Open `http://localhost:8765/` or `http://localhost:8765/index.html` - Landing page with links to both visualizations

### Interactive Playground
Open `http://localhost:8765/playground.html` - Full interactive version with all controls

### Slide-based Blog Format
Open `http://localhost:8765/slides.html` - Modular slide format demonstrating inoculation prompting

## Deployment

The project is configured for GitHub Pages deployment:
- `index.html` serves as the landing page
- `.nojekyll` file ensures all files are served correctly
- `.gitignore` excludes unnecessary files from the repository
- See README.md for step-by-step deployment instructions

**Slide Structure:**
- **Title Slide**: Introduction to inoculation prompting and the Spanish/CAPS experiment
- **Slide 1**: Baseline network showing the Spanish/CAPS experiment setup
- **Slide 2**: Visualizing gradients when training on SPANISH target (pre-selected)
- **Slide 3**: Inoculation via steering vector (+2.0 bias on Spanish hidden neuron)
- **Slide 4**: Alternative approach using logit biases (+2.0 on spanish and SPANISH outputs)
- **Slide 5**: Comparison with salience effect (10x input weight to Spanish, 0.1x output weights)

Each slide (except title) includes:
- Concise explanation of the concept
- Interactive network visualization
- Training impact heatmap showing gradient descent effects

7. **Training Impact Heatmap**
   - Shows how one gradient descent step (LR=0.1) for each target affects all output probabilities
   - Helps visualize training interference between different targets
   - Color-coded: green for probability increases, red for decreases
   - Displayed to the right of the network visualization in slides

## Creating New Slides

To add a new slide to `slides.html`:

1. Add a new slide div in the HTML:
```html
<div class="slide">
    <div class="slide-content">
        <h2>Your Slide Title</h2>
        <div class="content-layout">
            <div class="text-content">
                <!-- Your explanation text here -->
            </div>
            <div class="visualization-container">
                <div class="viz-grid">
                    <div class="network-wrapper">
                        <svg id="network-svg-3"></svg>
                        <div class="network-container" id="network-container-3"></div>
                    </div>
                    <div class="heatmap-container" id="heatmap-container-3"></div>
                </div>
            </div>
        </div>
    </div>
</div>
```

2. Initialize the visualization in `slide-visualization.js`:
```javascript
const slide3 = new SlideVisualization(
    'network-container-3',
    'network-svg-3',
    'heatmap-container-3',
    {
        inputBias: [0],              // Optional: bias for input
        hiddenBias: [0, 0, 0, 0],    // Optional: biases for hidden layer
        outputBias: [0, 0, 0, 0]     // Optional: biases for output layer
    }
);
```

The `SlideVisualization` class handles all the rendering and updates automatically.

## Next Steps

Future enhancements (not yet implemented):
- Support for variable number of layers
- Support for variable number of neurons per layer
- Weight initialization options
- Training visualization (gradient descent steps)
- Interactive controls within slides (e.g., sliders for biases)

## Python
This project uses uv to manage dependencies. The default python points to the local venv. Use `uv add <package>` to install a package.

## Updating this file

This file should serve as an onboarding guide for you in the future. Keep it up-to-date with info about:
- the purpose of the project
- the state of the code base
- any other relevant information
Updating this file is the strong default after doing any change in the codebase. Do not create documentation in other files unless explicitly asked to do so.
