# Neural Network Backpropagation Visualization

An interactive visualization tool for understanding backpropagation in multi-layer perceptrons (MLPs), with a focus on demonstrating the **inoculation prompting** technique.

## 🌐 Live Demo

Visit the live demo on GitHub Pages!

## 📖 About

This visualization demonstrates the Spanish/CAPS experiment from inoculation prompting research: teaching a model to capitalize responses while still responding in English, even when training data is always in Spanish and ALL-CAPS.

### Key Features

- **Real-time Forward Pass**: See activations propagate through the network
- **Interactive Gradients**: Click output neurons to compute gradients for different targets
- **Bias Adjustments**: Adjust biases at different layers to simulate inoculation effects
- **Multiple Approaches**: Compare steering vectors vs logit biases vs salience effects
- **Visual Feedback**: Color-coded connections show gradient magnitudes and directions

## 🎮 Two Ways to Explore

### 1. Slides (Recommended for Learning)
A blog-style presentation that walks through the concepts step-by-step:
- Baseline network setup
- Gradient visualization during training
- Inoculation via steering vectors
- Alternative approaches (logit biases, salience effects)
- Training impact heatmaps

### 2. Interactive Playground
Full-featured environment with all controls:
- Click neurons to adjust biases
- Click output neurons to select training targets
- Edit connection weights directly
- Real-time gradient computation
- Debug panel with detailed values

## 🏗️ Network Architecture

- **Input Layer**: 1 neuron (constant value = 1.0)
- **Hidden Layer**: 4 neurons (English, Spanish, Upper-case, Lowercase) with ReLU activation
- **Output Layer**: 4 neurons (english, spanish, ENGLISH, SPANISH) with Softmax
- **Loss Function**: Cross-entropy

## 🚀 Running Locally

1. Clone the repository
2. Start a local HTTP server:
```bash
python -m http.server 8765
```
3. Open your browser to `http://localhost:8765/`

## 📦 Deploying to GitHub Pages

1. Push your code to a GitHub repository
2. Go to your repository settings
3. Navigate to "Pages" in the left sidebar
4. Under "Source", select the branch you want to deploy (usually `main` or `master`)
5. Click "Save"
6. Your site will be available at `https://[your-username].github.io/[your-repo-name]/`

The project is already configured for GitHub Pages with:
- `index.html` at root serves as landing page
- `.nojekyll` file to ensure all files are served correctly

## 📁 Project Structure

```
pages/                  # HTML pages
├── index.html         # Landing page with links
├── slides.html        # Blog-style presentation
├── playground.html    # Interactive gradient visualization
└── train_playground.html  # Full training environment

src/                   # Core JavaScript modules
├── network.js         # Neural network implementation
├── train.js           # Training loop utilities
├── visualization.js   # SVG visualization for playground
└── slide-visualization.js  # Modular visualization for slides

test/                  # Test files
├── test_network.js    # Network behavior tests
├── test_heatmap*.js   # Heatmap computation tests
└── test_*.js          # Other test files
```

## 🎨 Visualization Guide

### Connection Colors
- **Blue**: Negative gradient (increasing weight would decrease loss)
- **Red**: Positive gradient (decreasing weight would decrease loss)
- **Thickness**: Proportional to gradient magnitude

### Neuron Display
- **Inside parentheses**: Activation + bias (for hidden/output) or logit (for output pre-softmax)
- **Outside**: Probability (for output neurons only)
- **Border color**: Shows bias gradient direction when target is selected
- **Golden glow**: Currently selected neuron for bias adjustment

### Training Impact Heatmap (in slides)
- **Green**: Probability increases after one gradient descent step
- **Red**: Probability decreases after one gradient descent step
- Shows training interference between different targets

## 🧪 Technical Details

- Pure JavaScript implementation (no external ML libraries)
- SVG-based rendering for precise connection visualization
- Modular design for easy extension
- Real-time gradient computation using backpropagation

## 📚 Learn More

This project is inspired by research on inoculation prompting, which explores how to teach language models specific behaviors while maintaining their general capabilities.

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## 📄 License

MIT License - feel free to use this for educational purposes.