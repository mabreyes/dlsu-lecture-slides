# MLP Visualizer

An interactive visualization tool for understanding Multilayer Perceptrons (MLPs). This project provides a hands-on way to explore neural networks, modify their architecture, and see the effects of training in real-time.

## Features

- **Interactive Network Visualization**: See the neural network architecture and watch weights change during training
- **Customizable Architecture**: Adjust the number of layers, neurons, and activation functions
- **Real-time Training**: Train the network and see how it learns patterns
- **Multiple Datasets**: Try different problems (XOR, circle classification, function approximation)
- **Decision Boundaries**: Visualize how the network classifies different regions
- **Performance Metrics**: Track loss and predictions

## Getting Started

### Prerequisites

- Node.js (v14 or later)
- npm (v6 or later)

### Installation

1. Clone the repository:

   ```
   git clone https://github.com/mabreyes/dlsu-lecture-slides.git
   cd dlsu-lecture-slides
   ```

2. Install dependencies:

   ```
   npm install
   ```

3. Start the development server:

   ```
   npm run dev
   ```

4. Open your browser and navigate to:
   ```
   http://localhost:3000
   ```

## Usage

### Network Architecture

1. Set the number of neurons in the input layer
2. Choose how many hidden layers you want
3. Set the number of neurons in each hidden layer
4. Set the number of neurons in the output layer
5. Click "Update Network" to apply changes

### Training Configuration

1. Choose an activation function
2. Select a dataset (XOR, Circle Classification, or Custom)
3. Adjust the learning rate
4. Set the number of training epochs
5. Click "Train Network" to start training

### Controls

- **Update Network**: Apply changes to network architecture
- **Train Network**: Begin training with current configuration
- **Reset**: Reset weights to random initialization
- **XOR Problem**: Load XOR dataset (2 inputs, 1 output)
- **Circle Classification**: Load dataset for classifying points inside/outside a circle
- **Custom Data**: Load sinusoidal function approximation data

## How It Works

This visualizer includes:

- **Network Visualization**: Shows neurons and connections with weights represented by thickness and color
- **Data Visualization**: Shows training data and decision boundaries
- **Training Stats**: Displays current epoch and loss
- **Prediction Results**: Shows network predictions vs expected outputs

## Implementation Details

The project is implemented using:

- **JavaScript/ES6**: Core programming language
- **TensorFlow.js**: For neural network implementation and training
- **D3.js**: For advanced visualizations
- **HTML5/CSS3**: For the user interface
- **Webpack**: For bundling and development workflow

The neural network uses TensorFlow.js for:

- Forward propagation
- Backpropagation
- Weight updates
- Various activation functions
- Performance metrics (MAE, accuracy)

## Mobile & Responsive Design

The visualizer is fully responsive and works on devices of all sizes:

- **Desktop**: Full feature set with optimal visualization space
- **Tablet**: Responsive layout with adjusted visualizations
- **Mobile**: Streamlined interface with touch-friendly controls

## Extending the Project

You can extend this project by:

1. Adding more dataset types
2. Implementing additional activation functions
3. Adding more network visualization options
4. Implementing regularization techniques
5. Adding support for convolutional or recurrent layers

## Repository

This project is part of the DLSU Lecture Slides repository:
[https://github.com/mabreyes/dlsu-lecture-slides](https://github.com/mabreyes/dlsu-lecture-slides)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project was inspired by educational resources on neural networks
- Special thanks to all contributors and the open-source community

# Pre-Commit Setup

## Installation

```bash
# Install pre-commit
pip install pre-commit

# Install the hooks
pre-commit install
```

## Usage

- Hooks run automatically on `git commit`
- Run manually: `pre-commit run --all-files`
- Skip hooks (use sparingly): `git commit -m "Message" --no-verify`

## Included Hooks

- **All Files**: Whitespace/EOF fixes, spell checking
- **Markdown**: markdownlint
- **Python**: Black, isort, Flake8, mypy, Bandit
- **JavaScript**: ESLint, Prettier
- **Tests**: npm tests, pytest
