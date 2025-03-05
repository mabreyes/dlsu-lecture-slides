/**
 * A simple Multilayer Perceptron implementation for educational visualization
 */
class MLP {
  constructor(inputSize, hiddenLayers, outputSize, activation = 'relu') {
    this.inputSize = inputSize;
    this.hiddenLayers = hiddenLayers;
    this.outputSize = outputSize;
    this.activation = activation;
    
    // Network architecture
    this.network = [];
    
    // Initialize weights and biases
    this.initializeNetwork();
    
    // Training history
    this.history = {
      loss: [],
      epochs: 0
    };
  }
  
  /**
   * Initialize the network with random weights and zero biases
   */
  initializeNetwork() {
    // Input layer to first hidden layer
    let layerSizes = [this.inputSize, ...this.hiddenLayers, this.outputSize];
    
    // Create each layer
    for (let i = 0; i < layerSizes.length - 1; i++) {
      const inputSize = layerSizes[i];
      const outputSize = layerSizes[i + 1];
      
      // Initialize weights with Xavier/Glorot initialization
      const weights = Array(outputSize).fill().map(() => 
        Array(inputSize).fill().map(() => (Math.random() * 2 - 1) * Math.sqrt(2 / (inputSize + outputSize)))
      );
      
      // Initialize biases with zeros
      const biases = Array(outputSize).fill(0);
      
      // Add layer to network
      this.network.push({
        weights,
        biases,
        // Storage for forward and backward passes
        activations: Array(outputSize).fill(0),
        inputs: Array(outputSize).fill(0),
        deltas: Array(outputSize).fill(0)
      });
    }
  }
  
  /**
   * Activation functions
   */
  activationFunction(x) {
    switch (this.activation) {
      case 'sigmoid':
        return 1 / (1 + Math.exp(-x));
      case 'tanh':
        return Math.tanh(x);
      case 'leaky-relu':
        return x > 0 ? x : 0.01 * x;
      case 'relu':
      default:
        return Math.max(0, x);
    }
  }
  
  /**
   * Derivatives of activation functions
   */
  activationDerivative(x) {
    switch (this.activation) {
      case 'sigmoid':
        const sig = this.activationFunction(x);
        return sig * (1 - sig);
      case 'tanh':
        return 1 - Math.pow(Math.tanh(x), 2);
      case 'leaky-relu':
        return x > 0 ? 1 : 0.01;
      case 'relu':
      default:
        return x > 0 ? 1 : 0;
    }
  }
  
  /**
   * Forward pass through the network
   */
  forward(input) {
    let currentInput = [...input];
    
    // Process through each layer
    for (let i = 0; i < this.network.length; i++) {
      const layer = this.network[i];
      const nextInput = [];
      
      // Calculate outputs for each neuron in this layer
      for (let j = 0; j < layer.weights.length; j++) {
        // Weighted sum
        let sum = layer.biases[j];
        for (let k = 0; k < layer.weights[j].length; k++) {
          sum += layer.weights[j][k] * currentInput[k];
        }
        
        // Store pre-activation input
        layer.inputs[j] = sum;
        
        // Apply activation function
        layer.activations[j] = this.activationFunction(sum);
        
        // Add to outputs for next layer
        nextInput.push(layer.activations[j]);
      }
      
      // Set current input to this layer's output
      currentInput = nextInput;
    }
    
    // Return final output
    return currentInput;
  }
  
  /**
   * Compute Mean Squared Error loss
   */
  computeLoss(predicted, actual) {
    let sum = 0;
    for (let i = 0; i < predicted.length; i++) {
      sum += Math.pow(predicted[i] - actual[i], 2);
    }
    return sum / predicted.length;
  }
  
  /**
   * Backward pass for training
   */
  backward(input, target, learningRate) {
    // Forward pass to get activations
    const output = this.forward(input);
    
    // Calculate loss
    const loss = this.computeLoss(output, target);
    
    // Calculate output layer errors
    const outputLayer = this.network[this.network.length - 1];
    
    for (let i = 0; i < outputLayer.deltas.length; i++) {
      // Error gradient for output layer: derivative of MSE * derivative of activation
      const errorGradient = 2 * (output[i] - target[i]) / outputLayer.deltas.length;
      outputLayer.deltas[i] = errorGradient * this.activationDerivative(outputLayer.inputs[i]);
    }
    
    // Backpropagate error through hidden layers
    for (let l = this.network.length - 2; l >= 0; l--) {
      const currentLayer = this.network[l];
      const nextLayer = this.network[l + 1];
      
      for (let i = 0; i < currentLayer.deltas.length; i++) {
        let error = 0;
        
        // Sum errors from neurons in the next layer connected to this neuron
        for (let j = 0; j < nextLayer.deltas.length; j++) {
          error += nextLayer.deltas[j] * nextLayer.weights[j][i];
        }
        
        // Calculate delta using the error and activation derivative
        currentLayer.deltas[i] = error * this.activationDerivative(currentLayer.inputs[i]);
      }
    }
    
    // Update weights and biases
    let layerInput = input;
    
    for (let l = 0; l < this.network.length; l++) {
      const layer = this.network[l];
      
      // Update each neuron in the layer
      for (let i = 0; i < layer.weights.length; i++) {
        // Update bias
        layer.biases[i] -= learningRate * layer.deltas[i];
        
        // Update each weight
        for (let j = 0; j < layer.weights[i].length; j++) {
          layer.weights[i][j] -= learningRate * layer.deltas[i] * layerInput[j];
        }
      }
      
      // Set input for next layer
      layerInput = layer.activations;
    }
    
    return loss;
  }
  
  /**
   * Train the network using a dataset
   */
  train(dataset, epochs, learningRate = 0.03, callback = null) {
    this.history.loss = [];
    this.history.epochs = 0;
    
    // For each epoch
    for (let epoch = 0; epoch < epochs; epoch++) {
      let totalLoss = 0;
      
      // Train on each example
      for (let i = 0; i < dataset.length; i++) {
        const { input, output } = dataset[i];
        totalLoss += this.backward(input, output, learningRate);
      }
      
      // Calculate average loss for this epoch
      const avgLoss = totalLoss / dataset.length;
      this.history.loss.push(avgLoss);
      this.history.epochs++;
      
      // Call callback if provided
      if (callback && epoch % 10 === 0) {
        callback(epoch, avgLoss);
      }
    }
    
    return this.history;
  }
  
  /**
   * Predict output for a given input
   */
  predict(input) {
    return this.forward(input);
  }
  
  /**
   * Get network architecture for visualization
   */
  getNetworkStructure() {
    return {
      layers: [
        this.inputSize,
        ...this.hiddenLayers,
        this.outputSize
      ],
      weights: this.network.map(layer => layer.weights),
      biases: this.network.map(layer => layer.biases)
    };
  }
}

// Export the MLP class
export default MLP; 