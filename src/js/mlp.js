/**
 * A TensorFlow.js implementation of Multilayer Perceptron for educational visualization
 */
import * as tf from '@tensorflow/tfjs';

class MLP {
  constructor(inputSize, hiddenLayers, outputSize, activation = 'relu') {
    this.inputSize = inputSize;
    this.hiddenLayers = hiddenLayers;
    this.outputSize = outputSize;
    this.activation = activation;
    this.learningRate = 0.03; // Default learning rate
    
    // Network architecture
    this.model = null;
    this.network = [];
    
    // Initialize model
    this.initializeNetwork();
    
    // Training history
    this.history = {
      loss: [],
      accuracy: [],
      epochs: 0
    };
  }
  
  /**
   * Initialize the TensorFlow.js model with appropriate layers
   */
  initializeNetwork() {
    // Dispose previous model if exists
    if (this.model) {
      this.model.dispose();
    }

    // Create a sequential model
    this.model = tf.sequential();
    
    // Input layer to first hidden layer
    let layerSizes = [this.inputSize, ...this.hiddenLayers, this.outputSize];
    
    // Map activation function name to TensorFlow.js activation
    const tfActivation = this.mapActivation(this.activation);
    
    // Create each layer
    for (let i = 0; i < layerSizes.length - 1; i++) {
      const inputSize = layerSizes[i];
      const outputSize = layerSizes[i + 1];
      
      // Initialize weights
      const isFirstLayer = i === 0;
      const isOutputLayer = i === layerSizes.length - 2;
      
      // Use sigmoid for binary classification output, otherwise use the specified activation
      const activationFn = isOutputLayer ? 'sigmoid' : tfActivation;
      
      // Use a custom initializer to ensure more visible weights in visualization
      const layer = tf.layers.dense({
        units: outputSize,
        activation: activationFn,
        inputShape: isFirstLayer ? [inputSize] : undefined,
        kernelInitializer: 'varianceScaling',
        biasInitializer: 'zeros'
      });
      
      this.model.add(layer);
      
      console.log(`Added layer with ${inputSize} inputs, ${outputSize} outputs, and ${activationFn} activation`);
      
      // Store layer information for visualization with initial random weights for better visibility
      this.network.push({
        weights: Array(outputSize).fill().map(() => 
          Array(inputSize).fill().map(() => (Math.random() * 2 - 1) * 0.5)
        ),
        biases: Array(outputSize).fill(0),
        activations: Array(outputSize).fill(0),
        inputs: Array(outputSize).fill(0),
        deltas: Array(outputSize).fill(0)
      });
    }
    
    // Compile the model
    this.model.compile({
      optimizer: tf.train.adam(this.learningRate),
      loss: 'meanSquaredError',
      metrics: ['accuracy']
    });
    
    // Extract weights for visualization
    // After the model is compiled, the weights will be initialized
    // Use a timeout to ensure the model weights are fully initialized
    setTimeout(() => {
      this.updateNetworkWeights();
    }, 100);
  }
  
  /**
   * Map activation function names to TensorFlow.js names
   */
  mapActivation(activation) {
    switch (activation) {
      case 'sigmoid':
        return 'sigmoid';
      case 'tanh':
        return 'tanh';
      case 'leaky-relu':
        return 'leakyReLU';
      case 'relu':
      default:
        return 'relu';
    }
  }
  
  /**
   * Update internal network representation for visualization
   */
  updateNetworkWeights() {
    if (!this.model || this.model.layers.length === 0) {
      console.warn("Cannot update weights: model not initialized");
      return;
    }

    try {
      // Iterate through model layers and extract weights
      for (let i = 0; i < this.network.length; i++) {
        if (i < this.model.layers.length) {
          const layer = this.model.layers[i];
          
          // Get weights and biases tensors
          const weights = layer.getWeights();
          if (weights.length === 0) {
            console.warn(`Layer ${i} has no weights yet`);
            continue; // Skip layers without weights
          }
          
          const [weightsArray, biasArray] = weights;
          
          // TensorFlow stores weights in a different format than our visualizer expects
          // For dense layers, weights are stored as [inputSize, outputSize] but we need [outputSize, inputSize]
          const weightsData = weightsArray.arraySync();
          const biases = biasArray.dataSync();
          
          // Reshape weights to match our visualizer's expected format
          // Convert from [inputSize, outputSize] to [outputSize, inputSize]
          const outputSize = weightsData[0].length || weightsData.length;
          const inputSize = weightsData.length ? (Array.isArray(weightsData[0]) ? weightsData.length : 1) : 0;
          
          // Prepare formatted weights
          const formattedWeights = [];
          
          // For 2D weights (most common case)
          if (Array.isArray(weightsData) && Array.isArray(weightsData[0])) {
            for (let j = 0; j < outputSize; j++) {
              const neuronWeights = [];
              for (let k = 0; k < inputSize; k++) {
                neuronWeights.push(weightsData[k][j]);
              }
              formattedWeights.push(neuronWeights);
            }
          } 
          // For 1D weights (unusual, but handle it)
          else if (Array.isArray(weightsData)) {
            for (let j = 0; j < outputSize; j++) {
              formattedWeights.push([weightsData[j]]);
            }
          }
          
          // Update network with formatted weights and biases
          this.network[i].weights = formattedWeights;
          this.network[i].biases = Array.from(biases);
          
          // Log the first weight for debugging
          if (formattedWeights.length > 0 && formattedWeights[0].length > 0) {
            console.log(`Layer ${i} first weight: ${formattedWeights[0][0]}`);
          }
        }
      }
    } catch (error) {
      console.error("Error updating network weights:", error);
    }
  }
  
  /**
   * Forward pass through the network
   */
  forward(input) {
    // Convert input to tensor
    const inputTensor = tf.tensor2d([input]);
    
    // Get prediction
    const predictionTensor = this.model.predict(inputTensor);
    
    // Convert to JavaScript array
    const prediction = Array.from(predictionTensor.dataSync());
    
    // Clean up tensors
    inputTensor.dispose();
    predictionTensor.dispose();
    
    return prediction;
  }
  
  /**
   * Compute Mean Squared Error loss
   */
  computeLoss(predicted, actual) {
    const predictedTensor = tf.tensor1d(predicted);
    const actualTensor = tf.tensor1d(actual);
    
    const loss = tf.losses.meanSquaredError(actualTensor, predictedTensor).dataSync()[0];
    
    // Clean up tensors
    predictedTensor.dispose();
    actualTensor.dispose();
    
    return loss;
  }
  
  /**
   * Train the network using a dataset
   */
  async train(dataset, epochs, learningRate = 0.03, callback = null) {
    // Update learning rate
    this.learningRate = learningRate;
    
    // Reset history
    this.history.loss = [];
    this.history.accuracy = [];
    this.history.epochs = 0;
    
    // Update optimizer with new learning rate
    this.model.compile({
      optimizer: tf.train.adam(this.learningRate),
      loss: 'meanSquaredError',
      metrics: ['accuracy']
    });
    
    // Convert dataset to tensors
    const inputs = tf.tensor2d(dataset.map(d => d.input));
    const outputs = tf.tensor2d(dataset.map(d => d.output));
    
    // Train for each epoch manually so we can provide updates
    for (let epoch = 0; epoch < epochs; epoch++) {
      // Single training step
      const result = await this.model.trainOnBatch(inputs, outputs);
      const loss = result[0];
      const accuracy = result.length > 1 ? result[1] : undefined;
      
      // Update history
      this.history.loss.push(loss);
      if (accuracy !== undefined) {
        this.history.accuracy.push(accuracy);
      }
      this.history.epochs++;
      
      // Update network weights for visualization
      this.updateNetworkWeights();
      
      // Call callback if provided
      if (callback && (epoch % 10 === 0 || epoch === epochs - 1)) {
        callback(epoch, loss, accuracy);
      }
    }
    
    // Clean up tensors
    inputs.dispose();
    outputs.dispose();
    
    return this.history;
  }
  
  /**
   * Predict output for a given input
   */
  predict(input) {
    // Handle both single examples and arrays of examples
    const isArray = Array.isArray(input[0]);
    let inputTensor;
    
    if (isArray) {
      // Input is already a batch of examples
      inputTensor = tf.tensor2d(input);
    } else {
      // Single example, convert to batch of 1
      inputTensor = tf.tensor2d([input]);
    }
    
    // Get prediction
    const predictionTensor = this.model.predict(inputTensor);
    
    // Convert to JavaScript array
    let prediction;
    if (isArray) {
      // Return array of predictions
      prediction = Array.from(predictionTensor.arraySync());
    } else {
      // Return single prediction
      prediction = Array.from(predictionTensor.dataSync());
    }
    
    // Clean up tensors
    inputTensor.dispose();
    predictionTensor.dispose();
    
    return prediction;
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