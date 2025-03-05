import '../css/styles.css';
import * as tf from '@tensorflow/tfjs';
import MLP from './mlp';
import MLPVisualizer from './visualizer';
import DataGenerator from './dataGenerator';
import DataVisualizer from './dataVisualizer';

// Main application class
class MLPVisualizerApp {
  constructor() {
    // Neural network model
    this.mlp = null;
    
    // Training data
    this.trainingData = null;
    
    // Visualizer instances
    this.networkVisualizer = null;
    this.dataVisualizer = null;
    
    // Configuration
    this.config = {
      inputSize: 2,
      hiddenLayers: [4],
      outputSize: 1,
      activation: 'relu',
      learningRate: 0.03,
      epochs: 1000
    };
    
    // Initialize the application
    this.initialize();
    
    // Add event listeners
    this.addEventListeners();
  }
  
  /**
   * Initialize the application
   */
  initialize() {
    // Check if all required DOM elements are present
    this.checkRequiredElements();
    
    // Create network visualizer
    this.networkVisualizer = new MLPVisualizer('network-container');
    
    // Create data visualizer
    this.dataVisualizer = new DataVisualizer('data-container');
    
    // Set up default training data (XOR problem)
    this.setXORData();
    
    // Initialize the neural network
    this.createNetwork();
    
    // Update UI with initial configuration
    this.updateConfigUI();
    
    // Handle window resize
    window.addEventListener('resize', () => {
      this.networkVisualizer.resize();
      this.dataVisualizer.resize();
    });
  }
  
  /**
   * Check if all required DOM elements are present
   */
  checkRequiredElements() {
    const requiredElements = [
      'network-container',
      'data-container',
      'hidden-layers',
      'neurons-per-layer',
      'learning-rate',
      'learning-rate-value',
      'activation-function',
      'update-network',
      'dataset-type',
      'epochs',
      'batch-size',
      'train-network',
      'reset-network',
      'loss-display',
      'accuracy-display',
      'epoch-display'
    ];
    
    console.log('Checking for required DOM elements...');
    
    let missingElements = [];
    
    requiredElements.forEach(id => {
      const element = document.getElementById(id);
      if (!element) {
        console.error(`Required element #${id} is missing`);
        missingElements.push(id);
      }
    });
    
    if (missingElements.length > 0) {
      console.error(`Missing elements: ${missingElements.join(', ')}`);
    } else {
      console.log('All required DOM elements are present');
    }
  }
  
  /**
   * Create the neural network based on current configuration
   */
  createNetwork() {
    // Create MLP instance
    this.mlp = new MLP(
      this.config.inputSize,
      this.config.hiddenLayers,
      this.config.outputSize,
      this.config.activation
    );
    
    // Get network structure for visualization
    const networkData = this.mlp.getNetworkStructure();
    
    // Update network visualizer
    this.networkVisualizer.setNetworkData(networkData);
    this.networkVisualizer.render();
    
    // Update data visualizer with the model
    this.dataVisualizer.setModel(this.mlp);
    this.dataVisualizer.render();
    
    // Update UI
    this.updateStatsUI();
  }
  
  /**
   * Set XOR problem data
   */
  setXORData() {
    // Get XOR data
    this.trainingData = DataGenerator.generateXORData();
    
    // Update visualizer
    this.dataVisualizer.setData(this.trainingData, '2d-scatter');
    this.dataVisualizer.render();
    
    // Update configuration
    this.config.inputSize = 2;
    this.config.outputSize = 1;
    this.updateConfigUI();
  }
  
  /**
   * Set circle classification data
   */
  setCircleData() {
    // Get circle classification data
    this.trainingData = DataGenerator.generateCircleData(100);
    
    // Update visualizer
    this.dataVisualizer.setData(this.trainingData, '2d-scatter');
    this.dataVisualizer.render();
    
    // Update configuration
    this.config.inputSize = 2;
    this.config.outputSize = 1;
    this.updateConfigUI();
  }
  
  /**
   * Set sinusoidal data for function approximation
   */
  setSinusoidalData() {
    console.log("Setting sinusoidal data...");
    
    // Get sinusoidal data with more points for smoother visualization
    this.trainingData = DataGenerator.generateSinusoidalData(100);
    
    // Log some data points for debugging
    console.log("Sample sine data:", this.trainingData.slice(0, 3));
    
    // Update visualizer
    this.dataVisualizer.setData(this.trainingData, 'line');
    this.dataVisualizer.render();
    
    // Update configuration - make sure to reset the network for 1D input
    this.config.inputSize = 1;
    this.config.outputSize = 1;
    this.config.hiddenLayers = [8, 8]; // More complex network for function approximation
    
    // Update UI and recreate network
    this.updateConfigUI();
    this.createNetwork();
    
    console.log("Sinusoidal data setup complete");
  }
  
  /**
   * Set spiral classification data
   */
  setSpiralData() {
    // Get spiral classification data
    this.trainingData = DataGenerator.generateSpiralData(200);
    
    // Update visualizer
    this.dataVisualizer.setData(this.trainingData, '2d-scatter');
    this.dataVisualizer.render();
    
    // Update configuration
    this.config.inputSize = 2;
    this.config.outputSize = 1;
    this.config.hiddenLayers = [8, 8]; // Spirals are harder to learn, use more neurons
    this.updateConfigUI();
    this.createNetwork();
  }
  
  /**
   * Set checkerboard pattern data
   */
  setCheckerboardData() {
    // Get checkerboard data
    this.trainingData = DataGenerator.generateCheckerboardData(200);
    
    // Update visualizer
    this.dataVisualizer.setData(this.trainingData, '2d-scatter');
    this.dataVisualizer.render();
    
    // Update configuration
    this.config.inputSize = 2;
    this.config.outputSize = 1;
    this.config.hiddenLayers = [8, 8]; // Checkerboard is harder to learn, use more neurons
    this.updateConfigUI();
    this.createNetwork();
  }
  
  /**
   * Train the neural network
   */
  async trainNetwork() {
    if (!this.mlp || !this.trainingData) {
      console.error("Missing MLP or training data");
      return;
    }
    
    // Get epochs from UI
    this.config.epochs = parseInt(document.getElementById('epochs').value);
    
    // Get batch size from UI
    const batchSize = parseInt(document.getElementById('batch-size').value);
    
    // Display training status
    const trainButton = document.getElementById('train-network');
    trainButton.disabled = true;
    trainButton.textContent = 'Training...';
    
    // Reset stats display
    document.getElementById('loss-display').textContent = 'Loss: 0';
    
    // Check if we're dealing with sine wave regression
    const isSineWave = this.config.inputSize === 1 && this.config.outputSize === 1;
    
    // Update the accuracy/MAE display based on the task type
    const metricElement = document.getElementById('accuracy-display');
    if (isSineWave) {
      metricElement.textContent = 'MAE: 0';
    } else {
      metricElement.textContent = 'Accuracy: 0%';
    }
    
    document.getElementById('epoch-display').textContent = 'Epoch: 0/0';
    
    // Use setTimeout to allow UI to update before training starts
    setTimeout(async () => {
      try {
        // Train the network
        const history = await this.mlp.train(
          this.trainingData,
          this.config.epochs,
          this.config.learningRate,
          (epoch, loss, metric, isMAE) => {
            // Update UI every 10 epochs
            if (epoch % 10 === 0 || epoch === this.config.epochs - 1) {
              // Update epoch counter
              document.getElementById('epoch-display').textContent = `Epoch: ${epoch + 1}/${this.config.epochs}`;
              
              // Update loss value
              document.getElementById('loss-display').textContent = `Loss: ${loss.toFixed(6)}`;
              
              // Update metric display (accuracy or MAE)
              if (isMAE) {
                // For regression (sine wave), display MAE
                metricElement.textContent = `MAE: ${metric.toFixed(6)}`;
              } else {
                // For classification, display accuracy percentage
                metricElement.textContent = `Accuracy: ${(metric * 100).toFixed(2)}%`;
              }
              
              // Update visualizations
              this.networkVisualizer.updateWeights(this.mlp.getNetworkStructure().weights);
              this.dataVisualizer.render();
            }
          }
        );
        
        // Update network visualization
        this.networkVisualizer.updateWeights(this.mlp.getNetworkStructure().weights);
        
        // Update data visualization
        this.dataVisualizer.updateModel(this.mlp);
        
        // Re-enable train button
        trainButton.disabled = false;
        trainButton.textContent = 'Train Network';
        
        // Update stats
        this.updateStatsUI();
      } catch (error) {
        console.error("Error during training:", error);
        trainButton.disabled = false;
        trainButton.textContent = 'Train Network';
      }
    }, 100);
  }
  
  /**
   * Update the network configuration based on UI inputs
   */
  updateNetwork() {
    // Get hidden layers configuration
    const hiddenLayersCount = parseInt(document.getElementById('hidden-layers').value);
    
    // Get neurons per layer
    const neuronsPerLayer = parseInt(document.getElementById('neurons-per-layer').value);
    const hiddenLayers = Array(hiddenLayersCount).fill(neuronsPerLayer);
    
    // Get learning rate
    const learningRate = parseFloat(document.getElementById('learning-rate').value);
    
    // Get activation function
    const activation = document.getElementById('activation-function').value;
    
    // Update configuration
    this.config.hiddenLayers = hiddenLayers;
    this.config.learningRate = learningRate;
    this.config.activation = activation;
    
    // Recreate network with new configuration
    this.createNetwork();
    
    // Force update visualizations
    const networkData = this.mlp.getNetworkStructure();
    this.networkVisualizer.setNetworkData(networkData);
    this.networkVisualizer.render();
    this.dataVisualizer.setModel(this.mlp);
    this.dataVisualizer.render();
    
    // Update UI
    this.updateConfigUI();
    
    console.log("Network updated with:", {
      inputSize: this.config.inputSize,
      hiddenLayers: this.config.hiddenLayers,
      outputSize: this.config.outputSize,
      activation: this.config.activation,
      learningRate: this.config.learningRate
    });
  }
  
  /**
   * Update the UI with current configuration
   */
  updateConfigUI() {
    // Update hidden layers
    document.getElementById('hidden-layers').value = this.config.hiddenLayers.length;
    
    // Update neurons per layer (assuming all layers have the same size)
    document.getElementById('neurons-per-layer').value = this.config.hiddenLayers[0] || 3;
    
    // Update learning rate
    document.getElementById('learning-rate').value = this.config.learningRate;
    document.getElementById('learning-rate-value').textContent = this.config.learningRate;
    
    // Update activation function
    document.getElementById('activation-function').value = this.config.activation;
  }
  
  /**
   * Update the stats UI with current network information
   */
  updateStatsUI() {
    if (!this.mlp || !this.mlp.history) {
      document.getElementById('loss-display').textContent = 'Loss: 0';
      document.getElementById('accuracy-display').textContent = 'Accuracy: 0%';
      document.getElementById('epoch-display').textContent = 'Epoch: 0/0';
      return;
    }
    
    // Update epoch counter
    const epochs = this.mlp.history.epochs || 0;
    document.getElementById('epoch-display').textContent = `Epoch: ${epochs}/${this.config.epochs}`;
    
    // Update loss value
    const lastLoss = this.mlp.history.loss?.length > 0 
      ? this.mlp.history.loss[this.mlp.history.loss.length - 1]
      : 0;
    document.getElementById('loss-display').textContent = `Loss: ${lastLoss.toFixed(6)}`;
    
    // Check if we're dealing with sine wave (regression)
    const isSineWave = this.config.inputSize === 1 && this.config.outputSize === 1;
    const metricElement = document.getElementById('accuracy-display');
    
    if (isSineWave) {
      // For regression, display MAE
      const lastMAE = this.mlp.history.mae?.length > 0
        ? this.mlp.history.mae[this.mlp.history.mae.length - 1]
        : 0;
      metricElement.textContent = `MAE: ${lastMAE.toFixed(6)}`;
    } else {
      // For classification, display Accuracy percentage
      const lastAccuracy = this.mlp.history.accuracy?.length > 0
        ? this.mlp.history.accuracy[this.mlp.history.accuracy.length - 1]
        : 0;
      metricElement.textContent = `Accuracy: ${(lastAccuracy * 100).toFixed(2)}%`;
    }
  }
  
  /**
   * Reset the neural network
   */
  resetNetwork() {
    console.log("Resetting network...");
    
    // Create a new network with the same configuration
    this.createNetwork();
    
    // Force visual update
    const networkData = this.mlp.getNetworkStructure();
    this.networkVisualizer.setNetworkData(networkData);
    this.networkVisualizer.render();
    
    // Update data visualization
    this.dataVisualizer.setModel(this.mlp);
    this.dataVisualizer.render();
    
    // Update UI
    this.updateStatsUI();
    
    console.log("Network reset complete");
  }
  
  /**
   * Add event listeners to UI elements
   */
  addEventListeners() {
    // Update network button
    document.getElementById('update-network').addEventListener('click', () => {
      this.updateNetwork();
    });
    
    // Train button
    document.getElementById('train-network').addEventListener('click', () => {
      this.trainNetwork();
    });
    
    // Reset button
    document.getElementById('reset-network').addEventListener('click', () => {
      this.resetNetwork();
    });
    
    // Dataset type selector
    document.getElementById('dataset-type').addEventListener('change', (e) => {
      const datasetType = e.target.value;
      
      switch (datasetType) {
        case 'xor':
          this.setXORData();
          break;
        case 'circle':
          this.setCircleData();
          break;
        case 'sine':
          this.setSinusoidalData();
          break;
        case 'spiral':
          this.setSpiralData();
          break;
        case 'checkerboard':
          this.setCheckerboardData();
          break;
      }
      
      this.createNetwork(); // Recreate network with updated configuration
    });
    
    // Learning rate input
    document.getElementById('learning-rate').addEventListener('input', (e) => {
      this.config.learningRate = parseFloat(e.target.value);
      document.getElementById('learning-rate-value').textContent = this.config.learningRate;
    });
    
    // Epochs input
    document.getElementById('epochs').addEventListener('input', (e) => {
      this.config.epochs = parseInt(e.target.value);
    });
    
    // Hidden layers count input
    document.getElementById('hidden-layers').addEventListener('input', (e) => {
      // No immediate action needed, will be handled by update-network button
    });
    
    // Neurons per layer input
    document.getElementById('neurons-per-layer').addEventListener('input', (e) => {
      // No immediate action needed, will be handled by update-network button
    });
  }
}

// Start the application when DOM is ready
document.addEventListener('DOMContentLoaded', async () => {
  try {
    console.log('Initializing MLP Visualizer App...');
    // Wait for TensorFlow.js to be ready
    await tf.ready();
    console.log('TensorFlow.js is ready');
    window.app = new MLPVisualizerApp();
    console.log('App initialized successfully');
  } catch (error) {
    console.error('Error initializing app:', error);
  }
}); 