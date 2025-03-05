/**
 * Data generator for training MLP
 */
class DataGenerator {
  /**
   * Generate XOR problem data
   * This is a classic problem for neural networks
   */
  static generateXORData() {
    return [
      { input: [0, 0], output: [0] },
      { input: [0, 1], output: [1] },
      { input: [1, 0], output: [1] },
      { input: [1, 1], output: [0] }
    ];
  }
  
  /**
   * Generate circle classification data
   * Points inside a circle are one class, outside are another
   */
  static generateCircleData(n = 100) {
    const data = [];
    
    // Generate n random points
    for (let i = 0; i < n; i++) {
      // Random position in a 2D space between -1 and 1
      const x = Math.random() * 2 - 1;
      const y = Math.random() * 2 - 1;
      
      // Distance from origin
      const distance = Math.sqrt(x * x + y * y);
      
      // If distance < 0.5, it's inside the circle (class 1)
      // Otherwise, it's outside (class 0)
      const insideCircle = distance < 0.5 ? 1 : 0;
      
      data.push({
        input: [x, y],
        output: [insideCircle]
      });
    }
    
    return data;
  }
  
  /**
   * Generate spiral data (two intertwined classes)
   */
  static generateSpiralData(n = 100) {
    const data = [];
    const turns = 3; // Number of spiral turns
    
    // Generate n/2 points for each spiral class
    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < n / 2; j++) {
        // Parametric equation for spiral
        const r = j / (n / 2) * 0.8; // Radius grows from 0 to 0.8
        const theta = turns * Math.PI * j / (n / 2) + (i * Math.PI); // Angle
        
        // Add some noise
        const noise = 0.05 * (Math.random() * 2 - 1);
        
        // Convert to Cartesian coordinates
        const x = r * Math.cos(theta) + noise;
        const y = r * Math.sin(theta) + noise;
        
        data.push({
          input: [x, y],
          output: i === 0 ? [1, 0] : [0, 1] // One-hot encoding of class
        });
      }
    }
    
    return data;
  }
  
  /**
   * Generate sine wave data for function approximation
   * y = sin(x) in the range [-π, π]
   */
  static generateSinusoidalData(n = 100) {
    const data = [];
    
    // Generate n points along x-axis
    for (let i = 0; i < n; i++) {
      const x = (i / (n - 1)) * (2 * Math.PI) - Math.PI; // Range [-π, π]
      const y = Math.sin(x); // sin(x)
      
      // Add some small noise to make training more robust
      const noise = (Math.random() * 0.05) - 0.025;
      
      data.push({
        input: [x],
        output: [y + noise]
      });
    }
    
    // Add some extra points around critical areas (peaks and troughs)
    const criticalPoints = [
      { x: -Math.PI, y: Math.sin(-Math.PI) },
      { x: -Math.PI/2, y: Math.sin(-Math.PI/2) },
      { x: 0, y: Math.sin(0) },
      { x: Math.PI/2, y: Math.sin(Math.PI/2) },
      { x: Math.PI, y: Math.sin(Math.PI) }
    ];
    
    for (const point of criticalPoints) {
      data.push({
        input: [point.x],
        output: [point.y]
      });
    }
    
    // Sort data by input value for better visualization
    data.sort((a, b) => a.input[0] - b.input[0]);
    
    return data;
  }
  
  /**
   * Generate a dataset for user-defined function
   */
  static generateCustomData(func, inputDim = 1, outputDim = 1, n = 100, range = [-1, 1]) {
    const data = [];
    
    for (let i = 0; i < n; i++) {
      // Generate random inputs within the specified range
      const input = Array(inputDim).fill().map(() => 
        range[0] + Math.random() * (range[1] - range[0])
      );
      
      // Calculate outputs using the provided function
      const output = func(input);
      
      data.push({
        input,
        output: Array.isArray(output) ? output : [output]
      });
    }
    
    return data;
  }
  
  /**
   * Normalize a dataset to have values between 0 and 1
   */
  static normalizeData(data) {
    // Find min and max for each input dimension
    const inputDim = data[0].input.length;
    const inputMins = Array(inputDim).fill(Number.MAX_VALUE);
    const inputMaxs = Array(inputDim).fill(Number.MIN_VALUE);
    
    // Find min and max for each output dimension
    const outputDim = data[0].output.length;
    const outputMins = Array(outputDim).fill(Number.MAX_VALUE);
    const outputMaxs = Array(outputDim).fill(Number.MIN_VALUE);
    
    // Find min and max values
    data.forEach(item => {
      // Input
      item.input.forEach((value, i) => {
        inputMins[i] = Math.min(inputMins[i], value);
        inputMaxs[i] = Math.max(inputMaxs[i], value);
      });
      
      // Output
      item.output.forEach((value, i) => {
        outputMins[i] = Math.min(outputMins[i], value);
        outputMaxs[i] = Math.max(outputMaxs[i], value);
      });
    });
    
    // Normalize data
    return data.map(item => {
      const normalizedInput = item.input.map((value, i) => {
        const range = inputMaxs[i] - inputMins[i];
        return range === 0 ? 0.5 : (value - inputMins[i]) / range;
      });
      
      const normalizedOutput = item.output.map((value, i) => {
        const range = outputMaxs[i] - outputMins[i];
        return range === 0 ? 0.5 : (value - outputMins[i]) / range;
      });
      
      return {
        input: normalizedInput,
        output: normalizedOutput,
        original: {
          input: [...item.input],
          output: [...item.output]
        }
      };
    });
  }
  
  /**
   * Split data into training and testing sets
   */
  static splitData(data, trainRatio = 0.8) {
    // Shuffle data
    const shuffled = [...data].sort(() => Math.random() - 0.5);
    
    const trainSize = Math.floor(data.length * trainRatio);
    const trainData = shuffled.slice(0, trainSize);
    const testData = shuffled.slice(trainSize);
    
    return { trainData, testData };
  }
}

export default DataGenerator; 