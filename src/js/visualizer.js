import * as d3 from 'd3';

/**
 * Neural Network Visualizer using D3.js
 */
class MLPVisualizer {
  constructor(containerId, width = null, height = null) {
    this.containerId = containerId;
    console.log(`MLPVisualizer initializing with container: #${containerId}`);
    this.container = d3.select(`#${containerId}`);

    if (this.container.empty()) {
      console.error(`Container #${containerId} not found in the DOM`);
      return;
    }

    console.log(`Container found, dimensions:`, this.container.node().getBoundingClientRect());

    this.width = width || this.container.node().getBoundingClientRect().width;
    this.height = height || this.container.node().getBoundingClientRect().height;

    // SVG setup
    this.svg = null;
    this.neuronRadius = 25;

    // Network data
    this.networkData = null;

    // Initialize SVG
    this.initSVG();
  }

  /**
   * Initialize the SVG element
   */
  initSVG() {
    this.container.selectAll('svg').remove();

    this.svg = this.container
      .append('svg')
      .attr('width', this.width)
      .attr('height', this.height)
      .attr('viewBox', [0, 0, this.width, this.height])
      .style('font', "10px -apple-system, BlinkMacSystemFont, 'Inter', sans-serif");
  }

  /**
   * Set network data for visualization
   */
  setNetworkData(networkData) {
    this.networkData = networkData;
  }

  /**
   * Create network visualization
   */
  render() {
    if (!this.networkData) {
      console.error('No network data available');
      return;
    }

    // Clear previous visualization
    this.svg.selectAll('*').remove();

    const { layers, weights } = this.networkData;

    // Calculate positions
    const layerPositions = this.calculateLayerPositions(layers);

    // Create connections (drawn first so they appear behind neurons)
    this.drawConnections(layerPositions, weights);

    // Create neurons
    this.drawNeurons(layerPositions);

    // Add layer labels
    this.addLayerLabels(layerPositions);
  }

  /**
   * Calculate positions for neurons in each layer
   */
  calculateLayerPositions(layers) {
    const padding = 50;
    const availableWidth = this.width - padding * 2;
    const availableHeight = this.height - padding * 2;

    // X positions for each layer
    const layerXPositions = d3
      .range(layers.length)
      .map(i => padding + i * (availableWidth / (layers.length - 1)));

    // For each layer, calculate Y positions of neurons
    const layerPositions = [];

    layers.forEach((neuronsCount, layerIndex) => {
      const layerHeight = Math.min(
        neuronsCount * (this.neuronRadius * 3),
        availableHeight - padding
      );

      const startY = (this.height - layerHeight) / 2;

      const neuronPositions = d3.range(neuronsCount).map(neuronIndex => {
        const y = startY + neuronIndex * (layerHeight / (neuronsCount - 1 || 1));
        return {
          x: layerXPositions[layerIndex],
          y: neuronsCount === 1 ? this.height / 2 : y,
          layerIndex,
          neuronIndex,
        };
      });

      layerPositions.push(neuronPositions);
    });

    return layerPositions;
  }

  /**
   * Draw neurons for each layer
   */
  drawNeurons(layerPositions) {
    // For each layer
    layerPositions.forEach((positions, layerIndex) => {
      // Draw neurons
      const neurons = this.svg
        .selectAll(`.layer-${layerIndex}`)
        .data(positions)
        .enter()
        .append('circle')
        .attr('class', d => {
          if (layerIndex === 0) return 'neuron input-neuron';
          if (layerIndex === layerPositions.length - 1) return 'neuron output-neuron';
          return 'neuron hidden-neuron';
        })
        .attr('cx', d => d.x)
        .attr('cy', d => d.y)
        .attr('r', this.neuronRadius)
        .attr('data-layer', layerIndex)
        .attr('data-neuron', d => d.neuronIndex);

      // Add neuron labels
      this.svg
        .selectAll(`.neuron-label-${layerIndex}`)
        .data(positions)
        .enter()
        .append('text')
        .attr('class', `neuron-label-${layerIndex}`)
        .attr('x', d => d.x)
        .attr('y', d => d.y + 4)
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'middle')
        .style('font-size', '12px')
        .style('font-weight', 'bold')
        .style('font-family', "-apple-system, BlinkMacSystemFont, 'Inter', sans-serif")
        .text(d => d.neuronIndex);
    });
  }

  /**
   * Draw connections between neurons
   */
  drawConnections(layerPositions, weights) {
    // For each layer except the last one
    for (let layerIndex = 0; layerIndex < layerPositions.length - 1; layerIndex++) {
      const layerWeights = weights[layerIndex];

      // Current layer neurons
      const currentLayerPositions = layerPositions[layerIndex];

      // Next layer neurons
      const nextLayerPositions = layerPositions[layerIndex + 1];

      // For each neuron in the next layer
      nextLayerPositions.forEach((nextNeuron, nextIndex) => {
        // For each neuron in the current layer
        currentLayerPositions.forEach((currentNeuron, currentIndex) => {
          // Get weight value
          const weight = layerWeights ? layerWeights[nextIndex][currentIndex] : 0;

          // Determine line width based on weight value
          const lineWidth = Math.abs(weight) * 7;

          // Determine line color based on weight sign
          const lineColor = weight > 0 ? '#4CAF50' : '#F44336';

          // Draw connection
          this.svg
            .append('line')
            .attr('class', 'connection')
            .attr('x1', currentNeuron.x)
            .attr('y1', currentNeuron.y)
            .attr('x2', nextNeuron.x)
            .attr('y2', nextNeuron.y)
            .style('stroke', lineColor)
            .style('stroke-width', Math.max(1.5, Math.min(lineWidth, 8)))
            .style('stroke-opacity', 0.7);
        });
      });
    }
  }

  /**
   * Add labels for each layer
   */
  addLayerLabels(layerPositions) {
    const layerNames = [
      'Input Layer',
      ...Array(layerPositions.length - 2)
        .fill()
        .map((_, i) => `Hidden Layer ${i + 1}`),
      'Output Layer',
    ];

    // For each layer, add a label at the top
    layerPositions.forEach((positions, index) => {
      const x = positions[0].x;

      this.svg
        .append('text')
        .attr('x', x)
        .attr('y', 25)
        .attr('text-anchor', 'middle')
        .attr('font-size', '16px')
        .text(layerNames[index]);
    });
  }

  /**
   * Update the visualization with new weights
   */
  updateWeights(weights) {
    if (!this.networkData) return;

    this.networkData.weights = weights;

    // Clear and redraw
    this.render();
  }

  /**
   * Resize visualization
   */
  resize(width, height) {
    this.width = width || this.container.node().getBoundingClientRect().width;
    this.height = height || this.container.node().getBoundingClientRect().height;

    this.svg
      .attr('width', this.width)
      .attr('height', this.height)
      .attr('viewBox', [0, 0, this.width, this.height]);

    if (this.networkData) {
      this.render();
    }
  }
}

export default MLPVisualizer;
