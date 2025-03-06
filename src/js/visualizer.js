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

    // Tooltip element
    this.tooltip = null;

    // Initialize SVG
    this.initSVG();

    // Create tooltip
    this.initTooltip();
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
   * Initialize tooltip div
   */
  initTooltip() {
    // Remove any existing tooltip
    d3.select(`#${this.containerId}-tooltip`).remove();
    
    // Create tooltip element with white background and proper styling
    this.tooltip = d3.select("body").append("div")
      .attr("id", `${this.containerId}-tooltip`)
      .attr("class", "neuron-tooltip")
      .style("position", "absolute")
      .style("pointer-events", "none")
      .style("background-color", "#ffffff")
      .style("padding", "14px 16px")
      .style("border-radius", "8px")
      .style("box-shadow", "0 3px 12px rgba(0, 0, 0, 0.1), 0 1px 4px rgba(0, 0, 0, 0.08)")
      .style("font-size", "10px") // Smaller text size
      .style("opacity", 0);
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
    const layerClasses = ["input-neuron", "hidden-neuron", "output-neuron"];
    
    // Flatten all neurons for easier selection
    let allNeurons = [];
    
    // Process each layer
    layerPositions.forEach((positions, layerIndex) => {
      // Determine neuron class based on layer index
      const neuronClass = layerIndex === 0 
        ? layerClasses[0] 
        : (layerIndex === layerPositions.length - 1 
          ? layerClasses[2] 
          : layerClasses[1]);
      
      // Add neurons to the flattened array
      positions.forEach(neuron => {
        neuron.layerIndex = layerIndex;
        neuron.class = neuronClass;
        allNeurons.push(neuron);
      });
    });
    
    // Create neurons
    allNeurons.forEach(neuron => {
      // Create neuron circle
      this.svg.append("circle")
        .attr("class", `neuron ${neuron.class}`)
        .attr("cx", neuron.x)
        .attr("cy", neuron.y)
        .attr("r", this.neuronRadius)
        .on("mouseover", (event) => this.showNeuronTooltip(event, neuron))
        .on("mouseout", () => this.hideTooltip());
      
      // Add label inside neuron
      this.svg.append("text")
        .attr("class", "neuron-label")
        .attr("x", neuron.x)
        .attr("y", neuron.y + 4)
        .attr("text-anchor", "middle")
        .attr("dominant-baseline", "middle")
        .style("font-size", "14px")
        .style("font-weight", "bold")
        .style("font-family", "-apple-system, BlinkMacSystemFont, 'Inter', sans-serif")
        .text(neuron.neuronIndex);
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

  /**
   * Show tooltip with neuron information
   */
  showNeuronTooltip(event, neuron) {
    if (!this.tooltip) return;
    
    // Basic neuron information
    let content = `<strong>Layer ${neuron.layerIndex + 1}, Neuron ${neuron.neuronIndex + 1}</strong><br>`;
    
    // Determine neuron type and background color
    let bgColor = "#ffffff";
    let textColor = "#333333";
    let neuronType = "";
    
    if (neuron.layerIndex === 0) {
      neuronType = "Input neuron";
      bgColor = "#EEF1FF"; // Light blue for input neurons
    } else if (neuron.layerIndex === this.networkData.layers.length - 1) {
      neuronType = "Output neuron";
      bgColor = "#EEFFF5"; // Light green for output neurons
    } else {
      neuronType = "Hidden neuron";
      bgColor = "#F0EAFF"; // Light purple for hidden neurons
    }
    
    content += `<span>${neuronType}</span>`;
    
    // Add weights information in a concise format
    if (this.networkData && this.networkData.weights) {
      // Input neurons - show outgoing weights
      if (neuron.layerIndex === 0 && this.networkData.weights[0]) {
        content += `<br><br><strong>Weights:</strong><br>`;
        
        // Count positive and negative outgoing weights
        let positiveCount = 0;
        let negativeCount = 0;
        let maxWeight = 0;
        let minWeight = 0;
        
        for (let i = 0; i < this.networkData.weights[0].length; i++) {
          const weight = this.networkData.weights[0][i][neuron.neuronIndex];
          if (weight > 0) {
            positiveCount++;
            maxWeight = Math.max(maxWeight, weight);
          } else if (weight < 0) {
            negativeCount++;
            minWeight = Math.min(minWeight, weight);
          }
        }
        
        // Show summary
        content += `<span class="positive">${positiveCount} positive</span> / <span class="negative">${negativeCount} negative</span><br>`;
        if (positiveCount > 0) {
          content += `Strongest positive: <span class="positive">${maxWeight.toFixed(3)}</span><br>`;
        }
        if (negativeCount > 0) {
          content += `Strongest negative: <span class="negative">${minWeight.toFixed(3)}</span>`;
        }
      }
      // Hidden neurons - show incoming weights
      else if (neuron.layerIndex > 0 && this.networkData.weights[neuron.layerIndex - 1]) {
        content += `<br><br><strong>Weights:</strong><br>`;
        
        // Count positive and negative incoming weights
        let positiveCount = 0;
        let negativeCount = 0;
        let maxWeight = 0;
        let minWeight = 0;
        
        const incomingWeights = this.networkData.weights[neuron.layerIndex - 1][neuron.neuronIndex];
        for (let i = 0; i < incomingWeights.length; i++) {
          const weight = incomingWeights[i];
          if (weight > 0) {
            positiveCount++;
            maxWeight = Math.max(maxWeight, weight);
          } else if (weight < 0) {
            negativeCount++;
            minWeight = Math.min(minWeight, weight);
          }
        }
        
        // Show summary
        content += `<span class="positive">${positiveCount} positive</span> / <span class="negative">${negativeCount} negative</span><br>`;
        if (positiveCount > 0) {
          content += `Strongest positive: <span class="positive">${maxWeight.toFixed(3)}</span><br>`;
        }
        if (negativeCount > 0) {
          content += `Strongest negative: <span class="negative">${minWeight.toFixed(3)}</span>`;
        }
      }
    }
    
    // Update tooltip content first
    this.tooltip.html(content);
    
    // Add an arrow to the tooltip that matches the background color
    const arrowStyle = `
      <style>
        #${this.containerId}-tooltip:after {
          content: '';
          position: absolute;
          bottom: -8px;
          left: 50%;
          margin-left: -8px;
          width: 0;
          height: 0;
          border-left: 8px solid transparent;
          border-right: 8px solid transparent;
          border-top: 8px solid ${bgColor};
          filter: drop-shadow(0 2px 2px rgba(0, 0, 0, 0.06));
        }
      </style>
    `;
    this.tooltip.node().insertAdjacentHTML('beforeend', arrowStyle);
    
    // Get the SVG container's position relative to the viewport
    const svgRect = this.svg.node().getBoundingClientRect();
    const scrollX = window.pageXOffset || document.documentElement.scrollLeft;
    const scrollY = window.pageYOffset || document.documentElement.scrollTop;
    
    // Calculate absolute position of the neuron in the document
    const neuronAbsX = svgRect.left + neuron.x + scrollX;
    const neuronAbsY = svgRect.top + neuron.y + scrollY;
    
    // Get the tooltip dimensions
    const tooltipRect = this.tooltip.node().getBoundingClientRect();
    
    // Position tooltip above the neuron with proper styling
    this.tooltip
      .style("position", "absolute")
      .style("background-color", bgColor)
      .style("color", textColor)
      .style("padding", "14px 16px")
      .style("border-radius", "8px")
      .style("box-shadow", "0 3px 12px rgba(0, 0, 0, 0.1), 0 1px 4px rgba(0, 0, 0, 0.08)")
      .style("font-size", "10px") // Smaller text size
      .style("left", (neuronAbsX - tooltipRect.width / 2) + "px")
      .style("top", (neuronAbsY - this.neuronRadius - tooltipRect.height - 10) + "px")
      .transition()
      .duration(200)
      .style("opacity", 1);
  }
  
  /**
   * Hide the tooltip
   */
  hideTooltip() {
    if (!this.tooltip) return;
    
    this.tooltip
      .transition()
      .duration(500)
      .style("opacity", 0)
      .on("end", () => {
        // Reset to default styles when hidden
        this.tooltip
          .style("left", "-9999px")
          .style("background-color", "#ffffff");
      });
  }
}

export default MLPVisualizer;
