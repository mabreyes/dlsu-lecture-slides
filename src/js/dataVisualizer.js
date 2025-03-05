import * as d3 from 'd3';

/**
 * Class for visualizing datasets and model predictions
 */
class DataVisualizer {
  constructor(containerId, width = null, height = null) {
    this.containerId = containerId;
    console.log(`DataVisualizer initializing with container: #${containerId}`);
    this.container = d3.select(`#${containerId}`);
    
    if (this.container.empty()) {
      console.error(`Container #${containerId} not found in the DOM`);
      return;
    }
    
    console.log(`Container found, dimensions:`, 
      this.container.node().getBoundingClientRect());
    
    this.width = width || this.container.node().getBoundingClientRect().width;
    this.height = height || this.container.node().getBoundingClientRect().height;
    
    // SVG setup
    this.svg = null;
    this.margin = { top: 20, right: 20, bottom: 40, left: 40 };
    
    // Scales
    this.xScale = null;
    this.yScale = null;
    
    // Data
    this.data = null;
    this.predictionData = null;
    
    // Model for predictions
    this.model = null;
    
    // Type of visualization
    this.visualizationType = '2d-scatter';
    
    // Initialize SVG
    this.initSVG();
  }
  
  /**
   * Initialize the SVG element
   */
  initSVG() {
    this.container.selectAll("svg").remove();
    
    this.svg = this.container.append("svg")
      .attr("width", this.width)
      .attr("height", this.height)
      .attr("viewBox", [0, 0, this.width, this.height])
      .style("font", "10px sans-serif");
  }
  
  /**
   * Set data to be visualized
   */
  setData(data, type = '2d-scatter') {
    this.data = data;
    this.visualizationType = type;
  }
  
  /**
   * Set model for making predictions
   */
  setModel(model) {
    this.model = model;
  }
  
  /**
   * Render visualization based on type
   */
  render() {
    if (!this.data) {
      console.error("No data available");
      return;
    }
    
    // Clear previous visualization
    this.svg.selectAll("*").remove();
    
    // Determine which visualization to use
    switch (this.visualizationType) {
      case '2d-scatter':
        this.render2DScatter();
        break;
      case 'line':
        this.renderLine();
        break;
      default:
        console.error("Unknown visualization type:", this.visualizationType);
    }
  }
  
  /**
   * Render 2D scatter plot
   */
  render2DScatter() {
    // Create scales
    const plotWidth = this.width - this.margin.left - this.margin.right;
    const plotHeight = this.height - this.margin.top - this.margin.bottom;
    
    // Check if we have 2D input data
    if (this.data[0].input.length !== 2) {
      console.error("2D scatter plot requires 2D input data");
      return;
    }
    
    // Extract x and y values from data
    const xValues = this.data.map(d => d.input[0]);
    const yValues = this.data.map(d => d.input[1]);
    
    // Create scales with some padding
    const xExtent = d3.extent(xValues);
    const yExtent = d3.extent(yValues);
    
    const xPadding = (xExtent[1] - xExtent[0]) * 0.1;
    const yPadding = (yExtent[1] - yExtent[0]) * 0.1;
    
    this.xScale = d3.scaleLinear()
      .domain([xExtent[0] - xPadding, xExtent[1] + xPadding])
      .range([this.margin.left, this.width - this.margin.right]);
    
    this.yScale = d3.scaleLinear()
      .domain([yExtent[0] - yPadding, yExtent[1] + yPadding])
      .range([this.height - this.margin.bottom, this.margin.top]);
    
    // Create plot container
    const plot = this.svg.append("g")
      .attr("class", "plot");
    
    // Add axes
    plot.append("g")
      .attr("class", "x-axis")
      .attr("transform", `translate(0,${this.height - this.margin.bottom})`)
      .call(d3.axisBottom(this.xScale));
    
    plot.append("g")
      .attr("class", "y-axis")
      .attr("transform", `translate(${this.margin.left},0)`)
      .call(d3.axisLeft(this.yScale));
    
    // If we have a model, render decision boundaries
    if (this.model) {
      this.renderDecisionBoundary(plot);
    }
    
    // Add data points
    plot.selectAll(".data-point")
      .data(this.data)
      .enter()
      .append("circle")
      .attr("class", "data-point")
      .attr("cx", d => this.xScale(d.input[0]))
      .attr("cy", d => this.yScale(d.input[1]))
      .attr("r", 5)
      .style("fill", d => {
        // Color based on output class (assuming binary classification)
        if (d.output.length === 1) {
          return d.output[0] > 0.5 ? "#3498db" : "#e74c3c";
        } else {
          // For multi-class, use index of max value
          const maxIndex = d.output.indexOf(Math.max(...d.output));
          const colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"];
          return colors[maxIndex % colors.length];
        }
      })
      .style("stroke", "#fff")
      .style("stroke-width", 1.5)
      .style("opacity", 0.8);
  }
  
  /**
   * Render decision boundary for 2D data
   */
  renderDecisionBoundary(plot) {
    // Create a grid of points
    const gridSize = 50;
    const xDomain = this.xScale.domain();
    const yDomain = this.yScale.domain();
    const xStep = (xDomain[1] - xDomain[0]) / gridSize;
    const yStep = (yDomain[1] - yDomain[0]) / gridSize;
    
    // Generate grid points
    const gridData = [];
    
    for (let x = xDomain[0]; x <= xDomain[1]; x += xStep) {
      for (let y = yDomain[0]; y <= yDomain[1]; y += yStep) {
        // Make prediction
        const prediction = this.model.predict([x, y]);
        
        gridData.push({
          x,
          y,
          prediction
        });
      }
    }
    
    // Create color scale
    const colorScale = d3.scaleSequential(d3.interpolateRdBu)
      .domain([1, 0]);
    
    // Draw boundary
    plot.selectAll(".grid-point")
      .data(gridData)
      .enter()
      .append("rect")
      .attr("class", "grid-point")
      .attr("x", d => this.xScale(d.x) - xStep / 2)
      .attr("y", d => this.yScale(d.y) - yStep / 2)
      .attr("width", Math.ceil(this.xScale(xDomain[0] + xStep) - this.xScale(xDomain[0])))
      .attr("height", Math.ceil(this.yScale(yDomain[0]) - this.yScale(yDomain[0] + yStep)))
      .style("fill", d => {
        // Determine color based on prediction
        if (d.prediction.length === 1) {
          return colorScale(d.prediction[0]);
        } else {
          // For multi-class, use index of max value for color
          const maxIndex = d.prediction.indexOf(Math.max(...d.prediction));
          const colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"];
          return colors[maxIndex % colors.length];
        }
      })
      .style("opacity", 0.2);
  }
  
  /**
   * Render line chart for 1D function approximation
   */
  renderLine() {
    // Create scales
    const plotWidth = this.width - this.margin.left - this.margin.right;
    const plotHeight = this.height - this.margin.top - this.margin.bottom;
    
    // Check if we have 1D input data
    if (this.data[0].input.length !== 1) {
      console.error("Line chart requires 1D input data");
      return;
    }
    
    // Extract x and y values from data
    const xValues = this.data.map(d => d.input[0]);
    const yValues = this.data.map(d => d.output[0]);
    
    // Create scales with some padding
    const xExtent = d3.extent(xValues);
    const yExtent = d3.extent(yValues);
    
    const xPadding = (xExtent[1] - xExtent[0]) * 0.1;
    const yPadding = Math.max((yExtent[1] - yExtent[0]) * 0.1, 0.1);
    
    this.xScale = d3.scaleLinear()
      .domain([xExtent[0] - xPadding, xExtent[1] + xPadding])
      .range([this.margin.left, this.width - this.margin.right]);
    
    this.yScale = d3.scaleLinear()
      .domain([yExtent[0] - yPadding, yExtent[1] + yPadding])
      .range([this.height - this.margin.bottom, this.margin.top]);
    
    // Create plot container
    const plot = this.svg.append("g")
      .attr("class", "plot");
    
    // Add axes
    plot.append("g")
      .attr("class", "x-axis")
      .attr("transform", `translate(0,${this.height - this.margin.bottom})`)
      .call(d3.axisBottom(this.xScale));
    
    plot.append("g")
      .attr("class", "y-axis")
      .attr("transform", `translate(${this.margin.left},0)`)
      .call(d3.axisLeft(this.yScale));
    
    // Add data points
    plot.selectAll(".data-point")
      .data(this.data)
      .enter()
      .append("circle")
      .attr("class", "data-point")
      .attr("cx", d => this.xScale(d.input[0]))
      .attr("cy", d => this.yScale(d.output[0]))
      .attr("r", 4)
      .style("fill", "#3498db")
      .style("stroke", "#fff")
      .style("stroke-width", 1)
      .style("opacity", 0.8);
    
    // If we have a model, render predictions
    if (this.model) {
      this.renderPredictionLine(plot);
    }
  }
  
  /**
   * Render prediction line for 1D function approximation
   */
  renderPredictionLine(plot) {
    // Generate predictions on a range of inputs
    const xDomain = this.xScale.domain();
    const numPoints = 100;
    const xStep = (xDomain[1] - xDomain[0]) / numPoints;
    
    const predictionData = [];
    
    for (let x = xDomain[0]; x <= xDomain[1]; x += xStep) {
      const prediction = this.model.predict([x])[0];
      predictionData.push({
        x,
        y: prediction
      });
    }
    
    // Sort prediction data by x for a clean line
    predictionData.sort((a, b) => a.x - b.x);
    
    // Create line generator
    const line = d3.line()
      .x(d => this.xScale(d.x))
      .y(d => this.yScale(d.y))
      .curve(d3.curveMonotoneX);
    
    // Draw prediction line
    plot.append("path")
      .datum(predictionData)
      .attr("class", "prediction-line")
      .attr("fill", "none")
      .attr("stroke", "#e74c3c")
      .attr("stroke-width", 2)
      .attr("stroke-dasharray", "5,5")
      .attr("d", line);
  }
  
  /**
   * Resize visualization
   */
  resize(width, height) {
    this.width = width || this.container.node().getBoundingClientRect().width;
    this.height = height || this.container.node().getBoundingClientRect().height;
    
    this.svg
      .attr("width", this.width)
      .attr("height", this.height)
      .attr("viewBox", [0, 0, this.width, this.height]);
    
    if (this.data) {
      this.render();
    }
  }
  
  /**
   * Update with new model for predictions
   */
  updateModel(model) {
    this.model = model;
    if (this.data) {
      this.render();
    }
  }
}

export default DataVisualizer; 