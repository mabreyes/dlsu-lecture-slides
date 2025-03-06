---
marp: true
theme: default
paginate: true
header: '**Multilayer Perceptrons** - Neural Network Fundamentals'
style: |
  @import url('./math-styles.css');
  @import url('./formula-styles.css');

  /* KaTeX styling for better formula rendering */
  @import url('https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css');

  /* Improve math rendering in Mermaid diagrams */
  .mermaid .label foreignObject {
    overflow: visible !important;
  }

  /* Style enhancements for math in diagrams */
  .mermaid text tspan {
    font-family: 'KaTeX_Math', 'Times New Roman', serif;
  }

  /* Allow math to overflow properly */
  .katex-display {
    overflow: visible !important;
    display: block;
  }

  /* Fix for math in diagram nodes */
  .mermaid .label div.math {
    display: inline-block;
  }

  /* Ensure math is centered in diagram nodes */
  .mermaid .node .label {
    text-align: center;
  }

  /* Add more padding around math in diagram nodes */
  .mermaid .node {
    padding: 10px;
  }

  /* Additional fixes for math in Mermaid diagrams */
  .mermaid .label .katex {
    font-size: 1em !important;
  }

  .mermaid .label .katex-html {
    text-align: center;
  }

  /* Support for display math in Mermaid diagrams */
  .mermaid .label .katex-display {
    margin: 0.5em 0;
  }
math: true
---

<!-- KaTeX and Mermaid Math Configuration -->
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"
        onload="renderMathInElement(document.body, {
          delimiters: [
            {left: '$$', right: '$$', display: true},
            {left: '$', right: '$', display: false}
          ],
          throwOnError: false,
          trust: true
        });"></script>
<script>
  // Configure Mermaid to properly handle math
  window.addEventListener('DOMContentLoaded', () => {
    if (typeof mermaid !== 'undefined') {
      mermaid.initialize({
        startOnLoad: true,
        theme: 'default',
        legacyMathML: true,
        securityLevel: 'loose',
        htmlLabels: true,
        flowchart: {
          htmlLabels: true,
          useMaxWidth: false,
          curve: 'basis'
        }
      });
    }

    // Function to re-render math after Mermaid renders diagrams
    const renderMermaidMath = () => {
      // Find all Mermaid diagram labels
      const mermaidLabels = document.querySelectorAll('.mermaid .label');

      mermaidLabels.forEach(label => {
        // Check if the label contains math ($ signs)
        if (label.textContent.includes('$')) {
          try {
            // Render math with KaTeX
            renderMathInElement(label, {
              delimiters: [
                {left: '$$', right: '$$', display: true},
                {left: '$', right: '$', display: false}
              ],
              throwOnError: false,
              output: 'html',
              trust: true
            });

            // Fix positioning
            const foreignObject = label.closest('foreignObject');
            if (foreignObject) {
              foreignObject.setAttribute('overflow', 'visible');
              // Increase foreignObject size to accommodate display math
              const width = parseInt(foreignObject.getAttribute('width') || 0);
              const height = parseInt(foreignObject.getAttribute('height') || 0);
              foreignObject.setAttribute('width', Math.max(width, 150));
              foreignObject.setAttribute('height', Math.max(height, 60));
            }
          } catch (error) {
            console.error('Failed to render math:', error);
          }
        }
      });
    };

    // Run initial render
    setTimeout(renderMermaidMath, 1000);
    // Run another render after diagrams are fully loaded
    setTimeout(renderMermaidMath, 2000);

    // Set up mutation observer to handle dynamically rendered diagrams
    const observer = new MutationObserver(mutations => {
      let mermaidDetected = false;

      mutations.forEach(mutation => {
        // Check if any added nodes contain a Mermaid diagram
        if (mutation.addedNodes.length) {
          mutation.addedNodes.forEach(node => {
            if (node.classList && node.classList.contains('mermaid')) {
              mermaidDetected = true;
            }
          });
        }
      });

      // If we found a new Mermaid diagram, re-render math
      if (mermaidDetected) {
        setTimeout(renderMermaidMath, 500);
      }
    });

    observer.observe(document.body, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ['class']
    });
  });
</script>

# Multilayer Perceptron (MLP)

## Neural Network Fundamentals

### A Comprehensive Introduction

**Marc Reyes**
*Lecturer*
*March 7, 2025*

---

# The Building Blocks of Deep Learning

<!-- Replace background image with a consistent MLP diagram -->
```mermaid
flowchart LR
    %% Input layer with nodes
    i1((x₁)) --> h1((h₁))
    i2((x₂)) --> h1
    i3((x₃)) --> h1
    i1 --> h2((h₂))
    i2 --> h2
    i3 --> h2
    i1 --> h3((h₃))
    i2 --> h3
    i3 --> h3

    %% Hidden to output connections
    h1 --> o1((ŷ₁))
    h1 --> o2((ŷ₂))
    h2 --> o1
    h2 --> o2
    h3 --> o1
    h3 --> o2

    %% Layer labels and grouping
    subgraph Input["Input Layer"]
        i1
        i2
        i3
    end

    subgraph Hidden["Hidden Layer"]
        h1
        h2
        h3
    end

    subgraph Output["Output Layer"]
        o1
        o2
    end

    %% Include annotations
    Input -.-> |"Features"| Input
    Hidden -.-> |"Feature Extraction"| Hidden
    Output -.-> |"Predictions"| Output

    %% Adding weight labels to some connections
    i1 --> |w₁₁| h1
    h2 --> |w₂₁| o1

    %% Styling
    classDef layer fill:#f5f5f5,stroke:#999,stroke-width:1px,rx:5px,ry:5px
    classDef node fill:white,stroke:#333,stroke-width:1px
    classDef input fill:#e3f2fd,stroke:#1565c0,color:#0d47a1
    classDef hidden fill:#f3e5f5,stroke:#6a1b9a,color:#4a148c
    classDef output fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20

    class Input layer
    class Hidden layer
    class Output layer
    class i1,i2,i3 input
    class h1,h2,h3 hidden
    class o1,o2 output
```

---

# The Building Blocks of Deep Learning

- Foundation of modern neural networks
- Versatile architecture for diverse problems
- Combines simplicity with powerful learning capabilities

---

# From Neurons to Networks

```mermaid
flowchart LR
    %% Left-to-right flow
    x1[x₁] -- w₁ --> sum((Σ))
    x2[x₂] -- w₂ --> sum
    x3[x₃] -- w₃ --> sum
    b((bias)) -.-> sum

    sum -- z --> af["Activation<br>Function"]
    af -- a --> output[Output]

    %% Optional dotted annotations (if desired):
    %% x1 -.-> |"Inputs"| x2
    %% af -.-> |"f(z)=σ(z)"| af

    %% Styling
    classDef node fill:#f9f9f9,stroke:#999,color:#333
    classDef io fill:#e1f5fe,stroke:#01579b,color:#01579b

    class x1,x2,x3,output io
    class sum,af node
```

---

# From Neurons to Networks

- **Biological inspiration**: Mimics brain's neural structure
  - Neurons receive, process, and transmit information
- **Artificial neuron**: Weighted sum + activation function
  - Processes inputs through mathematical operations
- **Network topology**: Input layer → Hidden layers → Output layer
  - Organized structure for information processing
- **Information flow**: Forward propagation for predictions
  - Data travels from input to output through the network

---

# The Perceptron Journey

<!-- Replace image with a timeline/gantt chart diagram -->
```mermaid
timeline
    title Neural Network Evolution
    section 1950s-1960s
        1957: Perceptron proposed by Frank Rosenblatt
        1958: First implementation of neural learning algorithm
        1969: "Perceptrons" book by Minsky & Papert exposed XOR limitation
    section 1970s-1980s
        1974: Backpropagation developed (Werbos)
        1986: Backpropagation popularized by Rumelhart, Hinton & Williams
        1989: Universal approximation theorem proven
    section 1990s-2000s
        1998: LeNet-5 for digit recognition
        1999: Vanishing gradient problem identified
        2006: Deep learning breakthrough (Hinton)
    section 2010s+
        2012: AlexNet revolutionizes computer vision
        2014: GANs introduced
        2017: Transformer architecture
        2025: Modern advancements continue
```

- **1958**: Rosenblatt's single-layer perceptron
  - First implementation of a neural learning algorithm
- **1969**: Minsky & Papert expose limitations (XOR problem)
  - Demonstrated that single-layer networks couldn't solve nonlinear problems
- **1986**: Rumelhart, Hinton & Williams introduce backpropagation
  - Breakthrough algorithm enabling training of multi-layer networks
- **Today**: Foundation for advanced architectures (CNNs, RNNs, Transformers)
  - Core concepts extended to specialized network designs

---

# MLP Architecture

<!-- Replace image with an MLP architecture diagram -->
```mermaid
flowchart LR
    %% Input layer with nodes
    i1((x₁)) --> h1((h₁))
    i2((x₂)) --> h1
    i3((x₃)) --> h1
    i1 --> h2((h₂))
    i2 --> h2
    i3 --> h2
    i1 --> h3((h₃))
    i2 --> h3
    i3 --> h3

    %% Hidden to output connections
    h1 --> o1((ŷ₁))
    h1 --> o2((ŷ₂))
    h2 --> o1
    h2 --> o2
    h3 --> o1
    h3 --> o2

    %% Layer labels and grouping
    subgraph Input["Input Layer"]
        i1
        i2
        i3
    end

    subgraph Hidden["Hidden Layer"]
        h1
        h2
        h3
    end

    subgraph Output["Output Layer"]
        o1
        o2
    end

    %% Include annotations
    Input -.-> |"Features"| Input
    Hidden -.-> |"Feature Extraction"| Hidden
    Output -.-> |"Predictions"| Output

    %% Adding weight labels to some connections
    i1 --> |w₁₁| h1
    h2 --> |w₂₁| o1

    %% Styling
    classDef layer fill:#f5f5f5,stroke:#999,stroke-width:1px,rx:5px,ry:5px
    classDef node fill:white,stroke:#333,stroke-width:1px
    classDef input fill:#e3f2fd,stroke:#1565c0,color:#0d47a1
    classDef hidden fill:#f3e5f5,stroke:#6a1b9a,color:#4a148c
    classDef output fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20

    class Input layer
    class Hidden layer
    class Output layer
    class i1,i2,i3 input
    class h1,h2,h3 hidden
    class o1,o2 output
```

---

## Key Components

- **Input layer**: Raw data reception
  - Receives and standardizes input features
- **Hidden layers**: Feature extraction and transformation
  - Learns hierarchical representations of data
- **Output layer**: Final prediction/classification
  - Produces the network's answer to the given problem

---

# MLP Architecture

<!-- Use the same MLP architecture diagram to remain consistent -->
```mermaid
flowchart LR
    %% Input layer with nodes
    i1((x₁)) --> h1((h₁))
    i2((x₂)) --> h1
    i3((x₃)) --> h1
    i1 --> h2((h₂))
    i2 --> h2
    i3 --> h2
    i1 --> h3((h₃))
    i2 --> h3
    i3 --> h3

    %% Hidden to output connections
    h1 --> o1((ŷ₁))
    h1 --> o2((ŷ₂))
    h2 --> o1
    h2 --> o2
    h3 --> o1
    h3 --> o2

    %% Layer labels and grouping
    subgraph Input["Input Layer"]
        i1
        i2
        i3
    end

    subgraph Hidden["Hidden Layer"]
        h1
        h2
        h3
    end

    subgraph Output["Output Layer"]
        o1
        o2
    end

    %% Include annotations
    Input -.-> |"Features"| Input
    Hidden -.-> |"Feature Extraction"| Hidden
    Output -.-> |"Predictions"| Output

    %% Adding weight labels to some connections
    i1 --> |w₁₁| h1
    h2 --> |w₂₁| o1

    %% Styling
    classDef layer fill:#f5f5f5,stroke:#999,stroke-width:1px,rx:5px,ry:5px
    classDef node fill:white,stroke:#333,stroke-width:1px
    classDef input fill:#e3f2fd,stroke:#1565c0,color:#0d47a1
    classDef hidden fill:#f3e5f5,stroke:#6a1b9a,color:#4a148c
    classDef output fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20

    class Input layer
    class Hidden layer
    class Output layer
    class i1,i2,i3 input
    class h1,h2,h3 hidden
    class o1,o2 output
```

---

## Key Components

- **Weights & biases**: Learnable parameters
  - Adjusted during training to minimize error
- **Activation functions**: Introduce non-linearity
  - Enable the network to learn complex patterns

---

# Activation Functions

<!-- Replace image with an activation functions diagram -->
```mermaid
flowchart TD
    %% Main activation function node with branches
    Root["Activation Functions"] --- SigClass["Sigmoid Class"]
    Root --- ReLUClass["ReLU Class"]
    Root --- OtherClass["Other Functions"]

    %% Sigmoid class activations - Using correct LaTeX notation
    SigClass --- Sigmoid["Sigmoid<br/>$$\sigma(x) = \frac{1}{1+e^{-x}}$$"]
    SigClass --- Tanh["Tanh<br/>$$\tanh(x) = \frac{e^x-e^{-x}}{e^x+e^{-x}}$$"]

    %% ReLU class activations - Using correct LaTeX notation
    ReLUClass --- ReLU["ReLU<br/>$$f(x) = \max(0,x)$$"]
    ReLUClass --- LeakyReLU["Leaky ReLU<br/>$$f(x) = \max(0.01x, x)$$"]
    ReLUClass --- PReLU["Parametric ReLU<br/>$$f(x) = \max(\alpha x, x)$$"]

    %% Other activation functions - Using correct LaTeX notation
    OtherClass --- Softmax["Softmax<br/>$$\sigma(x_i) = \frac{e^{x_i}}{\sum e^{x_j}}$$"]
    OtherClass --- GELU["GELU<br/>$$f(x) = x\cdot\Phi(x)$$"]

    %% Styling
    classDef root fill:#f5f5f5,stroke:#666,stroke-width:2px,color:#333,rx:5px,ry:5px
    classDef category fill:#e1f5fe,stroke:#0277bd,color:#0277bd,rx:5px,ry:5px
    classDef function fill:#fff,stroke:#999,color:#333,rx:5px,ry:5px,padding:15px

    class Root root
    class SigClass,ReLUClass,OtherClass category
    class Sigmoid,Tanh,ReLU,LeakyReLU,PReLU,Softmax,GELU function
```

| Function      | Formula                                               | Characteristics                     |
|---------------|-------------------------------------------------------|-------------------------------------|
| Sigmoid       | $\sigma(x) = \frac{1}{1+e^{-x}}$                       | Output range [0,1], vanishing gradient |
| Tanh          | $\tanh(x) = \frac{e^x-e^{-x}}{e^x+e^{-x}}$             | Output range [-1,1], zero-centered    |

---

# Activation Functions

<!-- Use the same activation functions diagram -->
```mermaid
flowchart TD
    %% Main activation function node with branches
    Root["Activation Functions"] --- SigClass["Sigmoid Class"]
    Root --- ReLUClass["ReLU Class"]
    Root --- OtherClass["Other Functions"]

    %% Sigmoid class activations - Using correct LaTeX notation
    SigClass --- Sigmoid["Sigmoid<br/>$$\sigma(x) = \frac{1}{1+e^{-x}}$$"]
    SigClass --- Tanh["Tanh<br/>$$\tanh(x) = \frac{e^x-e^{-x}}{e^x+e^{-x}}$$"]

    %% ReLU class activations - Using correct LaTeX notation
    ReLUClass --- ReLU["ReLU<br/>$$f(x) = \max(0,x)$$"]
    ReLUClass --- LeakyReLU["Leaky ReLU<br/>$$f(x) = \max(0.01x, x)$$"]
    ReLUClass --- PReLU["Parametric ReLU<br/>$$f(x) = \max(\alpha x, x)$$"]

    %% Other activation functions - Using correct LaTeX notation
    OtherClass --- Softmax["Softmax<br/>$$\sigma(x_i) = \frac{e^{x_i}}{\sum e^{x_j}}$$"]
    OtherClass --- GELU["GELU<br/>$$f(x) = x\cdot\Phi(x)$$"]

    %% Styling
    classDef root fill:#f5f5f5,stroke:#666,stroke-width:2px,color:#333,rx:5px,ry:5px
    classDef category fill:#e1f5fe,stroke:#0277bd,color:#0277bd,rx:5px,ry:5px
    classDef function fill:#fff,stroke:#999,color:#333,rx:5px,ry:5px,padding:15px

    class Root root
    class SigClass,ReLUClass,OtherClass category
    class Sigmoid,Tanh,ReLU,LeakyReLU,PReLU,Softmax,GELU function
```

| Function      | Formula                                               | Characteristics                          |
|---------------|-------------------------------------------------------|------------------------------------------|
| ReLU          | $f(x) = \max(0,x)$                                    | Computationally efficient, sparse activation |
| Leaky ReLU    | $f(x) = \max(0.01x, x)$                                | Prevents dying ReLU problem              |

---

# Forward Propagation

<!-- Replace image with a forward propagation diagram -->
```mermaid
flowchart LR
    %% Forward pass nodes
    x["Input<br/>X"] --> z1["Forward<br/>Propagation"]
    z1 --> a1["Hidden<br/>Activation"]
    a1 --> z2["Forward<br/>Propagation"]
    z2 --> a2["Output<br/>Activation"]
    a2 --> yhat["Prediction<br/>ŷ"]

    %% Error calculation
    yhat --> error["Loss<br/>L(ŷ,y)"]
    y["Actual<br/>y"] --> error

    %% Backward pass nodes with correct LaTeX notation
    error --> gradL["$$\frac{\partial L}{\partial \hat{y}}$$"]
    gradL --> gradZ2["$$\frac{\partial L}{\partial Z^{[2]}}$$"]
    gradZ2 --> gradW2["$$\frac{\partial L}{\partial W^{[2]}}$$"]
    gradZ2 --> gradA1["$$\frac{\partial L}{\partial A^{[1]}}$$"]
    gradA1 --> gradZ1["$$\frac{\partial L}{\partial Z^{[1]}}$$"]
    gradZ1 --> gradW1["$$\frac{\partial L}{\partial W^{[1]}}$$"]

    %% Parameter update
    gradW1 --> update1["Update W¹"]
    gradW2 --> update2["Update W²"]

    %% Style
    classDef input fill:#e3f2fd,stroke:#0d47a1,color:#0d47a1,rx:5px,ry:5px
    classDef forward fill:#e8f5e9,stroke:#1b5e20,color:#1b5e20,rx:5px,ry:5px
    classDef hidden fill:#f3e5f5,stroke:#4a148c,color:#4a148c,rx:5px,ry:5px
    classDef error fill:#ffebee,stroke:#c62828,color:#c62828,rx:5px,ry:5px
    classDef grad fill:#fff3e0,stroke:#e65100,color:#e65100,rx:5px,ry:5px,padding:10px
    classDef update fill:#e1f5fe,stroke:#01579b,color:#01579b,rx:5px,ry:5px

    class x,y input
    class z1,z2 forward
    class a1,a2 hidden
    class error error
    class gradL,gradA1,gradW1,gradW2,gradZ1,gradZ2 grad
    class update1,update2,yhat update
```

For each layer $l$:

$$Z^{[l]} = W^{[l]} \cdot A^{[l-1]} + b^{[l]}$$
$$A^{[l]} = g^{[l]}(Z^{[l]})$$

Where:

- $W^{[l]}$ = weights matrix
- $b^{[l]}$ = bias vector

---

# Forward Propagation

<!-- Use the same forward propagation diagram -->
```mermaid
flowchart LR
    %% Set up the structure of forward propagation
    Input["Input<br/>X"] --> WeightSum1["Weighted Sum<br/>$$Z^{[1]} = W^{[1]}X + b^{[1]}$$"]
    WeightSum1 --> Activation1["Activation<br/>$$A^{[1]} = g(Z^{[1]})$$"]
    Activation1 --> WeightSum2["Weighted Sum<br/>$$Z^{[2]} = W^{[2]}A^{[1]} + b^{[2]}$$"]
    WeightSum2 --> Activation2["Activation<br/>$$A^{[2]} = g(Z^{[2]})$$"]
    Activation2 --> Output["Output<br/>ŷ = A²"]

    %% Add layer labels above
    LayerInput["Input Layer"] -..- Input
    LayerHidden["Hidden Layer"] -..- Activation1
    LayerOutput["Output Layer"] -..- Output

    %% Add detailed labels below for weights
    Weights1["Weights W¹<br/>bias b¹"] -..- WeightSum1
    Weights2["Weights W²<br/>bias b²"] -..- WeightSum2

    %% Styling
    classDef input fill:#e3f2fd,stroke:#0d47a1,color:#0d47a1
    classDef forward fill:#e8f5e9,stroke:#1b5e20,color:#1b5e20,padding:10px
    classDef hidden fill:#f3e5f5,stroke:#4a148c,color:#4a148c,padding:10px
    classDef output fill:#fff3e0,stroke:#e65100,color:#e65100

    class Input input
    class WeightSum1,WeightSum2 forward
    class Activation1,Activation2 hidden
    class Output output
    class LayerInput,LayerHidden,LayerOutput layer
    class Weights1,Weights2 weights
```

Where:

- $g^{[l]}$ = activation function
- $A^{[l]}$ = activation output

---

# Backpropagation: Learning Process

<!-- Replace image with a backpropagation diagram -->
```mermaid
flowchart LR
    FP[Forward Pass]
    EC[Error Calculation]
    BP[Backward Pass]
    UP[Update Parameters]
    FP --> EC
    EC --> BP
    BP --> UP
```

1. **Forward pass**: Compute predictions
   - Process inputs through the network
2. **Error calculation**: Compare with ground truth

---

# Backpropagation: Learning Process

<!-- Use the same backpropagation diagram -->
```mermaid
flowchart LR
    %% Forward pass nodes
    x["Input<br/>X"] --> z1["Forward<br/>Propagation"]
    z1 --> a1["Hidden<br/>Activation"]
    a1 --> z2["Forward<br/>Propagation"]
    z2 --> a2["Output<br/>Activation"]
    a2 --> yhat["Prediction<br/>ŷ"]

    %% Error calculation
    yhat --> error["Loss<br/>L(ŷ,y)"]
    y["Actual<br/>y"] --> error

    %% Backward pass nodes with correct LaTeX notation
    error --> gradL["$$\frac{\partial L}{\partial \hat{y}}$$"]
    gradL --> gradZ2["$$\frac{\partial L}{\partial Z^{[2]}}$$"]
    gradZ2 --> gradW2["$$\frac{\partial L}{\partial W^{[2]}}$$"]
    gradZ2 --> gradA1["$$\frac{\partial L}{\partial A^{[1]}}$$"]
    gradA1 --> gradZ1["$$\frac{\partial L}{\partial Z^{[1]}}$$"]
    gradZ1 --> gradW1["$$\frac{\partial L}{\partial W^{[1]}}$$"]

    %% Parameter update
    gradW1 --> update1["Update W¹"]
    gradW2 --> update2["Update W²"]

    %% Style
    classDef input fill:#e3f2fd,stroke:#0d47a1,color:#0d47a1,rx:5px,ry:5px
    classDef forward fill:#e8f5e9,stroke:#1b5e20,color:#1b5e20,rx:5px,ry:5px
    classDef hidden fill:#f3e5f5,stroke:#4a148c,color:#4a148c,rx:5px,ry:5px
    classDef error fill:#ffebee,stroke:#c62828,color:#c62828,rx:5px,ry:5px
    classDef grad fill:#fff3e0,stroke:#e65100,color:#e65100,rx:5px,ry:5px,padding:10px
    classDef update fill:#e1f5fe,stroke:#01579b,color:#01579b,rx:5px,ry:5px

    class x,y input
    class z1,z2 forward
    class a1,a2 hidden
    class error error
    class gradL,gradA1,gradW1,gradW2,gradZ1,gradZ2 grad
    class update1,update2,yhat update
```

3. **Backward pass**: Compute gradients
4. **Parameter update**: Adjust weights and biases

$$W^{[l]} = W^{[l]} - \alpha \frac{\partial J}{\partial W^{[l]}}$$
$$b^{[l]} = b^{[l]} - \alpha \frac{\partial J}{\partial b^{[l]}}$$

---

# Loss Functions

| Task                     | Loss Function         | Formula |
|--------------------------|-----------------------|---------|
| Regression               | Mean Squared Error    | $\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$ |
| Binary Classification    | Binary Cross-Entropy  | $-\frac{1}{n}\sum_{i=1}^{n}[y_i\log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$ |

---

# Loss Functions

| Task                         | Loss Function             | Formula |
|------------------------------|---------------------------|---------|
| Multi-class Classification   | Categorical Cross-Entropy | $-\frac{1}{n}\sum_{i=1}^{n}\sum_{j=1}^{m}y_{ij}\log(\hat{y}_{ij})$ |

- Loss guides the learning process
- Different tasks use specialized error measurements
- Optimization aims to minimize loss

---

# Universal Approximation Theorem

<!-- Replace image with a UAT diagram -->
```mermaid
flowchart LR
    %% Input and output nodes
    Input[("Input<br/>Features")] --> Hidden
    Hidden --> Output[("Output<br/>Prediction")]

    %% Hidden layer with multiple neurons
    subgraph Hidden["Single Hidden Layer"]
        N1[("Neuron 1")]
        N2[("Neuron 2")]
        N3[("...")]
        N4[("Neuron n")]
    end

    %% Annotation for the theorem
    Hidden -.- Theorem["Can approximate<br/>any continuous function"]

    %% Styling
    classDef node fill:#f5f5f5,stroke:#666,stroke-width:1px,color:#333,rx:5px,ry:5px
    classDef hidden fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20,rx:10px,ry:10px
    classDef io fill:#e3f2fd,stroke:#1565c0,color:#0d47a1,rx:15px,ry:15px
    classDef theorem fill:none,stroke:none,color:#d32f2f,font-style:italic,font-weight:bold

    class Input,Output io
    class Hidden hidden
    class N1,N2,N3,N4 node
    class Theorem theorem
```

---

# Universal Approximation Theorem

> "A feedforward network with a single hidden layer containing a finite number of neurons can approximate any continuous function, under mild assumptions on the activation function."

- The theoretical foundation for MLP capabilities

---

# Universal Approximation Theorem

<!-- Use the same UAT diagram -->
```mermaid
flowchart LR
    %% Input and output nodes
    Input[("Input<br/>Features")] --> Hidden
    Hidden --> Output[("Output<br/>Prediction")]

    %% Hidden layer with multiple neurons
    subgraph Hidden["Single Hidden Layer"]
        N1[("Neuron 1")]
        N2[("Neuron 2")]
        N3[("...")]
        N4[("Neuron n")]
    end

    %% Annotation for the theorem
    Hidden -.- Theorem["Can approximate<br/>any continuous function"]

    %% Styling
    classDef node fill:#f5f5f5,stroke:#666,stroke-width:1px,color:#333,rx:5px,ry:5px
    classDef hidden fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20,rx:10px,ry:10px
    classDef io fill:#e3f2fd,stroke:#1565c0,color:#0d47a1,rx:15px,ry:15px
    classDef theorem fill:none,stroke:none,color:#d32f2f,font-style:italic,font-weight:bold

    class Input,Output io
    class Hidden hidden
    class N1,N2,N3,N4 node
    class Theorem theorem
```

- More complex functions may require more neurons
- Practical implementations must balance capacity and training challenges

---

# Visualizing Decision Boundaries

<!-- Replace image with the same diagram used for forward propagation -->
```mermaid
flowchart LR
    %% Set up the structure of forward propagation
    Input["Input<br/>X"] --> WeightSum1["Weighted Sum<br/>$$Z^{[1]} = W^{[1]}X + b^{[1]}$$"]
    WeightSum1 --> Activation1["Activation<br/>$$A^{[1]} = g(Z^{[1]})$$"]
    Activation1 --> WeightSum2["Weighted Sum<br/>$$Z^{[2]} = W^{[2]}A^{[1]} + b^{[2]}$$"]
    WeightSum2 --> Activation2["Activation<br/>$$A^{[2]} = g(Z^{[2]})$$"]
    Activation2 --> Output["Output<br/>ŷ = A²"]

    %% Add layer labels above
    LayerInput["Input Layer"] -..- Input
    LayerHidden["Hidden Layer"] -..- Activation1
    LayerOutput["Output Layer"] -..- Output

    %% Add detailed labels below for weights
    Weights1["Weights W¹<br/>bias b¹"] -..- WeightSum1
    Weights2["Weights W²<br/>bias b²"] -..- WeightSum2

    %% Styling
    classDef input fill:#e3f2fd,stroke:#0d47a1,color:#0d47a1
    classDef forward fill:#e8f5e9,stroke:#1b5e20,color:#1b5e20,padding:10px
    classDef hidden fill:#f3e5f5,stroke:#4a148c,color:#4a148c,padding:10px
    classDef output fill:#fff3e0,stroke:#e65100,color:#e65100

    class Input input
    class WeightSum1,WeightSum2 forward
    class Activation1,Activation2 hidden
    class Output output
    class LayerInput,LayerHidden,LayerOutput layer
    class Weights1,Weights2 weights
```

- **Linear boundaries**: Single-layer perceptrons
  - Separate data with straight lines
- **Non-linear boundaries**: MLPs with hidden layers
  - Can form complex separation surfaces

---

# Visualizing Decision Boundaries

<!-- Use the same forward propagation diagram -->
```mermaid
flowchart LR
    %% Set up the structure of forward propagation
    Input["Input<br/>X"] --> WeightSum1["Weighted Sum<br/>$$Z^{[1]} = W^{[1]}X + b^{[1]}$$"]
    WeightSum1 --> Activation1["Activation<br/>$$A^{[1]} = g(Z^{[1]})$$"]
    Activation1 --> WeightSum2["Weighted Sum<br/>$$Z^{[2]} = W^{[2]}A^{[1]} + b^{[2]}$$"]
    WeightSum2 --> Activation2["Activation<br/>$$A^{[2]} = g(Z^{[2]})$$"]
    Activation2 --> Output["Output<br/>ŷ = A²"]

    %% Add layer labels above
    LayerInput["Input Layer"] -..- Input
    LayerHidden["Hidden Layer"] -..- Activation1
    LayerOutput["Output Layer"] -..- Output

    %% Add detailed labels below for weights
    Weights1["Weights W¹<br/>bias b¹"] -..- WeightSum1
    Weights2["Weights W²<br/>bias b²"] -..- WeightSum2

    %% Styling
    classDef input fill:#e3f2fd,stroke:#0d47a1,color:#0d47a1
    classDef forward fill:#e8f5e9,stroke:#1b5e20,color:#1b5e20,padding:10px
    classDef hidden fill:#f3e5f5,stroke:#4a148c,color:#4a148c,padding:10px
    classDef output fill:#fff3e0,stroke:#e65100,color:#e65100

    class Input input
    class WeightSum1,WeightSum2 forward
    class Activation1,Activation2 hidden
    class Output output
    class LayerInput,LayerHidden,LayerOutput layer
    class Weights1,Weights2 weights
```

- **Complexity increases** with deeper architectures
- Explore an interactive demo at [perceptron.marcr.xyz](https://perceptron.marcr.xyz)

---

# Quick Quiz: Test Your Knowledge

## Which of these problems can a single-layer perceptron solve?

A) XOR problem
B) Linear classification
C) Image recognition
D) All of the above

*Use the poll feature to submit your answer!*

---

# Practical Implementation Challenges

## What's your biggest challenge with neural networks?

- Understanding the math
- Choosing the right architecture
- Overfitting/underfitting
- Computational resources
- Interpreting results

*Share your thoughts!*

---

# Thank You

## Contact Information

- Email: [hi@marcr.xyz](mailto:hi@marcr.xyz)

## Resources

- Interactive Demo: [perceptron.marcr.xyz](https://perceptron.marcr.xyz)
- Slides: [github.com/mabreyes/dlsu-lecture-slides](https://github.com/mabreyes/dlsu-lecture-slides)
