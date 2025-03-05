---
marp: true
theme: default
paginate: true
math: mathjax
---

# Multilayer Perceptrons: Unraveling the Neural Network Powerhouse

---

## Learning Objectives

By the end of this presentation, you will be able to:

- Understand the basic architecture and mathematics of Multilayer Perceptrons
- Explain the significance of activation functions and backpropagation
- Compare implementation approaches using raw Python vs. frameworks like TensorFlow
- Evaluate when to use different implementation strategies
- Apply MLPs to solve real-world problems

---

## Table of Contents

### Theory
- What is an MLP?
- Mathematical representation
- Architecture
- Backpropagation

### Practice
- Implementation approaches
- Python vs TensorFlow
- Applications
- Interactive challenges
- Future directions

---

## Multilayer Perceptrons: The Building Blocks of Deep Learning

```mermaid
%%{init: {"theme": "default"}}%%
flowchart LR
    subgraph "MLP Architecture"
        A["Input Layer"] --> B["Hidden Layers"]
        B --> C["Output Layer"]
    end
    style A fill:#ff9f43
    style B fill:#2ecc71
    style C fill:#54a0ff
```

---

## What is a Multilayer Perceptron?

- Definition: An MLP is a class of feedforward artificial neural network
- Key features:
  - Multiple layers of perceptrons
  - Nonlinear activation functions
  - Fully connected architecture

---

## MLP: Basic Structure

```mermaid
%%{init: {"theme": "default"}}%%
flowchart LR
    subgraph "MLP Architecture" 
        subgraph "Input Layer"
            I1((I1))
            I2((I2))
            I3((I3))
        end
        subgraph "Hidden Layer"
            H1((H1))
            H2((H2))
            H3((H3))
        end
        subgraph "Output Layer"
            O1((O1))
            O2((O2))
        end
        I1 & I2 & I3 --> H1 & H2 & H3 --> O1 & O2
    end
    style I1 fill:#ff9f43
    style I2 fill:#ff9f43
    style I3 fill:#ff9f43
    style H1 fill:#2ecc71
    style H2 fill:#2ecc71
    style H3 fill:#2ecc71
    style O1 fill:#54a0ff
    style O2 fill:#54a0ff
```

---

## Mathematical Representation of an MLP Neuron

$$f(x) = \phi\left(\sum_{i=1}^n w_i x_i + b\right)$$

Where:
- $f(x)$ is the output
- $\phi$ is the activation function
- $w_i$ are weights
- $x_i$ are inputs
- $b$ is the bias

---

## Common Activation Functions (1/2)

### ReLU
$$\phi(z) = \max(0, z)$$

### Sigmoid
$$\phi(z) = \frac{1}{1 + e^{-z}}$$

```mermaid
%%{init: {"theme": "default"}}%%
flowchart LR
    subgraph "Activation Functions"
        A["Input"] --> B["Activation Function"] --> C["Output"]
    end
    style B fill:#ff9f43
```

---

## Common Activation Functions (2/2)

### Tanh
$$\phi(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$

### Leaky ReLU
$$\phi(z) = \max(\alpha z, z), \alpha = 0.01$$

```mermaid
%%{init: {"theme": "default"}}%%
flowchart LR
    subgraph "Activation Functions"
        A["Input"] --> B["Activation Function"] --> C["Output"]
    end
    style B fill:#ff9f43
```

---

## Activation Function in Neural Processing

```mermaid
%%{init: {"theme": "default"}}%%
flowchart LR
    subgraph "Single Neuron"
        A["Input"] --> B["Weighted Sum"]
        B --> C["Activation Function"]
        C --> D["Output"]
    end
    style C fill:#ff9f43
```

---

## Why MLP?

- Advantages of MLPs:
  1. Universal function approximators
  2. Ability to learn nonlinear relationships
  3. Scalability to high-dimensional data
  4. Adaptability to various problem types

---

## MLP Applications Overview

```mermaid
%%{init: {"theme": "default"}}%%
flowchart TD
    subgraph "MLP Applications"
        A["Input Data"] --> B["MLP"]
        B --> C["Classification"]
        B --> D["Regression"]
        B --> E["Pattern Recognition"]
    end
    style B fill:#2ecc71
```

---

## Architecture of MLP

- Layers:
  1. Input Layer: Receives the raw data
  2. Hidden Layer(s): Performs the complex data processing
  3. Output Layer: Produces the final result

```mermaid
%%{init: {"theme": "default"}}%%
flowchart LR
    subgraph "MLP Architecture" 
        A["Input Layer"] --> B["Hidden Layers"]
        B --> C["Output Layer"]
    end
    style A fill:#ff9f43
    style B fill:#2ecc71
    style C fill:#54a0ff
```

---

## MLP Architecture: Detailed View

```mermaid
%%{init: {"theme": "default"}}%%
flowchart LR
    subgraph "MLP Architecture" 
        subgraph "Input Layer"
            I1((I1))
            I2((I2))
            I3((I3))
        end
        subgraph "Hidden Layer"
            H1((H1))
            H2((H2))
            H3((H3))
            H4((H4))
        end
        subgraph "Output Layer"
            O1((O1))
            O2((O2))
        end
        I1 & I2 & I3 --> H1 & H2 & H3 & H4 --> O1 & O2
    end
    style I1 fill:#ff9f43
    style I2 fill:#ff9f43
    style I3 fill:#ff9f43
    style H1 fill:#2ecc71
    style H2 fill:#2ecc71
    style H3 fill:#2ecc71
    style H4 fill:#2ecc71
    style O1 fill:#54a0ff
    style O2 fill:#54a0ff
```

---

## MLP Architecture Visualization

```mermaid
%%{init: {"theme": "default"}}%%
flowchart LR
    subgraph "MLP Architecture" 
        subgraph "Input Layer"
            I1((I1))
            I2((I2))
            I3((I3))
        end
        subgraph "Hidden Layer"
            H1((H1))
            H2((H2))
            H3((H3))
            H4((H4))
        end
        subgraph "Output Layer"
            O1((O1))
            O2((O2))
        end
        I1 --> H1 & H2 & H3 & H4
        I2 --> H1 & H2 & H3 & H4
        I3 --> H1 & H2 & H3 & H4
        H1 & H2 & H3 & H4 --> O1 & O2
    end
    style I1 fill:#ff9f43
    style I2 fill:#ff9f43
    style I3 fill:#ff9f43
    style H1 fill:#2ecc71
    style H2 fill:#2ecc71
    style H3 fill:#2ecc71
    style H4 fill:#2ecc71
    style O1 fill:#54a0ff
    style O2 fill:#54a0ff
```

---

## How MLPs Work

1. Forward Propagation: Input data flows through the network
2. Loss Calculation: Error between prediction and actual output is measured
3. Backpropagation: Error is propagated backward through the network
4. Weight Update: Parameters are updated to reduce error

---

## MLP Training Process

```mermaid
%%{init: {"theme": "default"}}%%
flowchart LR
    subgraph "MLP Training Cycle"
        A["Input"] --> B["Forward Pass"]
        B --> C["Calculate Loss"]
        C --> D["Backward Pass"]
        D --> E["Update Weights"]
        E -.-> B
    end
    style B fill:#54a0ff
    style D fill:#ff9f43
```

---

## Understanding Backpropagation

**Mathematical representation:**
$$\frac{\partial L}{\partial w_{ij}} = \frac{\partial L}{\partial a_j} \cdot \frac{\partial a_j}{\partial z_j} \cdot \frac{\partial z_j}{\partial w_{ij}}$$

Where:
- $L$ is the loss
- $a_j$ is the activation of neuron j
- $z_j$ is the weighted sum input to neuron j
- $w_{ij}$ is the weight from neuron i to neuron j

---

## Backpropagation Intuition

### Chain Rule Application
Backpropagation applies the chain rule of calculus to compute gradients efficiently through the network.

### Error Attribution
It determines how much each weight contributed to the final error.

---

## Backpropagation Flow

```mermaid
%%{init: {"theme": "default"}}%%
flowchart RL
    subgraph "Backpropagation"
        A["Error"] --> B["Output Layer"]
        B --> C["Hidden Layer"]
        C --> D["Input Layer"]
    end
    style A fill:#ff6b6b
    style B fill:#54a0ff
    style C fill:#2ecc71
    style D fill:#ff9f43
```

---

## Loss Functions in MLPs: Classification

- Cross-Entropy Loss
$$L = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)$$

- Binary Cross-Entropy
$$L = -y\log(\hat{y}) - (1-y)\log(1-\hat{y})$$

```mermaid
%%{init: {"theme": "default"}}%%
flowchart LR
    subgraph "Loss Calculation"
        A["Prediction"] --> B["Loss Function"]
        C["Ground Truth"] --> B
        B --> D["Loss Value"]
    end
    style B fill:#ff6b6b
```

---

## Loss Functions in MLPs: Regression

- Mean Squared Error
$$L = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

- Mean Absolute Error
$$L = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

```mermaid
%%{init: {"theme": "default"}}%%
flowchart LR
    subgraph "Loss Calculation"
        A["Prediction"] --> B["Loss Function"]
        C["Ground Truth"] --> B
        B --> D["Loss Value"]
    end
    style B fill:#ff6b6b
```

---

## Applications of MLP (1/2)

- Image and Speech Recognition
- Natural Language Processing
- Financial Forecasting
- Medical Diagnosis
- Robotics and Control Systems

```mermaid
%%{init: {"theme": "default"}}%%
flowchart TD
    subgraph "MLP Applications"
        A["Input Data"] --> B["MLP"]
        B --> C["Various Tasks"]
    end
    style B fill:#2ecc71
```

---

## Applications of MLP: Visual Examples

```mermaid
%%{init: {"theme": "default"}}%%
flowchart TD
    subgraph "MLP Applications by Domain"
        A["MLP"] --> B["Computer Vision"]
        A --> C["Audio Processing"]
        A --> D["Finance"]
        A --> E["Healthcare"]
    end
    style A fill:#2ecc71
```

---

## Applications of MLP (2/2)

```mermaid
%%{init: {"theme": "default"}}%%
mindmap
  root((MLP Applications))
    Computer Vision
      Image Classification
      Object Detection
    Natural Language Processing
      Sentiment Analysis
      Language Translation
    Finance
      Stock Prediction
      Fraud Detection
    Healthcare
      Disease Diagnosis
      Drug Discovery
    Robotics
      Motion Planning
      Sensor Fusion
```

---

## From Theory to Practice: MLP Implementation Approaches

Now that we understand the theory behind MLPs, let's explore how to implement them:

1. **From Scratch** - Build every component manually
   - Full control and understanding of each step
   - Educational but labor-intensive

```mermaid
%%{init: {"theme": "default"}}%%
flowchart TD
    subgraph "Implementation Approaches"
        A["MLP Implementation"] --> B["From Scratch"]
        A --> C["Using Frameworks"]
    end
    style B fill:#ff9f43
    style C fill:#54a0ff
```

---

## Implementation Approaches (continued)

2. **Using Frameworks** - Leverage high-level APIs
   - Faster development and optimization
   - Abstracts away implementation details

```mermaid
%%{init: {"theme": "default"}}%%
flowchart TD
    subgraph "Implementation Approaches"
        A["MLP Implementation"] --> B["From Scratch"]
        A --> C["Using Frameworks"]
    end
    style B fill:#ff9f43
    style C fill:#54a0ff
```

---

## Implementation Showdown: Raw Python vs TensorFlow

- We'll implement the same MLP in two ways:
  1. Pure Python to see the inner workings
  2. TensorFlow to leverage abstraction
- Key comparison points:
  - Code complexity
  - Performance
  - Flexibility

```mermaid
%%{init: {"theme": "default"}}%%
flowchart LR
    subgraph "Implementation Methods"
        A["Same MLP Model"] --> B["Python from Scratch"]
        A --> C["TensorFlow"]
    end
    style B fill:#ff9f43
    style C fill:#54a0ff
```

---

## Standard Approach: Building an MLP with TensorFlow

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(64, activation='relu', input_shape=(784,)),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', 
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

---

## TensorFlow Implementation (continued)

```python
# Train the model (assuming x_train and y_train are defined)
history = model.fit(x_train, y_train, epochs=10, validation_split=0.2)
```

```mermaid
%%{init: {"theme": "default"}}%%
flowchart TD
    subgraph "TensorFlow Training"
        A["Define Model"] --> B["Compile Model"]
        B --> C["Fit Model"]
        C --> D["Evaluate Results"]
    end
    style C fill:#54a0ff
```

---

## Visual Example: MNIST Classification with MLP

### MNIST Dataset
- 28×28 pixel grayscale images
- 10 classes (digits 0-9)
- 60,000 training, 10,000 test images

```mermaid
%%{init: {"theme": "default"}}%%
flowchart LR
    subgraph "MNIST Classification with MLP"
        A["Image Input"] --> B["MLP"]
        B --> C["Digit Classification"]
    end
    style B fill:#2ecc71
```

---

## MNIST Classification: Code Example

```python
# Flatten 28x28 images to 784-feature vectors
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
# 97% accuracy in seconds!
```

---

## Under the Hood: Implementing an MLP from Scratch (1/3)

```python
import numpy as np

class ScratchMLP:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
    
    def relu(self, Z):
        return np.maximum(0, Z)
```

---

## Under the Hood: Implementing an MLP from Scratch (2/3)
    
```python
    def forward(self, X):
        # Forward propagation
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        return self.Z2  # Linear output for regression
    
    def mse_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred)**2)
```

---

## Under the Hood: Implementing an MLP from Scratch (3/3)
    
```python
    def train(self, X, y, epochs=1000, lr=0.01):
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(X)
            
            # Backward pass (simplified)
            dL_dZ2 = 2*(y_pred - y)/len(y)
            dZ2_dW2 = self.A1.T
            dW2 = np.dot(dZ2_dW2, dL_dZ2)
            db2 = np.sum(dL_dZ2, axis=0, keepdims=True)
            
            dA1_dZ1 = self.Z1 > 0
            dZ1_dW1 = X.T
            dW1 = np.dot(dZ1_dW1, np.dot(dL_dZ2, self.W2.T) * dA1_dZ1)
            db1 = np.sum(np.dot(dL_dZ2, self.W2.T) * dA1_dZ1, axis=0)
            
            # Update parameters
            self.W2 -= lr * dW2
            self.b2 -= lr * db2
            self.W1 -= lr * dW1
            self.b1 -= lr * db1
```

---

## Scratch Implementation Key Points

50 lines of pure Python! We initialize weights manually, implement ReLU and MSE from scratch, and hand-code backpropagation using chain rule.

---

## Scratch Implementation Workflow

```mermaid
%%{init: {"theme": "default"}}%%
flowchart TD
    subgraph "Scratch Implementation"
        A["Manual Weight Initialization"] --> B["Forward Propagation"]
        B --> C["Loss Calculation"]
        C --> D["Manual Backpropagation"]
        D --> E["Weight Updates"]
    end
    style A fill:#ff9f43
    style D fill:#54a0ff
```

---

## Demo: Testing Both Implementations (1/2)

```python
# Generate synthetic data for XOR problem
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])  

# Train our scratch model
mlp = ScratchMLP(2, 4, 1)
print("Initial loss:", mlp.mse_loss(y, mlp.forward(X)))
mlp.train(X, y, epochs=1000)
print("Final loss:", mlp.mse_loss(y, mlp.forward(X)))
```

---

## Demo: Testing Both Implementations (2/2)

```python
# Now with TensorFlow
tf_mlp = Sequential([
    Dense(4, activation='relu', input_shape=(2,)),
    Dense(1)
])
tf_mlp.compile(optimizer='adam', loss='mse')
history = tf_mlp.fit(X, y, epochs=1000, verbose=0)
```

---

## XOR Problem Visualization

### XOR Logic
- Input (0,0) → Output 0
- Input (0,1) → Output 1
- Input (1,0) → Output 1
- Input (1,1) → Output 0

```mermaid
%%{init: {"theme": "default"}}%%
flowchart TD
    subgraph "XOR Problem"
        A["MLP Model"] --> B["XOR Classification"]
    end
    style A fill:#2ecc71
```

---

## Why XOR is Important for Neural Networks

### Why XOR is important
XOR is not linearly separable, requiring a hidden layer to solve.

```mermaid
%%{init: {"theme": "default"}}%%
flowchart TD
    subgraph "XOR Problem"
        A["(0,0) → 0"] 
        B["(0,1) → 1"]
        C["(1,0) → 1"]
        D["(1,1) → 0"]
    end
    style A fill:#ff9f43
    style B fill:#54a0ff
    style C fill:#54a0ff
    style D fill:#ff9f43
```

---

## Interactive Comparison Challenge

| Metric | Scratch MLP | TensorFlow MLP |
| :-- | :-- | :-- |
| Lines of Code | 50 | 5 |
| Training Time | ~2s | ~0.5s |
| Final Loss | 0.12 | 0.08 |
| Extensibility | Low | High |

---

## Discussion Points for Comparison

- **Team A**: Defend the scratch version!
- **Team B**: Champion TensorFlow!
- **Discussion point**: Why did TensorFlow achieve lower loss despite similar architecture? (Hint: Adam optimizer vs our basic SGD)

---

## Behind the Scenes - What TensorFlow Adds

```mermaid
%%{init: {"theme": "default"}}%%
flowchart TD
    subgraph "TensorFlow Advantages"
        A["Input"] --> B["Autograd"]
        B --> C["Optimizers"]
        C --> D["GPU Acceleration"]
        D --> E["Distributed Training"]
    end
    style B fill:#2ecc71
    style D fill:#54a0ff
```

---

## TensorFlow's Advantages

TensorFlow's advantages include automatic differentiation, advanced optimizers like Adam, GPU acceleration, and distributed training capabilities.

---

## Optimizers in Modern Deep Learning (1/2)

### Basic SGD (Our Scratch Version)
$$\theta_{t+1} = \theta_t - \eta \nabla J(\theta_t)$$

### Momentum
$$v_{t+1} = \gamma v_t + \eta \nabla J(\theta_t)$$
$$\theta_{t+1} = \theta_t - v_{t+1}$$

```mermaid
%%{init: {"theme": "default"}}%%
flowchart LR
    subgraph "Optimization Techniques"
        A["Gradients"] --> B["Optimizer"]
        B --> C["Parameter Updates"]
    end
    style B fill:#ff9f43
```

---

## Optimizers in Modern Deep Learning (2/2)

### RMSProp
$$E[g^2]_t = 0.9 E[g^2]_{t-1} + 0.1 g_t^2$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} g_t$$

### Adam (Used by TensorFlow)
Combines momentum and RMSProp with bias correction

```mermaid
%%{init: {"theme": "default"}}%%
flowchart LR
    subgraph "Optimization Techniques"
        A["Gradients"] --> B["Optimizer"]
        B --> C["Parameter Updates"]
    end
    style B fill:#ff9f43
```

---

## Interactive Challenge - Optimize the MLP

- Challenge: Improve the model's accuracy
- Options:
  1. Add more layers
  2. Increase neurons in hidden layers
  3. Change activation functions
  4. Adjust learning rate

```mermaid
%%{init: {"theme": "default"}}%%
flowchart TD
    subgraph "MLP Optimization"
        A["Original MLP"] --> B["Modified Architecture"]
        B --> C["Improved Performance"]
    end
    style B fill:#2ecc71
```

---

## Optimization Implementation

- Let's implement these optimizations in both our scratch model and TensorFlow!

---

## Interactive Activity - Gradient Calculation

**Problem**:
Given:

$$\text{Loss} = (y - (W_2 \cdot \text{ReLU}(W_1 X + b_1) + b_2))^2$$

Calculate 
$$\frac{\partial \text{Loss}}{\partial W_1}$$ 

for single sample X=[0.5, 0.3]

---

## Gradient Calculation Hint

**Hint**: Use chain rule through both layers! (ReLU derivative is 1 when input >0 else 0)

```mermaid
%%{init: {"theme": "default"}}%%
flowchart RL
    subgraph "Backpropagation"
        A["Error"] --> B["Output Layer"]
        B --> C["Hidden Layer"]
        C --> D["Input Layer"]
    end
    style A fill:#ff6b6b
    style B fill:#54a0ff
    style C fill:#2ecc71
    style D fill:#ff9f43
```

---

## Gradient Calculation Solution

$$\frac{\partial \text{Loss}}{\partial W_1} = -2(y - (W_2 \cdot \text{ReLU}(W_1 X + b_1) + b_2)) \cdot W_2 \cdot \frac{\partial \text{ReLU}(W_1 X + b_1)}{\partial (W_1 X + b_1)} \cdot X^T$$

Where:
- $$\frac{\partial \text{ReLU}(z)}{\partial z} = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z \leq 0 \end{cases}$$

---

## The Verdict - When to Use Each Approach

```mermaid
%%{init: {"theme": "default"}}%%
pie
    title "MLP Implementation Choices"
    "Education/Learning" : 30
    "Prototyping" : 60
    "Production" : 10
    "Research" : 20
```

---

## Practical Implementation Recommendations

- **Scratch coding**: Best for education and understanding fundamentals
- **TensorFlow**: Dominates real-world use cases
- **Knowing both**: Makes you a better engineer!

```mermaid
%%{init: {"theme": "default"}}%%
flowchart LR
    subgraph "Implementation Choice"
        A["Learning"] --> B["Scratch"]
        C["Production"] --> D["Framework"]
    end
    style B fill:#ff9f43
    style D fill:#54a0ff
```

---

## Handling Overfitting in MLPs (1/2)

### Common Techniques
- Dropout: Randomly disable neurons
- L1/L2 Regularization: Penalize large weights
- Early Stopping: Stop when validation error increases

```mermaid
%%{init: {"theme": "default"}}%%
flowchart LR
    subgraph "Regularization"
        A["MLP Model"] --> B["Regularization Techniques"]
        B --> C["Reduced Overfitting"]
    end
    style B fill:#ff9f43
```

---

## Handling Overfitting in MLPs (2/2)

### Implementation Example
```python
# TensorFlow Dropout Example
model = Sequential([
    Dense(64, activation='relu'),
    Dropout(0.5),  # 50% dropout
    Dense(32, activation='relu'),
    Dropout(0.3),  # 30% dropout
    Dense(10, activation='softmax')
])
```

---

## Future Directions in MLP Development

- Emerging trends:
  1. Transfer Learning
  2. Attention Mechanisms
  3. Neural Architecture Search
  4. Automated hyperparameter optimization
  5. Explainable AI techniques

```mermaid
%%{init: {"theme": "default"}}%%
flowchart TD
    subgraph "Future Directions"
        A["Current MLPs"] --> B["Advanced Techniques"]
        B --> C["Next-Gen Models"]
    end
    style B fill:#2ecc71
```

---

## MLPs vs Other Neural Networks (1/2)

### MLP
- Fully connected layers
- No inherent spatial awareness
- Good for tabular data

### CNN
- Convolutional layers
- Spatial awareness
- Excellent for images

```mermaid
%%{init: {"theme": "default"}}%%
flowchart LR
    subgraph "Neural Network Types"
        A["Input Data"] --> B["Different NN Architectures"]
        B --> C["Task-Specific Outputs"]
    end
    style B fill:#2ecc71
```

---

## MLPs vs Other Neural Networks (2/2)

### RNN
- Recurrent connections
- Temporal awareness
- Good for sequences

### Transformer
- Attention mechanisms
- Parallelizable
- State-of-the-art for NLP

```mermaid
%%{init: {"theme": "default"}}%%
flowchart LR
    subgraph "Neural Network Types"
        A["Input Data"] --> B["Different NN Architectures"]
        B --> C["Task-Specific Outputs"]
    end
    style B fill:#2ecc71
```

---

## Final Challenge - Real-world MLP Design

- Scenario: Design an MLP for a real-world problem
- Teams brainstorm and present their designs:
  - Define architecture
  - Choose implementation approach
  - Plan validation strategy
  - Consider deployment challenges

```mermaid
%%{init: {"theme": "default"}}%%
flowchart TD
    subgraph "MLP Design Challenge"
        A["Problem Definition"] --> B["MLP Design"]
        B --> C["Implementation & Validation"]
    end
    style B fill:#2ecc71
```

---

## Key Insights

1. **Mathematical Foundation**: Both implementations rely on the same underlying principles
2. **Optimization**: TensorFlow's advanced optimizers outperform basic implementations
3. **Ecosystem**: Production needs demand framework tooling
4. **Education**: Scratch code reveals hidden details

---

## Implementation Challenge

**Challenge**: Try modifying the scratch code to add Adam optimizer. Share your attempts with #MLPFromScratch!

```mermaid
%%{init: {"theme": "default"}}%%
flowchart LR
    subgraph "Implementation Challenge"
        A["Scratch MLP"] --> B["Add Adam Optimizer"]
        B --> C["Improved Performance"]
    end
    style B fill:#ff9f43
```

---

## Summary and Conclusion

### What We've Learned
- MLP architecture and mathematics
- Implementation approaches
- Practical applications
- Performance optimization
- Industry best practices

```mermaid
%%{init: {"theme": "default"}}%%
flowchart LR
    subgraph "MLP Knowledge Journey"
        A["Theory"] --> B["Implementation"]
        B --> C["Applications"]
    end
    style A fill:#ff9f43
    style B fill:#2ecc71
    style C fill:#54a0ff
```

---

## Next Steps

### Moving Forward
- Explore deeper architectures
- Apply to your own projects
- Investigate specialized networks
- Join the neural network community
- Contribute to open-source frameworks

---

## References and Further Reading

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
- Chollet, F. (2021). Deep Learning with Python. Manning Publications.
- TensorFlow Documentation: [tensorflow.org](https://www.tensorflow.org)
- Stanford CS231n: Convolutional Neural Networks for Visual Recognition
- Kaggle Competitions: [kaggle.com](https://www.kaggle.com)

---

## Thank You!

Questions? Contact: your.email@example.com
GitHub: [github.com/yourusername](https://github.com/yourusername)

```mermaid
%%{init: {"theme": "default"}}%%
flowchart LR
    subgraph "MLP Architecture" 
        A["Input Layer"] --> B["Hidden Layers"]
        B --> C["Output Layer"]
    end
    style A fill:#ff9f43
    style B fill:#2ecc71
    style C fill:#54a0ff
```

