import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

class SimpleCNN:
    def __init__(self, input_size, kernel_size, kernel_number):
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.kernel_number = kernel_number
        
        self.weights = np.random.randn(kernel_number, kernel_size, kernel_size) / np.sqrt(kernel_size * kernel_size)
        self.bias = np.zeros((kernel_number, 1))
        
    def forward(self, x):
        self.x = x
        batch_size, _, height, width = x.shape
        self.output_height = height - self.kernel_size + 1
        self.output_width = width - self.kernel_size + 1
        
        self.convolved = np.zeros((batch_size, self.kernel_number, self.output_height, self.output_width))
        for i in range(self.output_height):
            for j in range(self.output_width):
                for k in range(self.kernel_number):
                    self.convolved[:, k, i, j] = np.sum(self.x[:, :, i:i+self.kernel_size, j:j+self.kernel_size] * self.weights[k], axis=(1, 2, 3))
                self.convolved[:, :, i, j] += self.bias[:, 0]
        
        self.activations = np.maximum(0, self.convolved)  # ReLU activation
        
        return self.activations
    
    def backward(self, grad_output):
        grad_activations = np.where(self.convolved > 0, grad_output, 0)  # Gradient of ReLU
        
        grad_weights = np.zeros_like(self.weights)
        grad_bias = np.sum(grad_activations, axis=(0, 2, 3)).reshape(-1, 1)
        
        batch_size = grad_output.shape[0]
        grad_input = np.zeros_like(self.x)
        
        for i in range(self.output_height):
            for j in range(self.output_width):
                for k in range(self.kernel_number):
                    grad_weights[k] += np.sum(self.x[:, :, i:i+self.kernel_size, j:j+self.kernel_size] * grad_activations[:, k, i, j].reshape(-1, 1, 1, 1), axis=0)
                    grad_input[:, :, i:i+self.kernel_size, j:j+self.kernel_size] += self.weights[k] * grad_activations[:, k, i, j].reshape(-1, 1, 1, 1)
        
        self.weights -= grad_weights * 0.001  # Learning rate for updating weights
        self.bias -= grad_bias * 0.001  # Learning rate for updating bias
        
        return grad_input



# Load and preprocess MNIST dataset
mnist = fetch_openml("mnist_784")
X, y = mnist["data"], mnist["target"]
X = X.astype(float) / 255.0
y = y.astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# One-hot encode labels
encoder = OneHotEncoder(sparse=False)
y_train_encoded = encoder.fit_transform(y_train.reshape(-1, 1))
y_test_encoded = encoder.transform(y_test.reshape(-1, 1))

# Create the CNN model
input_size = 784
kernel_size = 5
kernel_number = 100
cnn = SimpleCNN(input_size, kernel_size, kernel_number)

# Training parameters
learning_rate = 0.01
num_epochs = 10
batch_size = 32

# Training loop
for epoch in range(num_epochs):
    for i in range(0, len(X_train), batch_size):
        batch_X = X_train[i:i+batch_size].reshape(-1, 1, 28, 28)
        batch_y = y_train_encoded[i:i+batch_size]
        
        # Forward pass
        activations = cnn.forward(batch_X)
        
        # Calculate loss (softmax with cross-entropy)
        exp_scores = np.exp(activations - np.max(activations, axis=1, keepdims=True))
        softmax_probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        loss = -np.sum(batch_y * np.log(softmax_probs)) / batch_size
        
        # Backward pass
        grad_output = (softmax_probs - batch_y) / batch_size
        grad_input = cnn.backward(grad_output)
        
        # Update weights and bias using SGD
        cnn.weights -= learning_rate * cnn.weights
        cnn.bias -= learning_rate * cnn.bias
        
        print(f"Epoch {epoch+1}/{num_epochs}, Batch {i//batch_size+1}/{len(X_train)//batch_size}, Loss: {loss:.4f}")



