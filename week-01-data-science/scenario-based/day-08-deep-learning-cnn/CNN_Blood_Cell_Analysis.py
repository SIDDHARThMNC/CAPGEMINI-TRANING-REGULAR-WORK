"""
CNN for Blood Cell Analysis
Biomedical Image Processing using Convolutional Neural Network

Input: 64×64 grayscale images (1 channel)
Convolutional Layer: 16 filters, 5×5 size
Stride: 1, Padding: Same
Activation: ReLU
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

class BloodCellCNN:
    def __init__(self):
        """
        Initialize CNN for blood cell analysis
        Input: 64×64 grayscale images
        Conv Layer: 16 filters of 5×5, stride=1, padding=same
        """
        self.input_shape = (64, 64, 1)
        self.num_filters = 16
        self.filter_size = 5
        self.stride = 1
        self.padding = 'same'
        
        # Initialize filters (weights) with Xavier initialization
        self.filters = np.random.randn(self.num_filters, self.filter_size, self.filter_size) * np.sqrt(2.0 / (self.filter_size * self.filter_size))
        self.biases = np.zeros(self.num_filters)
        
        print("=" * 60)
        print("CNN Architecture for Blood Cell Analysis")
        print("=" * 60)
        print(f"Input Shape: {self.input_shape}")
        print(f"Number of Filters: {self.num_filters}")
        print(f"Filter Size: {self.filter_size}×{self.filter_size}")
        print(f"Stride: {self.stride}")
        print(f"Padding: {self.padding}")
        print(f"Activation: ReLU")
        print("=" * 60)
    
    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def convolve2d(self, image, kernel):
        """
        Perform 2D convolution with 'same' padding
        """
        return signal.correlate2d(image, kernel, mode='same', boundary='fill')
    
    def forward_pass(self, image):
        """
        Forward pass through the convolutional layer
        
        Parameters:
        - image: Input grayscale image (64×64)
        
        Returns:
        - feature_maps: Output feature maps after convolution and ReLU
        """
        # Ensure image is 64×64
        if image.shape != (64, 64):
            raise ValueError(f"Expected image shape (64, 64), got {image.shape}")
        
        # Initialize output feature maps
        output_height = 64  # Same padding maintains dimensions
        output_width = 64
        feature_maps = np.zeros((self.num_filters, output_height, output_width))
        
        # Apply each filter
        for i in range(self.num_filters):
            # Convolve image with filter
            conv_output = self.convolve2d(image, self.filters[i])
            
            # Add bias
            conv_output += self.biases[i]
            
            # Apply ReLU activation
            feature_maps[i] = self.relu(conv_output)
        
        return feature_maps
    
    def visualize_filters(self):
        """Visualize the learned filters"""
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        fig.suptitle('16 Convolutional Filters (5×5)', fontsize=16)
        
        for i in range(self.num_filters):
            ax = axes[i // 4, i % 4]
            ax.imshow(self.filters[i], cmap='gray')
            ax.set_title(f'Filter {i+1}')
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('cnn_filters.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_feature_maps(self, feature_maps, original_image):
        """Visualize the output feature maps"""
        fig = plt.figure(figsize=(16, 10))
        
        # Show original image
        ax = plt.subplot(4, 5, 1)
        ax.imshow(original_image, cmap='gray')
        ax.set_title('Original Image\n(64×64)', fontsize=10)
        ax.axis('off')
        
        # Show first 16 feature maps
        for i in range(min(16, self.num_filters)):
            ax = plt.subplot(4, 5, i + 2)
            ax.imshow(feature_maps[i], cmap='viridis')
            ax.set_title(f'Feature Map {i+1}', fontsize=9)
            ax.axis('off')
        
        plt.suptitle('CNN Feature Maps - Blood Cell Analysis', fontsize=16)
        plt.tight_layout()
        plt.savefig('cnn_feature_maps.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_output_shape(self):
        """Calculate and display output dimensions"""
        print("\n" + "=" * 60)
        print("Output Shape Calculation")
        print("=" * 60)
        
        input_h, input_w = 64, 64
        
        # With 'same' padding: output_size = input_size
        output_h = input_h
        output_w = input_w
        
        print(f"Input: {input_h}×{input_w}×1")
        print(f"After Conv Layer: {output_h}×{output_w}×{self.num_filters}")
        print(f"\nTotal parameters in Conv Layer:")
        print(f"  Weights: {self.num_filters} × {self.filter_size} × {self.filter_size} = {self.num_filters * self.filter_size * self.filter_size}")
        print(f"  Biases: {self.num_filters}")
        print(f"  Total: {self.num_filters * self.filter_size * self.filter_size + self.num_filters}")
        print("=" * 60)


def generate_synthetic_blood_cell():
    """Generate a synthetic blood cell image for demonstration"""
    image = np.zeros((64, 64))
    
    # Create circular cell
    center = (32, 32)
    radius = 20
    
    for i in range(64):
        for j in range(64):
            dist = np.sqrt((i - center[0])**2 + (j - center[1])**2)
            if dist < radius:
                # Cell interior
                image[i, j] = 0.7 + 0.2 * np.random.randn()
            elif dist < radius + 2:
                # Cell boundary
                image[i, j] = 0.3
            else:
                # Background
                image[i, j] = 0.1 * np.random.randn()
    
    # Add some texture variations
    noise = np.random.randn(64, 64) * 0.05
    image += noise
    
    # Normalize to [0, 1]
    image = np.clip(image, 0, 1)
    
    return image


def generate_abnormal_blood_cell():
    """Generate an abnormal blood cell with irregular features"""
    image = np.zeros((64, 64))
    
    # Create irregular cell shape
    center = (32, 32)
    
    for i in range(64):
        for j in range(64):
            dist = np.sqrt((i - center[0])**2 + (j - center[1])**2)
            angle = np.arctan2(i - center[0], j - center[1])
            
            # Irregular radius
            radius = 18 + 5 * np.sin(3 * angle)
            
            if dist < radius:
                # Abnormal cell interior with irregular texture
                image[i, j] = 0.5 + 0.3 * np.random.randn()
            elif dist < radius + 3:
                # Thick irregular boundary
                image[i, j] = 0.2
            else:
                # Background
                image[i, j] = 0.1 * np.random.randn()
    
    # Add abnormal spots
    for _ in range(3):
        spot_x = np.random.randint(25, 40)
        spot_y = np.random.randint(25, 40)
        for i in range(-2, 3):
            for j in range(-2, 3):
                if 0 <= spot_x + i < 64 and 0 <= spot_y + j < 64:
                    image[spot_x + i, spot_y + j] = 0.9
    
    # Normalize
    image = np.clip(image, 0, 1)
    
    return image


def main():
    """Main function to demonstrate CNN blood cell analysis"""
    
    # Initialize CNN
    cnn = BloodCellCNN()
    
    # Analyze output shape
    cnn.analyze_output_shape()
    
    # Visualize filters
    print("\nVisualizing convolutional filters...")
    cnn.visualize_filters()
    
    # Generate synthetic blood cell images
    print("\nGenerating synthetic blood cell images...")
    healthy_cell = generate_synthetic_blood_cell()
    abnormal_cell = generate_abnormal_blood_cell()
    
    # Process healthy cell
    print("\nProcessing healthy blood cell...")
    healthy_features = cnn.forward_pass(healthy_cell)
    print(f"Output feature maps shape: {healthy_features.shape}")
    cnn.visualize_feature_maps(healthy_features, healthy_cell)
    
    # Process abnormal cell
    print("\nProcessing abnormal blood cell...")
    abnormal_features = cnn.forward_pass(abnormal_cell)
    
    # Compare feature statistics
    print("\n" + "=" * 60)
    print("Feature Extraction Analysis")
    print("=" * 60)
    print(f"Healthy Cell - Mean activation: {np.mean(healthy_features):.4f}")
    print(f"Healthy Cell - Max activation: {np.max(healthy_features):.4f}")
    print(f"Healthy Cell - Active neurons: {np.sum(healthy_features > 0)}")
    print()
    print(f"Abnormal Cell - Mean activation: {np.mean(abnormal_features):.4f}")
    print(f"Abnormal Cell - Max activation: {np.max(abnormal_features):.4f}")
    print(f"Abnormal Cell - Active neurons: {np.sum(abnormal_features > 0)}")
    print("=" * 60)
    
    # Visualize comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Healthy cell
    axes[0, 0].imshow(healthy_cell, cmap='gray')
    axes[0, 0].set_title('Healthy Cell\n(Original)', fontsize=12)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(healthy_features[0], cmap='viridis')
    axes[0, 1].set_title('Feature Map 1', fontsize=12)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(healthy_features[5], cmap='viridis')
    axes[0, 2].set_title('Feature Map 6', fontsize=12)
    axes[0, 2].axis('off')
    
    # Abnormal cell
    axes[1, 0].imshow(abnormal_cell, cmap='gray')
    axes[1, 0].set_title('Abnormal Cell\n(Original)', fontsize=12)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(abnormal_features[0], cmap='viridis')
    axes[1, 1].set_title('Feature Map 1', fontsize=12)
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(abnormal_features[5], cmap='viridis')
    axes[1, 2].set_title('Feature Map 6', fontsize=12)
    axes[1, 2].axis('off')
    
    plt.suptitle('Blood Cell Analysis - Healthy vs Abnormal', fontsize=16)
    plt.tight_layout()
    plt.savefig('blood_cell_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n✓ Analysis complete! Images saved.")
    print("  - cnn_filters.png")
    print("  - cnn_feature_maps.png")
    print("  - blood_cell_comparison.png")


if __name__ == "__main__":
    main()
