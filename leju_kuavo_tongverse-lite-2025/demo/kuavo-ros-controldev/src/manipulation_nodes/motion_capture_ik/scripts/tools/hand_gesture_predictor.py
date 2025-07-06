import subprocess
import sys
import importlib.util
import json
import time
import numpy as np

def install_package(packages):
    """
    Install one or more packages using pip if they're not already installed
    Args:
        packages: str or list of str, package name(s) to install
    """
    if isinstance(packages, str):
        packages = [packages]
    
    for package in packages:
        if importlib.util.find_spec(package.split('==')[0]) is None:
            try:
                print(f"{package} not found. Installing...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"Successfully installed {package}")
            except subprocess.CalledProcessError as e:
                print(f"Failed to install {package}: {e}")
                sys.exit(1)

# Check and install required packages
required_packages = [
    "onnxruntime",
    "numpy"
]

install_package(required_packages)

import onnxruntime as ort

class HandGesturePredictor:
    def __init__(self, model_path):
        """
        Initialize the hand gesture predictor
        Args:
            model_path: Path to the ONNX model file
        """
        # Load metadata
        metadata_path = model_path.replace('.onnx', '_metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        self.gesture_names = metadata['gesture_names']
        print(f"Loaded gesture names: {self.gesture_names}")
        
        # Initialize ONNX runtime
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = 4  # Set number of threads
        
        # Select execution provider
        available_providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' in available_providers:
            provider = 'CUDAExecutionProvider'
            print("Using CUDA for inference")
        else:
            provider = 'CPUExecutionProvider'
            print("Using CPU for inference")

        # Load model
        self.session = ort.InferenceSession(
            model_path, 
            providers=[provider],
            sess_options=options
        )
        self.input_name = self.session.get_inputs()[0].name
        print("Model loaded successfully")

    def predict(self, frame_poses):
        """
        Predict hand gesture
        Args:
            frame_poses: Array of length 63 containing position and rotation information for 9 transforms
        Returns:
            gesture: Predicted gesture name
            confidence: Prediction confidence
            inference_time: Inference time in milliseconds
        """
        if not isinstance(frame_poses, (list, np.ndarray)) or len(frame_poses) != 63:
            raise ValueError(f"frame_poses must be a list or array of length 63")
            
        start_time = time.time()
        
        # Prepare input data
        input_data = np.array([frame_poses], dtype=np.float32)
        
        # Run inference
        outputs = self.session.run(None, {self.input_name: input_data})
        
        # Replace PyTorch softmax with NumPy implementation
        def softmax(x):
            exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
            return exp_x / exp_x.sum(axis=1, keepdims=True)
            
        probabilities = softmax(outputs[0]).squeeze()
        
        # Get top-2 predictions and their probabilities
        top2_idx = np.argsort(probabilities)[-2:][::-1]
        confidence = probabilities[top2_idx[0]]
        second_confidence = probabilities[top2_idx[1]]
        
        # Calculate prediction uncertainty metrics
        margin = confidence - second_confidence
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        
        # Calculate inference time in milliseconds
        inference_time = (time.time() - start_time) * 1000
        
        # Decision thresholds
        CONFIDENCE_THRESHOLD = 0.80
        MARGIN_THRESHOLD = 0.3
        ENTROPY_THRESHOLD = 0.8
        
        if (confidence < CONFIDENCE_THRESHOLD or 
            margin < MARGIN_THRESHOLD or 
            entropy > ENTROPY_THRESHOLD):
            return "other", confidence, inference_time
            
        return self.gesture_names[top2_idx[0]], confidence, inference_time

    def get_gesture_names(self):
        """Get a list of all possible gesture names"""
        return self.gesture_names.copy()

# # Usage example
# if __name__ == "__main__":
#     # Create predictor instance
#     predictor = HandGesturePredictor("hand_gesture_model.onnx")
    
#     # Example data: 63 random values
#     example_poses = np.random.randn(63)
    
#     try:
#         # Make prediction
#         gesture, confidence, inference_time = predictor.predict(example_poses)
#         print(f"Predicted gesture: {gesture}")
#         print(f"Confidence: {confidence:.2%}")
#         print(f"Inference time: {inference_time:.2f}ms")
#     except Exception as e:
#         print(f"Error during prediction: {e}")