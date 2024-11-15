import numpy as np
import torch
import torch.nn.functional as F
from pyhive import PyHive
import time
from typing import List, Tuple
import pandas as pd
from PIL import Image

class PyHiveExamples:
    """Collection of PyHive usage examples"""
    
    def __init__(self):
        self.hive = PyHive(num_cpu_workers=4, num_cuda_workers=1)
        self.hive.start()
    
    def cleanup(self):
        self.hive.stop()

    def example_1_basic_math(self):
        """Basic mathematical operations example"""
        print("\n=== Example 1: Basic Mathematical Operations ===")
        
        # CPU operation: Matrix multiplication
        def cpu_matrix_mult(data):
            return np.dot(data, data.T)
        
        # CPU task
        matrix = np.random.rand(100, 100)
        task_id = self.hive.submit(matrix, cpu_matrix_mult, device='cpu')
        results = self.hive.get_results()
        print(f"Matrix multiplication result shape: {results[0][1].shape}")

    def example_2_image_processing(self):
        """Image processing example"""
        print("\n=== Example 2: Image Processing ===")
        
        # Simulate image data
        images = np.random.rand(10, 224, 224, 3)  # 10 RGB images
        
        def process_image_batch(batch):
            # Simulate image processing
            # Normalize
            normalized = batch / 255.0
            # Add brightness
            brightened = normalized * 1.2
            # Clip values
            return np.clip(brightened, 0, 1)
        
        task_id = self.hive.submit(
            images,
            process_image_batch,
            device='cpu',
            batch_size=2
        )
        
        results = self.hive.get_results()
        print(f"Processed {len(images)} images")

    def example_3_cuda_tensor_operations(self):
        """CUDA tensor operations example"""
        print("\n=== Example 3: CUDA Tensor Operations ===")
        
        if not torch.cuda.is_available():
            print("CUDA not available, skipping example 3")
            return
            
        def cuda_operation(tensor):
            # Complex tensor operations
            x = F.relu(tensor)
            x = F.max_pool2d(x, kernel_size=2)
            return x
        
        # Create random tensor
        data = torch.randn(100, 3, 32, 32)  # Batch of 100 RGB 32x32 images
        task_id = self.hive.submit(data, cuda_operation, device='cuda')
        results = self.hive.get_results()
        print(f"CUDA tensor operation output shape: {results[0][1].shape}")

    def example_4_parallel_data_processing(self):
        """Parallel data processing example"""
        print("\n=== Example 4: Parallel Data Processing ===")
        
        # Create sample dataset
        df = pd.DataFrame({
            'A': np.random.randn(1000),
            'B': np.random.randn(1000),
            'C': np.random.choice(['X', 'Y', 'Z'], 1000)
        })
        
        def process_chunk(data):
            # Simulate complex data processing
            result = data.copy()
            result['D'] = result['A'] + result['B']
            result['E'] = result['D'].apply(lambda x: x**2)
            return result
        
        # Split data into chunks and process
        chunks = np.array_split(df, 4)
        task_ids = []
        for chunk in chunks:
            task_id = self.hive.submit(chunk, process_chunk, device='cpu')
            task_ids.append(task_id)
        
        results = self.hive.get_results()
        print(f"Processed {len(chunks)} data chunks in parallel")

    def example_5_mixed_device_workflow(self):
        """Example of mixing CPU and GPU operations"""
        print("\n=== Example 5: Mixed Device Workflow ===")
        
        if not torch.cuda.is_available():
            print("CUDA not available, skipping example 5")
            return
        
        # Step 1: CPU data preparation
        def prepare_data(data):
            # Simulate data preparation on CPU
            normalized = data / data.max()
            return normalized
        
        # Step 2: GPU processing
        def gpu_processing(tensor):
            # Simulate GPU-intensive computation
            return torch.nn.functional.softmax(tensor, dim=1)
        
        # Create sample data
        data = np.random.randn(1000, 10)
        
        # Submit CPU task
        cpu_task_id = self.hive.submit(data, prepare_data, device='cpu')
        results = self.hive.get_results()
        
        # Get CPU results and submit to GPU
        if results:
            prepared_data = torch.tensor(results[0][1])
            gpu_task_id = self.hive.submit(
                prepared_data,
                gpu_processing,
                device='cuda'
            )
            
            final_results = self.hive.get_results()
            print("Completed mixed device workflow")

    def example_6_error_handling(self):
        """Error handling example"""
        print("\n=== Example 6: Error Handling ===")
        
        def faulty_operation(data):
            # Deliberately cause an error
            return data / 0
        
        try:
            task_id = self.hive.submit(
                np.array([1, 2, 3]),
                faulty_operation,
                device='cpu'
            )
            results = self.hive.get_results()
        except Exception as e:
            print(f"Caught error successfully: {str(e)}")

def run_all_examples():
    """Run all PyHive examples"""
    examples = PyHiveExamples()
    
    try:
        examples.example_1_basic_math()
        examples.example_2_image_processing()
        examples.example_3_cuda_tensor_operations()
        examples.example_4_parallel_data_processing()
        examples.example_5_mixed_device_workflow()
        examples.example_6_error_handling()
    finally:
        examples.cleanup()

if __name__ == "__main__":
    run_all_examples()
