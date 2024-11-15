# PyHive 

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![CUDA Support](https://img.shields.io/badge/CUDA-enabled-brightgreen)
![Version](https://img.shields.io/badge/version-1.0.0-blue)

PyHive is a high-performance distributed computing framework for Python that orchestrates computational tasks across multiple CPU threads and CUDA-enabled GPUs. Like a well-organized beehive, PyHive efficiently distributes workloads across your system's computational resources, maximizing throughput and minimizing processing time.

## 🚀 Features

- **Multi-threaded Processing**: Distribute tasks across multiple CPU cores
- **CUDA Acceleration**: Leverage GPU power for compute-intensive operations
- **Smart Task Distribution**: Automatic workload balancing across available resources
- **Flexible Data Handling**: Seamless support for NumPy arrays and PyTorch tensors
- **Robust Error Management**: Built-in error handling and comprehensive logging
- **Thread-safe Operations**: Secure task queuing and result collection
- **Easy Integration**: Simple, intuitive API for quick implementation

## 🛠️ Installation

```bash
git clone https://github.com/jomardyan/pyhive.git
cd pyhive
pip install -r requirements.txt
```

## 🏃‍♂️ Quick Start

```python
from pyhive import PyHive
import numpy as np
import torch

# Initialize PyHive
hive = PyHive(num_cpu_workers=4, num_cuda_workers=1)
hive.start()

# Define operations
def cpu_operation(data):
    return data * 2

def cuda_operation(tensor):
    return tensor.pow(2)

# Submit CPU tasks
cpu_data = np.array([1, 2, 3, 4, 5])
task_id1 = hive.submit(cpu_data, cpu_operation, device='cpu')

# Submit CUDA tasks
cuda_data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
task_id2 = hive.submit(cuda_data, cuda_operation, device='cuda')

# Collect results
results = hive.get_results()

# Cleanup
hive.stop()
```

## 🎯 Advanced Usage

### Batch Processing

```python
def batch_operation(batch):
    return batch.mean(axis=0)

# Process large datasets in batches
large_data = np.random.randn(1000, 100)
task_id = hive.submit(
    large_data,
    batch_operation,
    device='cuda',
    batch_size=32
)
```

### Error Handling

```python
try:
    hive.submit(data, operation, device='cuda')
except RuntimeError as e:
    print(f"CUDA error occurred: {e}")
    # Automatically fallback to CPU processing
    hive.submit(data, operation, device='cpu')
```

## 📊 Performance Tips

### CPU Workers Best For:
- I/O-bound operations
- Small to medium data processing
- Task coordination and management
- Sequential operations

### CUDA Workers Excel At:
- Large matrix computations
- Deep learning operations
- Parallel numerical processing
- Batch data transformations

## 🔧 Configuration

```python
hive = PyHive(
    num_cpu_workers=4,      # Number of CPU threads
    num_cuda_workers=1,     # Number of GPU workers
    task_queue_size=1000    # Maximum queue size
)
```


## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 Citation

If you use PyHive in your research, please cite:

```bibtex
@software{pyhive2024,
  author = {Hayk Jomardyan},
  title = {PyHive: Distributed Computing Made Simple},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/jomardyan/PyHive}
}
```
