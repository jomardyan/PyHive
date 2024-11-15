import threading
import queue
import numpy as np
import torch
from typing import Callable, Any, List, Union, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pyhive")

@dataclass
class HiveTask:
    """Represents a computational task within the PyHive framework"""
    id: int
    data: Any
    operation: Callable
    device: str = 'cpu'  # 'cpu' or 'cuda'
    batch_size: Optional[int] = None

class HiveWorker:
    """Worker class that handles computation on either CPU or CUDA device"""
    def __init__(self, device: str = 'cpu', cuda_device_id: int = 0):
        self.device = device
        self.cuda_device_id = cuda_device_id
        
        if device == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available on this system")
            self.torch_device = torch.device(f'cuda:{cuda_device_id}')
        else:
            self.torch_device = torch.device('cpu')
    
    def process_task(self, task: HiveTask) -> Any:
        """Process a single task"""
        try:
            if isinstance(task.data, np.ndarray):
                # Convert NumPy arrays to PyTorch tensors
                data = torch.from_numpy(task.data).to(self.torch_device)
            elif isinstance(task.data, torch.Tensor):
                data = task.data.to(self.torch_device)
            else:
                data = task.data
            
            # Apply the operation
            result = task.operation(data)
            
            # Move result back to CPU if needed
            if isinstance(result, torch.Tensor):
                result = result.cpu()
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing task {task.id}: {str(e)}")
            raise

class HiveScheduler:
    """Manages task distribution and execution across workers"""
    def __init__(self, 
                 num_cpu_workers: int = 4,
                 num_cuda_workers: int = 1,
                 task_queue_size: int = 1000):
        
        self.task_queue = queue.Queue(maxsize=task_queue_size)
        self.result_queue = queue.Queue()
        
        # Initialize workers
        self.cpu_workers = [HiveWorker(device='cpu') for _ in range(num_cpu_workers)]
        self.cuda_workers = [
            HiveWorker(device='cuda', cuda_device_id=i) 
            for i in range(min(num_cuda_workers, torch.cuda.device_count() if torch.cuda.is_available() else 0))
        ]
        
        self.thread_pool = ThreadPoolExecutor(
            max_workers=num_cpu_workers + len(self.cuda_workers)
        )
        
        self.is_running = False
        self.scheduler_thread = None
    
    def submit_task(self, task: HiveTask) -> None:
        """Submit a task to the queue"""
        self.task_queue.put(task)
    
    def _worker_loop(self, worker: HiveWorker) -> None:
        """Main loop for each worker"""
        while self.is_running:
            try:
                task = self.task_queue.get(timeout=1)
                if task.device == worker.device:
                    result = worker.process_task(task)
                    self.result_queue.put((task.id, result))
                else:
                    # Put back tasks that don't match the worker's device
                    self.task_queue.put(task)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker error: {str(e)}")
                continue
    
    def start(self) -> None:
        """Start the scheduler and workers"""
        self.is_running = True
        
        # Start worker threads
        for worker in self.cpu_workers + self.cuda_workers:
            self.thread_pool.submit(self._worker_loop, worker)
        
        logger.info(f"Started PyHive scheduler with {len(self.cpu_workers)} CPU workers and "
                   f"{len(self.cuda_workers)} CUDA workers")
    
    def stop(self) -> None:
        """Stop the scheduler and workers"""
        self.is_running = False
        self.thread_pool.shutdown(wait=True)
        logger.info("Stopped PyHive scheduler and all workers")
    
    def get_results(self, timeout: Optional[float] = None) -> List[tuple]:
        """Get all available results"""
        results = []
        try:
            while True:
                result = self.result_queue.get(timeout=timeout)
                results.append(result)
                self.result_queue.task_done()
        except queue.Empty:
            pass
        return results

class PyHive:
    """Main interface for PyHive distributed computing framework"""
    def __init__(self, 
                 num_cpu_workers: int = 4,
                 num_cuda_workers: int = 1,
                 task_queue_size: int = 1000):
        
        self.scheduler = HiveScheduler(
            num_cpu_workers=num_cpu_workers,
            num_cuda_workers=num_cuda_workers,
            task_queue_size=task_queue_size
        )
        self.task_counter = 0
    
    def start(self) -> None:
        """Start the PyHive computation framework"""
        self.scheduler.start()
    
    def stop(self) -> None:
        """Stop the PyHive computation framework"""
        self.scheduler.stop()
    
    def submit(self, 
               data: Any,
               operation: Callable,
               device: str = 'cpu',
               batch_size: Optional[int] = None) -> int:
        """Submit data for computation"""
        task_id = self.task_counter
        self.task_counter += 1
        
        task = HiveTask(
            id=task_id,
            data=data,
            operation=operation,
            device=device,
            batch_size=batch_size
        )
        
        self.scheduler.submit_task(task)
        return task_id
    
    def get_results(self, timeout: Optional[float] = None) -> List[tuple]:
        """Get available results"""
        return self.scheduler.get_results(timeout=timeout)

# Example usage
if __name__ == "__main__":
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

    # Submit CUDA tasks (if available)
    if torch.cuda.is_available():
        cuda_data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        task_id2 = hive.submit(cuda_data, cuda_operation, device='cuda')

    # Get results
    results = hive.get_results()
    print("Results:", results)

    # Clean up
    hive.stop()
