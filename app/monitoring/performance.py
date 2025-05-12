import time
import threading
import psutil
import GPUtil
from typing import Dict, List, Any, Optional
import json
import os
from datetime import datetime
from prometheus_client import Gauge, Histogram, Counter

# Initialize Prometheus metrics
GPU_MEMORY_USAGE = Gauge('gpu_memory_usage_bytes', 'GPU memory usage in bytes', ['device'])
GPU_UTILIZATION = Gauge('gpu_utilization_percent', 'GPU utilization percentage', ['device'])
CPU_UTILIZATION = Gauge('cpu_utilization_percent', 'CPU utilization percentage')
MEMORY_UTILIZATION = Gauge('memory_utilization_percent', 'Memory utilization percentage')
BATCH_SIZE_GAUGE = Gauge('current_batch_size', 'Current batch size', ['operation'])
THROUGHPUT_COUNTER = Counter('throughput_items', 'Throughput in items per second', ['operation'])

class PerformanceMonitor:
    """
    Monitors and logs system performance metrics
    """
    def __init__(self, log_dir: str = "logs", interval: float = 5.0):
        self.log_dir = log_dir
        self.interval = interval
        self.stop_event = threading.Event()
        self.monitor_thread = None
        self.performance_data = []
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
    
    def start_monitoring(self):
        """Start the performance monitoring thread"""
        if self.monitor_thread is None or not self.monitor_thread.is_alive():
            self.stop_event.clear()
            self.monitor_thread = threading.Thread(target=self._monitor_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop the performance monitoring thread"""
        self.stop_event.set()
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        
        # Save collected data
        self.save_performance_data()
    
    def _monitor_loop(self):
        """Background worker that collects performance metrics"""
        while not self.stop_event.is_set():
            # Collect metrics
            metrics = self.collect_metrics()
            
            # Store metrics
            self.performance_data.append(metrics)
            
            # Update Prometheus metrics
            self._update_prometheus_metrics(metrics)
            
            # Sleep until next collection
            time.sleep(self.interval)
    
    def collect_metrics(self) -> Dict[str, Any]:
        """Collect system performance metrics"""
        # Get timestamp
        timestamp = datetime.now().isoformat()
        
        # Collect CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Collect memory metrics
        memory = psutil.virtual_memory()
        memory_used_percent = memory.percent
        memory_used_gb = memory.used / (1024 ** 3)  # Convert to GB
        memory_total_gb = memory.total / (1024 ** 3)  # Convert to GB
        
        # Collect GPU metrics if available
        gpu_metrics = []
        try:
            gpus = GPUtil.getGPUs()
            for i, gpu in enumerate(gpus):
                gpu_metrics.append({
                    'id': i,
                    'name': gpu.name,
                    'load': gpu.load * 100,  # Convert to percentage
                    'memory_used': gpu.memoryUsed,  # In MB
                    'memory_total': gpu.memoryTotal,  # In MB
                    'memory_used_percent': gpu.memoryUtil * 100  # Convert to percentage
                })
        except Exception as e:
            # GPUtil may not be available or GPU monitoring might fail
            gpu_metrics = [{'error': str(e)}]
        
        # Collect disk metrics
        disk = psutil.disk_usage('/')
        disk_used_percent = disk.percent
        disk_used_gb = disk.used / (1024 ** 3)  # Convert to GB
        disk_total_gb = disk.total / (1024 ** 3)  # Convert to GB
        
        # Collect network metrics
        net_io = psutil.net_io_counters()
        net_sent_mb = net_io.bytes_sent / (1024 ** 2)  # Convert to MB
        net_recv_mb = net_io.bytes_recv / (1024 ** 2)  # Convert to MB
        
        # Return all metrics
        return {
            'timestamp': timestamp,
            'cpu': {
                'percent': cpu_percent
            },
            'memory': {
                'percent': memory_used_percent,
                'used_gb': memory_used_gb,
                'total_gb': memory_total_gb
            },
            'gpu': gpu_metrics,
            'disk': {
                'percent': disk_used_percent,
                'used_gb': disk_used_gb,
                'total_gb': disk_total_gb
            },
            'network': {
                'sent_mb': net_sent_mb,
                'recv_mb': net_recv_mb
            }
        }
    
    def _update_prometheus_metrics(self, metrics: Dict[str, Any]):
        """Update Prometheus metrics with current values"""
        # Update CPU metrics
        CPU_UTILIZATION.set(metrics['cpu']['percent'])
        
        # Update memory metrics
        MEMORY_UTILIZATION.set(metrics['memory']['percent'])
        
        # Update GPU metrics
        for gpu in metrics['gpu']:
            if 'id' in gpu:  # Skip if error
                device_id = str(gpu['id'])
                GPU_UTILIZATION.labels(device=device_id).set(gpu['load'])
                GPU_MEMORY_USAGE.labels(device=device_id).set(gpu['memory_used'] * 1024 * 1024)  # Convert MB to bytes
    
    def save_performance_data(self):
        """Save collected performance data to disk"""
        if not self.performance_data:
            return
        
        # Create filename with timestamp
        filename = f"performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        file_path = os.path.join(self.log_dir, filename)
        
        # Write to file
        with open(file_path, 'w') as f:
            json.dump(self.performance_data, f, indent=2)
        
        # Clear performance data
        self.performance_data = []
        
        print(f"Performance data saved to {file_path}")
    
    def log_operation_metrics(self, operation: str, batch_size: int, latency: float, items_processed: int):
        """
        Log metrics for a specific operation
        
        Args:
            operation: Name of the operation (e.g., "text_embedding", "image_search")
            batch_size: Batch size used
            latency: Operation latency in seconds
            items_processed: Number of items processed
        """
        # Update Prometheus metrics
        BATCH_SIZE_GAUGE.labels(operation=operation).set(batch_size)
        THROUGHPUT_COUNTER.labels(operation=operation).inc(items_processed)
        
        # Calculate throughput
        throughput = items_processed / latency if latency > 0 else 0
        
        # Log to console
        print(f"Operation: {operation}, Batch Size: {batch_size}, Latency: {latency:.4f}s, "
              f"Items: {items_processed}, Throughput: {throughput:.2f} items/s")