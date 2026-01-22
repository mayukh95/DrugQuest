"""
Resource management for parallel optimization tasks.
"""

import psutil
import threading
import time
import numpy as np
from typing import Dict, Optional, List
from dataclasses import dataclass


@dataclass
class ResourceUsage:
    """Track resource usage for a job."""
    job_id: str
    cpu_percent: float
    memory_mb: float
    start_time: float
    estimated_completion: Optional[float] = None


class ResourceManager:
    """Manage computational resources for parallel optimization."""
    
    def __init__(self, max_memory_gb=16, max_workers=4, 
                 cpu_threshold=90, memory_threshold=85):
        """Initialize resource manager."""
        self.max_memory_gb = max_memory_gb
        self.max_workers = max_workers
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        
        self.active_jobs = {}
        self.resource_history = []
        self.lock = threading.RLock()
        
        self._monitoring = False
        self._monitor_thread = None
    
    def start_monitoring(self, interval=5.0):
        """Start resource monitoring thread."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_resources, 
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self._monitoring = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join()
    
    def _monitor_resources(self, interval):
        """Monitor system resources continuously."""
        while self._monitoring:
            try:
                # Get system stats
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                timestamp = time.time()
                
                with self.lock:
                    self.resource_history.append({
                        'timestamp': timestamp,
                        'cpu_percent': cpu_percent,
                        'memory_percent': memory.percent,
                        'memory_used_gb': memory.used / (1024**3),
                        'active_jobs': len(self.active_jobs)
                    })
                    
                    # Keep only recent history (last hour)
                    cutoff = timestamp - 3600
                    self.resource_history = [
                        entry for entry in self.resource_history 
                        if entry['timestamp'] > cutoff
                    ]
                
                time.sleep(interval)
                
            except Exception as e:
                print(f"Resource monitoring error: {e}")
                time.sleep(interval)
    
    def can_allocate_job(self, estimated_memory_gb=2.0):
        """Check if resources are available for a new job."""
        with self.lock:
            # Check worker limit
            if len(self.active_jobs) >= self.max_workers:
                return False
            
            # Check memory availability
            memory = psutil.virtual_memory()
            available_memory_gb = memory.available / (1024**3)
            
            if available_memory_gb < estimated_memory_gb:
                return False
            
            # Check if total allocated memory would exceed limit
            current_allocated = sum(
                job.memory_mb for job in self.active_jobs.values()
            ) / 1024  # Convert to GB
            
            if (current_allocated + estimated_memory_gb) > self.max_memory_gb:
                return False
            
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            if cpu_percent > self.cpu_threshold:
                return False
            
            return True
    
    def allocate_job(self, job_id: str, estimated_memory_gb=2.0):
        """Allocate resources for a job."""
        with self.lock:
            if not self.can_allocate_job(estimated_memory_gb):
                raise RuntimeError(f"Cannot allocate resources for job {job_id}")
            
            self.active_jobs[job_id] = ResourceUsage(
                job_id=job_id,
                cpu_percent=0.0,
                memory_mb=estimated_memory_gb * 1024,
                start_time=time.time()
            )
    
    def release_job(self, job_id: str):
        """Release resources for a completed job."""
        with self.lock:
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]
    
    def update_job_usage(self, job_id: str, cpu_percent: float, memory_mb: float):
        """Update resource usage for a running job."""
        with self.lock:
            if job_id in self.active_jobs:
                job = self.active_jobs[job_id]
                job.cpu_percent = cpu_percent
                job.memory_mb = memory_mb
    
    def get_system_status(self):
        """Get current system resource status."""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        with self.lock:
            active_job_memory = sum(
                job.memory_mb for job in self.active_jobs.values()
            )
        
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_used_gb': memory.used / (1024**3),
            'memory_available_gb': memory.available / (1024**3),
            'active_jobs': len(self.active_jobs),
            'active_job_memory_mb': active_job_memory,
            'cpu_overloaded': cpu_percent > self.cpu_threshold,
            'memory_overloaded': memory.percent > self.memory_threshold
        }
    
    def get_job_status(self, job_id: str):
        """Get resource status for a specific job."""
        with self.lock:
            if job_id not in self.active_jobs:
                return None
            
            job = self.active_jobs[job_id]
            runtime = time.time() - job.start_time
            
            return {
                'job_id': job_id,
                'cpu_percent': job.cpu_percent,
                'memory_mb': job.memory_mb,
                'runtime': runtime,
                'estimated_completion': job.estimated_completion
            }
    
    def estimate_completion_time(self, job_id: str, progress_fraction: float):
        """Estimate job completion time based on progress."""
        with self.lock:
            if job_id not in self.active_jobs:
                return None
            
            job = self.active_jobs[job_id]
            runtime = time.time() - job.start_time
            
            if progress_fraction > 0:
                total_estimated_time = runtime / progress_fraction
                remaining_time = total_estimated_time - runtime
                job.estimated_completion = time.time() + remaining_time
                return remaining_time
            
            return None
    
    def get_optimal_worker_count(self):
        """Suggest optimal number of workers based on system resources."""
        cpu_count = psutil.cpu_count(logical=False)
        memory = psutil.virtual_memory()
        
        # CPU-based limit
        cpu_workers = max(1, min(cpu_count, self.max_workers))
        
        # Memory-based limit (assuming 2GB per job)
        memory_workers = max(1, int(memory.available / (2 * 1024**3)))
        
        return min(cpu_workers, memory_workers, self.max_workers)
    
    def get_resource_recommendations(self):
        """Get recommendations for resource optimization."""
        status = self.get_system_status()
        recommendations = []
        
        if status['cpu_overloaded']:
            recommendations.append("CPU usage is high. Consider reducing max_workers.")
        
        if status['memory_overloaded']:
            recommendations.append("Memory usage is high. Consider reducing memory_limit_gb.")
        
        if len(self.active_jobs) < self.max_workers and not status['cpu_overloaded']:
            recommendations.append("System can handle more parallel jobs.")
        
        optimal_workers = self.get_optimal_worker_count()
        if optimal_workers != self.max_workers:
            recommendations.append(
                f"Consider setting max_workers to {optimal_workers} for optimal performance."
            )
        
        return recommendations
    
    def get_performance_metrics(self, time_window_hours=1):
        """Get performance metrics over a time window."""
        cutoff = time.time() - (time_window_hours * 3600)
        
        with self.lock:
            recent_history = [
                entry for entry in self.resource_history 
                if entry['timestamp'] > cutoff
            ]
        
        if not recent_history:
            return None
        
        cpu_values = [entry['cpu_percent'] for entry in recent_history]
        memory_values = [entry['memory_percent'] for entry in recent_history]
        job_counts = [entry['active_jobs'] for entry in recent_history]
        
        return {
            'time_window_hours': time_window_hours,
            'avg_cpu_percent': np.mean(cpu_values),
            'max_cpu_percent': np.max(cpu_values),
            'avg_memory_percent': np.mean(memory_values),
            'max_memory_percent': np.max(memory_values),
            'avg_active_jobs': np.mean(job_counts),
            'max_active_jobs': np.max(job_counts),
            'cpu_overload_fraction': np.mean(np.array(cpu_values) > self.cpu_threshold),
            'memory_overload_fraction': np.mean(np.array(memory_values) > self.memory_threshold)
        }
    
    def adjust_limits_automatically(self):
        """Automatically adjust resource limits based on performance."""
        metrics = self.get_performance_metrics()
        
        if metrics is None:
            return
        
        adjustments = {}
        
        # Adjust worker count based on CPU usage
        if metrics['cpu_overload_fraction'] > 0.5:  # CPU overloaded more than 50% of time
            new_workers = max(1, self.max_workers - 1)
            adjustments['max_workers'] = new_workers
        elif metrics['cpu_overload_fraction'] < 0.1 and metrics['avg_cpu_percent'] < 50:
            optimal_workers = self.get_optimal_worker_count()
            if optimal_workers > self.max_workers:
                adjustments['max_workers'] = min(self.max_workers + 1, optimal_workers)
        
        # Adjust memory limit based on memory usage
        if metrics['memory_overload_fraction'] > 0.3:
            new_memory = max(4, self.max_memory_gb - 2)
            adjustments['max_memory_gb'] = new_memory
        
        return adjustments
    
    def export_performance_report(self, output_file):
        """Export performance report to file."""
        status = self.get_system_status()
        metrics = self.get_performance_metrics()
        recommendations = self.get_resource_recommendations()
        
        with open(output_file, 'w') as f:
            f.write("RESOURCE MANAGER PERFORMANCE REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("CURRENT STATUS\n")
            f.write("-" * 15 + "\n")
            f.write(f"CPU Usage: {status['cpu_percent']:.1f}%\n")
            f.write(f"Memory Usage: {status['memory_percent']:.1f}%\n")
            f.write(f"Available Memory: {status['memory_available_gb']:.1f} GB\n")
            f.write(f"Active Jobs: {status['active_jobs']}/{self.max_workers}\n")
            f.write(f"Job Memory Usage: {status['active_job_memory_mb']:.0f} MB\n\n")
            
            if metrics:
                f.write("PERFORMANCE METRICS (Last Hour)\n")
                f.write("-" * 30 + "\n")
                f.write(f"Average CPU: {metrics['avg_cpu_percent']:.1f}%\n")
                f.write(f"Peak CPU: {metrics['max_cpu_percent']:.1f}%\n")
                f.write(f"Average Memory: {metrics['avg_memory_percent']:.1f}%\n")
                f.write(f"Peak Memory: {metrics['max_memory_percent']:.1f}%\n")
                f.write(f"CPU Overload: {metrics['cpu_overload_fraction']*100:.1f}% of time\n")
                f.write(f"Memory Overload: {metrics['memory_overload_fraction']*100:.1f}% of time\n\n")
            
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 15 + "\n")
            if recommendations:
                for rec in recommendations:
                    f.write(f"• {rec}\n")
            else:
                f.write("• No specific recommendations at this time.\n")
        
        print(f"✓ Performance report saved to {output_file}")
    
    def __enter__(self):
        """Context manager entry."""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_monitoring()