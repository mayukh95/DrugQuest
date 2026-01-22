"""
Progress tracking and monitoring for optimization tasks.
"""

import time
import threading
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
import plotly.graph_objects as go
from plotly.subplots import make_subplots


@dataclass
class OptimizationProgress:
    """Track progress of an optimization task."""
    job_id: str
    start_time: float
    current_iteration: int = 0
    max_iterations: int = 100
    current_energy: Optional[float] = None
    gradient_norm: Optional[float] = None
    convergence_target: float = 1e-6
    status: str = 'running'
    estimated_completion: Optional[float] = None
    last_update: Optional[float] = None


class OptimizationTracker:
    """Track and monitor optimization progress across multiple jobs."""
    
    def __init__(self, update_interval=1.0):
        """Initialize optimization tracker."""
        self.update_interval = update_interval
        self.jobs = {}
        self.completed_jobs = {}
        self.lock = threading.RLock()
        
        self.update_callbacks = []
        self.completion_callbacks = []
        
        self._monitoring = False
        self._monitor_thread = None
    
    def register_job(self, job_id: str, max_iterations=100, 
                    convergence_target=1e-6):
        """Register a new job for tracking."""
        with self.lock:
            self.jobs[job_id] = OptimizationProgress(
                job_id=job_id,
                start_time=time.time(),
                max_iterations=max_iterations,
                convergence_target=convergence_target
            )
    
    def update_progress(self, job_id: str, iteration: int, 
                       energy: float, gradient_norm: float):
        """Update progress for a specific job."""
        with self.lock:
            if job_id not in self.jobs:
                return
            
            progress = self.jobs[job_id]
            progress.current_iteration = iteration
            progress.current_energy = energy
            progress.gradient_norm = gradient_norm
            progress.last_update = time.time()
            
            # Update status
            if gradient_norm < progress.convergence_target:
                progress.status = 'converged'
            elif iteration >= progress.max_iterations:
                progress.status = 'max_iterations'
            else:
                progress.status = 'running'
            
            # Estimate completion time
            if iteration > 0:
                elapsed = time.time() - progress.start_time
                time_per_iteration = elapsed / iteration
                remaining_iterations = progress.max_iterations - iteration
                progress.estimated_completion = time.time() + (time_per_iteration * remaining_iterations)
        
        # Notify callbacks
        self._notify_update_callbacks(job_id)
        
        # Check for completion
        if progress.status in ['converged', 'max_iterations', 'failed']:
            self.complete_job(job_id)
    
    def complete_job(self, job_id: str, status: str = None):
        """Mark a job as completed."""
        with self.lock:
            if job_id not in self.jobs:
                return
            
            progress = self.jobs[job_id]
            if status:
                progress.status = status
            
            # Move to completed jobs
            self.completed_jobs[job_id] = progress
            del self.jobs[job_id]
        
        # Notify completion callbacks
        self._notify_completion_callbacks(job_id, progress)
    
    def get_job_progress(self, job_id: str):
        """Get progress information for a specific job."""
        with self.lock:
            if job_id in self.jobs:
                return asdict(self.jobs[job_id])
            elif job_id in self.completed_jobs:
                return asdict(self.completed_jobs[job_id])
            else:
                return None
    
    def get_all_progress(self):
        """Get progress for all jobs."""
        with self.lock:
            active = {job_id: asdict(progress) for job_id, progress in self.jobs.items()}
            completed = {job_id: asdict(progress) for job_id, progress in self.completed_jobs.items()}
        
        return {'active': active, 'completed': completed}
    
    def get_summary_statistics(self):
        """Get summary statistics across all jobs."""
        with self.lock:
            all_jobs = list(self.jobs.values()) + list(self.completed_jobs.values())
        
        if not all_jobs:
            return {}
        
        # Calculate statistics
        total_jobs = len(all_jobs)
        completed_count = len(self.completed_jobs)
        active_count = len(self.jobs)
        
        # Status breakdown
        status_counts = {}
        for job in all_jobs:
            status_counts[job.status] = status_counts.get(job.status, 0) + 1
        
        # Timing statistics for completed jobs
        completed_jobs = list(self.completed_jobs.values())
        if completed_jobs:
            runtimes = []
            for job in completed_jobs:
                if job.last_update:
                    runtime = job.last_update - job.start_time
                    runtimes.append(runtime)
            
            avg_runtime = sum(runtimes) / len(runtimes) if runtimes else 0
            
        else:
            avg_runtime = 0
        
        # Convergence statistics
        converged_jobs = [job for job in completed_jobs if job.status == 'converged']
        convergence_rate = len(converged_jobs) / completed_count if completed_count > 0 else 0
        
        return {
            'total_jobs': total_jobs,
            'active_jobs': active_count,
            'completed_jobs': completed_count,
            'status_breakdown': status_counts,
            'convergence_rate': convergence_rate,
            'average_runtime': avg_runtime,
            'estimated_total_completion': self._estimate_total_completion()
        }
    
    def _estimate_total_completion(self):
        """Estimate when all active jobs will complete."""
        with self.lock:
            if not self.jobs:
                return None
            
            completion_times = []
            for progress in self.jobs.values():
                if progress.estimated_completion:
                    completion_times.append(progress.estimated_completion)
            
            if completion_times:
                return max(completion_times)
            else:
                return None
    
    def start_monitoring(self):
        """Start monitoring thread for automatic updates."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self._monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring thread."""
        self._monitoring = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join()
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self._monitoring:
            try:
                # Check for stalled jobs
                current_time = time.time()
                stall_threshold = 300  # 5 minutes
                
                with self.lock:
                    for job_id, progress in list(self.jobs.items()):
                        if progress.last_update:
                            time_since_update = current_time - progress.last_update
                            if time_since_update > stall_threshold:
                                progress.status = 'stalled'
                                self._notify_update_callbacks(job_id)
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(self.update_interval)
    
    def register_update_callback(self, callback: Callable[[str], None]):
        """Register callback for job updates."""
        self.update_callbacks.append(callback)
    
    def register_completion_callback(self, callback: Callable[[str, OptimizationProgress], None]):
        """Register callback for job completions."""
        self.completion_callbacks.append(callback)
    
    def _notify_update_callbacks(self, job_id: str):
        """Notify all update callbacks."""
        for callback in self.update_callbacks:
            try:
                callback(job_id)
            except Exception as e:
                print(f"Update callback error: {e}")
    
    def _notify_completion_callbacks(self, job_id: str, progress: OptimizationProgress):
        """Notify all completion callbacks."""
        for callback in self.completion_callbacks:
            try:
                callback(job_id, progress)
            except Exception as e:
                print(f"Completion callback error: {e}")
    
    def plot_progress_overview(self, title="Optimization Progress Overview"):
        """Create overview plot of all optimization progress."""
        with self.lock:
            all_jobs = list(self.jobs.values()) + list(self.completed_jobs.values())
        
        if not all_jobs:
            return None
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Job Status Distribution', 'Energy Progress', 
                          'Convergence Progress', 'Runtime Distribution'],
            specs=[[{'type': 'pie'}, {'type': 'scatter'}],
                   [{'type': 'scatter'}, {'type': 'histogram'}]]
        )
        
        # Status distribution pie chart
        status_counts = {}
        for job in all_jobs:
            status_counts[job.status] = status_counts.get(job.status, 0) + 1
        
        fig.add_trace(
            go.Pie(
                labels=list(status_counts.keys()),
                values=list(status_counts.values()),
                name="Status"
            ),
            row=1, col=1
        )
        
        # Energy progress for active jobs
        for job in self.jobs.values():
            if job.current_energy is not None:
                fig.add_trace(
                    go.Scatter(
                        x=[job.current_iteration],
                        y=[job.current_energy],
                        mode='markers',
                        name=f"Job {job.job_id}",
                        showlegend=False
                    ),
                    row=1, col=2
                )
        
        # Convergence progress
        for job in all_jobs:
            if job.gradient_norm is not None:
                color = 'green' if job.status == 'converged' else 'red' if job.status == 'failed' else 'blue'
                fig.add_trace(
                    go.Scatter(
                        x=[job.current_iteration],
                        y=[job.gradient_norm],
                        mode='markers',
                        marker=dict(color=color),
                        name=f"Job {job.job_id}",
                        showlegend=False
                    ),
                    row=2, col=1
                )
        
        # Runtime distribution for completed jobs
        completed_jobs = list(self.completed_jobs.values())
        if completed_jobs:
            runtimes = []
            for job in completed_jobs:
                if job.last_update:
                    runtime = job.last_update - job.start_time
                    runtimes.append(runtime / 60)  # Convert to minutes
            
            if runtimes:
                fig.add_trace(
                    go.Histogram(
                        x=runtimes,
                        name="Runtime",
                        showlegend=False
                    ),
                    row=2, col=2
                )
        
        # Add convergence threshold line
        fig.add_hline(
            y=1e-6,
            line_dash="dash",
            line_color="red",
            annotation_text="Convergence Threshold",
            row=2, col=1
        )
        
        fig.update_layout(
            title=title,
            height=800,
            width=1000
        )
        
        fig.update_xaxes(title_text="Iteration", row=1, col=2)
        fig.update_yaxes(title_text="Energy", row=1, col=2)
        fig.update_xaxes(title_text="Iteration", row=2, col=1)
        fig.update_yaxes(title_text="Gradient Norm", row=2, col=1, type="log")
        fig.update_xaxes(title_text="Runtime (minutes)", row=2, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=2)
        
        return fig
    
    def plot_job_timeline(self, job_ids: List[str] = None, title="Job Timeline"):
        """Create timeline plot for specific jobs."""
        if job_ids is None:
            with self.lock:
                job_ids = list(self.jobs.keys()) + list(self.completed_jobs.keys())
        
        fig = go.Figure()
        
        for i, job_id in enumerate(job_ids):
            progress = self.get_job_progress(job_id)
            if not progress:
                continue
            
            start_time = datetime.fromtimestamp(progress['start_time'])
            
            if progress['last_update']:
                end_time = datetime.fromtimestamp(progress['last_update'])
            else:
                end_time = datetime.now()
            
            # Color based on status
            color_map = {
                'running': 'blue',
                'converged': 'green',
                'failed': 'red',
                'max_iterations': 'orange',
                'stalled': 'gray'
            }
            color = color_map.get(progress['status'], 'blue')
            
            fig.add_trace(
                go.Scatter(
                    x=[start_time, end_time],
                    y=[i, i],
                    mode='lines+markers',
                    line=dict(width=10, color=color),
                    name=f"{job_id} ({progress['status']})",
                    hovertemplate=f"Job: {job_id}<br>Status: {progress['status']}<br>Start: {start_time}<br>Duration: {end_time - start_time}<extra></extra>"
                )
            )
        
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Job",
            yaxis=dict(
                tickmode='array',
                tickvals=list(range(len(job_ids))),
                ticktext=job_ids
            ),
            height=max(400, len(job_ids) * 50),
            width=1000
        )
        
        return fig
    
    def export_progress_report(self, output_file):
        """Export detailed progress report to file."""
        summary = self.get_summary_statistics()
        all_progress = self.get_all_progress()
        
        with open(output_file, 'w') as f:
            json.dump({
                'timestamp': time.time(),
                'summary': summary,
                'progress': all_progress
            }, f, indent=2)
        
        print(f"✓ Progress report exported to {output_file}")
    
    def load_progress_report(self, input_file):
        """Load progress report from file."""
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # Restore progress data
        with self.lock:
            for job_id, progress_data in data['progress']['active'].items():
                progress = OptimizationProgress(**progress_data)
                self.jobs[job_id] = progress
            
            for job_id, progress_data in data['progress']['completed'].items():
                progress = OptimizationProgress(**progress_data)
                self.completed_jobs[job_id] = progress
        
        print(f"✓ Loaded progress for {len(data['progress']['active'])} active and {len(data['progress']['completed'])} completed jobs")
    
    def clear_completed_jobs(self):
        """Clear completed jobs from memory."""
        with self.lock:
            count = len(self.completed_jobs)
            self.completed_jobs.clear()
        
        print(f"✓ Cleared {count} completed jobs from memory")
    
    def __enter__(self):
        """Context manager entry."""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_monitoring()