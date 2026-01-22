"""
Batch optimization tools for running multiple DFT calculations in parallel.
"""

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Callable, Any
import json
import numpy as np

from ..core.molecule import DrugMolecule
from ..quantum.dft_calculator import EnhancedDFTCalculator
from ..quantum.geometry_optimizer import NewtonRaphsonOptimizer
from .resource_manager import ResourceManager


@dataclass
class OptimizationJob:
    """Single optimization job definition."""
    job_id: str
    molecule: DrugMolecule
    method: str = 'B3LYP'
    basis: str = 'aug-cc-pVTZ'
    max_iterations: int = 100
    convergence_threshold: float = 1e-6
    priority: int = 0
    status: str = 'pending'
    result: Optional[Dict] = None
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None


class BatchOptimizer:
    """Manage batch optimization of multiple molecules."""
    
    def __init__(self, max_workers=4, use_gpu=False, memory_limit_gb=16):
        """Initialize batch optimizer."""
        self.max_workers = max_workers
        self.use_gpu = use_gpu
        self.memory_limit_gb = memory_limit_gb
        
        self.jobs = {}
        self.job_queue = []
        self.completed_jobs = []
        self.failed_jobs = []
        
        self.resource_manager = ResourceManager(
            max_memory_gb=memory_limit_gb,
            max_workers=max_workers
        )
        
        self.calculator = EnhancedDFTCalculator()
        self.optimizer = NewtonRaphsonOptimizer()
        
        self._progress_callbacks = []
        self._is_running = False
        self._executor = None
    
    def add_job(self, molecule: DrugMolecule, job_id: str = None, 
                method='B3LYP', basis='aug-cc-pVTZ', 
                max_iterations=100, priority=0):
        """Add optimization job to queue."""
        if job_id is None:
            job_id = f"job_{len(self.jobs):04d}"
        
        job = OptimizationJob(
            job_id=job_id,
            molecule=molecule,
            method=method,
            basis=basis,
            max_iterations=max_iterations,
            priority=priority
        )
        
        self.jobs[job_id] = job
        self.job_queue.append(job_id)
        
        # Sort queue by priority
        self.job_queue.sort(key=lambda jid: self.jobs[jid].priority, reverse=True)
        
        return job_id
    
    def add_molecule_list(self, molecules: List[DrugMolecule], 
                         job_prefix="batch", **kwargs):
        """Add multiple molecules as batch jobs."""
        job_ids = []
        
        for i, molecule in enumerate(molecules):
            job_id = f"{job_prefix}_{i:04d}"
            jid = self.add_job(molecule, job_id=job_id, **kwargs)
            job_ids.append(jid)
        
        return job_ids
    
    def run_batch(self, use_multiprocessing=True, 
                  progress_callback: Callable = None):
        """Run all jobs in the queue."""
        if not self.job_queue:
            print("No jobs in queue.")
            return
        
        if progress_callback:
            self._progress_callbacks.append(progress_callback)
        
        self._is_running = True
        
        try:
            if use_multiprocessing:
                self._run_with_multiprocessing()
            else:
                self._run_with_threading()
        finally:
            self._is_running = False
    
    def _run_with_multiprocessing(self):
        """Run jobs using ProcessPoolExecutor."""
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            self._executor = executor
            
            # Submit all jobs
            future_to_job = {}
            
            for job_id in self.job_queue[:]:
                if not self.resource_manager.can_allocate_job():
                    break
                
                job = self.jobs[job_id]
                job.status = 'running'
                job.start_time = time.time()
                
                future = executor.submit(self._run_single_job, job_id)
                future_to_job[future] = job_id
                
                self.job_queue.remove(job_id)
                self.resource_manager.allocate_job(job_id)
            
            # Collect results
            for future in as_completed(future_to_job):
                job_id = future_to_job[future]
                job = self.jobs[job_id]
                
                try:
                    result = future.result()
                    job.result = result
                    job.status = 'completed'
                    self.completed_jobs.append(job_id)
                    
                except Exception as e:
                    job.error = str(e)
                    job.status = 'failed'
                    self.failed_jobs.append(job_id)
                
                finally:
                    job.end_time = time.time()
                    self.resource_manager.release_job(job_id)
                    self._notify_progress()
                
                # Submit next job if available
                if self.job_queue and self.resource_manager.can_allocate_job():
                    next_job_id = self.job_queue.pop(0)
                    next_job = self.jobs[next_job_id]
                    next_job.status = 'running'
                    next_job.start_time = time.time()
                    
                    future = executor.submit(self._run_single_job, next_job_id)
                    future_to_job[future] = next_job_id
                    self.resource_manager.allocate_job(next_job_id)
    
    def _run_with_threading(self):
        """Run jobs using ThreadPoolExecutor."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            self._executor = executor
            
            future_to_job = {}
            
            for job_id in self.job_queue[:]:
                job = self.jobs[job_id]
                job.status = 'running'
                job.start_time = time.time()
                
                future = executor.submit(self._run_single_job, job_id)
                future_to_job[future] = job_id
                
                self.job_queue.remove(job_id)
            
            for future in as_completed(future_to_job):
                job_id = future_to_job[future]
                job = self.jobs[job_id]
                
                try:
                    result = future.result()
                    job.result = result
                    job.status = 'completed'
                    self.completed_jobs.append(job_id)
                    
                except Exception as e:
                    job.error = str(e)
                    job.status = 'failed'
                    self.failed_jobs.append(job_id)
                
                finally:
                    job.end_time = time.time()
                    self._notify_progress()
    
    def _run_single_job(self, job_id: str):
        """Run a single optimization job."""
        job = self.jobs[job_id]
        molecule = job.molecule
        
        # Setup calculator
        self.calculator.setup_calculation(
            molecule.mol, 
            method=job.method,
            basis=job.basis
        )
        
        # Setup optimizer
        self.optimizer.setup(
            max_iterations=job.max_iterations,
            convergence_threshold=job.convergence_threshold
        )
        
        # Run optimization
        initial_coords = self.calculator.get_coordinates()
        
        def energy_gradient_func(coords):
            self.calculator.mol.SetPositions(coords.reshape(-1, 3))
            energy = self.calculator.calculate_energy()
            gradient = self.calculator.calculate_gradient()
            return energy, gradient.flatten()
        
        # Optimize geometry
        opt_coords, opt_energy, trajectory = self.optimizer.optimize(
            initial_coords.flatten(),
            energy_gradient_func
        )
        
        # Final calculations
        self.calculator.mol.SetPositions(opt_coords.reshape(-1, 3))
        final_energy = self.calculator.calculate_energy()
        final_gradient = self.calculator.calculate_gradient()
        
        # Compile results
        result = {
            'job_id': job_id,
            'initial_energy': self.calculator.calculate_energy(initial_coords),
            'final_energy': final_energy,
            'optimized_coordinates': opt_coords.tolist(),
            'final_gradient_norm': np.linalg.norm(final_gradient),
            'optimization_steps': len(trajectory),
            'convergence_achieved': np.linalg.norm(final_gradient) < job.convergence_threshold,
            'trajectory': [step.tolist() for step in trajectory]
        }
        
        return result
    
    def _notify_progress(self):
        """Notify progress callbacks."""
        progress_info = {
            'total_jobs': len(self.jobs),
            'completed': len(self.completed_jobs),
            'failed': len(self.failed_jobs),
            'running': sum(1 for job in self.jobs.values() if job.status == 'running'),
            'pending': len(self.job_queue)
        }
        
        for callback in self._progress_callbacks:
            try:
                callback(progress_info)
            except Exception as e:
                print(f"Progress callback error: {e}")
    
    def get_status(self):
        """Get overall batch status."""
        return {
            'total_jobs': len(self.jobs),
            'pending': len(self.job_queue),
            'running': sum(1 for job in self.jobs.values() if job.status == 'running'),
            'completed': len(self.completed_jobs),
            'failed': len(self.failed_jobs),
            'is_running': self._is_running
        }
    
    def get_job_results(self, job_ids: List[str] = None):
        """Get results for specified jobs or all jobs."""
        if job_ids is None:
            job_ids = list(self.jobs.keys())
        
        results = {}
        for job_id in job_ids:
            if job_id in self.jobs:
                job = self.jobs[job_id]
                results[job_id] = {
                    'status': job.status,
                    'result': job.result,
                    'error': job.error,
                    'runtime': (job.end_time - job.start_time) if job.end_time and job.start_time else None
                }
        
        return results
    
    def cancel_jobs(self, job_ids: List[str] = None):
        """Cancel pending or running jobs."""
        if job_ids is None:
            job_ids = [jid for jid, job in self.jobs.items() 
                      if job.status in ['pending', 'running']]
        
        cancelled = []
        for job_id in job_ids:
            if job_id in self.jobs:
                job = self.jobs[job_id]
                if job.status == 'pending':
                    job.status = 'cancelled'
                    if job_id in self.job_queue:
                        self.job_queue.remove(job_id)
                    cancelled.append(job_id)
        
        return cancelled
    
    def save_results(self, output_file: Path):
        """Save all job results to file."""
        results = {
            'batch_info': {
                'total_jobs': len(self.jobs),
                'completed': len(self.completed_jobs),
                'failed': len(self.failed_jobs),
                'timestamp': time.time()
            },
            'jobs': {}
        }
        
        for job_id, job in self.jobs.items():
            results['jobs'][job_id] = {
                'status': job.status,
                'method': job.method,
                'basis': job.basis,
                'priority': job.priority,
                'result': job.result,
                'error': job.error,
                'runtime': (job.end_time - job.start_time) if job.end_time and job.start_time else None
            }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✓ Batch results saved to {output_file}")
    
    def load_results(self, input_file: Path):
        """Load job results from file."""
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        for job_id, job_data in data['jobs'].items():
            if job_id in self.jobs:
                job = self.jobs[job_id]
                job.status = job_data['status']
                job.result = job_data['result']
                job.error = job_data['error']
        
        print(f"✓ Loaded results for {len(data['jobs'])} jobs")
    
    def clear_completed(self):
        """Remove completed and failed jobs from memory."""
        to_remove = self.completed_jobs + self.failed_jobs
        
        for job_id in to_remove:
            if job_id in self.jobs:
                del self.jobs[job_id]
        
        self.completed_jobs.clear()
        self.failed_jobs.clear()
        
        print(f"✓ Cleared {len(to_remove)} completed jobs from memory")
    
    def estimate_remaining_time(self):
        """Estimate remaining computation time."""
        completed_times = []
        
        for job_id in self.completed_jobs:
            job = self.jobs.get(job_id)
            if job and job.start_time and job.end_time:
                completed_times.append(job.end_time - job.start_time)
        
        if not completed_times:
            return None
        
        avg_time = np.mean(completed_times)
        remaining_jobs = len(self.job_queue) + sum(1 for job in self.jobs.values() 
                                                 if job.status == 'running')
        
        return avg_time * remaining_jobs / self.max_workers