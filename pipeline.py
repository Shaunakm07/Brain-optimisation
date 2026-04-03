"""
NeuroStim Optimization Pipeline
Complete end-to-end system for neuro-optimization with TRIBE V2
"""

import numpy as np
import torch
import yaml
from pathlib import Path
from typing import Dict, Optional
import logging
import json
from datetime import datetime

# Import project modules
from tribe_wrapper import TribeV2Wrapper
from stimulus_generator import ImageGenerator, VideoGenerator, AudioGenerator
from reward_function import RewardFunction
from optimization_engine import LatentOptimizer
from visualization import NeuroVisualization

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class NeuroStimPipeline:
    """
    Complete NeuroStim optimization pipeline.
    
    Workflow:
    1. Load configuration
    2. Initialize TRIBE V2 brain model
    3. Initialize stimulus generator(s)
    4. Define reward function
    5. Run optimization
    6. Visualize results
    """
    
    def __init__(self, config_path: str = "neurostim_config.yaml"):
        """
        Initialize pipeline from configuration file.
        
        Args:
            config_path: Path to YAML configuration
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Setup directories
        self.output_dir = Path(self.config["visualization"]["output_dir"])
        self.checkpoint_dir = Path(self.config["logging"]["checkpoint_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized NeuroStimPipeline from {config_path}")
        
        # Initialize components
        self.tribe = None
        self.generator = None
        self.reward_fn = None
        self.optimizer = None
        self.visualizer = None
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration: {config['experiment']['name']}")
        return config
    
    def setup(self) -> None:
        """Initialize all pipeline components."""
        logger.info("=" * 70)
        logger.info("SETTING UP NEUROSTIM PIPELINE")
        logger.info("=" * 70)
        
        # 1. Initialize TRIBE V2
        self._setup_tribe()
        
        # 2. Initialize stimulus generator
        self._setup_generator()
        
        # 3. Initialize reward function
        self._setup_reward()
        
        # 4. Initialize optimizer
        self._setup_optimizer()
        
        # 5. Initialize visualization
        self._setup_visualization()
        
        logger.info("Pipeline setup complete!")
    
    def _setup_tribe(self) -> None:
        """Initialize TRIBE V2 brain model."""
        logger.info("Initializing TRIBE V2...")
        
        tribe_config = self.config.get("tribe", {})
        self.tribe = TribeV2Wrapper(
            model_name=tribe_config.get("model_name", "facebook/tribev2"),
            device=tribe_config.get("device", "cuda")
        )
        
        logger.info(f"✓ TRIBE V2 initialized")
    
    def _setup_generator(self) -> None:
        """Initialize stimulus generator based on modality."""
        logger.info("Initializing stimulus generator...")
        
        modality = self.config["experiment"]["modality"]
        gen_config = self.config["generator"]
        
        if modality == "image":
            self.generator = ImageGenerator(
                model_name=gen_config.get("model_name", "runwayml/stable-diffusion-v1-5"),
                device=gen_config.get("device", "cuda"),
                num_inference_steps=gen_config.get("num_inference_steps", 50),
                guidance_scale=gen_config.get("guidance_scale", 7.5)
            )
        elif modality == "video":
            self.generator = VideoGenerator(
                device=gen_config.get("device", "cuda"),
                num_frames=self.config["experiment"].get("duration_frames", 16)
            )
        elif modality == "audio":
            self.generator = AudioGenerator(
                device=gen_config.get("device", "cuda"),
                sample_rate=self.config["experiment"].get("sample_rate", 16000)
            )
        else:
            raise ValueError(f"Unknown modality: {modality}")
        
        logger.info(f"✓ {modality.capitalize()} generator initialized")
    
    def _setup_reward(self) -> None:
        """Initialize reward function."""
        logger.info("Initializing reward function...")
        
        reward_config = self.config["reward"]
        
        self.reward_fn = RewardFunction(
            target_roi=self.config["target_roi"],
            off_target_rois=self.config["off_target_rois"],
            target_weight=reward_config.get("target_activation_weight", 1.0),
            off_target_weight=reward_config.get("off_target_suppression_weight", 0.3),
            energy_weight=reward_config.get("energy_penalty_weight", 0.05),
            temporal_weight=reward_config.get("temporal_consistency_weight", 0.1),
            temporal_aggregation=reward_config.get("temporal_aggregation", "mean")
        )
        
        logger.info(f"✓ Reward function initialized")
    
    def _setup_optimizer(self) -> None:
        """Initialize optimization engine."""
        logger.info("Initializing optimizer...")
        
        self.optimizer = LatentOptimizer(
            generator=self.generator,
            tribe_wrapper=self.tribe,
            reward_function=self.reward_fn,
            latent_dim=256,
            device=self.config["hardware"]["device"]
        )
        
        logger.info(f"✓ Optimizer initialized")
    
    def _setup_visualization(self) -> None:
        """Initialize visualization module."""
        self.visualizer = NeuroVisualization(
            output_dir=str(self.output_dir)
        )
    
    def run_experiment(self) -> Dict:
        """
        Run the complete optimization experiment.
        
        Returns:
            results: Dictionary with optimization results
        """
        logger.info("=" * 70)
        logger.info("STARTING OPTIMIZATION EXPERIMENT")
        logger.info("=" * 70)
        
        exp_config = self.config["experiment"]
        opt_config = self.config["optimization"]
        
        logger.info(f"Experiment: {exp_config['name']}")
        logger.info(f"Modality: {exp_config['modality']}")
        logger.info(f"Target ROI: {self.config['target_roi']['name']}")
        
        # Run optimization
        method = opt_config.get("method", "evolutionary")
        
        if method == "cma_es":
            state = self._run_cmaes_optimization()
        elif method == "evolutionary":
            state = self._run_evolutionary_optimization()
        elif method == "ppo":
            state = self._run_ppo_optimization()
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        # Save results
        results = self._save_results(state)
        
        # Visualize results
        self._visualize_results(state)
        
        logger.info("=" * 70)
        logger.info("OPTIMIZATION COMPLETE")
        logger.info("=" * 70)
        
        return results
    
    def _run_evolutionary_optimization(self) -> "OptimizationState":
        """Run evolutionary strategy optimization."""
        opt_config = self.config["optimization"]
        
        logger.info("Running Evolutionary Strategy optimization...")
        
        state = self.optimizer.optimize_evolutionary(
            modality=self.config["experiment"]["modality"],
            num_iterations=opt_config.get("num_iterations", 100),
            population_size=opt_config.get("population_size", 8),
            mutation_std=0.1,
            elite_fraction=0.2,
            seed=42
        )
        
        return state
    
    def _run_cmaes_optimization(self) -> "OptimizationState":
        """Run CMA-ES optimization."""
        opt_config = self.config["optimization"]
        
        logger.info("Running CMA-ES optimization...")
        
        state = self.optimizer.optimize_cmaes(
            modality=self.config["experiment"]["modality"],
            num_iterations=opt_config.get("num_iterations", 100),
            population_size=opt_config.get("population_size", 8),
            seed=42
        )
        
        return state
    
    def _run_ppo_optimization(self) -> "OptimizationState":
        """Run PPO optimization (placeholder)."""
        logger.warning("PPO not yet implemented, using evolutionary instead")
        return self._run_evolutionary_optimization()
    
    def _save_results(self, state: "OptimizationState") -> Dict:
        """Save optimization results to disk."""
        logger.info("Saving results...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results = {
            "experiment": self.config["experiment"]["name"],
            "timestamp": timestamp,
            "final_reward": float(state.best_reward),
            "mean_reward": float(state.mean_reward),
            "num_iterations": state.iteration,
            "rewards_over_time": [float(r) for r in state.rewards],
            "target_roi": self.config["target_roi"]["name"],
            "modality": self.config["experiment"]["modality"],
            "optimization_method": self.config["optimization"]["method"]
        }
        
        # Save JSON
        results_file = self.output_dir / f"results_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to: {results_file}")
        
        return results
    
    def _visualize_results(self, state: "OptimizationState") -> None:
        """Generate visualizations of results."""
        logger.info("Generating visualizations...")
        
        # 1. Optimization progress
        self.visualizer.plot_optimization_progress(
            state.rewards,
            title=f"Optimization Progress - {self.config['experiment']['name']}",
            save_path=str(self.output_dir / "optimization_progress.png")
        )
        
        # 2. Reward breakdown (sample)
        sample_preds = np.random.randn(60, 40962)  # Dummy predictions
        reward_dict = self.reward_fn.compute_reward(
            sample_preds,
            self.tribe,
            return_components=True
        )
        
        self.visualizer.plot_reward_breakdown(
            reward_dict,
            save_path=str(self.output_dir / "reward_breakdown.png")
        )
        
        logger.info(f"Visualizations saved to: {self.output_dir}")
    
    def generate_report(self, results: Dict) -> None:
        """Generate a text summary report."""
        report_path = self.output_dir / "experiment_report.txt"
        
        with open(report_path, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("NEUROSTIM OPTIMIZATION EXPERIMENT REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("EXPERIMENT CONFIGURATION\n")
            f.write("-" * 70 + "\n")
            f.write(f"Name: {results['experiment']}\n")
            f.write(f"Timestamp: {results['timestamp']}\n")
            f.write(f"Modality: {results['modality']}\n")
            f.write(f"Target ROI: {results['target_roi']}\n")
            f.write(f"Method: {results['optimization_method']}\n\n")
            
            f.write("RESULTS\n")
            f.write("-" * 70 + "\n")
            f.write(f"Final Reward: {results['final_reward']:.4f}\n")
            f.write(f"Mean Reward: {results['mean_reward']:.4f}\n")
            f.write(f"Iterations: {results['num_iterations']}\n")
            f.write(f"Best Reward: {max(results['rewards_over_time']):.4f}\n")
            f.write(f"Initial Reward: {results['rewards_over_time'][0]:.4f}\n")
            f.write(f"Improvement: {max(results['rewards_over_time']) - results['rewards_over_time'][0]:.4f}\n")
            f.write("\n")
            
            f.write("CONFIGURATION DETAILS\n")
            f.write("-" * 70 + "\n")
            for key, value in self.config.items():
                f.write(f"\n{key}:\n")
                if isinstance(value, dict):
                    for k, v in value.items():
                        f.write(f"  {k}: {v}\n")
                else:
                    f.write(f"  {value}\n")
        
        logger.info(f"Report saved to: {report_path}")


def main():
    """Main entry point for the pipeline."""
    logger.info("Starting NeuroStim Optimization Engine...")
    
    # Initialize pipeline
    pipeline = NeuroStimPipeline("neurostim_config.yaml")
    
    # Setup all components
    pipeline.setup()
    
    # Run experiment
    results = pipeline.run_experiment()
    
    # Generate report
    pipeline.generate_report(results)
    
    logger.info("Pipeline execution complete!")
    logger.info(f"Results saved to: {pipeline.output_dir}")
    
    return results


if __name__ == "__main__":
    results = main()
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Best Reward: {results['final_reward']:.4f}")
    print(f"Iterations: {results['num_iterations']}")
    print(f"Method: {results['optimization_method']}")
