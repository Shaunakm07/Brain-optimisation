"""
Quick Start Guide - NeuroStim Optimization Engine

This script demonstrates basic usage patterns for the system.
"""

import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_1_basic_pipeline():
    """
    Example 1: Run the basic optimization pipeline with default config
    """
    logger.info("\n" + "=" * 70)
    logger.info("EXAMPLE 1: Basic Pipeline")
    logger.info("=" * 70)
    
    from pipeline import NeuroStimPipeline
    
    # Initialize pipeline with default config
    pipeline = NeuroStimPipeline("neurostim_config.yaml")
    
    # Setup all components
    logger.info("Setting up pipeline...")
    pipeline.setup()
    
    # Run optimization
    logger.info("Starting optimization...")
    results = pipeline.run_experiment()
    
    # Generate report
    pipeline.generate_report(results)
    
    logger.info(f"✓ Results saved to {pipeline.output_dir}")
    return results


def example_2_custom_roi():
    """
    Example 2: Optimize for a custom ROI
    """
    logger.info("\n" + "=" * 70)
    logger.info("EXAMPLE 2: Custom ROI Configuration")
    logger.info("=" * 70)
    
    from pipeline import NeuroStimPipeline
    import yaml
    
    # Create custom configuration
    custom_config = {
        "experiment": {
            "name": "custom_roi_optimization",
            "modality": "image",
            "description": "Optimize for a custom brain region"
        },
        "target_roi": {
            "name": "Superior Temporal Sulcus (STS)",
            "description": "Theory of Mind region",
            "vertex_ranges": {
                "left": [7500, 8500],
                "right": [27500, 28500]
            }
        },
        "off_target_rois": [
            {
                "name": "Primary Visual Cortex",
                "vertex_ranges": {"left": [0, 1000], "right": [20000, 21000]},
                "weight": 0.2
            }
        ],
        "generator": {
            "type": "stable_diffusion",
            "device": "cuda",
            "num_inference_steps": 50
        },
        "optimization": {
            "method": "evolutionary",
            "num_iterations": 50,
            "population_size": 8
        },
        "reward": {
            "target_activation_weight": 1.0,
            "off_target_suppression_weight": 0.3,
            "energy_penalty_weight": 0.05,
            "temporal_consistency_weight": 0.1
        },
        "tribe": {"model_name": "facebook/tribev2", "device": "cuda"},
        "visualization": {"output_dir": "./outputs/custom_roi"},
        "hardware": {"device": "cuda", "mixed_precision": True}
    }
    
    # Save custom config
    config_path = Path("custom_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(custom_config, f)
    
    logger.info(f"Created custom config: {config_path}")
    
    # Run with custom config
    pipeline = NeuroStimPipeline(str(config_path))
    pipeline.setup()
    results = pipeline.run_experiment()
    
    logger.info(f"✓ Custom ROI optimization complete")
    return results


def example_3_compare_methods():
    """
    Example 3: Compare different optimization methods
    """
    logger.info("\n" + "=" * 70)
    logger.info("EXAMPLE 3: Comparing Optimization Methods")
    logger.info("=" * 70)
    
    from pipeline import NeuroStimPipeline
    import yaml
    
    methods = ["evolutionary", "cma_es"]
    results_by_method = {}
    
    for method in methods:
        logger.info(f"\nTesting method: {method.upper()}")
        
        # Load base config
        with open("neurostim_config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        # Modify for this method
        config["optimization"]["method"] = method
        config["optimization"]["num_iterations"] = 30  # Quick test
        config["visualization"]["output_dir"] = f"./outputs/{method}_comparison"
        
        # Save modified config
        config_path = Path(f"config_{method}.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        
        # Run optimization
        pipeline = NeuroStimPipeline(str(config_path))
        pipeline.setup()
        results = pipeline.run_experiment()
        
        results_by_method[method] = results
    
    # Compare results
    logger.info("\n" + "=" * 70)
    logger.info("COMPARISON RESULTS")
    logger.info("=" * 70)
    for method, results in results_by_method.items():
        logger.info(
            f"{method:15} - "
            f"Best reward: {results['final_reward']:.4f}, "
            f"Iterations: {results['num_iterations']}"
        )
    
    return results_by_method


def example_4_multi_modal():
    """
    Example 4: Run experiments with different modalities
    """
    logger.info("\n" + "=" * 70)
    logger.info("EXAMPLE 4: Multi-Modal Experiments")
    logger.info("=" * 70)
    
    from pipeline import NeuroStimPipeline
    from example_experiments import load_experiment
    
    modalities = {
        "ffa": "Face optimization with video",
        "auditory": "Auditory cortex optimization",
        "quick": "Quick test (all modalities)"
    }
    
    results_by_modality = {}
    
    for exp_name, description in modalities.items():
        logger.info(f"\n{description}...")
        
        try:
            # This assumes configs exist from example_experiments.py --setup
            config = load_experiment(exp_name)
            
            # Modify for quick testing
            config["optimization"]["num_iterations"] = 20
            
            # Create pipeline and run
            import tempfile
            import yaml
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(config, f)
                config_path = f.name
            
            pipeline = NeuroStimPipeline(config_path)
            pipeline.setup()
            results = pipeline.run_experiment()
            
            results_by_modality[exp_name] = results
            
            # Cleanup
            Path(config_path).unlink()
            
        except Exception as e:
            logger.warning(f"Could not run {exp_name}: {e}")
    
    return results_by_modality


def example_5_custom_reward():
    """
    Example 5: Implement custom reward function
    """
    logger.info("\n" + "=" * 70)
    logger.info("EXAMPLE 5: Custom Reward Function")
    logger.info("=" * 70)
    
    from reward_function import RewardFunction
    import numpy as np
    
    # Create custom reward function
    class CustomRewardFunction(RewardFunction):
        """
        Custom reward that emphasizes temporal consistency
        and peak activation rather than mean.
        """
        
        def compute_reward(self, preds, tribe_wrapper, return_components=False):
            # Get target activation (use peak instead of mean)
            target_activation = self._compute_target_activation_peak(preds, tribe_wrapper)
            target_reward = self.w_target * target_activation
            
            # Get off-target suppression
            off_target = self._compute_off_target_activation(preds, tribe_wrapper)
            off_target_penalty = self.w_off_target * off_target
            
            # Temporal smoothness
            temporal_penalty = self.w_temporal * self._compute_temporal_penalty(preds)
            
            total_reward = target_reward - off_target_penalty - temporal_penalty
            
            if return_components:
                return {
                    "total": float(total_reward),
                    "target_activation": float(target_activation),
                    "target_reward": float(target_reward),
                    "off_target_penalty": float(off_target_penalty),
                    "temporal_penalty": float(temporal_penalty)
                }
            
            return float(total_reward)
        
        def _compute_target_activation_peak(self, preds, tribe_wrapper):
            """Use peak activation instead of mean."""
            roi_vertex_ranges = self.target_roi["vertex_ranges"]
            roi_activity = tribe_wrapper.get_roi_activation(
                preds, roi_vertex_ranges, aggregation="max"  # Use max instead of mean
            )
            
            activations = list(roi_activity.values())
            mean_activation = np.mean(activations)
            normalized = np.clip(mean_activation / 3.0, 0, 1)
            
            return float(normalized)
    
    logger.info("✓ Custom reward function created")
    logger.info("  Features:")
    logger.info("  - Uses peak activation instead of mean")
    logger.info("  - Emphasizes temporal smoothness")
    logger.info("  - Can be swapped into any pipeline")
    
    return CustomRewardFunction


def example_6_visualization():
    """
    Example 6: Generate visualizations from results
    """
    logger.info("\n" + "=" * 70)
    logger.info("EXAMPLE 6: Visualization Generation")
    logger.info("=" * 70)
    
    from visualization import NeuroVisualization
    import numpy as np
    
    # Create visualizer
    viz = NeuroVisualization(output_dir="./outputs/visualizations")
    
    # Generate synthetic data for demo
    rewards = np.array([0.1 * (1 + 0.01 * i + 0.02 * np.sin(i/10)) 
                        for i in range(50)])
    
    # Generate plots
    logger.info("Generating optimization progress plot...")
    viz.plot_optimization_progress(
        rewards,
        title="Optimization Progress",
        save_path="./outputs/visualizations/progress.png"
    )
    
    logger.info("Generating reward breakdown...")
    reward_components = {
        "target_activation": 0.8,
        "off_target_suppression": -0.3,
        "energy_penalty": -0.05,
        "temporal_penalty": -0.1
    }
    viz.plot_reward_breakdown(
        reward_components,
        save_path="./outputs/visualizations/breakdown.png"
    )
    
    logger.info("✓ Visualizations generated")


def example_7_batch_experiment():
    """
    Example 7: Run batch of experiments with different configs
    """
    logger.info("\n" + "=" * 70)
    logger.info("EXAMPLE 7: Batch Experiments")
    logger.info("=" * 70)
    
    from pipeline import NeuroStimPipeline
    import yaml
    import json
    
    # Define experiment grid
    roi_configs = {
        "ffa": {"left": [4500, 5200], "right": [24500, 25200]},
        "v1": {"left": [0, 1000], "right": [20000, 21000]},
        "auditory": {"left": [8000, 9000], "right": [28000, 29000]},
    }
    
    weight_configs = [
        {"target": 1.0, "off_target": 0.2},
        {"target": 1.0, "off_target": 0.5},
        {"target": 1.5, "off_target": 0.3},
    ]
    
    results_matrix = {}
    
    logger.info(f"Running {len(roi_configs)} × {len(weight_configs)} experiments...")
    
    for roi_name, roi_vertices in roi_configs.items():
        results_matrix[roi_name] = {}
        
        for weight_idx, weights in enumerate(weight_configs):
            exp_name = f"{roi_name}_weights{weight_idx}"
            logger.info(f"  Running {exp_name}...")
            
            # Create config
            config = {
                "experiment": {
                    "name": exp_name,
                    "modality": "image"
                },
                "target_roi": {
                    "name": roi_name,
                    "vertex_ranges": roi_vertices
                },
                "off_target_rois": [],
                "generator": {"type": "stable_diffusion", "device": "cuda"},
                "optimization": {
                    "method": "evolutionary",
                    "num_iterations": 10  # Quick
                },
                "reward": {
                    "target_activation_weight": weights["target"],
                    "off_target_suppression_weight": weights["off_target"],
                    "energy_penalty_weight": 0.05,
                    "temporal_consistency_weight": 0.1
                },
                "tribe": {"model_name": "facebook/tribev2", "device": "cuda"},
                "visualization": {"output_dir": f"./outputs/batch/{exp_name}"},
                "hardware": {"device": "cuda"}
            }
            
            # Save and run
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(config, f)
                config_path = f.name
            
            try:
                pipeline = NeuroStimPipeline(config_path)
                pipeline.setup()
                results = pipeline.run_experiment()
                results_matrix[roi_name][f"weights{weight_idx}"] = results["final_reward"]
            except Exception as e:
                logger.warning(f"Error in {exp_name}: {e}")
            finally:
                Path(config_path).unlink()
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("BATCH RESULTS SUMMARY")
    logger.info("=" * 70)
    for roi, results in results_matrix.items():
        logger.info(f"{roi:15} {results}")
    
    return results_matrix


def main():
    """Run all examples."""
    import argparse
    
    parser = argparse.ArgumentParser(description="NeuroStim Quick Start Examples")
    parser.add_argument(
        "--example",
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7],
        help="Run specific example (1-7)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all examples"
    )
    
    args = parser.parse_args()
    
    examples = {
        1: ("Basic Pipeline", example_1_basic_pipeline),
        2: ("Custom ROI", example_2_custom_roi),
        3: ("Compare Methods", example_3_compare_methods),
        4: ("Multi-Modal", example_4_multi_modal),
        5: ("Custom Reward", example_5_custom_reward),
        6: ("Visualization", example_6_visualization),
        7: ("Batch Experiments", example_7_batch_experiment),
    }
    
    if args.example:
        title, func = examples[args.example]
        logger.info(f"\nRunning Example {args.example}: {title}")
        try:
            func()
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
    
    elif args.all:
        for num, (title, func) in examples.items():
            logger.info(f"\n\nRunning Example {num}: {title}")
            try:
                func()
            except Exception as e:
                logger.warning(f"Error in Example {num}: {e}")
    
    else:
        logger.info("\nAvailable Examples:")
        for num, (title, _) in examples.items():
            logger.info(f"  {num}. {title}")
        logger.info("\nUsage:")
        logger.info("  python quick_start.py --example 1")
        logger.info("  python quick_start.py --all")


if __name__ == "__main__":
    main()
