"""
Example Experiments
Pre-configured experiments for common neuro-optimization tasks
"""

import yaml
from pathlib import Path


# ============================================================================
# EXPERIMENT 1: Face Region Optimization (FFA)
# ============================================================================

FACE_OPTIMIZATION_CONFIG = """
# Face Region Optimization - Target Fusiform Face Area (FFA)
experiment:
  name: "ffa_face_optimization"
  description: "Optimize video stimuli to maximize activation in FFA"
  modality: "video"
  duration_frames: 60
  fps: 30

target_roi:
  name: "Fusiform Face Area (FFA)"
  description: "Face-selective region in ventral temporal cortex"
  hemisphere: "bilateral"
  vertex_ranges:
    left: [4500, 5200]
    right: [24500, 25200]
  activation_threshold: 0.5

off_target_rois:
  - name: "Primary Visual Cortex (V1)"
    vertex_ranges:
      left: [0, 1000]
      right: [20000, 21000]
    weight: 0.2
  - name: "Middle Temporal Area (MT)"
    vertex_ranges:
      left: [5500, 6500]
      right: [25500, 26500]
    weight: 0.1

generator:
  type: "stable_diffusion"
  model_name: "runwayml/stable-diffusion-v1-5"
  device: "cuda"
  num_inference_steps: 50
  guidance_scale: 7.5
  seed: 42

optimization:
  method: "evolutionary"
  num_iterations: 150
  population_size: 16
  mutation_std: 0.1
  elite_fraction: 0.25
  learning_rate: 0.001
  entropy_coef: 0.01

reward:
  target_activation_weight: 1.0
  off_target_suppression_weight: 0.4
  energy_penalty_weight: 0.05
  temporal_consistency_weight: 0.1
  temporal_aggregation: "mean"

tribe:
  model_name: "facebook/tribev2"
  device: "cuda"
  batch_size: 4
  cache_predictions: true
  average_timesteps: true

visualization:
  save_frequency: 20
  plot_brain_surface: true
  plot_roi_timecourse: true
  plot_stimulus_evolution: true
  output_dir: "./outputs/ffa_optimization"

hardware:
  device: "cuda"
  num_workers: 4
  mixed_precision: true
"""


# ============================================================================
# EXPERIMENT 2: Auditory Cortex Optimization
# ============================================================================

AUDIO_OPTIMIZATION_CONFIG = """
# Auditory Cortex Optimization
experiment:
  name: "auditory_cortex_optimization"
  description: "Optimize audio stimuli to maximize activation in primary auditory cortex"
  modality: "audio"
  sample_rate: 16000
  duration: 5.0

target_roi:
  name: "Primary Auditory Cortex (A1)"
  hemisphere: "bilateral"
  vertex_ranges:
    left: [8000, 9200]
    right: [28000, 29200]
  activation_threshold: 0.5

off_target_rois:
  - name: "Visual Cortex V1"
    vertex_ranges:
      left: [0, 1000]
      right: [20000, 21000]
    weight: 0.15
  - name: "Superior Temporal Sulcus (STS)"
    vertex_ranges:
      left: [7000, 8000]
      right: [27000, 28000]
    weight: 0.1

generator:
  type: "audio_spectrogram"
  device: "cuda"
  sample_rate: 16000
  duration: 5.0

optimization:
  method: "evolutionary"
  num_iterations: 100
  population_size: 12
  mutation_std: 0.15
  elite_fraction: 0.2

reward:
  target_activation_weight: 1.0
  off_target_suppression_weight: 0.3
  energy_penalty_weight: 0.05
  temporal_consistency_weight: 0.15

visualization:
  output_dir: "./outputs/auditory_optimization"

hardware:
  device: "cuda"
  mixed_precision: true
"""


# ============================================================================
# EXPERIMENT 3: Language Region Optimization (Broca's Area)
# ============================================================================

LANGUAGE_OPTIMIZATION_CONFIG = """
# Language Region Optimization - Broca's Area
experiment:
  name: "language_region_optimization"
  description: "Optimize stimuli to maximize activation in language areas"
  modality: "video"
  duration_frames: 90
  fps: 30

target_roi:
  name: "Broca's Area (IFG)"
  hemisphere: "left"
  vertex_ranges:
    left: [10000, 11500]
    right: [30000, 31500]
  activation_threshold: 0.5

off_target_rois:
  - name: "Visual Cortex"
    vertex_ranges:
      left: [0, 1500]
      right: [20000, 21500]
    weight: 0.2
  - name: "Motor Cortex"
    vertex_ranges:
      left: [15000, 16000]
      right: [35000, 36000]
    weight: 0.1

generator:
  type: "stable_diffusion"
  model_name: "runwayml/stable-diffusion-v1-5"
  device: "cuda"
  num_inference_steps: 40
  guidance_scale: 7.5

optimization:
  method: "cma_es"
  num_iterations: 120
  population_size: 16
  sigma: 0.5

reward:
  target_activation_weight: 1.0
  off_target_suppression_weight: 0.3
  energy_penalty_weight: 0.08
  temporal_consistency_weight: 0.12

visualization:
  output_dir: "./outputs/language_optimization"

hardware:
  device: "cuda"
  mixed_precision: true
"""


# ============================================================================
# EXPERIMENT 4: Multi-Objective (Balance Face + Suppress Other)
# ============================================================================

MULTI_OBJECTIVE_CONFIG = """
# Multi-Objective Optimization
experiment:
  name: "multi_objective_optimization"
  description: "Balance face activation with suppression of other regions"
  modality: "image"

target_roi:
  name: "Fusiform Face Area"
  vertex_ranges:
    left: [4500, 5200]
    right: [24500, 25200]

off_target_rois:
  - name: "Visual Cortex V1"
    vertex_ranges:
      left: [0, 1000]
      right: [20000, 21000]
    weight: 0.25
  - name: "Place Area (PPA)"
    vertex_ranges:
      left: [5700, 6500]
      right: [25700, 26500]
    weight: 0.3
  - name: "Object Area (LOa)"
    vertex_ranges:
      left: [6600, 7500]
      right: [26600, 27500]
    weight: 0.25

generator:
  type: "stable_diffusion"
  model_name: "runwayml/stable-diffusion-v1-5"
  device: "cuda"
  num_inference_steps: 50
  guidance_scale: 7.5

optimization:
  method: "evolutionary"
  num_iterations: 200
  population_size: 20
  mutation_std: 0.12
  elite_fraction: 0.25

reward:
  target_activation_weight: 1.5  # Emphasize target
  off_target_suppression_weight: 0.6  # Stronger suppression
  energy_penalty_weight: 0.1  # Avoid extreme activations
  temporal_consistency_weight: 0.05

visualization:
  output_dir: "./outputs/multi_objective"

hardware:
  device: "cuda"
  mixed_precision: true
"""


# ============================================================================
# EXPERIMENT 5: Quick Test (Low Computational Cost)
# ============================================================================

QUICK_TEST_CONFIG = """
# Quick Test Configuration
# For testing without long computation
experiment:
  name: "quick_test"
  description: "Fast test run with minimal iterations"
  modality: "image"

target_roi:
  name: "FFA"
  vertex_ranges:
    left: [4500, 5200]
    right: [24500, 25200]

off_target_rois:
  - name: "V1"
    vertex_ranges:
      left: [0, 1000]
      right: [20000, 21000]
    weight: 0.2

generator:
  type: "stable_diffusion"
  model_name: "runwayml/stable-diffusion-v1-5"
  device: "cuda"
  num_inference_steps: 30
  guidance_scale: 7.5

optimization:
  method: "evolutionary"
  num_iterations: 20  # Very quick
  population_size: 8

reward:
  target_activation_weight: 1.0
  off_target_suppression_weight: 0.3
  energy_penalty_weight: 0.05
  temporal_consistency_weight: 0.1

visualization:
  output_dir: "./outputs/quick_test"

hardware:
  device: "cuda"
  mixed_precision: true
"""


# ============================================================================
# EXPERIMENT 6: Video with CMA-ES (Advanced)
# ============================================================================

VIDEO_CMAES_CONFIG = """
# Advanced Video Optimization with CMA-ES
experiment:
  name: "video_cmaes_advanced"
  description: "High-quality video optimization using CMA-ES"
  modality: "video"
  duration_frames: 120
  fps: 30

target_roi:
  name: "Fusiform Face Area"
  vertex_ranges:
    left: [4500, 5200]
    right: [24500, 25200]

off_target_rois:
  - name: "V1"
    vertex_ranges:
      left: [0, 1000]
      right: [20000, 21000]
    weight: 0.15

generator:
  type: "video_diffusion"
  device: "cuda"
  num_frames: 120
  fps: 30

optimization:
  method: "cma_es"
  num_iterations: 200
  population_size: 24
  sigma: 0.5  # CMA-ES initial stddev

reward:
  target_activation_weight: 1.0
  off_target_suppression_weight: 0.3
  energy_penalty_weight: 0.05
  temporal_consistency_weight: 0.2  # Emphasize temporal smoothness

tribe:
  cache_predictions: true
  average_timesteps: true

visualization:
  save_frequency: 10
  output_dir: "./outputs/video_cmaes"

hardware:
  device: "cuda"
  mixed_precision: true
  num_workers: 8
"""


def save_experiment(name: str, config_str: str) -> Path:
    """Save experiment config to YAML file."""
    path = Path(f"config_{name}.yaml")
    with open(path, "w") as f:
        f.write(config_str)
    print(f"✓ Saved: {path}")
    return path


def load_experiment(name: str) -> dict:
    """Load experiment config."""
    path = Path(f"config_{name}.yaml")
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def list_experiments() -> None:
    """Print available experiments."""
    experiments = {
        "ffa": ("Face Region Optimization", FACE_OPTIMIZATION_CONFIG),
        "auditory": ("Auditory Cortex Optimization", AUDIO_OPTIMIZATION_CONFIG),
        "language": ("Language Region Optimization", LANGUAGE_OPTIMIZATION_CONFIG),
        "multi": ("Multi-Objective Optimization", MULTI_OBJECTIVE_CONFIG),
        "quick": ("Quick Test", QUICK_TEST_CONFIG),
        "video_cmaes": ("Video with CMA-ES", VIDEO_CMAES_CONFIG),
    }
    
    print("\n" + "=" * 70)
    print("AVAILABLE EXPERIMENTS")
    print("=" * 70)
    for key, (title, _) in experiments.items():
        print(f"  {key:15} - {title}")
    print("=" * 70 + "\n")


def setup_all_experiments() -> None:
    """Create config files for all experiments."""
    experiments = {
        "ffa": FACE_OPTIMIZATION_CONFIG,
        "auditory": AUDIO_OPTIMIZATION_CONFIG,
        "language": LANGUAGE_OPTIMIZATION_CONFIG,
        "multi": MULTI_OBJECTIVE_CONFIG,
        "quick": QUICK_TEST_CONFIG,
        "video_cmaes": VIDEO_CMAES_CONFIG,
    }
    
    print("\nSetting up example experiments...\n")
    for name, config in experiments.items():
        save_experiment(name, config)
    
    print("\n✓ All experiment configs created!")
    print("Usage: python pipeline.py --config config_ffa.yaml\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Experiment configuration utility")
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Create all experiment config files"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available experiments"
    )
    
    args = parser.parse_args()
    
    if args.setup:
        setup_all_experiments()
    elif args.list:
        list_experiments()
    else:
        list_experiments()
        print("Use --setup to create all config files")
        print("Use --list to see available experiments\n")
