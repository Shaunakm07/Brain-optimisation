# NeuroStim Project Structure & Architecture

## 📁 Project Layout

```
neurostim_optimization/
│
├── 🎯 MAIN FILES (Core Execution)
│   ├── pipeline.py                    # Main orchestration engine
│   ├── neurostim_config.yaml         # Default configuration
│   └── requirements.txt               # Dependencies
│
├── 🧠 CORE MODULES
│   ├── tribe_wrapper.py              # TRIBE V2 interface
│   ├── stimulus_generator.py          # Image/video/audio generation
│   ├── reward_function.py             # Multi-objective reward
│   └── optimization_engine.py         # Evolutionary/CMA-ES optimization
│
├── 📊 UTILITIES
│   ├── visualization.py               # Brain maps & plots
│   └── example_experiments.py         # Pre-configured experiments
│
├── 🚀 SCRIPTS
│   ├── quick_start.py                # Usage examples (7 examples)
│   └── install.sh                    # Installation script
│
├── 📚 DOCUMENTATION
│   ├── README.md                     # Complete guide
│   └── PROJECT_STRUCTURE.md          # This file
│
├── 📁 OUTPUTS (Created after running)
│   ├── outputs/                      # Experiment results
│   │   ├── ffa_optimization/
│   │   ├── auditory_optimization/
│   │   └── ...
│   │
│   └── checkpoints/                  # Saved optimizer states
│
└── 📁 CONFIGS (Generated)
    ├── config_ffa.yaml
    ├── config_auditory.yaml
    ├── config_language.yaml
    └── ...
```

---

## 🔧 Module Details

### 1. **pipeline.py** - Main Orchestrator
```
NeuroStimPipeline
├── __init__(config_path)              # Load YAML config
├── setup()                             # Initialize all components
│   ├── _setup_tribe()                 # Initialize TRIBE V2 model
│   ├── _setup_generator()             # Initialize stimulus generator
│   ├── _setup_reward()                # Initialize reward function
│   ├── _setup_optimizer()             # Initialize optimizer
│   └── _setup_visualization()         # Initialize visualization
├── run_experiment()                    # Execute optimization
│   ├── _run_evolutionary_optimization()
│   ├── _run_cmaes_optimization()
│   └── _run_ppo_optimization()
├── _save_results()                     # Save to disk
├── _visualize_results()                # Create plots
└── generate_report()                   # Text report
```

**Usage:**
```python
from pipeline import NeuroStimPipeline
pipeline = NeuroStimPipeline("neurostim_config.yaml")
pipeline.setup()
results = pipeline.run_experiment()
```

---

### 2. **tribe_wrapper.py** - Brain Model Interface
```
TribeV2Wrapper
├── __init__(model_name, device)       # Load official TRIBE V2
├── predict_from_video()               # (timesteps, vertices)
├── predict_from_audio()               # Brain predictions
├── predict_from_image()               # from different modalities
├── predict_from_text()
├── get_roi_activation()               # Extract ROI timecourses
├── get_whole_brain_summary()          # Whole-brain statistics
└── [Mock methods for testing]
```

**Key Features:**
- Official Facebook Research TRIBE V2 model
- fsaverage cortical surface (~40k vertices bilateral)
- Multi-modal support: video, audio, image, text
- Mock mode for development without real installation

**Output Shape:**
- `preds`: (timesteps, num_vertices) = (T, 40962)
- Typically: T=30-120 frames

---

### 3. **stimulus_generator.py** - Media Generation
```
StimulusGenerator (abstract)
│
├── ImageGenerator
│   ├── generate_from_prompt()         # Text→Image
│   ├── modify_image()                 # Image modification
│   └── [Mock image generation]
│
├── VideoGenerator
│   ├── generate_from_prompt()         # Text→Video
│   ├── create_video_from_images()     # Sequence→Video
│   └── [Mock video generation]
│
└── AudioGenerator
    ├── generate_from_prompt()         # Text→Audio
    ├── modify_audio()                 # Audio modification
    └── [Mock audio generation]
```

**Supported Models:**
- Images: Stable Diffusion v1.5, v2, SDXL
- Video: Text-to-video, latent video diffusion
- Audio: Spectrogram-based, waveform synthesis

---

### 4. **reward_function.py** - Optimization Objective
```
RewardFunction
├── compute_reward()                   # Main reward computation
│   ├── _compute_target_activation()   # Maximize target ROI
│   ├── _compute_off_target_activation() # Suppress off-target
│   ├── _compute_energy_penalty()      # L2 regularization
│   └── _compute_temporal_penalty()    # Smooth over time
├── get_roi_activation()               # Helper functions
└── log_reward_breakdown()
```

**Reward Formula:**
```
R = w_target × target_act
  - w_off_target × off_target_act
  - w_energy × ||preds||²
  - w_temporal × smoothness_penalty
```

**Customization:**
- Adjust weights for different objectives
- Subclass to implement custom logic
- Support multi-objective optimization

---

### 5. **optimization_engine.py** - Search Algorithm
```
LatentOptimizer
├── optimize_evolutionary()            # ES optimization
│   ├── Population initialization
│   ├── Fitness evaluation
│   ├── Selection (elitism)
│   └── Mutation
├── optimize_cmaes()                   # CMA-ES optimization
│   ├── Covariance matrix adaptation
│   └── Step-size control
├── _evaluate_latent()                 # Helper functions
├── _decode_stimulus()
└── _save_stimulus()
```

**Optimization Methods:**
1. **Evolutionary Strategy (ES)**
   - Simple, gradient-free
   - Good for exploration
   - Population-based

2. **CMA-ES**
   - Adaptive covariance matrix
   - Faster convergence
   - Better for exploitation

3. **PPO** (placeholder)
   - Reinforcement learning approach
   - For future implementation

**Key Features:**
- Black-box optimization (no gradients required)
- Population-based search
- Parallelizable evaluation

---

### 6. **visualization.py** - Results Visualization
```
NeuroVisualization
├── plot_roi_activation()              # Timecourse plots
├── plot_brain_surface()               # Cortical surface maps
├── plot_optimization_progress()       # Reward curves
├── plot_reward_breakdown()            # Component breakdown
├── plot_stimulus_comparison()         # Side-by-side stimuli
└── plot_summary_report()              # Comprehensive report
```

**Output Types:**
- ROI activation timecourses
- Brain surface activation maps (fsaverage projection)
- Optimization convergence curves
- Reward function breakdown
- Stimulus evolution comparison

---

## 🔄 Data Flow Pipeline

### Complete Optimization Loop

```
┌─────────────────────────────────────────────────────────────┐
│          OPTIMIZATION ITERATION (t)                         │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │ Sample Population              │
        │ N = population_size            │
        │ Each: latent vector ~ 𝒩(0,σ²) │
        └───────┬───────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────┐
│         EVALUATE EACH SOLUTION (Parallel Possible)          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  For each latent vector z:                                 │
│    1. z → [Generator] → stimulus                          │
│       (decode latent to image/video/audio)                │
│                                                             │
│    2. stimulus → [Save to file]                           │
│       (TRIBE V2 requires file input)                       │
│                                                             │
│    3. file → [TRIBE V2] → predictions                     │
│       Shape: (T, 40962) - timesteps × vertices            │
│                                                             │
│    4. predictions → [ROI Extraction]                      │
│       Extract target and off-target ROI activations       │
│                                                             │
│    5. ROI → [Reward Function] → scalar reward             │
│       reward = f(target, off_target, energy, temporal)    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │ Collect Rewards               │
        │ rewards = [r₁, r₂, ..., rₙ]   │
        └───────┬───────────────────────┘
                │
                ▼
        ┌───────────────────────────────┐
        │ Update Population              │
        │ [Optimizer Strategy]           │
        │ ├─ ES: mutation+selection      │
        │ └─ CMA-ES: adapt covariance    │
        └───────┬───────────────────────┘
                │
                ▼
        ┌───────────────────────────────┐
        │ Check Convergence             │
        │ ├─ Iterations done?           │
        │ └─ Improvement stalled?       │
        └───────┬───────────────────────┘
                │
        ╔═══════╤════════╗
        ║       │        ║
      YES      NO     CONVERGE
        │       │        │
        ▼       │        ▼
     [SAVE]   [LOOP]  [REPORT]
        │       │        │
        └───────┘        │
                         ▼
                ┌──────────────────────┐
                │ Generate Results     │
                │ ├─ JSON metrics      │
                │ ├─ Visualizations    │
                │ └─ Report            │
                └──────────────────────┘
```

---

## 📊 Configuration System

### Hierarchical Config Structure

```yaml
experiment:              # Task definition
  - name: identifier
  - modality: video/audio/image/text
  - duration: frames/seconds

target_roi:             # What to maximize
  - name: brain region
  - vertex_ranges: {left: [start, end], right: [start, end]}

off_target_rois:        # What to suppress
  - name: region 1
  - name: region 2
  - weight: importance

generator:              # Stimulus generation
  - type: model architecture
  - model_name: pretrained model
  - hyperparameters

optimization:           # Search algorithm
  - method: evolutionary / cma_es / ppo
  - num_iterations: generations/steps
  - population_size: individuals
  - hyperparameters

reward:                 # Objective function
  - weights for different objectives

tribe:                  # Brain model settings
  - model_name: facebook/tribev2
  - batch_size
  - device

visualization:          # Output generation
  - frequencies
  - output_dir

hardware:               # Compute resources
  - device: cuda/cpu
  - mixed_precision
```

### Validation & Defaults

- All required fields have defaults
- Config validated on load
- YAML parsing with error messages
- Environment variable overrides

---

## 🎯 ROI Vertex Mapping

### fsaverage Cortical Surface

The TRIBE V2 model operates on the **fsaverage** standard surface:
- **Total**: ~40,962 vertices (bilateral)
- **Per hemisphere**: ~20,481 vertices
- **Spatial resolution**: ~4mm between vertices

### Common ROI Ranges

```python
ROI_ATLAS = {
    "V1": {
        "left": (0, 1000),
        "right": (20000, 21000)
    },
    "V4": {
        "left": (1500, 2500),
        "right": (21500, 22500)
    },
    "FFA": {
        "left": (4500, 5200),
        "right": (24500, 25200)
    },
    "PPA": {
        "left": (5700, 6500),
        "right": (25700, 26500)
    },
    "A1": {
        "left": (8000, 9000),
        "right": (28000, 29000)
    },
    "Broca": {
        "left": (10000, 11000),
        "right": (30000, 31000)
    },
}
```

---

## 💾 Output Structure

### File Organization

```
outputs/
├── {experiment_name}/
│   ├── results_{timestamp}.json          # Metrics
│   ├── optimization_progress.png         # Curves
│   ├── reward_breakdown.png              # Components
│   ├── brain_activation_map.png          # Surface
│   ├── roi_timecourse.png                # Time series
│   ├── stimulus_evolution.png            # Progression
│   ├── summary_report.png                # Dashboard
│   └── experiment_report.txt             # Text summary
│
checkpoints/
├── optim_state_iter_50.pkl               # Optimizer state
├── optim_state_iter_100.pkl              # For resuming
└── best_latent.npy                       # Best solution
```

### Results JSON Format

```json
{
  "experiment": "ffa_optimization",
  "timestamp": "20240115_143022",
  "final_reward": 0.8234,
  "mean_reward": 0.5123,
  "num_iterations": 100,
  "target_roi": "Fusiform Face Area",
  "modality": "video",
  "optimization_method": "evolutionary",
  "rewards_over_time": [0.1, 0.15, 0.22, ...],
  "best_latent_path": "checkpoints/best_latent.npy",
  "config": {...}
}
```

---

## 🔌 Extension Points

### 1. Custom Generators
```python
class MyGenerator(StimulusGenerator):
    def generate(self, prompt):
        # Your implementation
        pass
```

### 2. Custom Rewards
```python
class MyReward(RewardFunction):
    def compute_reward(self, preds, tribe):
        # Your objective
        pass
```

### 3. Custom Optimizers
```python
class MyOptimizer(LatentOptimizer):
    def optimize(self, ...):
        # Your algorithm
        pass
```

### 4. Custom ROI Extraction
```python
def extract_custom_roi(preds, roi_definition):
    # Your ROI mapping
    pass
```

---

## 📈 Performance Considerations

### Computational Cost

| Task | Time | Memory |
|------|------|--------|
| Generate image (SD) | 10-30s | 8GB |
| Predict with TRIBE | 5-10s | 12GB |
| Compute reward | <1s | 2GB |
| Single evaluation | ~20-40s | 12GB |
| 100 iterations, pop=16 | 8-12 hours | 12GB |

### Optimization Tips

1. **Batch evaluation** - Use GPU parallelism
2. **Reduce steps** - Fewer inference steps for generators
3. **CMA-ES** - Faster convergence than ES
4. **Smaller models** - Use base versions if available
5. **Mixed precision** - Enable in config
6. **Caching** - Cache TRIBE predictions if possible

---

## 🐛 Debugging & Logging

### Logging Levels

```python
import logging
logging.basicConfig(level=logging.DEBUG)
# Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
```

### Common Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| OOM | Large batch | Reduce population_size |
| Slow | Small steps | Reduce inference_steps |
| Poor convergence | Bad ROI | Check vertex ranges |
| Crashes | Missing model | Check config paths |

---

## 🚀 Production Deployment

### Checklist

- [ ] Install all dependencies
- [ ] Download TRIBE V2 model
- [ ] Test with quick_start.py
- [ ] Validate configs
- [ ] Monitor GPU usage
- [ ] Save checkpoints regularly
- [ ] Version control results
- [ ] Document findings

---

## 📚 References & Resources

- **TRIBE V2**: https://github.com/facebookresearch/tribev2
- **Diffusers**: https://github.com/huggingface/diffusers
- **CMA-ES**: https://cma-es.github.io/
- **fsaverage**: https://surfer.nmr.mgh.harvard.edu/fswiki/FsAverage
- **fMRI**: https://www.fil.ion.ucl.ac.uk/spm/

---

**Last Updated**: 2024-01  
**Version**: 1.0.0
