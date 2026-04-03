# NeuroStim Optimization Engine with TRIBE V2

A complete end-to-end system for **optimizing visual, audio, and video stimuli** to maximize activation in specific brain regions using **TRIBE V2** as a simulated fMRI brain model.

---

## 🧠 Overview

### What This Does

This project implements a **closed-loop neuro-optimization system** where:

1. **Generative AI** (Stable Diffusion, video/audio diffusion) creates or modifies stimuli
2. **TRIBE V2** (Meta's brain encoding model) predicts neural fMRI responses
3. **Reward function** quantifies activation in target brain regions
4. **Evolutionary/CMA-ES optimization** iteratively improves stimuli

**Result**: Automatically discover stimuli that activate specific brain regions (e.g., face area, auditory cortex, language centers).

### Key Features

✅ **Multi-modal support**: Images, video, audio, text  
✅ **TRIBE V2 integration**: Uses official Facebook Research model  
✅ **Multiple optimization methods**: Evolutionary strategies, CMA-ES, PPO  
✅ **Flexible reward function**: Multi-objective optimization with ROI targeting  
✅ **Rich visualization**: Brain activation maps, ROI timecourses, stimulus evolution  
✅ **Modular architecture**: Easy to extend and customize  
✅ **Mock mode**: Test without real TRIBE installation  

---

## 📋 Requirements

### System Requirements
- Python 3.9+
- GPU recommended (CUDA 11.8+) for diffusion models
- At least 16GB VRAM for concurrent model inference
- Linux/macOS recommended (Windows supported with minor adjustments)

### Dependencies
See `requirements.txt`:
- PyTorch + CUDA
- TRIBE V2 (official package)
- Diffusers (Stable Diffusion, video/audio models)
- CMA-ES optimizer
- Matplotlib/Seaborn for visualization
- YAML configuration

---

## 🚀 Installation

### 1. Clone and Setup

```bash
cd /path/to/project
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install TRIBE V2

```bash
# From GitHub (official)
pip install git+https://github.com/facebookresearch/tribev2.git



### 4. Verify Installation

```python
python -c "from tribe_wrapper import TribeV2Wrapper; print('✓ TRIBE V2 wrapper loaded')"
python -c "from stimulus_generator import ImageGenerator; print('✓ Stimulus generator loaded')"
```

---

## 🎯 Quick Start

### Basic Usage (5 minutes)

```python
from pipeline import NeuroStimPipeline

# Initialize with config
pipeline = NeuroStimPipeline("neurostim_config.yaml")

# Setup all components
pipeline.setup()

# Run optimization
results = pipeline.run_experiment()

# View results
print(f"Best reward: {results['final_reward']:.4f}")
print(f"Results saved to: {pipeline.output_dir}")
```

### Command Line

```bash
# Run with default config
python pipeline.py

# Run with custom config
python pipeline.py --config my_experiment.yaml
```

---

## ⚙️ Configuration

### Main Config File: `neurostim_config.yaml`

```yaml
# Experiment setup
experiment:
  name: "face_region_optimization"
  modality: "video"  # video, audio, image, text
  duration_frames: 60
  fps: 30

# Target brain region to activate
target_roi:
  name: "Fusiform Face Area"
  hemisphere: "bilateral"
  vertex_ranges:
    left: [4500, 5200]
    right: [24500, 25200]

# Regions to suppress activation
off_target_rois:
  - name: "Visual Cortex V1"
    vertex_ranges:
      left: [0, 1000]
      right: [20000, 21000]
    weight: 0.1

# Stimulus generation
generator:
  type: "stable_diffusion"
  model_name: "runwayml/stable-diffusion-v1-5"
  device: "cuda"
  num_inference_steps: 50

# Optimization algorithm
optimization:
  method: "evolutionary"  # evolutionary, cma_es, ppo
  num_iterations: 100
  population_size: 8
  learning_rate: 0.001

# Reward function weights
reward:
  target_activation_weight: 1.0
  off_target_suppression_weight: 0.3
  energy_penalty_weight: 0.05
  temporal_consistency_weight: 0.1
```

### ROI Vertex Ranges

TRIBE V2 uses **fsaverage** cortical surface with ~20k vertices per hemisphere.

**Common ROI vertex ranges** (approximate):
```
- Left V1 (visual cortex):      0 - 1000
- Right V1:                     20000 - 21000
- Left FFA (faces):             4500 - 5200
- Right FFA:                    24500 - 25200
- Left auditory cortex:         8000 - 9000
- Right auditory cortex:        28000 - 29000
- Left Broca's area (language): 10000 - 11000
- Right Broca's area:           30000 - 31000
```

---

## 🏗️ Architecture

### Module Overview

```
neurostim_pipeline/
├── tribe_wrapper.py           # TRIBE V2 interface
├── stimulus_generator.py       # Image/video/audio generators
├── reward_function.py          # Multi-objective reward
├── optimization_engine.py      # Evolutionary/CMA-ES optimizer
├── visualization.py            # Brain maps and plots
├── pipeline.py                 # Main orchestration
├── neurostim_config.yaml       # Configuration
└── requirements.txt
```

### Data Flow

```
┌─────────────────────────────────────────────────────┐
│                   OPTIMIZATION LOOP                 │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌──────────────┐         ┌──────────────┐          │
│  │   Generator  │         │  TRIBE V2    │          │
│  │ (Diffusion)  ├────────>│ (Brain Model)│          │
│  └──────────────┘         └──────┬───────┘          │
│                                  │                  │
│                           ┌──────v──────┐           │
│                           │  Predictions │          │
│                           │  (~40k verts)│          │
│                           └──────┬───────┘          │
│                                  │                  │ 
│                           ┌──────v──────┐           │
│                           │ROI Extraction│          │
│                           └──────┬───────┘          │ 
│                                  │                  │
│                           ┌──────v──────────┐       │ 
│                           │ Reward Function  │      │
│                           │  (scalar reward) │      │
│                           └──────┬──────────┘       │
│                                  │                  │
│                           ┌──────v──────────┐       │
│                           │   Optimizer      │      │
│                           │ (Evo/CMA-ES/PPO) │      │
│                           └──────┬──────────┘       │
│                                  │                  │
│                           (update generator)        │
│                                  │                  │
│                                  └────────┐         │
│                                           │         │
│                    ┌──────────────────────┘         │
│                    │ Repeat iterations              │
└────────────────────┼────────────────────────────────┘
                     │
              ┌──────v──────┐
              │   Results    │
              │Visualization│
              └──────────────┘
```

---

## 🔬 Usage Examples

### Example 1: Optimize for Face Region Activation

```python
from pipeline import NeuroStimPipeline
import yaml

# Load config
config = {
    "experiment": {"name": "face_opt", "modality": "video"},
    "target_roi": {
        "name": "FFA",
        "vertex_ranges": {"left": [4500, 5200], "right": [24500, 25200]}
    },
    # ... rest of config
}

# Run
pipeline = NeuroStimPipeline()
pipeline.config = config
pipeline.setup()
results = pipeline.run_experiment()
```

### Example 2: Audio to Auditory Cortex

```yaml
# In config file:
experiment:
  modality: "audio"
  
target_roi:
  name: "Primary Auditory Cortex"
  vertex_ranges:
    left: [8000, 9000]
    right: [28000, 29000]
```

### Example 3: Multi-Objective (Face + Suppress Visual)

```yaml
target_roi:
  name: "FFA"
  vertex_ranges: {left: [4500, 5200], right: [24500, 25200]}

off_target_rois:
  - name: "V1"
    vertex_ranges: {left: [0, 1000], right: [20000, 21000]}
    weight: 0.3
  - name: "V4"
    vertex_ranges: {left: [1500, 2500], right: [21500, 22500]}
    weight: 0.1

reward:
  target_activation_weight: 1.0
  off_target_suppression_weight: 0.5
```

---

## 📊 Outputs

### Directory Structure After Run

```
outputs/
├── optimization_progress.png          # Reward curve
├── reward_breakdown.png               # Reward components
├── brain_activation_map.png           # Cortical surface
├── roi_timecourse.png                 # ROI activation over time
└── results_20240101_120000.json       # Numerical results

checkpoints/
└── optim_state_iter_50.pkl            # Optimizer checkpoints
```

### Results JSON

```json
{
  "experiment": "face_region_optimization",
  "timestamp": "20240101_120000",
  "final_reward": 0.8234,
  "mean_reward": 0.5123,
  "num_iterations": 100,
  "rewards_over_time": [0.1, 0.15, 0.22, ...],
  "target_roi": "Fusiform Face Area",
  "modality": "video",
  "optimization_method": "evolutionary"
}
```

---

## 🔧 Advanced Usage

### Custom Reward Function

```python
from reward_function import RewardFunction

class CustomRewardFunction(RewardFunction):
    def compute_reward(self, preds, tribe_wrapper, **kwargs):
        # Custom logic here
        target = self._compute_target_activation(preds, tribe_wrapper)
        return target  # Or any custom formula
```

### Custom Optimizer

```python
from optimization_engine import LatentOptimizer
import optax  # JAX optimizers

class GradientOptimizer(LatentOptimizer):
    def optimize_gradient(self, modality, num_steps):
        # Use JAX/PyTorch for differentiable optimization
        pass
```

### Multiple Experiments

```python
configs = [
    "config_face.yaml",
    "config_auditory.yaml",
    "config_language.yaml"
]

for cfg in configs:
    pipeline = NeuroStimPipeline(cfg)
    pipeline.setup()
    results = pipeline.run_experiment()
    print(f"{cfg}: Best reward = {results['final_reward']}")
```

---

## 🐛 Troubleshooting

### Issue: "TRIBE V2 not installed"

**Solution**: Install from GitHub
```bash
pip install git+https://github.com/facebookresearch/tribev2.git
```

Mock mode will work for testing without real installation.

### Issue: Out of Memory

**Solutions**:
1. Reduce `population_size` in config
2. Reduce `num_inference_steps` for generators
3. Use smaller models (e.g., `stable-diffusion-2-base`)
4. Enable `mixed_precision: true`

### Issue: Slow optimization

**Solutions**:
1. Use CMA-ES instead of evolutionary (faster convergence)
2. Reduce `duration_frames` for videos
3. Batch evaluation of population
4. Use smaller TRIBE model if available

### Issue: Poor reward improvement

**Check**:
1. ROI vertex ranges are correct for your target
2. Reward weights are balanced
3. Generator can produce diverse outputs
4. Enough iterations (try 200+)

---

## 📚 API Reference

### TribeV2Wrapper

```python
from tribe_wrapper import TribeV2Wrapper

tribe = TribeV2Wrapper(model_name="facebook/tribev2")

# Predict from stimulus
preds, metadata = tribe.predict_from_video("video.mp4")
preds, metadata = tribe.predict_from_audio("audio.wav")
preds, metadata = tribe.predict_from_image("image.png")

# Extract ROI activation
roi_activity = tribe.get_roi_activation(
    preds,
    roi_vertex_ranges={"left": [4500, 5200], "right": [24500, 25200]}
)
```

### RewardFunction

```python
from reward_function import RewardFunction

reward_fn = RewardFunction(
    target_roi=config["target_roi"],
    off_target_rois=config["off_target_rois"],
    target_weight=1.0,
    off_target_weight=0.3
)

reward = reward_fn.compute_reward(preds, tribe_wrapper)

# Get breakdown
breakdown = reward_fn.compute_reward(
    preds, tribe_wrapper,
    return_components=True
)
print(breakdown)
```

### Generators

```python
from stimulus_generator import ImageGenerator, VideoGenerator, AudioGenerator

# Image
img_gen = ImageGenerator()
images = img_gen.generate_from_prompt("a smiling face", height=512, width=512)
modified = img_gen.modify_image(images[0], "make it happier", strength=0.5)

# Video
vid_gen = VideoGenerator(num_frames=16)
video = vid_gen.generate_from_prompt("person walking")

# Audio
audio_gen = AudioGenerator(duration=5.0)
audio = audio_gen.generate_from_prompt("speech with music")
```

### Optimization

```python
from optimization_engine import LatentOptimizer

optimizer = LatentOptimizer(generator, tribe, reward_fn)

# Evolutionary Strategy
state = optimizer.optimize_evolutionary(
    modality="video",
    num_iterations=100,
    population_size=16
)

# CMA-ES
state = optimizer.optimize_cmaes(
    modality="video",
    num_iterations=100,
    population_size=16
)

print(f"Best reward: {state.best_reward}")
```

---

## 🎓 Research Applications

This system enables:

1. **In-silico neuroscience**: Study brain responses without fMRI scanners
2. **Stimulus optimization**: Design maximally effective visual/audio stimuli
3. **Brain encoding models**: Understand how perception works
4. **Accessibility research**: Optimize stimuli for different perceptual abilities
5. **Neurofeedback**: Generate stimuli for brain-computer interfaces

---

## ⚠️ Important Notes

### Limitations

- **Predicted, not real fMRI**: TRIBE V2 predicts average brain responses, not individual-specific
- **Research prototype**: Not for clinical use
- **No real cognitive control**: Optimized stimuli don't imply real-world behavioral effects
- **Computational cost**: Full optimization can take hours on GPU

### Ethical Considerations

This system is designed for **research and educational purposes**. When using stimuli optimization:

✓ Use responsibly in research settings  
✓ Disclose use of AI-optimized stimuli  
✗ Do not attempt to manipulate or harm  
✗ Do not use without informed consent in human studies  

---

## 📖 References

### TRIBE V2
- **Paper**: Meta AI's brain encoding model
- **GitHub**: https://github.com/facebookresearch/tribev2
- **HuggingFace**: https://huggingface.co/facebook/tribev2

### Related Work
- Diffusion models: https://github.com/huggingface/diffusers
- fMRI analysis: FSL, SPM, nilearn
- Brain surface visualization: Freesurfer, HCP

---

## 📝 Citation

If you use this system in research:

```bibtex
@software{neurostim2024,
  title={NeuroStim Optimization Engine with TRIBE V2},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/neurostim}
}
```

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- [ ] Gradient-based optimization (if TRIBE becomes differentiable)
- [ ] Individual-subject customization
- [ ] Real-time interactive optimization
- [ ] Integration with actual fMRI scanners
- [ ] Additional optimization algorithms (Bayesian, RL)
- [ ] Web interface for visualization

---

## 📄 License

[Your License Here]

---

## 🆘 Support

For issues and questions:

1. Check troubleshooting section above
2. Review TRIBE V2 documentation: https://github.com/facebookresearch/tribev2
3. Open GitHub issue with config and error logs
4. Join neuroscience ML communities

---

**Built with ❤️ for neuro-optimization research**
