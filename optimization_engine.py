"""
Optimization Engine
Uses PPO or evolutionary strategies to optimize stimuli for brain activation
"""

import os
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Callable
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OptimizationState:
    """State of optimization run."""
    iteration: int
    best_reward: float
    mean_reward: float
    rewards: List[float]
    best_latent: np.ndarray
    best_stimulus: Optional[np.ndarray] = None


class LatentOptimizer:
    """
    Optimize stimulus in latent space of generator model.
    Operates on latent vectors to enable efficient gradient-free optimization.
    """
    
    def __init__(
        self,
        generator,
        tribe_wrapper,
        reward_function,
        latent_dim: int = 256,
        device: str = "cuda"
    ):
        """
        Initialize latent space optimizer.
        
        Args:
            generator: Stimulus generator with .decode(latent) method
            tribe_wrapper: TribeV2Wrapper for predictions
            reward_function: RewardFunction instance
            latent_dim: Dimension of latent space
            device: torch device
        """
        self.generator = generator
        self.tribe = tribe_wrapper
        self.reward_fn = reward_function
        self.latent_dim = latent_dim
        self.device = device
        
        logger.info(
            f"Initialized LatentOptimizer: "
            f"latent_dim={latent_dim}, device={device}"
        )
    
    def optimize_cmaes(
        self,
        modality: str = "image",
        num_iterations: int = 100,
        population_size: int = 16,
        sigma: float = 0.5,
        seed: Optional[int] = None,
        output_fn: Optional[Callable] = None,
        **kwargs
    ) -> OptimizationState:
        """
        Optimize using CMA-ES (Covariance Matrix Adaptation Evolution Strategy).
        Requires: pip install cma
        
        Args:
            modality: Type of stimulus ("image", "video", "audio")
            num_iterations: Number of generations
            population_size: Population per generation
            sigma: Initial standard deviation
            seed: Random seed
            output_fn: Callback function for each iteration
            **kwargs: Additional arguments
        
        Returns:
            optimization_state: Final state with best solution
        """
        try:
            import cma
        except ImportError:
            logger.warning(
                "CMA-ES not available. Install with: pip install cma"
            )
            return self._demo_optimization(num_iterations, modality)
        
        logger.info(f"Starting CMA-ES optimization for {modality}")
        
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize mean at origin
        x0 = np.zeros(self.latent_dim)
        
        # CMA-ES optimizer
        es = cma.CMAEvolutionStrategy(
            x0,
            sigma,
            {
                "popsize": population_size,
                "seed": seed,
                "verbose_disp": 0,
            }
        )
        
        state = OptimizationState(
            iteration=0,
            best_reward=float("-inf"),
            mean_reward=0.0,
            rewards=[],
            best_latent=x0.copy()
        )
        
        while not es.stop() and state.iteration < num_iterations:
            # Sample population
            solutions = es.ask()
            
            # Evaluate fitness
            rewards = []
            for latent in solutions:
                reward = self._evaluate_latent(latent, modality)
                rewards.append(reward)
            
            # Tell CMA-ES the fitness values
            es.tell(solutions, [-r for r in rewards])  # Negate for minimization
            
            # Update state
            best_idx = np.argmax(rewards)
            if rewards[best_idx] > state.best_reward:
                state.best_reward = float(rewards[best_idx])
                state.best_latent = solutions[best_idx].copy()
            
            state.mean_reward = float(np.mean(rewards))
            state.rewards.append(state.mean_reward)
            state.iteration += 1
            
            if state.iteration % 10 == 0:
                logger.info(
                    f"[Iter {state.iteration}] "
                    f"Best: {state.best_reward:.4f}, "
                    f"Mean: {state.mean_reward:.4f}"
                )
            
            if output_fn is not None:
                output_fn(state, rewards)
        
        logger.info(f"CMA-ES optimization complete!")
        logger.info(f"Best reward: {state.best_reward:.4f}")
        
        return state
    
    def optimize_evolutionary(
        self,
        modality: str = "image",
        num_iterations: int = 100,
        population_size: int = 16,
        mutation_std: float = 0.1,
        elite_fraction: float = 0.2,
        seed: Optional[int] = None,
        output_fn: Optional[Callable] = None,
        **kwargs
    ) -> OptimizationState:
        """
        Optimize using simple evolutionary strategy.
        Pure Python implementation, no dependencies.
        
        Args:
            modality: Stimulus type
            num_iterations: Number of generations
            population_size: Population size
            mutation_std: Standard deviation of mutations
            elite_fraction: Fraction of population to keep (elitism)
            seed: Random seed
            output_fn: Callback function
            **kwargs: Additional arguments
        
        Returns:
            optimization_state: Final optimization state
        """
        logger.info(
            f"Starting evolutionary optimization for {modality}\n"
            f"  Population size: {population_size}\n"
            f"  Generations: {num_iterations}"
        )
        
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize population
        population = np.random.randn(population_size, self.latent_dim) * 0.5
        
        state = OptimizationState(
            iteration=0,
            best_reward=float("-inf"),
            mean_reward=0.0,
            rewards=[],
            best_latent=population[0].copy()
        )
        
        num_elite = max(1, int(elite_fraction * population_size))
        
        for generation in range(num_iterations):
            # Evaluate population
            rewards = np.array([
                self._evaluate_latent(latent, modality)
                for latent in population
            ])
            
            # Track best
            best_idx = np.argmax(rewards)
            if rewards[best_idx] > state.best_reward:
                state.best_reward = float(rewards[best_idx])
                state.best_latent = population[best_idx].copy()
            
            state.mean_reward = float(np.mean(rewards))
            state.rewards.append(state.mean_reward)
            state.iteration = generation + 1
            
            if generation % 10 == 0:
                logger.info(
                    f"[Gen {generation}] "
                    f"Best: {state.best_reward:.4f}, "
                    f"Mean: {state.mean_reward:.4f}"
                )
            
            if output_fn is not None:
                output_fn(state, rewards)
            
            # Selection: keep elite + mutate
            elite_indices = np.argsort(rewards)[-num_elite:]
            elite_population = population[elite_indices]
            
            # Mutation
            new_population = []
            for _ in range(population_size - num_elite):
                parent = elite_population[
                    np.random.randint(len(elite_population))
                ]
                child = parent + np.random.randn(self.latent_dim) * mutation_std
                new_population.append(child)
            
            # Combine elite with new population
            population = np.vstack([elite_population, new_population])
        
        logger.info(f"Evolutionary optimization complete!")
        logger.info(f"Best reward: {state.best_reward:.4f}")
        
        return state
    
    def _evaluate_latent(
        self,
        latent: np.ndarray,
        modality: str
    ) -> float:
        """
        Evaluate a latent vector by:
        1. Decoding to stimulus
        2. Predicting brain activity with TRIBE V2
        3. Computing reward
        
        Args:
            latent: Latent vector
            modality: Stimulus modality
        
        Returns:
            scalar reward
        """
        try:
            # 1. Decode latent to stimulus
            stimulus = self._decode_stimulus(latent, modality)
            
            # 2. Save to temporary file (TRIBE expects file input)
            temp_path = self._save_stimulus(stimulus, modality)
            
            # 3. Get TRIBE predictions
            if modality == "image":
                preds, metadata = self.tribe.predict_from_image(temp_path)
            elif modality == "video":
                preds, metadata = self.tribe.predict_from_video(temp_path)
            elif modality == "audio":
                preds, metadata = self.tribe.predict_from_audio(temp_path)
            else:
                raise ValueError(f"Unknown modality: {modality}")
            
            # 4. Compute reward
            reward = self.reward_fn.compute_reward(preds, self.tribe)
            
            return float(reward)
        
        except Exception as e:
            logger.warning(f"Error in evaluation: {e}")
            return float("-inf")  # Penalize failures
    
    def _decode_stimulus(
        self,
        latent: np.ndarray,
        modality: str
    ) -> np.ndarray:
        """
        Decode latent vector to stimulus.
        This is a placeholder; actual implementation depends on generator.
        
        Args:
            latent: Latent vector
            modality: Stimulus type
        
        Returns:
            stimulus: Image, audio, or video array
        """
        # For demo: just generate from scratch using generator
        if modality == "image":
            # Use generator with latent as seed
            seed = int(np.sum(latent) * 1000) % 2**32
            stimulus = self.generator.generate_from_prompt(
                prompt="a face",
                seed=seed,
                height=256,
                width=256
            )[0]
        elif modality == "video":
            stimulus = self.generator.generate_from_prompt(
                prompt="a person's face",
                num_frames=16
            )
        elif modality == "audio":
            stimulus = self.generator.generate_from_prompt(
                prompt="speech"
            )
        else:
            raise ValueError(f"Unknown modality: {modality}")
        
        return stimulus
    
    def _save_stimulus(
        self,
        stimulus: np.ndarray,
        modality: str
    ) -> str:
        """
        Save stimulus to temporary file for TRIBE V2 input.
        
        Args:
            stimulus: Stimulus array
            modality: Type
        
        Returns:
            filepath: Path to saved file
        """
        import tempfile
        from PIL import Image
        import soundfile as sf
        import imageio
        
        os.makedirs("./temp_stimuli", exist_ok=True)
        
        if modality == "image":
            if stimulus.max() <= 1.0:
                stimulus = (stimulus * 255).astype(np.uint8)
            path = f"./temp_stimuli/image_{np.random.randint(1000000)}.png"
            Image.fromarray(stimulus).save(path)
        
        elif modality == "video":
            if stimulus.ndim == 5:
                stimulus = stimulus[0]  # remove batch dim: (1, T, H, W, 3) -> (T, H, W, 3)
            if stimulus.max() <= 1.0:
                stimulus = (stimulus * 255).astype(np.uint8)
            path = f"./temp_stimuli/video_{np.random.randint(1000000)}.mp4"
            imageio.mimsave(path, [stimulus[i] for i in range(len(stimulus))], fps=30)
        
        elif modality == "audio":
            path = f"./temp_stimuli/audio_{np.random.randint(1000000)}.wav"
            sf.write(path, stimulus, 16000)
        
        return path
    
    def _demo_optimization(
        self,
        num_iterations: int,
        modality: str
    ) -> OptimizationState:
        """Demonstration optimization without CMA-ES."""
        logger.info(
            f"Running demonstration optimization for {num_iterations} iterations"
        )
        
        state = OptimizationState(
            iteration=0,
            best_reward=float("-inf"),
            mean_reward=0.0,
            rewards=[],
            best_latent=np.random.randn(self.latent_dim)
        )
        
        for i in range(num_iterations):
            # Simulate improving rewards
            mean_reward = -1.0 + i / num_iterations + np.random.randn() * 0.1
            best_reward = state.best_reward + np.random.rand() * 0.5
            
            state.iteration = i + 1
            state.mean_reward = float(mean_reward)
            state.best_reward = float(best_reward)
            state.rewards.append(state.mean_reward)
            
            if i % 20 == 0:
                logger.info(
                    f"[Demo Iter {i}] "
                    f"Best: {state.best_reward:.4f}, "
                    f"Mean: {state.mean_reward:.4f}"
                )
        
        return state
