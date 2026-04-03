"""
Reward Function for NeuroStim Optimization
Computes scalar reward from TRIBE V2 predictions and ROI activations
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class RewardFunction:
    """
    Multi-objective reward function for neuro-optimization.
    
    Reward = (
        w_target * target_activation
        - w_off_target * off_target_activation
        - w_energy * energy_penalty
        - w_temporal * temporal_consistency_penalty
    )
    """
    
    def __init__(
        self,
        target_roi: Dict,
        off_target_rois: List[Dict],
        target_weight: float = 1.0,
        off_target_weight: float = 0.3,
        energy_weight: float = 0.05,
        temporal_weight: float = 0.1,
        temporal_aggregation: str = "mean"
    ):
        """
        Initialize reward function.
        
        Args:
            target_roi: Target ROI config with 'name' and 'vertex_ranges'
            off_target_rois: List of off-target ROI configs
            target_weight: Weight for target activation (positive reward)
            off_target_weight: Weight for off-target suppression (penalty)
            energy_weight: Weight for brain energy penalty
            temporal_weight: Weight for temporal smoothness penalty
            temporal_aggregation: How to aggregate over time ("mean", "max", "peak")
        """
        self.target_roi = target_roi
        self.off_target_rois = off_target_rois
        
        self.w_target = target_weight
        self.w_off_target = off_target_weight
        self.w_energy = energy_weight
        self.w_temporal = temporal_weight
        self.temporal_aggregation = temporal_aggregation
        
        logger.info(
            f"Reward function initialized:\n"
            f"  Target ROI: {target_roi['name']}\n"
            f"  Off-target ROIs: {len(off_target_rois)}\n"
            f"  Weights: target={target_weight}, "
            f"off_target={off_target_weight}, "
            f"energy={energy_weight}, temporal={temporal_weight}"
        )
    
    def compute_reward(
        self,
        preds: np.ndarray,
        tribe_wrapper,
        return_components: bool = False
    ) -> float:
        """
        Compute total reward from TRIBE V2 predictions.
        
        Args:
            preds: (timesteps, num_vertices) predictions from TRIBE V2
            tribe_wrapper: TribeV2Wrapper instance
            return_components: If True, return dict with reward breakdown
        
        Returns:
            scalar reward (or dict with components if return_components=True)
        """
        
        # 1. TARGET ACTIVATION (maximize)
        target_activation = self._compute_target_activation(
            preds, tribe_wrapper
        )
        target_reward = self.w_target * target_activation
        
        # 2. OFF-TARGET SUPPRESSION (minimize)
        off_target_activation = self._compute_off_target_activation(
            preds, tribe_wrapper
        )
        off_target_penalty = self.w_off_target * off_target_activation
        
        # 3. ENERGY PENALTY (minimize whole-brain activity)
        energy_penalty = self.w_energy * self._compute_energy_penalty(preds)
        
        # 4. TEMPORAL SMOOTHNESS (minimize sudden changes)
        temporal_penalty = self.w_temporal * self._compute_temporal_penalty(preds)
        
        # Total reward
        total_reward = (
            target_reward - off_target_penalty - energy_penalty - temporal_penalty
        )
        
        if return_components:
            return {
                "total": float(total_reward),
                "target_activation": float(target_activation),
                "target_reward": float(target_reward),
                "off_target_activation": float(off_target_activation),
                "off_target_penalty": float(off_target_penalty),
                "energy_penalty": float(energy_penalty),
                "temporal_penalty": float(temporal_penalty)
            }
        
        return float(total_reward)
    
    def _compute_target_activation(
        self,
        preds: np.ndarray,
        tribe_wrapper
    ) -> float:
        """
        Compute mean activation in target ROI.
        
        Args:
            preds: (timesteps, num_vertices) predictions
            tribe_wrapper: TribeV2Wrapper for ROI extraction
        
        Returns:
            scalar activation measure
        """
        roi_vertex_ranges = self.target_roi["vertex_ranges"]
        roi_activity = tribe_wrapper.get_roi_activation(
            preds,
            roi_vertex_ranges,
            aggregation="mean"
        )
        
        # Average across hemispheres and time
        activations = list(roi_activity.values())
        mean_activation = np.mean(activations)
        
        # Normalize to [0, 1] range (assuming predictions ~ [-1, 3])
        normalized_activation = np.clip(mean_activation / 3.0, 0, 1)
        
        return float(normalized_activation)
    
    def _compute_off_target_activation(
        self,
        preds: np.ndarray,
        tribe_wrapper
    ) -> float:
        """
        Compute activation in off-target ROIs.
        
        Args:
            preds: Predictions
            tribe_wrapper: TribeV2Wrapper
        
        Returns:
            scalar penalty (should be minimized)
        """
        total_off_target = 0.0
        
        for off_roi in self.off_target_rois:
            roi_vertex_ranges = off_roi["vertex_ranges"]
            weight = off_roi.get("weight", 1.0)
            
            roi_activity = tribe_wrapper.get_roi_activation(
                preds,
                roi_vertex_ranges,
                aggregation="mean"
            )
            
            activations = list(roi_activity.values())
            mean_activation = np.mean(activations)
            normalized = np.clip(mean_activation / 3.0, 0, 1)
            
            total_off_target += weight * normalized
        
        # Average across off-target ROIs
        if self.off_target_rois:
            total_off_target /= len(self.off_target_rois)
        
        return float(total_off_target)
    
    def _compute_energy_penalty(self, preds: np.ndarray) -> float:
        """
        Compute energy penalty (L2 norm of activation).
        Encourages sparse, efficient activations.
        
        Args:
            preds: (timesteps, num_vertices) predictions
        
        Returns:
            scalar penalty
        """
        # Mean squared activation across brain
        energy = np.mean(np.square(preds))
        
        # Normalize to roughly [0, 1]
        normalized_energy = np.clip(energy / 2.0, 0, 1)
        
        return float(normalized_energy)
    
    def _compute_temporal_penalty(self, preds: np.ndarray) -> float:
        """
        Compute temporal smoothness penalty.
        Penalizes sharp transitions across time.
        
        Args:
            preds: (timesteps, num_vertices) predictions
        
        Returns:
            scalar penalty
        """
        if preds.shape[0] < 2:
            return 0.0
        
        # Compute temporal differences (first derivative)
        temporal_diffs = np.diff(preds, axis=0)
        
        # Mean squared temporal gradient
        temporal_roughness = np.mean(np.square(temporal_diffs))
        
        # Normalize
        normalized_roughness = np.clip(temporal_roughness / 2.0, 0, 1)
        
        return float(normalized_roughness)
    
    def log_reward_breakdown(self, reward_dict: Dict) -> None:
        """Log a detailed breakdown of reward components."""
        logger.info(
            f"Reward breakdown:\n"
            f"  Total reward: {reward_dict['total']:.4f}\n"
            f"  Target activation: {reward_dict['target_activation']:.4f} "
            f"(reward: {reward_dict['target_reward']:.4f})\n"
            f"  Off-target activation: {reward_dict['off_target_activation']:.4f} "
            f"(penalty: {reward_dict['off_target_penalty']:.4f})\n"
            f"  Energy penalty: {reward_dict['energy_penalty']:.4f}\n"
            f"  Temporal penalty: {reward_dict['temporal_penalty']:.4f}"
        )
