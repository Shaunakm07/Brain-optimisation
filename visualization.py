"""
Visualization Module
Plots brain activation maps, ROI timecourses, and stimulus evolution
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class NeuroVisualization:
    """
    Visualization utilities for neurostimulation optimization.
    Creates plots of brain activation, ROI timecourses, and stimulus evolution.
    """
    
    def __init__(self, output_dir: str = "./visualizations"):
        """
        Initialize visualization module.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Visualization output dir: {self.output_dir}")
    
    def plot_roi_activation(
        self,
        roi_activities: Dict[str, np.ndarray],
        target_roi_name: str,
        off_target_roi_names: List[str],
        title: str = "ROI Activation Over Time",
        save_path: Optional[str] = None,
        show: bool = False
    ) -> plt.Figure:
        """
        Plot activation timecourse for multiple ROIs.
        
        Args:
            roi_activities: Dict mapping ROI name to (timesteps,) timecourse
            target_roi_name: Name of target ROI (highlighted)
            off_target_roi_names: Names of off-target ROIs
            title: Plot title
            save_path: Path to save figure
            show: Whether to display
        
        Returns:
            figure: Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot target ROI with emphasis
        if target_roi_name in roi_activities:
            timecourse = roi_activities[target_roi_name]
            ax.plot(
                timecourse,
                linewidth=3,
                label=f"{target_roi_name} (TARGET)",
                color="red",
                alpha=0.8
            )
            ax.fill_between(
                range(len(timecourse)),
                timecourse,
                alpha=0.2,
                color="red"
            )
        
        # Plot off-target ROIs
        colors = plt.cm.Set2(np.linspace(0, 1, len(off_target_roi_names)))
        for off_roi, color in zip(off_target_roi_names, colors):
            if off_roi in roi_activities:
                ax.plot(
                    roi_activities[off_roi],
                    linewidth=1.5,
                    label=f"{off_roi} (off-target)",
                    color=color,
                    alpha=0.6,
                    linestyle="--"
                )
        
        ax.set_xlabel("Time (frames)", fontsize=12)
        ax.set_ylabel("Activation (a.u.)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved ROI activation plot: {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_brain_surface(
        self,
        preds: np.ndarray,
        target_roi_vertices: Dict[str, Tuple[int, int]],
        timepoint: int = -1,
        title: str = "Brain Activation Map",
        save_path: Optional[str] = None,
        show: bool = False
    ) -> plt.Figure:
        """
        Plot brain activation on cortical surface (simplified 2D projection).
        
        Args:
            preds: (timesteps, num_vertices) predictions
            target_roi_vertices: Vertex range for target ROI
            timepoint: Which timepoint to plot (-1 for last)
            title: Plot title
            save_path: Path to save
            show: Whether to display
        
        Returns:
            figure: Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Use final timepoint
        activation = preds[timepoint]
        
        # Separate hemispheres
        num_vertices = len(activation)
        vertices_per_hemi = num_vertices // 2
        
        for ax, hemi, start in zip(
            axes,
            ["Left", "Right"],
            [0, vertices_per_hemi]
        ):
            hemi_activation = activation[start:start + vertices_per_hemi]
            
            # Reshape for visualization (simplified grid)
            grid_size = int(np.sqrt(len(hemi_activation)))
            try:
                grid = hemi_activation[:grid_size**2].reshape(grid_size, grid_size)
                
                im = ax.imshow(grid, cmap="hot", aspect="auto")
                ax.set_title(f"{hemi} Hemisphere", fontweight="bold")
                ax.axis("off")
                
                # Highlight target ROI if applicable
                roi_key = "left" if hemi == "Left" else "right"
                if roi_key in target_roi_vertices:
                    roi_start, roi_end = target_roi_vertices[roi_key]
                    # Roughly mark ROI region
                    roi_idx = roi_start % (grid_size ** 2)
                    roi_y, roi_x = divmod(roi_idx, grid_size)
                    rect = mpatches.Rectangle(
                        (roi_x - 5, roi_y - 5), 10, 10,
                        linewidth=2, edgecolor="cyan", facecolor="none"
                    )
                    ax.add_patch(rect)
                
                plt.colorbar(im, ax=ax, label="Activation")
            
            except Exception as e:
                logger.warning(f"Error plotting hemisphere: {e}")
                ax.text(0.5, 0.5, f"Error plotting {hemi} hemisphere")
        
        fig.suptitle(title, fontsize=14, fontweight="bold")
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved brain surface plot: {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_optimization_progress(
        self,
        rewards: List[float],
        title: str = "Optimization Progress",
        save_path: Optional[str] = None,
        show: bool = False
    ) -> plt.Figure:
        """
        Plot reward progression during optimization.
        
        Args:
            rewards: List of rewards per iteration
            title: Plot title
            save_path: Path to save
            show: Whether to display
        
        Returns:
            figure: Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        iterations = np.arange(len(rewards))
        ax.plot(iterations, rewards, linewidth=2, marker="o", markersize=4)
        
        # Add rolling mean
        if len(rewards) > 5:
            rolling_mean = np.convolve(rewards, np.ones(5)/5, mode="valid")
            ax.plot(
                iterations[2:-2],
                rolling_mean,
                linewidth=2,
                label="Rolling mean (window=5)",
                linestyle="--",
                color="red"
            )
        
        ax.fill_between(iterations, rewards, alpha=0.2)
        ax.set_xlabel("Iteration", fontsize=12)
        ax.set_ylabel("Reward", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved optimization progress plot: {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_reward_breakdown(
        self,
        reward_components: Dict[str, float],
        title: str = "Reward Components",
        save_path: Optional[str] = None,
        show: bool = False
    ) -> plt.Figure:
        """
        Plot breakdown of reward function components.
        
        Args:
            reward_components: Dict with component values
            title: Plot title
            save_path: Path to save
            show: Whether to display
        
        Returns:
            figure: Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        components = list(reward_components.keys())
        values = list(reward_components.values())
        
        # Color bars by positive/negative
        colors = ["green" if v > 0 else "red" for v in values]
        
        bars = ax.barh(components, values, color=colors, alpha=0.7)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(val, i, f"  {val:.4f}", va="center")
        
        ax.set_xlabel("Value", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.axvline(x=0, color="black", linestyle="-", linewidth=0.8)
        ax.grid(True, alpha=0.3, axis="x")
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved reward breakdown plot: {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_stimulus_comparison(
        self,
        stimuli: Dict[str, np.ndarray],
        title: str = "Stimulus Evolution",
        save_path: Optional[str] = None,
        show: bool = False
    ) -> plt.Figure:
        """
        Compare multiple stimuli side-by-side (for images).
        
        Args:
            stimuli: Dict mapping label to image array
            title: Plot title
            save_path: Path to save
            show: Whether to display
        
        Returns:
            figure: Matplotlib figure
        """
        num_stimuli = len(stimuli)
        fig, axes = plt.subplots(1, num_stimuli, figsize=(4*num_stimuli, 4))
        
        if num_stimuli == 1:
            axes = [axes]
        
        for ax, (label, stimulus) in zip(axes, stimuli.items()):
            # Normalize for display
            if stimulus.max() <= 1.0:
                display_stim = stimulus
            else:
                display_stim = stimulus / 255.0
            
            ax.imshow(display_stim, cmap="gray")
            ax.set_title(label, fontweight="bold")
            ax.axis("off")
        
        fig.suptitle(title, fontsize=14, fontweight="bold")
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved stimulus comparison plot: {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_summary_report(
        self,
        optimization_state,
        reward_components: Dict,
        roi_names: List[str],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a comprehensive summary report.
        
        Args:
            optimization_state: OptimizationState from optimizer
            reward_components: Breakdown of reward
            roi_names: List of ROI names
            save_path: Path to save
        
        Returns:
            figure: Matplotlib figure
        """
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Optimization progress
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(optimization_state.rewards, linewidth=2, marker="o")
        ax1.set_title("Optimization Progress", fontweight="bold")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Reward")
        ax1.grid(True, alpha=0.3)
        
        # 2. Reward breakdown
        ax2 = fig.add_subplot(gs[1, 0])
        components = list(reward_components.keys())
        values = list(reward_components.values())
        colors = ["green" if v > 0 else "red" for v in values]
        ax2.barh(components, values, color=colors, alpha=0.7)
        ax2.set_title("Reward Components", fontweight="bold")
        ax2.grid(True, alpha=0.3, axis="x")
        
        # 3. Summary statistics
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.axis("off")
        summary_text = f"""
        OPTIMIZATION SUMMARY
        
        Final Reward: {optimization_state.best_reward:.4f}
        Iterations: {optimization_state.iteration}
        Mean Reward: {optimization_state.mean_reward:.4f}
        Max Reward: {max(optimization_state.rewards):.4f}
        
        Improvement: {(max(optimization_state.rewards) - optimization_state.rewards[0]):.4f}
        """
        ax3.text(0.1, 0.5, summary_text, fontsize=11, family="monospace",
                verticalalignment="center")
        
        # 4. ROI distribution (simplified)
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis("off")
        ax4.text(0.1, 0.5, f"Target ROIs: {', '.join(roi_names)}", fontsize=11)
        
        fig.suptitle("NeuroStim Optimization Summary", fontsize=16, fontweight="bold")
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved summary report: {save_path}")
        
        return fig
