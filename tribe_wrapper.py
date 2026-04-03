"""
TRIBE V2 Brain Model Wrapper
Provides a unified interface to TRIBE V2 for fMRI prediction and ROI extraction
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TribeV2Wrapper:
    """
    Wraps the official TRIBE V2 model from Facebook Research.
    Handles input conversion, prediction, and ROI extraction.
    
    GitHub: https://github.com/facebookresearch/tribev2
    """
    
    def __init__(self, model_name: str = "facebook/tribev2", device: str = "cuda"):
        """
        Initialize TRIBE V2 model.
        
        Args:
            model_name: HuggingFace model identifier
            device: torch device (cuda or cpu)
        """
        self.device = device
        self.model_name = model_name
        
        try:
            # Import TRIBE V2 (assumes installation via: pip install tribev2)
            from tribev2 import TribeModel
            self.TribeModel = TribeModel
            logger.info(f"Loading TRIBE V2 model: {model_name}")
            self.model = TribeModel.from_pretrained(model_name)
            if hasattr(self.model, 'to'):
                self.model.to(device)
            if hasattr(self.model, 'eval'):
                self.model.eval()
        except ImportError:
            logger.warning(
                "TRIBE V2 not installed. Using mock model for demonstration.\n"
                "Install with: pip install tribev2\n"
                "GitHub: https://github.com/facebookresearch/tribev2"
            )
            self.model = None
            self._setup_mock_model()
    
    def _setup_mock_model(self):
        """Setup a mock TRIBE model for testing without real installation."""
        logger.info("Using mock TRIBE V2 model for development/testing")
        self.mock_mode = True
    
    def predict_from_video(
        self,
        video_path: str,
        fps: int = 30,
        **kwargs
    ) -> Tuple[np.ndarray, Dict]:
        """
        Predict brain activity from video stimulus.
        
        Args:
            video_path: Path to video file
            fps: Frames per second
            **kwargs: Additional arguments for TRIBE
        
        Returns:
            preds: (timesteps, num_vertices) brain activity predictions
            metadata: Dictionary with prediction metadata
        """
        if self.model is None:
            return self._mock_predict_video(video_path)
        
        try:
            # Create events dataframe from video
            df = self.model.get_events_dataframe(video_path=video_path)
            logger.info(f"Video events shape: {df.shape}")
            
            # Run prediction
            with torch.no_grad():
                preds, segments = self.model.predict(events=df)
            
            # preds shape: (timesteps, brain_vertices)
            # fsaverage: ~20k vertices per hemisphere, bilateral = ~40k total
            preds = np.array(preds)
            
            metadata = {
                "modality": "video",
                "num_timesteps": preds.shape[0],
                "num_vertices": preds.shape[1],
                "fps": fps,
                "video_path": video_path
            }
            
            logger.info(
                f"Prediction shape: {preds.shape} "
                f"(timesteps x vertices)"
            )
            
            return preds, metadata
        
        except Exception as e:
            logger.error(f"Error in video prediction: {e}")
            raise
    
    def predict_from_audio(
        self,
        audio_path: str,
        sr: int = 16000,
        **kwargs
    ) -> Tuple[np.ndarray, Dict]:
        """
        Predict brain activity from audio stimulus.
        
        Args:
            audio_path: Path to audio file
            sr: Sample rate
            **kwargs: Additional arguments
        
        Returns:
            preds: (timesteps, num_vertices) brain activity
            metadata: Prediction metadata
        """
        if self.model is None:
            return self._mock_predict_audio(audio_path)
        
        try:
            df = self.model.get_events_dataframe(audio_path=audio_path, sr=sr)
            
            with torch.no_grad():
                preds, segments = self.model.predict(events=df)
            
            preds = np.array(preds)
            
            metadata = {
                "modality": "audio",
                "num_timesteps": preds.shape[0],
                "num_vertices": preds.shape[1],
                "sample_rate": sr,
                "audio_path": audio_path
            }
            
            return preds, metadata
        
        except Exception as e:
            logger.error(f"Error in audio prediction: {e}")
            raise
    
    def predict_from_image(
        self,
        image_path: str,
        **kwargs
    ) -> Tuple[np.ndarray, Dict]:
        """
        Predict brain activity from static image.
        
        Args:
            image_path: Path to image file
            **kwargs: Additional arguments
        
        Returns:
            preds: (1, num_vertices) brain activity (single timepoint)
            metadata: Prediction metadata
        """
        if self.model is None:
            return self._mock_predict_image(image_path)
        
        try:
            df = self.model.get_events_dataframe(image_path=image_path)
            
            with torch.no_grad():
                preds, segments = self.model.predict(events=df)
            
            preds = np.array(preds)
            
            metadata = {
                "modality": "image",
                "num_timesteps": preds.shape[0],
                "num_vertices": preds.shape[1],
                "image_path": image_path
            }
            
            return preds, metadata
        
        except Exception as e:
            logger.error(f"Error in image prediction: {e}")
            raise
    
    def predict_from_text(
        self,
        text: str,
        **kwargs
    ) -> Tuple[np.ndarray, Dict]:
        """
        Predict brain activity from text (auto-converted to speech by TRIBE).
        
        Args:
            text: Text string
            **kwargs: Additional arguments
        
        Returns:
            preds: Brain activity predictions
            metadata: Prediction metadata
        """
        if self.model is None:
            return self._mock_predict_text(text)
        
        try:
            df = self.model.get_events_dataframe(text=text)
            
            with torch.no_grad():
                preds, segments = self.model.predict(events=df)
            
            preds = np.array(preds)
            
            metadata = {
                "modality": "text",
                "num_timesteps": preds.shape[0],
                "num_vertices": preds.shape[1],
                "text": text[:100]  # Store first 100 chars
            }
            
            return preds, metadata
        
        except Exception as e:
            logger.error(f"Error in text prediction: {e}")
            raise
    
    # ========== MOCK PREDICTIONS FOR TESTING ==========
    
    def _mock_predict_video(self, video_path: str) -> Tuple[np.ndarray, Dict]:
        """Generate synthetic predictions for testing."""
        timesteps = 60  # ~2 seconds at 30 fps
        num_vertices = 40962  # fsaverage surface (bilateral)
        
        # Simulate structured noise: stronger in visual regions
        preds = np.random.normal(0, 0.5, (timesteps, num_vertices))
        
        # Boost visual cortex and face regions
        preds[:, 4500:5200] += 0.5  # Left FFA
        preds[:, 24500:25200] += 0.5  # Right FFA
        preds[:, 0:1000] += 0.3  # V1
        
        metadata = {
            "modality": "video",
            "num_timesteps": timesteps,
            "num_vertices": num_vertices,
            "video_path": video_path,
            "mock": True
        }
        
        return preds, metadata
    
    def _mock_predict_audio(self, audio_path: str) -> Tuple[np.ndarray, Dict]:
        """Generate synthetic audio predictions."""
        timesteps = 40
        num_vertices = 40962
        
        preds = np.random.normal(0, 0.5, (timesteps, num_vertices))
        preds[:, 8000:9000] += 0.6  # Auditory cortex
        
        metadata = {
            "modality": "audio",
            "num_timesteps": timesteps,
            "num_vertices": num_vertices,
            "audio_path": audio_path,
            "mock": True
        }
        
        return preds, metadata
    
    def _mock_predict_image(self, image_path: str) -> Tuple[np.ndarray, Dict]:
        """Generate synthetic image predictions."""
        timesteps = 1
        num_vertices = 40962
        
        preds = np.random.normal(0, 0.5, (timesteps, num_vertices))
        preds[:, 4500:5200] += 0.4
        
        metadata = {
            "modality": "image",
            "num_timesteps": timesteps,
            "num_vertices": num_vertices,
            "image_path": image_path,
            "mock": True
        }
        
        return preds, metadata
    
    def _mock_predict_text(self, text: str) -> Tuple[np.ndarray, Dict]:
        """Generate synthetic text predictions."""
        timesteps = 30
        num_vertices = 40962
        
        preds = np.random.normal(0, 0.5, (timesteps, num_vertices))
        preds[:, 12000:13000] += 0.5  # Language regions
        
        metadata = {
            "modality": "text",
            "num_timesteps": timesteps,
            "num_vertices": num_vertices,
            "text": text[:100],
            "mock": True
        }
        
        return preds, metadata
    
    # ========== ROI UTILITIES ==========
    
    def get_roi_activation(
        self,
        preds: np.ndarray,
        roi_vertex_ranges: Dict[str, Tuple[int, int]],
        aggregation: str = "mean"
    ) -> Dict[str, np.ndarray]:
        """
        Extract activation in specific ROI vertex ranges.
        
        Args:
            preds: (timesteps, num_vertices) predictions
            roi_vertex_ranges: Dict mapping hemisphere to (start, end) vertex indices
            aggregation: How to combine vertices ("mean", "max", "std")
        
        Returns:
            roi_timecourse: Dict with aggregated timecourse per hemisphere
        """
        roi_activity = {}
        
        for hemisphere, (start, end) in roi_vertex_ranges.items():
            roi_verts = preds[:, start:end]
            
            if aggregation == "mean":
                roi_activity[hemisphere] = np.mean(roi_verts, axis=1)
            elif aggregation == "max":
                roi_activity[hemisphere] = np.max(roi_verts, axis=1)
            elif aggregation == "std":
                roi_activity[hemisphere] = np.std(roi_verts, axis=1)
            else:
                raise ValueError(f"Unknown aggregation: {aggregation}")
        
        return roi_activity
    
    def get_whole_brain_summary(
        self,
        preds: np.ndarray,
        aggregation: str = "mean"
    ) -> float:
        """
        Get whole-brain summary statistic.
        
        Args:
            preds: (timesteps, num_vertices) predictions
            aggregation: Aggregation method
        
        Returns:
            scalar summary value
        """
        if aggregation == "mean":
            return float(np.mean(preds))
        elif aggregation == "max":
            return float(np.max(preds))
        elif aggregation == "std":
            return float(np.std(preds))
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
