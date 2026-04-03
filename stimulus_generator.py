"""
Stimulus Generator
Generates and manipulates images, audio, and video using diffusion models
"""

import numpy as np
import torch
import os
from typing import Union, List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class StimulusGenerator:
    """
    Base class for stimulus generation.
    Subclasses implement image, audio, and video generation.
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
    
    def generate(self, *args, **kwargs):
        raise NotImplementedError


class ImageGenerator(StimulusGenerator):
    """
    Generate or manipulate images using diffusion models.
    Uses Stable Diffusion or similar.
    """
    
    def __init__(
        self,
        model_name: str = "runwayml/stable-diffusion-v1-5",
        device: str = "cuda",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5
    ):
        """
        Initialize image generator.
        
        Args:
            model_name: HuggingFace model identifier
            device: torch device
            num_inference_steps: Number of diffusion steps
            guidance_scale: Classifier-free guidance scale
        """
        super().__init__(device)
        self.model_name = model_name
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        
        try:
            from diffusers import StableDiffusionPipeline
            logger.info(f"Loading image generator: {model_name}")
            
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False
            )
            self.pipe = self.pipe.to(device)
            self.pipe.enable_attention_slicing()
            
        except ImportError:
            logger.warning(
                "Diffusers not installed. Using mock image generator.\n"
                "Install with: pip install diffusers transformers torch"
            )
            self.pipe = None
            self.mock_mode = True
    
    def generate_from_prompt(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        height: int = 512,
        width: int = 512,
        num_images: int = 1,
        seed: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Generate image from text prompt.
        
        Args:
            prompt: Positive prompt
            negative_prompt: Negative prompt (what to avoid)
            height: Output height
            width: Output width
            num_images: Number of images to generate
            seed: Random seed for reproducibility
            **kwargs: Additional arguments
        
        Returns:
            images: (batch, height, width, 3) uint8 array
        """
        if self.pipe is None:
            return self._mock_generate_image(height, width, num_images)
        
        try:
            generator = torch.Generator(device=self.device)
            if seed is not None:
                generator.manual_seed(seed)
            
            with torch.no_grad():
                output = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=height,
                    width=width,
                    num_inference_steps=self.num_inference_steps,
                    guidance_scale=self.guidance_scale,
                    num_images_per_prompt=num_images,
                    generator=generator,
                    output_type="np"
                )
            
            images = output.images
            logger.info(f"Generated {images.shape[0]} images with shape {images.shape}")
            
            return images
        
        except Exception as e:
            logger.error(f"Error in image generation: {e}")
            raise
    
    def modify_image(
        self,
        image: np.ndarray,
        prompt: str,
        strength: float = 0.75,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Modify existing image using img2img diffusion.
        
        Args:
            image: Input image (H, W, 3) in [0, 1] or [0, 255]
            prompt: Modification prompt
            strength: Denoising strength (0=no change, 1=full regeneration)
            negative_prompt: Negative prompt
            seed: Random seed
            **kwargs: Additional arguments
        
        Returns:
            modified_image: (H, W, 3) uint8 array
        """
        if self.pipe is None:
            return self._mock_modify_image(image)
        
        try:
            from diffusers import StableDiffusionImg2ImgPipeline
            from PIL import Image
            
            # Convert numpy to PIL Image
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            
            pil_image = Image.fromarray(image)
            
            # Use img2img pipeline
            pipe_img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False
            )
            pipe_img2img = pipe_img2img.to(self.device)
            
            generator = torch.Generator(device=self.device)
            if seed is not None:
                generator.manual_seed(seed)
            
            with torch.no_grad():
                output = pipe_img2img(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=pil_image,
                    strength=strength,
                    guidance_scale=self.guidance_scale,
                    num_inference_steps=int(self.num_inference_steps * strength),
                    generator=generator,
                    output_type="np"
                )
            
            modified = output.images[0]
            logger.info(f"Modified image shape: {modified.shape}")
            
            return modified
        
        except Exception as e:
            logger.error(f"Error in image modification: {e}")
            raise
    
    def _mock_generate_image(
        self,
        height: int,
        width: int,
        num_images: int
    ) -> np.ndarray:
        """Generate synthetic images for testing."""
        # Create colorful random images
        images = np.random.randint(0, 256, (num_images, height, width, 3), dtype=np.uint8)
        
        # Add some structure (not pure noise)
        for i in range(num_images):
            # Add gradient
            for c in range(3):
                images[i, :, :, c] += np.linspace(0, 50, width)[np.newaxis, :]
        
        return images / 255.0  # Normalize to [0, 1]
    
    def _mock_modify_image(self, image: np.ndarray) -> np.ndarray:
        """Mock image modification for testing."""
        # Add slight noise and color shift
        modified = image.copy()
        modified += np.random.normal(0, 0.05, modified.shape)
        modified = np.clip(modified, 0, 1)
        return modified


class VideoGenerator(StimulusGenerator):
    """
    Generate video sequences using latent video diffusion.
    Can also be constructed from image sequences.
    """
    
    def __init__(
        self,
        model_name: str = "damo-vilab/text-to-video-ms-1.7b",
        device: str = "cuda",
        num_frames: int = 16,
        fps: int = 8
    ):
        """
        Initialize video generator.
        
        Args:
            model_name: HuggingFace model identifier
            device: torch device
            num_frames: Number of frames per video
            fps: Frames per second
        """
        super().__init__(device)
        self.model_name = model_name
        self.num_frames = num_frames
        self.fps = fps
        
        try:
            from diffusers import TextToVideoSDPipeline
            logger.info(f"Loading video generator: {model_name}")
            
            self.pipe = TextToVideoSDPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float16
            )
            self.pipe = self.pipe.to(device)
            
        except ImportError:
            logger.warning(
                "Video diffusion not available. Using mock video generator.\n"
                "Install with: pip install diffusers"
            )
            self.pipe = None
            self.mock_mode = True
    
    def generate_from_prompt(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_frames: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Generate video from text prompt.
        
        Args:
            prompt: Text description
            negative_prompt: What to avoid
            num_frames: Number of frames (default from init)
            seed: Random seed
            **kwargs: Additional arguments
        
        Returns:
            video: (num_frames, height, width, 3) in [0, 1]
        """
        if num_frames is None:
            num_frames = self.num_frames
        
        if self.pipe is None:
            return self._mock_generate_video(num_frames)
        
        try:
            generator = torch.Generator(device=self.device)
            if seed is not None:
                generator.manual_seed(seed)
            
            with torch.no_grad():
                frames = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_frames=num_frames,
                    num_inference_steps=50,
                    guidance_scale=7.5,
                    generator=generator
                ).frames
            
            # Convert to numpy
            video = np.stack([np.array(f) for f in frames])
            logger.info(f"Generated video shape: {video.shape}")
            
            return video
        
        except Exception as e:
            logger.error(f"Error in video generation: {e}")
            raise
    
    def create_video_from_images(
        self,
        images: List[np.ndarray],
        loop: bool = False
    ) -> np.ndarray:
        """
        Create video from sequence of images.
        
        Args:
            images: List of (H, W, 3) images in [0, 1] or [0, 255]
            loop: Whether to add reverse frames for looping
        
        Returns:
            video: (num_frames, H, W, 3) in [0, 1]
        """
        # Normalize to [0, 1]
        normalized_images = []
        for img in images:
            if img.max() > 1.0:
                img = img / 255.0
            normalized_images.append(img)
        
        video = np.stack(normalized_images)
        
        if loop and len(video) > 1:
            # Add reverse sequence for loop
            video = np.concatenate([video, video[::-1][1:-1]])
        
        logger.info(f"Created video from {len(images)} images: {video.shape}")
        
        return video
    
    def _mock_generate_video(self, num_frames: int) -> np.ndarray:
        """Generate synthetic video for testing."""
        height, width = 256, 256
        
        video = np.zeros((num_frames, height, width, 3))
        
        for t in range(num_frames):
            # Create moving patterns
            for c in range(3):
                # Sinusoidal pattern that evolves over time
                x = np.linspace(-np.pi, np.pi, width)
                y = np.linspace(-np.pi, np.pi, height)
                X, Y = np.meshgrid(x, y)
                
                phase = 2 * np.pi * t / num_frames
                pattern = np.sin(X + phase) * np.cos(Y + phase)
                pattern = (pattern + 1) / 2  # Normalize to [0, 1]
                
                video[t, :, :, c] = pattern
        
        return video


class AudioGenerator(StimulusGenerator):
    """
    Generate or modify audio using diffusion models on spectrograms.
    """
    
    def __init__(
        self,
        device: str = "cuda",
        sample_rate: int = 16000,
        duration: float = 5.0
    ):
        """
        Initialize audio generator.
        
        Args:
            device: torch device
            sample_rate: Audio sample rate (Hz)
            duration: Duration in seconds
        """
        super().__init__(device)
        self.sample_rate = sample_rate
        self.duration = duration
        self.num_samples = int(sample_rate * duration)
        
        logger.info(
            f"Initialized audio generator: "
            f"sr={sample_rate}, duration={duration}s"
        )
    
    def generate_from_prompt(
        self,
        prompt: str,
        duration: Optional[float] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Generate audio from text description.
        
        Args:
            prompt: Audio description
            duration: Duration in seconds
            seed: Random seed
            **kwargs: Additional arguments
        
        Returns:
            audio: (num_samples,) waveform in [-1, 1]
        """
        if duration is None:
            duration = self.duration
        
        # For now, return mock audio
        return self._mock_generate_audio(duration)
    
    def modify_audio(
        self,
        audio: np.ndarray,
        prompt: str,
        strength: float = 0.5,
        seed: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Modify existing audio.
        
        Args:
            audio: Input waveform
            prompt: Modification description
            strength: Modification strength
            seed: Random seed
            **kwargs: Additional arguments
        
        Returns:
            modified_audio: (num_samples,) waveform
        """
        return self._mock_modify_audio(audio, strength)
    
    def _mock_generate_audio(self, duration: float) -> np.ndarray:
        """Generate synthetic audio for testing."""
        num_samples = int(self.sample_rate * duration)
        
        # Create a simple synthetic sound: combination of sinusoids
        t = np.arange(num_samples) / self.sample_rate
        
        # Mix of frequencies (like a musical tone)
        f1, f2 = 440, 554  # Musical notes
        audio = (
            0.5 * np.sin(2 * np.pi * f1 * t) +
            0.3 * np.sin(2 * np.pi * f2 * t)
        )
        
        # Add amplitude envelope (fade in/out)
        envelope = np.ones_like(t)
        fade_samples = int(0.1 * num_samples)
        envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
        envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
        
        audio = audio * envelope
        
        # Normalize
        audio = audio / (np.max(np.abs(audio)) + 1e-7)
        
        return audio
    
    def _mock_modify_audio(
        self,
        audio: np.ndarray,
        strength: float
    ) -> np.ndarray:
        """Mock audio modification."""
        # Add slight distortion
        modified = audio.copy()
        noise = np.random.normal(0, strength * 0.1, audio.shape)
        modified = modified + noise
        modified = np.clip(modified, -1, 1)
        return modified
