#!/bin/bash
# NeuroStim Installation Script
# Complete setup for NeuroStim Optimization Engine

set -e

echo "=========================================="
echo "NeuroStim Installation Script"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python --version 2>&1 | grep -oP '\d+\.\d+' || echo "unknown")
echo "✓ Python $python_version detected"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate || . venv/Scripts/activate
echo "✓ Virtual environment activated"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel --quiet
echo "✓ Pip upgraded"

# Install PyTorch
echo ""
echo "Installing PyTorch..."
echo "Choose your CUDA version:"
echo "  1) CPU only"
echo "  2) CUDA 11.8"
echo "  3) CUDA 12.1"
read -p "Enter choice (1-3): " cuda_choice

case $cuda_choice in
    1)
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet
        echo "✓ PyTorch CPU installed"
        ;;
    2)
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --quiet
        echo "✓ PyTorch CUDA 11.8 installed"
        ;;
    3)
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet
        echo "✓ PyTorch CUDA 12.1 installed"
        ;;
    *)
        echo "Invalid choice, installing CPU version"
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet
        ;;
esac

# Install other requirements
echo ""
echo "Installing requirements from requirements.txt..."
pip install -r requirements.txt --quiet
echo "✓ All requirements installed"

# Install TRIBE V2
echo ""
echo "Installing TRIBE V2..."
pip install git+https://github.com/facebookresearch/tribev2.git --quiet 2>/dev/null || {
    echo "⚠ TRIBE V2 installation from GitHub failed"
    echo "  (This is OK - mock mode will work for testing)"
    echo "  To install later: pip install git+https://github.com/facebookresearch/tribev2.git"
}

# Verify installations
echo ""
echo "Verifying installations..."
echo ""

python -c "import torch; print('✓ PyTorch', torch.__version__)" && verified_torch=1 || verified_torch=0
python -c "from diffusers import StableDiffusionPipeline; print('✓ Diffusers installed')" && verified_diffusers=1 || verified_diffusers=0
python -c "import yaml; print('✓ PyYAML installed')" && verified_yaml=1 || verified_yaml=0
python -c "import matplotlib; print('✓ Matplotlib installed')" && verified_mpl=1 || verified_mpl=0

if [ "$verified_torch" -eq 1 ] && [ "$verified_yaml" -eq 1 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Installation Complete!"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "  1. Setup example configs:"
    echo "     python example_experiments.py --setup"
    echo ""
    echo "  2. Run a quick test:"
    echo "     python pipeline.py --config config_quick.yaml"
    echo ""
    echo "  3. View available examples:"
    echo "     python quick_start.py --example 1"
    echo ""
    echo "  4. Read the README:"
    echo "     cat README.md"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "⚠ Installation had some issues"
    echo "=========================================="
    echo "Some packages failed to install. Check above messages."
    echo "Core modules (PyTorch, YAML) are essential."
fi

echo ""
echo "=========================================="
echo "Project Structure:"
echo "=========================================="
echo ""
ls -la *.py *.yaml 2>/dev/null | awk '{print "  " $NF}'
echo ""
