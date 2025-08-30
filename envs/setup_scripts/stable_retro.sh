#!/bin/bash
# Setup script for stable-retro environment

echo "Setting up stable-retro for DreamerV3..."

# Activate the conda environment
echo "Activating dreamerv3-torch environment..."
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
conda activate dreamerv3-torch

# Check if stable-retro is already installed
if python -c "import retro" 2>/dev/null; then
    echo "stable-retro is already installed!"
else
    echo "Installing stable-retro..."
    pip install stable-retro
fi

# Test the installation
echo "Testing stable-retro installation..."
python -c "import retro; print(f'stable-retro version: {retro.__version__}')"

# Check available games
echo "Available games with ROMs:"
python -c "
import retro
games = retro.data.list_games()
print(f'Found {len(games)} games:')
for game in sorted(games)[:10]:  # Show first 10 games
    print(f'  - {game}')
if len(games) > 10:
    print(f'  ... and {len(games) - 10} more')
"

echo "Setup complete!"
echo ""
echo "To test the integration, run:"
echo "  python test_stable_retro.py"
echo ""
echo "To train DreamerV3 on a retro game, run:"
echo "  python dreamer.py --configs retro --task retro_Airstriker-Genesis --logdir ./logdir/retro_airstriker"
