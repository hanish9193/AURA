# AURA Complete Setup Guide

## Prerequisites

- Python 3.9+
- NVIDIA GPU (optional but recommended for speed)
- 20GB free disk space (for models + data)
- Ollama + Mistral (for interactive Q&A)

## Step 1: Clone & Create Virtual Environment

\`\`\`bash
git clone https://github.com/yourusername/aura.git
cd aura

# Create environment (Python 3.11)
conda create -n aura_env python=3.11
conda activate aura_env
\`\`\`

## Step 2: Install Dependencies

\`\`\`bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
\`\`\`

If you get build errors (especially on Windows):
\`\`\`bash
# Option A: Pre-built wheels only
pip install --only-binary :all: -r requirements.txt

# Option B: Install individually
pip install pandas numpy matplotlib seaborn scikit-learn pillow tensorflow requests torch
\`\`\`

## Step 3: Setup Pre-trained VisionTextBridge Model

\`\`\`bash
# Create model folder
mkdir -p TextVisionBridge

# Place your trained best_model.h5 here
cp /path/to/best_model.h5 TextVisionBridge/

# Verify
ls TextVisionBridge/best_model.h5  # Should exist
\`\`\`

## Step 4: Install Ollama & Mistral (Optional but Recommended)

For interactive Q&A with Mistral LLM:

\`\`\`bash
# Download from https://ollama.ai/ and install
# OR use package manager:

# macOS
brew install ollama

# Linux
curl https://ollama.ai/install.sh | sh

# Then pull Mistral
ollama pull mistral

# Keep Ollama running (in separate terminal)
ollama serve
\`\`\`

If Ollama not installed, AURA uses fallback text analysis (slower but works).

## Step 5: Test Installation

\`\`\`bash
python examples/quick_start.py
\`\`\`

Expected output:
\`\`\`
✓ Loaded data: 1000 rows, 10 columns
✓ Created 15 graphs
✓ Extracted embeddings (2560-D)
✓ Generated text descriptions
✓ Q&A engine ready
🎨 Launching GUI...
\`\`\`

## Step 6: Use AURA in Your Project

\`\`\`python
from aura import Aura

# Initialize
aura = Aura()

# Load your CSV data
aura.load_data("your_data.csv")

# Generate insights (15 graphs + embeddings + descriptions)
aura.generate_insights()

# Launch interactive Tkinter GUI
aura.launch_gui()
\`\`\`

The GUI will show:
- 💬 Chat interface powered by Mistral
- 📊 View all 15 generated graphs
- ❓ Ask questions about your data
- 🔍 Get intelligent answers based on visual patterns

## Complete Pipeline Flow

\`\`\`
[1. Load CSV]
        ↓
[2. Validate Data]
        ↓
[3. Generate 15 Graphs]
    • Correlation heatmap
    • Distribution plots
    • Scatter plots
    • Box plots (outliers)
    • Category analysis
    • Data quality
    • Feature importance
        ↓
[4. Extract Features (EfficientNetB7)]
    • 2560-D embeddings per graph
    • Pre-trained on ImageNet
        ↓
[5. Convert to Text (VisionTextBridge)]
    • Neural model: embeddings → descriptions
    • Trained on 100k PlotQA graphs
    • Outputs: "shows positive trend", "has outliers", etc.
        ↓
[6. Interactive Q&A (Mistral LLM)]
    • Takes text descriptions + your question
    • Generates intelligent answers
    • Runs locally via Ollama
        ↓
[7. Tkinter GUI Chat Interface]
    • Ask questions
    • View graphs
    • Get instant answers
\`\`\`

## Troubleshooting

### "Module not found: aura"
\`\`\`bash
# Make sure you're in the correct environment
conda activate aura_env
# Install in development mode
pip install -e .
\`\`\`

### "TensorFlow GPU not found"
\`\`\`bash
# Install CUDA support (RTX/GTX only)
pip install tensorflow[and-cuda]==2.13.0

# Test GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
\`\`\`

### "VisionTextBridge model not found"
\`\`\`bash
# Make sure best_model.h5 is in the right place
ls -la TextVisionBridge/best_model.h5

# If not there, copy it:
cp /your/path/to/best_model.h5 TextVisionBridge/
\`\`\`

### "Ollama connection refused"
\`\`\`bash
# Make sure Ollama is running in another terminal
ollama serve

# Test connection
curl http://localhost:11434/api/tags
\`\`\`

### "Out of Memory"
\`\`\`python
# In feature_extractor.py, reduce batch size from 16 to 8
# Or use CPU only (slower but uses less memory)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
\`\`\`

### Numpy/Pandas build errors on Windows
\`\`\`bash
# Delete old environment
conda env remove -n aura_env

# Recreate with Python 3.10
conda create -n aura_env python=3.10
conda activate aura_env

# Install again
pip install -r requirements.txt
\`\`\`

## Performance Benchmarks

| Component | Speed (GPU) | Speed (CPU) |
|-----------|------------|-----------|
| Load CSV | <1s | <1s |
| Generate 15 graphs | 2-5s | 3-5s |
| Extract embeddings | 10-20s | 120s |
| VisionTextBridge | 5-10s | 20-30s |
| Mistral Q&A | 0.5-1s | 0.5-1s |
| **Total** | **20-40s** | **150-160s** |

## File Structure

\`\`\`
aura/
├── __init__.py                      # Package entry point
├── core.py                          # Main Aura class
├── graph_generator.py               # 15 graph types
├── feature_extractor.py             # EfficientNetB7 embeddings
├── vision_text_bridge_loader.py     # Load best_model.h5
├── qa_engine.py                     # VisionTextBridge + Mistral
├── gui.py                           # Tkinter GUI
└── examples/
    └── quick_start.py               # Complete example

TextVisionBridge/
└── best_model.h5                    # Pre-trained model (you add this)

requirements.txt                     # All dependencies
README.md                           # User guide
SETUP_INSTRUCTIONS.md               # This file
AURA_IEEE_PAPER.md                 # Research paper
\`\`\`

## Next Steps

1. ✅ Complete setup as above
2. 📖 Read `README.md` for usage guide
3. 📚 Read `AURA_IEEE_PAPER.md` for methodology
4. 🚀 Run `python examples/quick_start.py`
5. 💾 Push to GitHub
6. 📝 Cite in your research

## Support

Issues? Check:
- `README.md` - Common questions
- `AURA_IEEE_PAPER.md` - How it works
- GitHub Issues - Community help

---

**Version**: 1.0.0  
**Last Updated**: November 2025  
**Status**: Ready for Production
