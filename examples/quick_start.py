"""
Quick start example - 2 minute setup
"""

import pandas as pd
import tempfile
import os

csv_path = "examples/Dataset.csv"


if __name__ == "__main__":
    import sys
    # Ensure local directory is first in sys.path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    import aura
    
    from aura import Aura

    # Use AURA
    print("🚀 AURA Quick Start\n")

    aura = Aura()
    aura.load_data(csv_path)

    insights = aura.generate_insights()
    print(f"\n📊 Insights: {insights}\n")

    print("\n🎨 Launching AURA Interactive Dashboard...\n")
    aura.launch_gui()
