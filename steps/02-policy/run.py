"""Load an actual checkpoint, evaluate a circuit and export its decision trace."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.experiments.evaluate import main

if __name__ == "__main__":
    main()
