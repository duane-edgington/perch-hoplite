"""Put the repository root on sys.path so `from src import train` resolves
when pytest is run from anywhere in the repo."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
