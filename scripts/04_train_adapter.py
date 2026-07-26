"""Step 4：训练 adapter。默认 nested=true(->adapter_mrl)；
消融跑：python scripts/04_train_adapter.py --set loss.nested=false --set experiment.name=adapter_plain"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.config import parse_cli
from src.train import train

if __name__ == "__main__":
    train(parse_cli())
