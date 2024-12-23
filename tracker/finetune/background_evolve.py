import os
import sys

os.system("nohup sh -c '" + sys.executable + f" evolve.py > optuna.txt 2>&1' &")