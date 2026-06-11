"""Run the full A/B and print the result table.

    python run_ab.py            # full 5-seed run
    python run_ab.py --fast     # quick smoke run
"""
import sys
from reservoir_catcher import evaluate, format_table

if __name__ == "__main__":
    fast = "--fast" in sys.argv
    res = evaluate(n=150, seeds=2, T=2400, split=1500, wash=150) if fast else evaluate()
    print(format_table(res))
