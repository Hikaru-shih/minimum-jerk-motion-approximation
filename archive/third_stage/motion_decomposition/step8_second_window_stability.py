import json
from pathlib import Path

from .step7_initialization_stability import analyze_saved

ROOT = Path(__file__).resolve().parents[1]


def main():
    result = analyze_saved(ROOT/'results/step8_second_window',
                           ROOT/'results/step8_second_window_stability',
                           prefix_template='w1e-06_run{index}',window_label='30-33 s')
    print('All 45 pairs, maximum differences:',json.dumps(result['pairwise_maxima'],indent=2))


if __name__=='__main__':
    main()
