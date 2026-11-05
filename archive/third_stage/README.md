# Motion decomposition research

Archived third-stage layout before promotion to the repository root. The [current project](../../README.md) contains the latest runnable code and report. Commands below describe the historical nested layout.

目前完整報告：[可編輯 Word 草稿](report/final_motion_decomposition_report.docx)。目前主線：[教授要求主線與完成結果](docs/mainline_results.md)。

使用速率谷值提出分量數量與時間區間，在區間附近擬合開始／結束時間，每次以固定位置端點的線性最小平方法求位移。允許重疊，不使用抵銷懲罰或舊彎折選數。

已完成同一 clip 的兩段分析、不同數量與同 K 均勻基準比較。第一段位置改善但速度較差；數量規則仍對偵測參數敏感。這是已完成的實驗回應，不是正確人體單元數的證明。

## 重現

```bash
cd research/motion_decomposition
python3 -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m motion_decomposition.final_pipeline
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m motion_decomposition.complete_mainline
python3 -m motion_decomposition.export_mainline_goals
python3 -m unittest discover -s tests -q
```

需先有根目錄 `results/wrist_trajectory.npz`，其產生方法保留在根目錄 README。主線結果在 `results/mainline/`；無限制擬合控制在 `results/final_pipeline/`。累積 goals 不一定是實際通過點。

## 檔案

- [主線結果與要求對照](docs/mainline_results.md)
- [時間鄰域係數說明與敏感度](docs/timing_bound_sensitivity.md)、[資料一致性核對](results/timing_bound_sensitivity/audit.json)
- [主線程式](motion_decomposition/complete_mainline.py)、[時間擬合](motion_decomposition/valley_fit.py)
- [完整資料](results/mainline/summary.json)、[數量比較](results/mainline/count_sweep.csv)、[目標位置與時間序列](results/mainline/goals.csv)
- [先前實驗索引](docs/experiment_index.md)、[結構驗證](docs/movement_unit_validation.md)、[References](docs/references.md)

敏感度與一致性收尾已完成：54 項測試通過，30 筆主比較、56 次主線局部時間擬合、6 個敏感度條件／24 次求解與 7 個 goals 均已核對。較寬範圍會降低本例誤差，但不據此換掉原主設定或宣稱係數最佳。

新版完整 Word 草稿已產生，包含本次主線、敏感度與歷史診斷。先前七頁 Word／PDF 及前兩階段報告保留作歷史紀錄。未推送或寄信。
