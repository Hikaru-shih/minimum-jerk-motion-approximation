# 第三階段完整報告

最新可編輯檔案：[final_motion_decomposition_report.docx](final_motion_decomposition_report.docx)。英文完整草稿，共 25 頁，含 21 個原生 Word 方程式、9 張不同圖（含兩張流程圖）、22 張表，以及演算法、實驗設定、數量與時間鄰域敏感度、失敗分析、重現方式及 references。

正文以目前無抵銷懲罰的谷值主線為準；先前自由時間、抵銷懲罰與彎折選數放在歷史探索及診斷章節，未混用主成績。詳細原始結果見[主線結果](../docs/mainline_results.md)與[時間鄰域敏感度](../docs/timing_bound_sensitivity.md)。

## 編輯與驗證

文字、表格及方程式可在 Word 編輯。圖表以圖片嵌入，繪圖來源與流程圖建置保留在本目錄。報告已由本機 Word 正常開啟並產生 25 頁預覽，逐頁檢查。另核對主比較、敏感度、完整數量表與來源雜湊；九張圖片未重複嵌入。核對紀錄在 [document_audit.json](final_report_assets/document_audit.json)。

預覽 PDF 與逐頁 PNG 僅在專案 tmp/ 下作排版檢查，沒有當作正式交付。使用者編輯後可自行另存最終 PDF。

## 重建

在研究目錄使用 WSL 數值環境，從已儲存數據繪圖，不重新擬合：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 report/plot_final_report.py
```

接著在有 python-docx 與 Pillow 的文件環境執行 `python report/build_final_report.py`。重建會覆寫這份產生的 DOCX；若已手動修改，請先保留修改版本。原生公式為 OMML，圖表資產與來源清單在 final_report_assets/。

## 舊版報告

[fitting_and_goal_selection_report.docx](fitting_and_goal_selection_report.docx) 與[舊 PDF](fitting_and_goal_selection_report.pdf)為較早七頁版本，保留作歷史紀錄，不是目前完整主線報告。原 build_report.py 只重建該舊版本。
