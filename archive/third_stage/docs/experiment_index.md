# 實驗與診斷索引

目前主線與最新結果：[教授要求主線與完成結果](mainline_results.md)。已完成速率谷值數量提案、局部時間擬合、線性最小平方位置擬合，以及同 K 固定間隔與數量比較。未使用舊抵銷懲罰或彎折選數。第一段速度誤差未改善、數量對參數敏感，均已明列；不宣稱得到唯一正確的動作單元數。

以下保留先前階段紀錄。既有 Word／PDF 未包含這次新主線，最新報告請讀上述連結。

主要閱讀文件是[研究總結](research_summary.md)。以下依用途分組，保留舊資料，不是待辦清單，也不表示每個原型都是最終方法。

| 用途 | 文件 |
|---|---|
| 基本公式與固定端點 | [已知生成例子](fitting_step1.md)、[固定端點](fitting_step2_fixed_endpoints.md)、[時間擬合](fitting_step4_timing.md) |
| 初猜與真實片段的早期檢查 | [速度低點](fitting_step3.md)、[初猜](fitting_step5_initialization.md)、[第一段擬合](fitting_step5_real_fit.md)、[分量解讀](fitting_step5_components.md) |
| 反向貢獻控制 | [小例子](fitting_step6_opposition.md)、[共同求解](fitting_step6_joint.md)、[真實資料控制](fitting_step6_real.md)、[分量診斷](fitting_step6_components.md) |
| 權重、初猜與另一片段 | [權重敏感度](fitting_step7_weights.md)、[第一段四分量初猜](fitting_step7_stability.md)、[第二段控制](fitting_step8_second_window.md)、[第二段初猜](fitting_step8_stability.md) |
| 數量比較的主要證據 | [K=1–6](fitting_step9_counts.md)、[A/B 原型](fitting_step10_count_selection.md)、[K=1–12 延伸](fitting_step14_extended_counts.md) |
| 為何沒有直接採用其他規則 | [B 的限制](fitting_step11_valley_diagnostic.md)、[既有轉折曲線](fitting_step12_count_evidence.md)、[BIC 形式評分假設](fitting_step13_information_score.md) |
| 本輪候選檢查與總結 | [研究總結](research_summary.md)、results/candidate_review/ |
| 交付收尾 | [同條件均勻基準比較](baseline_comparison.md)、[英文 Word](../report/fitting_and_goal_selection_report.docx)、[英文 PDF](../report/fitting_and_goal_selection_report.pdf) |
| 文獻 | [分量數文獻筆記](literature_component_count.md)、[References](references.md) |

各文件都保留對應程式與輸出連結。程式在 motion_decomposition/，結果在 results/；資料提取仍依賴專案根目錄的 results/wrist_trajectory.npz，沒有移動前兩次實驗的程式、資料或 PDF。
