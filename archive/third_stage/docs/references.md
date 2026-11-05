# References and research notes

這份清單記錄目前討論過的研究來源與使用範圍。舊版 follow-up PDF 保留原樣；新研究報告已依實際使用內容納入 Sato、Liao 與 Flash 的文獻，其餘項目是研究／診斷筆記，不表示全部方法都實作。

## 1. 原專案的表示方式

Sato, H., Mitake, H., & Hasegawa, S. (2025). Artist-Directable Motion Generation Using Overlapping Minimum Jerk Trajectories for Interactive Embodied Social Agent Adapting to Dynamic Environments. *Computer Animation and Virtual Worlds, 36*, e70075. https://doi.org/10.1002/cav.70075

用途：minimum jerk 位移重疊、goal 表示與互動動作生成。與本計畫關係最直接的原論文。

## 2. Minimum jerk 的人體運動學依據

Flash, T., & Hogan, N. (1985). The coordination of arm movements: An experimentally confirmed mathematical model. *The Journal of Neuroscience, 5*(7), 1688–1703. https://doi.org/10.1523/JNEUROSCI.05-07-01688.1985

原始研究頁：https://pubmed.ncbi.nlm.nih.gov/4020415/

用途：到達運動的 minimum jerk 模型及其運動學證據。不能據此把任意擬合成分視為肌肉命令或證明其力學可行性。

## 3. 重疊 submovement 的擬合與限制

Liao, J. Y., & Kirsch, R. F. (2014). Characterizing and Predicting Submovements during Human Three-Dimensional Arm Reaches. *PLOS ONE, 9*(7), e103387. https://doi.org/10.1371/journal.pone.0103387

全文：https://pmc.ncbi.nlm.nih.gov/articles/PMC4110007/

用途：Methods C 將實測到達運動分解為相加的 minimum jerk 成分，擬合成分參數，並在代價函數中處理同時反向成分的抵銷問題。它提供後續方法研究的依據。

使用限制：本計畫沒有完整重現該文的反向成分代價或神經網路；已實作改用本計畫總代價的轉折選數原型。它的實驗條件與本計畫的會話手勢不同；不能直接套用該文門檻或宣稱每條擬合位移已具有人體動作意義。

步驟 6 核對 Methods C：該文以速度擬合並處理同時反向成分。本專案的「速度向量負內積平方懲罰」搭配位置誤差、固定端點與明示尺度，並非重現該文公式 (3)–(4)。未直接採用其代價權重或 0.1 秒限制；後續選數借用正規化曲線最大弦線距離的思路，但換成本計畫代價與候選範圍。推導與驗證見 [固定時間驗證](fitting_step6_opposition.md)、[共同求解驗證](fitting_step6_joint.md)、[真實短段比較](fitting_step6_real.md) 與[研究總結](research_summary.md)。

## 4. 速度低點的解釋限制

Shmuelof, L., Krakauer, J. W., & Mazzoni, P. (2012). How is a motor skill learned? Change and invariance at the levels of task success and trajectory control. *Journal of Neurophysiology, 108*(2), 578–594. https://doi.org/10.1152/jn.00856.2011

全文：https://pmc.ncbi.nlm.nih.gov/articles/PMC3404800/

用途：Submovement Analysis 討論重疊可能使速度峰融合，速度低點未必能揭示所有成分。這是將速度低點視為候選而非唯一正確邊界的參考。

使用限制：目前沒有採用該文的 jerk 偵測規則或門檻。

## 本次固定終點求解的來源定位

固定終點版本使用本專案重建式與指定端點的等式條件，消去最後一條位移後求解線性最小平方問題。這個代數推導已寫在 [第二步說明](fitting_step2_fixed_endpoints.md)，不是宣稱重現上述論文的完整分解演算法。

## 分量數與可辨識性的補充文獻（2026-11-05 查閱）

完整搜尋筆記、原始來源、讀取範圍與適用限制見 [分量數相關文獻](literature_component_count.md)。後續已借鑑 Liao 的正規化曲線思路與 minimum–maximum–minimum 候選概念實作 [兩種原型比較](fitting_step10_count_selection.md)，不是完整重現任一篇的方法，未加入舊 PDF。

- Chen, L. L., Lee, D., Fukushima, K., & Fukushima, J. (2012). *Submovement Composition of Head Movement*. PLOS ONE, 7(11), e47565. https://doi.org/10.1371/journal.pone.0047565
- Horowitz, J., Majeed, Y. A., & Patton, J. (2016). *A Fresh Perspective on Dissecting Action Into Discrete Submotions*. EMBC, 5684–5688. https://doi.org/10.1109/EMBC.2016.7592017
- Rohrer, B., & Hogan, N. (2003). *Avoiding spurious submovement decompositions: a globally optimal algorithm*. Biological Cybernetics, 89, 190–199. https://doi.org/10.1007/s00422-003-0428-4
- Rohrer, B., & Hogan, N. (2006). *Avoiding spurious submovement decompositions II: a scattershot algorithm*. Biological Cybernetics, 94, 409–414. https://doi.org/10.1007/s00422-006-0055-y
- Gowda, S., Overduin, S. A., Chen, M., Chang, Y.-H., Tomlin, C. J., & Carmena, J. M. (2015). *Accelerating Submovement Decomposition With Search-Space Reduction Heuristics*. IEEE Transactions on Biomedical Engineering, 62(10), 2508–2515. https://doi.org/10.1109/TBME.2015.2434595
- Prados, A., Hermus, J., Barber, R., & Calinon, S. (2026). *Identifiable Decomposition of Submovements in Human Hand Trajectories*. arXiv:2609.40012v1，預印本，提交日期 2026-09-30。https://arxiv.org/abs/2609.40012
- Rudakov, E., Shock, J., Lappi, O., & Cowley, B. U. (2025). *SSSUMO: Real-Time Semi-Supervised Submovement Decomposition*. arXiv:2507.08028v1。https://arxiv.org/abs/2507.08028

## 模型數量評分的適用性

- Schwarz, G. (1978). *Estimating the Dimension of a Model*. The Annals of Statistics, 6(2), 461–464. [原文](https://www.andrew.cmu.edu/user/kk3n/simplicity/schwarzbic.pdf)。第 13 步僅試算獨立高斯假設下的 BIC 形式分數，未認定其假設成立。
- Watanabe, S. (2013). *A Widely Applicable Bayesian Information Criterion*. JMLR, 14(27), 867–897. [期刊頁](https://www.jmlr.org/papers/v14/watanabe13a.html)。說明奇異模型下標準 BIC 的限制；未實作 WBIC。

## 數值工具（求解器）

SciPy 1.15.2. `scipy.optimize.least_squares` 官方文件：https://docs.scipy.org/doc/scipy-1.15.2/reference/generated/scipy.optimize.least_squares.html

用途：第四步使用有界非線性最小平方求解時間，採 `trf` 與一般平方損失。官方文件指出這是局部求解；求解器回報成功不代表恢復真實參數。

SciPy 1.15.2 的 [find_peaks](https://docs.scipy.org/doc/scipy-1.15.2/reference/generated/scipy.signal.find_peaks.html) 與 [savgol_filter](https://docs.scipy.org/doc/scipy-1.15.2/reference/generated/scipy.signal.savgol_filter.html) 官方文件：第十步使用 prominence、distance 與 Savitzky–Golay 導數估計，具體閾值為明示的本專案實驗設定，非套件或生理標準。
