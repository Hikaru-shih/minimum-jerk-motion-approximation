# 分量數與分解辨識：文獻搜尋筆記

查閱日期：2026-11-05。搜尋 minimum-jerk / submovement decomposition、number selection、identifiability。以下依論文原文、作者預印本與 PubMed 原摘要整理，不表示已重現演算法，也不是完整系統性文獻回顧。本文記錄搜尋階段；後續的原型實驗見 [A/B 比較](fitting_step10_count_selection.md)，未選定最終 K。

## 直接相關的方法

| 論文 | 選數量或辨識的方式 | 對目前專案的用途與限制 |
|---|---|---|
| Liao & Kirsch (2014), *Characterizing and Predicting Submovements during Human Three-Dimensional Arm Reaches* | Methods C、Eq.5、Fig.2：比較不同數量的代價曲線，正規化後取距對角線最遠點 | 最接近目前 K 掃描，可作數量選擇基準；原文擬合速度且代價不同，轉折不等於生理真值 |
| Chen et al. (2012), *Submovement Composition of Head Movement* | 從一分量開始增加，直到速度 R²>0.985，再檢查不合理組合 | 直接顯示誤差門檻配合結構限制的做法；頭部運動、目視初猜與研究特定的重疊排除規則不能直接移植 |
| Horowitz, Majeed & Patton (2016), *A Fresh Perspective on Dissecting Action Into Discrete Submotions* | 利用速度峰方向及方向變化辨識分量，再從殘差逐次提取；以剩餘速度與訊噪條件停止 | 直接回應「不要只追逐擬合誤差」；有外力擾動及意圖估計，與本專案只有腕部位置不同，依賴峰方向可代表單一分量的假設 |
| Prados et al. (2026), *Identifiable Decomposition of Submovements in Human Hand Trajectories*，arXiv:2609.40012v1 | 速度峰初猜、殘差增量、adaptive ridge，並用時空核相關性討論可辨識性；增量仍有相對速度誤差門檻 | 對目前近相同時間、大分量抵銷特別相關；不是無門檻的數量答案，也不是本專案已實作的方法。本次查閱為 2026-09-30 預印本，未另外確認正式期刊發表 |

原始來源：

- [Liao & Kirsch 2014 全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC4110007/)
- [Chen et al. 2012 出版社全文](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0047565)
- [Horowitz et al. 2016 全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC8734945/)；PMC 的一般開啟有時出現驗證頁，本次搜尋擷取取得 Methods 與限制，書目另以 [PubMed](https://pubmed.ncbi.nlm.nih.gov/28269545/) 核對。
- [Prados et al. 2026 作者全文](https://arxiv.org/html/2609.40012v1)、[版本與提交日期](https://arxiv.org/abs/2609.40012)

Chen 原文的 R²>0.985、反向/高度重疊排除規則，是該研究設定，不是建議直接套用於會話手勢。Horowitz 的殘差速度停止條件也有其量測訊噪依據。Prados 的殘差停止以速度 RMS 相對最大速度約 2% 為條件，同樣需要門檻；不能因為論文使用就把它當成我們資料的已驗證標準。

## 其他有用來源，和數量選擇分開看

- Rohrer & Hogan (2003), *Avoiding spurious submovement decompositions: a globally optimal algorithm*. DOI: 10.1007/s00422-003-0428-4。[PubMed 原摘要](https://pubmed.ncbi.nlm.nih.gov/14504938/)。討論看起來合理卻錯誤的分解與 branch-and-bound。本次以摘要核對，不聲稱已讀取完整求解細節，也不把「全域最小化」等同於已驗證人體真值。
- Rohrer & Hogan (2006), *Avoiding spurious submovement decompositions II: a scattershot algorithm*. DOI: 10.1007/s00422-006-0055-y。[PubMed 原摘要](https://pubmed.ncbi.nlm.nih.gov/16570179/)。多初猜分解的經典來源；本次未由原文完整核對所有停止設定，不把後續論文的改版設定直接當原文。
- Gowda et al. (2015), *Accelerating Submovement Decomposition With Search-Space Reduction Heuristics*. DOI: 10.1109/TBME.2015.2434595。[PubMed 原摘要](https://pubmed.ncbi.nlm.nih.gov/26011861/)、[作者機構研究頁](https://ohsu.elsevierpure.com/en/publications/accelerating-submovement-decomposition-with-search-space-reductio/)。幅度與形狀參數分開求解、集中搜尋高誤差區，和現有位移/時間分離思路相關；不能只憑摘要說它提供無任意門檻的數量選擇。
- Rudakov et al. (2025), *SSSUMO: Real-Time Semi-Supervised Submovement Decomposition*. [arXiv:2507.08028](https://arxiv.org/abs/2507.08028)、[作者全文](https://arxiv.org/html/2507.08028v1)。由 minimum jerk 人造資料及未標註真實資料學習分解，是後續學習路線參考；目前不建議因這次搜尋就改做神經網路。

## 對下一步的判斷

有直接文獻依據可研究自動數量，不必從任意公分門檻開始。但文獻也顯示，誤差轉折、門檻、方向辨識及正則化各有假設，沒有哪篇可直接證明我們目前應選幾個分量。

建議先細讀 Liao 的數量選取式，作為有限、可比較的基準；同時讀 Horowitz 的反例與假設，以及 Prados 的可辨識性診斷。先列出這些方法和本專案的位置目標、固定端點、懲罰與局部求解差異，再決定要實作哪一種。尤其不能直接把目前含懲罰的總代價跨 K 送進轉折公式，卻忽略配對數、參數數量、K 上限和未收斂結果。

以上為方法適用性的判斷，不是已完成的數量選擇實驗。既有 K=1–6 結果保持原樣。
