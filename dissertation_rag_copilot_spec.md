# 論文寫作 RAG Copilot — 專案規劃規格書

## 0. 專案一句話定義

針對 James 的 dissertation 文獻庫（nature-based interventions / fMRI / ISC-ISFC / predictive processing 相關 PDF），
建立一個**嚴格 citation-grounded** 的檢索增強生成助理，並用**可量化的評估流程**證明它的幻覺率低到可以信任。

核心價值主張（也是履歷/面試素材）：
- Prompt & context engineering（chunking 策略、citation-forcing prompt 設計）
- RAG 系統設計（retrieval pipeline、hybrid search、rerank）
- 評估方法設計（groundedness/hallucination benchmark，非只是「感覺準不準」）

---

## 1. 專案目標與範圍

### 1.1 要解決的具體痛點
寫 dissertation 時常見的真實問題（用來定義成功標準）：
- 「我記得某篇提過 predictive processing 跟 precision weighting 的關係，但是哪一段、哪一篇？」
- 「我這句話的說法有文獻支持嗎，還是我自己腦補的？」
- 「幫我找出所有討論 ISC/ISFC 方法限制的段落，我要寫 limitation。」
- 「這段是我 paraphrase 某篇文獻，但語氣/主張是否超出原文的證據強度？」

### 1.2 Scope（V1 邊界，先小後大）
**V1 包含：**
- 針對你自己的 PDF 文獻庫（先抓 30–50 篇你最常引用的，不用一次全庫）建 RAG
- 支援問答 + 每個論點附段落級別 citation（含頁碼/段落 ID）
- 支援「幫我檢查這段話有沒有幻覺」的反向查核模式（你貼一段你寫的文字，系統回報哪些主張有支持、哪些沒有）
- 有一套可重複執行的評估腳本，輸出 groundedness / hallucination rate 報表

**V1 不包含（先不做，避免專案膨脹）：**
- 全自動幫你寫段落草稿（先做「查證」比「生成」更保守也更有價值）
- 跨語言檢索（先假設文獻多為英文）
- Web UI（先用 CLI + Jupyter/簡單 script 即可，晚點再包 Streamlit）

### 1.3 成功標準（要能量化，這是評估流程的核心）
- **Retrieval 準確率**：對於你手工標註的 30–50 題「已知答案在哪一篇哪一段」的測試集，top-k retrieval hit rate ≥ 90%
- **Citation grounding 準確率**：模型產生的每個「論點+引用」pair，人工或自動判斷「引用段落是否真的支持該論點」，目標 ≥ 95% 為 supported
- **幻覈偵測**：對於刻意注入的「無中生有」測試題（你的庫裡根本沒有支持的說法），系統應該要能明確回答「查無依據」而不是硬掰，目標 refusal/flag rate ≥ 90%

這三個指標就是你可以在履歷/面試直接講的「評估方法設計」成果。

---

## 2. 系統架構

```
┌─────────────────────────────────────────────────────────────┐
│  1. Ingestion Layer                                          │
│     PDF → 文字/結構抽取 → chunking → metadata 標註            │
├─────────────────────────────────────────────────────────────┤
│  2. Indexing Layer                                            │
│     Embedding → Vector DB（+ 可選 BM25 sparse index）          │
├─────────────────────────────────────────────────────────────┤
│  3. Retrieval Layer                                           │
│     Query → Hybrid search（dense + sparse）→ Rerank → top-k   │
├─────────────────────────────────────────────────────────────┤
│  4. Generation Layer（Citation-Grounded Prompting）            │
│     強制格式：每個論點必須附 [chunk_id] 引用                    │
│     若無支持證據 → 必須明確說「查無依據」，禁止腦補                │
├─────────────────────────────────────────────────────────────┤
│  5. Verification Layer（這是本專案的靈魂）                       │
│     對每個生成的 (claim, citation) pair 做二次查核：             │
│       - 抓出引用的原文段落                                      │
│       - 用另一個 LLM call（或 NLI 模型）判斷 entailment          │
│       - 標記 supported / partially supported / unsupported     │
├─────────────────────────────────────────────────────────────┤
│  6. Evaluation Harness                                        │
│     Golden set 測試題 → 跑 pipeline → 產出量化報表               │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. 各層細節設計

### 3.1 Ingestion Layer（PDF → Chunk）

**挑戰**：學術 PDF 常見雙欄排版、圖表穿插、references 混在正文附近，naive text extraction 容易把句子切斷或欄位錯位。

- PDF 解析工具選項：`PyMuPDF (fitz)`、`pdfplumber`、或更強的 `Grobid`（專門處理學術論文結構，能抽出 section、references、citation span）
  - 建議：先用 PyMuPDF 快速跑通 MVP，若發現雙欄排版錯亂嚴重，再導入 Grobid 做結構化抽取
- **Metadata 一定要保留**：paper title、authors、year、section（Introduction/Methods/Discussion...）、page number、chunk 在原文的字元 offset
  - 這些 metadata 是 citation grounding 能「可追溯到原文段落」的關鍵，不能只存純文字
- **Chunking 策略**（這是 prompt/context engineering 的重點之一，值得做 ablation）：
  - Baseline：fixed-size chunking（e.g. 500 tokens, 50 token overlap）
  - 進階：semantic/structure-aware chunking（依段落、依 section 邊界切，避免把一個論證從中間切斷）
  - 建議做法：先實作 fixed-size 版本作為 baseline，再實作 section-aware 版本，用第 5 節的 evaluation harness 比較兩者對 retrieval hit rate 和 citation grounding 準確率的影響 —— 這個 ablation 本身就是很好的「方法設計」成果展示

### 3.2 Indexing Layer

- Embedding model 選項：
  - 本地/開源：`bge-large-en`, `e5-large-v2`（學術文獻英文為主，這兩個效果不錯且免費）
  - API：OpenAI `text-embedding-3-large` 或用 Claude API 搭配其他 embedding provider
- Vector DB 選項（依你想要的複雜度）：
  - 輕量：`Chroma`（本地檔案型，最快上手，適合 30–50 篇文獻規模）
  - 進階：`Qdrant` 或 `Weaviate`（支援 hybrid search、metadata filtering 更完整）
  - 建議：V1 用 Chroma 就夠，庫規模不大，重點在方法而非工程規模

### 3.3 Retrieval Layer

- **Hybrid search**：dense（語意）+ sparse/BM25（關鍵字，對專有名詞如 "DiFuMo", "ISFC", "TFCE" 這種術語特別重要，純語意 embedding 有時抓不準專有名詞）
- **Rerank**：拿到 top-20 candidates 後用 cross-encoder（如 `bge-reranker`）重排到 top-5，能顯著提升 precision
- **Query 改寫**：dissertation 問題常常口語化或跨段落，可以加一層「query expansion」把使用者問題轉成更適合檢索的形式（這也是 prompt engineering 的一環）

### 3.4 Generation Layer — Citation-Grounded Prompting（本專案的核心技術重點）

這是你要花最多心思設計的部分。關鍵原則：

**強制輸出格式**（不是「建議」引用，而是結構化強制）：
```
每個論點都要用這個格式回答：
<claim>
  <statement>你的論點</statement>
  <citation chunk_id="xxx" paper="xxx" page="xxx">支持這個論點的原文段落（直接摘錄或緊密改寫）</citation>
</claim>

如果沒有檢索到的段落支持某個論點，不要生成該論點，
改為回答：「我在目前的文獻庫中查無直接支持此說法的段落。」
```

- 用 XML tag 或 JSON schema 強制結構化輸出，方便後續 Verification Layer 用程式解析、逐一查核
- Prompt 裡要明確告訴模型「寧可少講、不可腦補」，並給 few-shot 範例展示「查無依據時該怎麼誠實回答」
- 建議也做一版「無 grounding 限制」的 baseline prompt，之後在評估階段比較兩者幻覺率差異 —— 這個對比本身就是很有說服力的作品集素材

### 3.5 Verification Layer（讓「嚴格」兩個字有依據，不是嘴巴說說）

這一層獨立於生成，是專門「回頭查核」剛剛生成的每個 (claim, citation) pair：

1. 抓出 citation 對應的原文 chunk 全文
2. 用第二次 LLM call（獨立 prompt，不看第一次的推理過程，避免自我確認偏誤）問：
   「這段原文是否真的支持這個論點？回答 supported / partially_supported / unsupported，並說明理由」
3. 也可以額外導入 NLI（Natural Language Inference）模型（如 `roberta-large-mnli`）做輕量、免 API 成本的第一層過濾，LLM 判斷作為第二層更精細的複核
4. 產出一份「hallucination report」：列出所有 unsupported/partially_supported 的 claim，供你人工複審

### 3.6 Evaluation Harness（這是讓專案從「玩具」變成「可信工具」的關鍵）

**Golden test set 建立方式**：
- 從你已經寫好的 dissertation 段落中，挑 30–50 句你自己確定「有明確文獻支持」的句子，記錄該句子對應的來源論文與段落 → 這是 retrieval 的 ground truth
- 額外設計 10–15 題「陷阱題」：問一些你的文獻庫裡**沒有**明確答案的問題（例如刻意問一個庫裡沒收錄的研究方法細節），測試系統是否會老實說「查無依據」而非硬掰

**評估指標與跑法**：
| 指標 | 定義 | 跑法 |
|---|---|---|
| Retrieval Hit Rate@k | golden set 問題中，正確 chunk 出現在 top-k 檢索結果的比例 | 自動化腳本比對 chunk_id |
| Citation Precision | 生成的 claim 中，citation 真的支持該 claim 的比例 | Verification Layer 產出 + 抽樣人工複審校準 |
| Hallucination Rate | unsupported claim 數 / 總 claim 數 | 同上 |
| Refusal Correctness | 陷阱題中，系統正確回答「查無依據」的比例 | 自動比對是否觸發 refusal 模板 |
| Chunking Ablation | fixed-size vs. section-aware chunking 對上述指標的影響 | 同一 golden set 分別跑兩套 index 比較 |

**產出物**：一份自動產生的 evaluation report（markdown 或簡單 dashboard），這份報告本身就是你可以放進作品集/面試展示的東西。

---

## 4. 技術棧建議（先求跑得動，再求精緻）

| 模組 | 建議選項 | 備註 |
|---|---|---|
| 語言 | Python | 你已熟悉，且生態成熟 |
| PDF 解析 | PyMuPDF（先）/ Grobid（進階） | |
| Embedding | bge-large-en 或 e5-large-v2（本地）| 免 API 費用，適合迭代實驗 |
| Vector DB | ChromaDB | 輕量、本地、易上手 |
| Sparse index | rank_bm25 或 Elasticsearch（輕量可跳過用前者）| |
| Rerank | bge-reranker-base | |
| LLM（生成+查核）| Claude API（Sonnet） | 你本身在用 Claude 生態，介面熟悉，也符合「Claude in Claude」示範價值 |
| NLI（可選）| roberta-large-mnli via HuggingFace | 免費輕量的第一層幻覺過濾 |
| Evaluation | 自寫 Python script + pandas 產表 | 不需要額外框架，重點是指標設計本身 |

---

## 5. 建議分階段時程（每階段都要有可展示成果）

### Phase 0（0.5 天）：資料準備
- 挑 10–15 篇你最核心的文獻（先小規模），整理成 PDF 資料夾
- 手工建立 10 題 golden set（先求有，之後再擴充到 30–50 題）

### Phase 1（1–2 天）：Ingestion + Indexing MVP
- PDF → chunk → embed → 存入 ChromaDB
- 先用最簡單的 fixed-size chunking 跑通全流程

### Phase 2（1 天）：Retrieval + 基礎 Generation
- Hybrid search + citation-forcing prompt
- 先手動跑 5–10 個問題，肉眼檢查輸出格式與引用是否合理

### Phase 3（1–2 天）：Verification Layer
- 建立二次查核機制（LLM-as-judge 或 NLI）
- 產出第一版 hallucination report

### Phase 4（1 天）：Evaluation Harness 正式化
- Golden set 擴充到 30–50 題（含陷阱題）
- 自動化跑分腳本，產出量化報表
- 做 chunking ablation（fixed vs. section-aware）比較實驗

### Phase 5（彈性，視時間）：Chunking/Prompt 優化迭代
- 根據 Phase 4 報表發現的弱點，針對性改善（例如某類問題 retrieval hit rate 特別低，就調整 chunk 大小或加 query rewriting）
- 每次迭代都重跑 evaluation harness，形成「迭代-評估」閉環，這個閉環本身就是最值得寫進履歷的部分

---

## 6. 與職缺技能點的對應（給你自己備忘，面試時可以直接用這張表講故事）

| 職缺常見要求 | 這個專案對應的具體成果 |
|---|---|
| Prompt / context engineering | Citation-forcing 結構化 prompt 設計、chunking 策略 ablation |
| RAG 系統設計 | Hybrid search + rerank + verification layer 的完整 pipeline |
| 評估方法設計 | Golden set 建構方法論、多指標評估框架（retrieval/grounding/hallucination/refusal）|
| 端到端專案執行力 | 從資料準備到迭代優化的完整閉環，且是解決自己真實痛點的專案 |

---

## 7. 開始跟 agent 討論時，建議先確認的幾個決策點

1. 文獻庫規模：先用多少篇 PDF 起跑？（建議 10–15 篇先跑通）
2. Embedding/LLM 要走本地免費模型還是直接用 API（成本 vs. 方便）？
3. Golden set 你要自己標，還是要我先幫你設計標註規範/範本？
4. 要不要順便把「幻覺率評估」這套方法論寫成一篇可以放進履歷/作品集的技術短文？

---

## 附錄：Citation-Grounded Prompt 範例草稿

```
System prompt（草稿，可再調整）：

你是一個嚴格的學術文獻查證助理。你只能根據下方提供的檢索段落回答問題。

規則：
1. 每一個論點都必須緊接一個 <citation> 標籤，標明來源 chunk_id。
2. 不可以使用任何檢索段落以外的知識或推論來補足論點。
3. 如果檢索到的段落無法直接支持某個你想講的論點，禁止生成該論點；
   改為誠實回答：「查無直接支持此說法的段落。」
4. 若使用者要求的資訊部分有支持、部分沒有，必須明確拆開標示哪部分有依據、哪部分沒有。

輸出格式：
<answer>
  <claim citation_ids="[chunk_id, ...]">論點內容</claim>
  ...
  <unsupported_note>（若有查無依據的部分，在此列出）</unsupported_note>
</answer>
```
