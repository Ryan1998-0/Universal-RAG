# Qwen / Claude RAG 回答品質評分

## 目的

同一份最終 RAG system prompt、user prompt 與檢索證據同時提供給 Qwen 與 Claude。Claude 的回答分數固定正規化為 100，Qwen 依固定 rubric 評分。這是相對品質基準，不代表 Claude 回答已被人工證明為絕對正確；檢索證據的優先級仍高於 Claude 的寫法。

## 100 分評分標準

| 評分項目 | 配分 | 判斷重點 |
|---|---:|---|
| 事實正確與證據一致 | 35 | 每個實質主張均有證據支持，沒有杜撰步驟、欄位、數值或結果。 |
| 關鍵資訊完整性 | 25 | 涵蓋問題所需的關鍵步驟、條件與例外，不遺漏重要證據。 |
| 引用與可追溯性 | 15 | 引用編號有效，且來源能直接支持相鄰主張。 |
| 回答規則遵循 | 10 | 遵守最終 prompt 的限制、格式與證據不足處理。 |
| 清楚與可操作性 | 10 | 順序明確、具體，使用者可依答案完成操作。 |
| 繁中與術語品質 | 5 | 使用自然繁體中文，正確保留介面名稱、代碼與術語。 |

## 重大錯誤上限

- 與檢索證據明確矛盾：最高 49 分。
- 加入會影響操作或結論的無證據重大主張：最高 59 分。
- 證據不足卻未依規則拒答：最高 40 分。

多個上限同時觸發時採最低者。文字風格與 Claude 不同不構成扣分理由。

## 執行方式

```bash
python scripts/evaluate_qwen_with_claude.py \
  --conversation-id <conversation-id> \
  --folder "Gmail 精簡版"
```

評測成品會寫入 `evals/qwen_claude/runs/`，完整保留實際送給 Claude 的最終 prompt、Claude 參考答案、Qwen 原始答案、各項分數、扣分理由與 prompt SHA-256。

## 網頁即時評測

啟動服務前設定：

```bash
RAG_CLAUDE_REFERENCE_EVALUATION=1 \
RAG_CLAUDE_REFERENCE_MODEL=sonnet \
python -m rag_demo.web_app
```

啟用後，只有具備檢索證據且由 Ollama/Qwen 生成的回答會送往 Claude 評測。Claude CLI 不可用或呼叫失敗時，Qwen 回答仍會正常顯示，評分區會標示無法使用。

## 資料邊界

啟用 Claude 評測會把該次最終 prompt 與檢索片段送往 Anthropic 服務。公司機密文件應先確認資料政策；不需要雲端評測時，保持 `RAG_CLAUDE_REFERENCE_EVALUATION=0`。
