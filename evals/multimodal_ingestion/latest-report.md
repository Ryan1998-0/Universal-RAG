# Multimodal ingestion evaluation

- Result: 10/10 passed
- Generated: 2026-07-31T08:34:44.687427+00:00

## Uploads

| File | Type | Method | Chunks | Default selected |
| --- | --- | --- | ---: | --- |
| benefits.txt | text | text-decode | 1 | False |
| operations.md | markdown | markdown-headings | 2 | False |
| customer.json | json | json-path-flatten | 1 | False |
| claims-guide.docx | word | docx-xml | 1 | False |
| invoice.png | image | macos-vision | 1 | False |
| financial-policy.pdf | pdf | pdf-text | 1 | False |
| scanned-invoice.pdf | pdf | pdf-ocr | 1 | False |

## Cases

- PASS: `selected:benefits.txt`
- PASS: `selected:operations.md`
- PASS: `selected:customer.json`
- PASS: `selected:claims-guide.docx`
- PASS: `selected:invoice.png`
- PASS: `selected:financial-policy.pdf`
- PASS: `selected:scanned-invoice.pdf`
- PASS: `explicit-empty-selection`
- PASS: `cross-document-isolation`
- PASS: `persisted-document-list`
