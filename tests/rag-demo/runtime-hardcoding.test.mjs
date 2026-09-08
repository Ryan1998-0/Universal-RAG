import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const APP_PATH = new URL("../../docs/rag-demo/assets/app.js", import.meta.url);
const INDEX_PATH = new URL("../../docs/rag-demo/index.html", import.meta.url);

test("active frontend does not import the legacy canned retrieval module", async () => {
  const source = await readFile(APP_PATH, "utf8");

  assert.doesNotMatch(source, /retrieval-core\.js/);
  assert.doesNotMatch(source, /\bifrs17\b|three_body_trilogy|三體/i);
});

test("public frontend uses a generic Chinese identity", async () => {
  const source = await readFile(INDEX_PATH, "utf8");

  assert.match(source, /泛用 RAG 工作台/);
  assert.match(source, /href="\.\/architecture\.html"/);
  assert.doesNotMatch(source, /IFRS\s*17|IFRS17-RAG|MaiGPT/i);
  assert.doesNotMatch(source, /<option value="en">/);
});

test("profile and model selects are populated by runtime config", async () => {
  const source = await readFile(INDEX_PATH, "utf8");
  const knowledgeBaseSelect = source.match(/<select id="knowledgeBaseSelect"[^>]*>([\s\S]*?)<\/select>/)?.[1];
  const modelSelect = source.match(/<select id="qaModelSelect"[^>]*>([\s\S]*?)<\/select>/)?.[1];

  assert.equal(String(knowledgeBaseSelect || "").trim(), "");
  assert.equal(String(modelSelect || "").trim(), "");
});
