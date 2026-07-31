import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const APP_PATH = new URL("../../docs/ifrs17-demo/assets/app.js", import.meta.url);
const INDEX_PATH = new URL("../../docs/ifrs17-demo/index.html", import.meta.url);

test("active frontend does not import the legacy canned retrieval module", async () => {
  const source = await readFile(APP_PATH, "utf8");

  assert.doesNotMatch(source, /retrieval-core\.js/);
  assert.doesNotMatch(source, /\bifrs17\b|three_body_trilogy|三體/i);
});

test("profile and model selects are populated by runtime config", async () => {
  const source = await readFile(INDEX_PATH, "utf8");
  const knowledgeBaseSelect = source.match(/<select id="knowledgeBaseSelect"[^>]*>([\s\S]*?)<\/select>/)?.[1];
  const modelSelect = source.match(/<select id="qaModelSelect"[^>]*>([\s\S]*?)<\/select>/)?.[1];

  assert.equal(String(knowledgeBaseSelect || "").trim(), "");
  assert.equal(String(modelSelect || "").trim(), "");
});
