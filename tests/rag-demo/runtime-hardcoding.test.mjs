import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

test("public page explains the Chainlit local entry point", async () => {
  const source = await readFile(new URL("../../docs/rag-demo/index.html", import.meta.url), "utf8");
  assert.match(source, /Chainlit/);
  assert.match(source, /泛用 RAG 工作台/);
  assert.match(source, /architecture.html/);
});
