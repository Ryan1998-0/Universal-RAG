import test from "node:test";
import assert from "node:assert/strict";

import {
  architectureViews,
  calculateAnchoredScroll,
  calculateProgress,
  clampZoom,
} from "../../docs/rag-demo/assets/architecture-workbench.js";

test("architecture starts with seven coarse domains", () => {
  assert.equal(architectureViews.root.nodes.length, 7);
  for (const domain of architectureViews.root.nodes) {
    assert.ok(domain.drilldown, `${domain.id} needs a drilldown target`);
    assert.ok(architectureViews[domain.drilldown], `${domain.id} target must exist`);
    assert.ok(architectureViews[domain.drilldown].nodes.length >= 7);
  }
});

test("every domain opens an independent architecture diagram", () => {
  const targets = architectureViews.root.nodes.map((domain) => domain.drilldown);
  assert.equal(new Set(targets).size, 7);
  for (const target of targets) {
    const view = architectureViews[target];
    assert.ok(view.edges.length > 0, `${target} needs connected flow nodes`);
    assert.ok(view.width > 0);
    assert.ok(view.height > 0);
  }
});

test("every architecture node has checkable task goals and traceable I/O", () => {
  const nodes = Object.values(architectureViews).flatMap((view) => view.nodes);
  assert.ok(nodes.length >= 50);
  assert.equal(new Set(nodes.map((node) => node.id)).size, nodes.length);
  for (const node of nodes) {
    assert.ok(node.title);
    assert.ok(node.input);
    assert.ok(node.output);
    assert.ok(Array.isArray(node.goals));
    assert.ok(node.goals.length >= 2, `${node.id} needs at least two goals`);
  }
});

test("task progress counts checked goals across all hierarchy levels", () => {
  const checked = {
    "domain-query:0": true,
    "ret-rrf:1": true,
    "ev-gate:1": true,
  };
  const progress = calculateProgress(architectureViews, checked);
  assert.equal(progress.done, 3);
  assert.ok(progress.total > progress.done);
  assert.equal(progress.percent, Math.round((3 / progress.total) * 100));
});

test("pointer-centered zoom preserves the content under the cursor", () => {
  const old = {
    scrollLeft: 240,
    scrollTop: 120,
    pointerX: 300,
    pointerY: 220,
    oldZoom: 60,
    newZoom: 95,
  };
  const beforeX = (old.scrollLeft + old.pointerX) / (old.oldZoom / 100);
  const beforeY = (old.scrollTop + old.pointerY) / (old.oldZoom / 100);
  const next = calculateAnchoredScroll(old);
  const afterX = (next.left + old.pointerX) / (old.newZoom / 100);
  const afterY = (next.top + old.pointerY) / (old.newZoom / 100);
  assert.ok(Math.abs(beforeX - afterX) < 0.0001);
  assert.ok(Math.abs(beforeY - afterY) < 0.0001);
});

test("canvas zoom remains in an accessible operating range", () => {
  assert.equal(clampZoom(5), 30);
  assert.equal(clampZoom(95), 95);
  assert.equal(clampZoom(300), 160);
});
