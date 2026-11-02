import assert from "node:assert/strict";
import { readFileSync, statSync } from "node:fs";
import { test } from "node:test";

import {
  calculate,
  normCdf,
  normPpf,
  studentCdf,
  studentPpf,
} from "../../docs/javascripts/sharpe_calc.js";

const scriptUrl = new URL("../../docs/javascripts/sharpe_calc.js", import.meta.url);
const casesUrl = new URL("../../tests/fixtures/js_cases.json", import.meta.url);
const cases = JSON.parse(readFileSync(casesUrl, "utf-8"));

const NUMERIC = {
  normCdf: ({ x }) => normCdf(x),
  normPpf: ({ p }) => normPpf(p),
  studentCdf: ({ df, t }) => studentCdf(df, t),
  studentPpf: ({ df, p }) => studentPpf(df, p),
};

function actual({ fn, inputs }) {
  return fn in NUMERIC ? NUMERIC[fn](inputs) : calculate(inputs)[fn];
}

function worstMiss(selected) {
  let worst = { miss: 0, id: "" };
  for (const item of selected) {
    const miss = Math.abs(actual(item) - item.expected);
    assert.ok(Number.isFinite(miss), `${item.id} is not finite`);
    if (miss > worst.miss) {
      worst = { miss, id: item.id };
    }
  }
  return worst;
}

test("the fixture covers every output", () => {
  const fns = new Set(cases.map((item) => item.fn));
  for (const fn of Object.keys(NUMERIC)) {
    assert.ok(fns.has(fn), fn);
  }
  const keys = Object.keys(calculate(cases.at(-1).inputs)).filter((key) => key !== "problems");
  for (const key of keys) {
    assert.ok(fns.has(key), key);
  }
});

test("normal CDF and PPF agree with Python within 1e-9 on 200 points each", () => {
  for (const fn of ["normCdf", "normPpf"]) {
    const selected = cases.filter((item) => item.fn === fn);
    assert.equal(selected.length, 200);
    const worst = worstMiss(selected);
    assert.ok(worst.miss <= 1e-9, `${worst.id}: ${worst.miss}`);
  }
});

test("every fixed case agrees with Python within 1e-6", () => {
  const worst = worstMiss(cases);
  assert.ok(cases.length >= 1000);
  assert.ok(worst.miss <= 1e-6, `${worst.id}: ${worst.miss}`);
});

test("the script stays within 50 KB", () => {
  assert.ok(statSync(scriptUrl).size <= 50 * 1024);
});

test("the script makes no network calls", () => {
  const source = readFileSync(scriptUrl, "utf-8");
  for (const word of ["fetch(", "XMLHttpRequest", "WebSocket", "sendBeacon", "import("]) {
    assert.ok(!source.includes(word), word);
  }
});

test("the page's fields and outputs match the script", () => {
  const page = readFileSync(new URL("../../docs/calculator.md", import.meta.url), "utf-8");
  const names = [...page.matchAll(/<(?:input|select) name="(\w+)"/g)].map((match) => match[1]);
  assert.equal(names.length, 10);
  const form = ["length", "name", "action", "method", "target", "elements", "item"];
  for (const name of names) {
    assert.ok(!form.includes(name), `${name} hides a form property`);
  }
  const keys = Object.keys(calculate({}));
  for (const [, metric] of page.matchAll(/data-metric="(\w+)"/g)) {
    assert.ok(keys.includes(metric), metric);
  }
});

const VALID = {
  sr_annual: 1,
  years: 5,
  periods_per_year: 252,
  skew: 0,
  kurtosis: 3,
  n_trials: 10,
  var_trials_annual: 0.5,
  n_tests: 10,
  autocorr: 0,
};

test("valid inputs raise no problems", () => {
  assert.deepEqual(calculate(VALID).problems, []);
});

test("a blank field empties the outputs and says why", () => {
  const result = calculate({ ...VALID, skew: Number.NaN });
  assert.equal(result.dsr, null);
  assert.equal(result.problems.length, 1);
});

test("out-of-range inputs are reported", () => {
  for (const bad of [
    { kurtosis: 0.5 },
    { n_trials: 0 },
    { var_trials_annual: -1 },
    { n_tests: 1.5 },
    { autocorr: 1 },
    { years: 0 },
  ]) {
    const result = calculate({ ...VALID, ...bad });
    assert.equal(result.dsr, null, JSON.stringify(bad));
    assert.equal(result.problems.length, 1, JSON.stringify(bad));
  }
});

test("a non-positive Sharpe ratio has no MinTRL and a full haircut", () => {
  const result = calculate({ ...VALID, sr_annual: -0.5 });
  assert.equal(result.minTrlObs, null);
  assert.equal(result.minBtlYears, null);
  assert.equal(result.haircutSrAnnual, 0);
  assert.equal(result.haircutPct, 100);
});

test("a sample under two months has no haircut", () => {
  const result = calculate({ ...VALID, years: 0.1 });
  assert.equal(result.haircutPct, null);
  assert.equal(result.problems.length, 1);
  assert.notEqual(result.dsr, null);
});
