// Result rendering: the summary table and the details modal.

import { el } from "./dom.js";

function formatMs(ms) {
  if (ms == null) return null;
  if (ms < 1) return ms.toFixed(2) + " ms";
  if (ms < 10) return ms.toFixed(1) + " ms";
  return Math.round(ms) + " ms";
}

function tokenTable(score) {
  const table = el("table", "pt");
  const headRow = el("tr");
  [["token", null], ["token id", "num"], ["logprob", "num"]].forEach(([h, cls]) => headRow.appendChild(el("th", cls, h)));
  const thead = el("thead");
  thead.appendChild(headRow);
  const tbody = el("tbody");
  for (const t of score.tokens) {
    const tr = el("tr");
    tr.appendChild(el("td", null, JSON.stringify(t.token)));
    tr.appendChild(el("td", "num", String(t.token_id)));
    tr.appendChild(el("td", "num", t.logprob.toFixed(4)));
    tbody.appendChild(tr);
  }
  const eosRow = el("tr");
  eosRow.appendChild(el("td", null, "<eos>"));
  eosRow.appendChild(el("td", "num", ""));
  eosRow.appendChild(el("td", "num", score.eos_logprob.toFixed(4)));
  tbody.appendChild(eosRow);
  table.append(thead, tbody);
  return table;
}

function optionDisplay(s) {
  return s.continuation && s.continuation !== s.option ? s.option + " (" + s.continuation + ")" : s.option;
}

function instructionsFor(questionMap, name) {
  const spec = questionMap && questionMap[name];
  if (!spec || spec.instructions == null) return "";
  return typeof spec.instructions === "string" ? spec.instructions : JSON.stringify(spec.instructions);
}

function chipLabel(r, s) {
  if (r.type === "score" && r.legend && r.legend[s.option] !== undefined) {
    return s.option + " " + r.legend[s.option];
  }
  return optionDisplay(s);
}

function topChips(r, n) {
  return [...r.scores]
    .sort((a, b) => b.probability - a.probability)
    .slice(0, n)
    .map((s, i) => {
      const chip = el("span", "chip" + (i === 0 ? " top" : ""));
      chip.textContent = chipLabel(r, s) + " " + (s.probability * 100).toFixed(1) + "%";
      return chip;
    });
}

function renderAnswer(r) {
  const main = el("div", "result-main");
  if (r.type === "noul") {
    const yes = (r.noul ?? 0) >= 0.5;
    const value = yes ? (r.noul ?? 0) : 1 - (r.noul ?? 0);
    main.classList.add(yes ? "yes" : "no");
    main.appendChild(document.createTextNode(yes ? "Yes" : "No"));
    main.appendChild(el("span", "result-pct", " " + (value * 100).toFixed(1) + "%"));
  } else if (r.type === "score") {
    const num = r.score ?? 0;
    main.appendChild(document.createTextNode(num.toFixed(2)));
    const level = r.legend ? r.legend[String(Math.round(num))] : null;
    if (level) main.appendChild(el("span", "result-level", "  " + level));
  } else {
    main.appendChild(document.createTextNode(r.choice ?? "(none)"));
    const top = [...r.scores].sort((a, b) => b.probability - a.probability)[0];
    if (top) main.appendChild(el("span", "result-pct", " " + (top.probability * 100).toFixed(1) + "%"));
  }
  return main;
}

function renderDetails(r) {
  const wrap = el("div");
  const chips = el("div", "chips");
  for (const chip of topChips(r, 5)) chips.appendChild(chip);
  wrap.appendChild(chips);
  const meta = el("div", "meta-line");
  const parts = [];
  if (r.confidence !== undefined && r.confidence !== null) parts.push("confidence " + r.confidence.toFixed(2));
  parts.push("details ↗");
  meta.textContent = parts.join("    ");
  wrap.appendChild(meta);
  return wrap;
}

export function renderResults(data, questionMap) {
  const table = el("table", "results-table");
  const headRow = el("tr");
  headRow.appendChild(el("th", null, "Question"));
  headRow.appendChild(el("th", null, "Answer"));
  headRow.appendChild(el("th", null, "Details"));
  const thead = el("thead");
  thead.appendChild(headRow);
  const tbody = el("tbody");

  const usage = data.usage || {};
  const timing = formatMs(usage.timing_ms);

  for (const r of data.results) {
    const tr = el("tr");
    tr.addEventListener("click", () => openModal(r, questionMap, timing));

    const qcell = el("td", "q");
    qcell.appendChild(el("div", "qname", r.name));
    const desc = instructionsFor(questionMap, r.name);
    qcell.appendChild(el("div", "qdesc", (desc ? desc + "  ·  " : "") + r.type));

    const acell = el("td", "ans");
    acell.appendChild(renderAnswer(r));

    const dcell = el("td", "det");
    dcell.appendChild(renderDetails(r));

    tr.append(qcell, acell, dcell);
    tbody.appendChild(tr);
  }

  table.append(thead, tbody);
  return table;
}

function openModal(r, questionMap, timing) {
  const modal = document.getElementById("modal");
  const body = document.getElementById("modalBody");
  body.innerHTML = "";

  const head = el("div", "modal-head");
  head.appendChild(el("h3", null, r.name));
  const close = el("button", "close", "×");
  close.addEventListener("click", () => modal.classList.remove("open"));
  head.appendChild(close);
  body.appendChild(head);

  const desc = instructionsFor(questionMap, r.name);
  body.appendChild(el("div", "qdesc", (desc ? desc + "  ·  " : "") + r.type));

  let headline = "";
  if (r.type === "choice") headline = "choice: " + (r.choice ?? "(none)");
  else if (r.type === "noul") headline = "noul (P(yes)): " + (r.noul ?? 0).toFixed(4);
  else if (r.type === "score") headline = "score: " + (r.score ?? 0).toFixed(4);
  if (r.confidence !== undefined && r.confidence !== null) {
    headline += "     confidence: " + r.confidence.toFixed(3);
  }
  body.appendChild(el("div", "choice", headline));

  if (timing) {
    body.appendChild(el("div", "opt-meta", "request timing: " + timing));
  }

  if (r.legend) {
    body.appendChild(el("div", "tick", Object.entries(r.legend).map(([k, v]) => k + ": " + v).join("   ")));
  }

  for (const s of [...r.scores].sort((a, b) => b.probability - a.probability)) {
    const row = el("div", "opt");
    const optHead = el("div", "opt-head");
    optHead.appendChild(el("span", "opt-name", optionDisplay(s)));
    optHead.appendChild(el("span", "opt-pct", (s.probability * 100).toFixed(2) + "%"));

    const wrap = el("div", "bar-wrap");
    const bar = el("div", "bar");
    bar.style.width = Math.max(0, Math.min(1, s.probability)) * 100 + "%";
    wrap.appendChild(bar);

    const nTokens = (s.tokens || []).length + 1;
    const meta = el("div", "opt-meta");
    meta.textContent =
      (s.logprob === null ? "logp n/a" : "logp " + s.logprob.toFixed(3)) +
      "  ·  " + nTokens + " token" + (nTokens === 1 ? "" : "s");

    const details = el("details", "tok");
    details.appendChild(el("summary", null, "token probabilities"));
    details.appendChild(tokenTable(s));
    details.appendChild(el("div", "opt-meta", "total logp = " + (s.logprob === null ? "n/a" : s.logprob.toFixed(4))));

    row.append(optHead, wrap, meta, details);
    body.appendChild(row);
  }

  modal.classList.add("open");
}

export function initModal() {
  document.getElementById("modal").addEventListener("click", (e) => {
    if (e.target.id === "modal") e.target.classList.remove("open");
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") document.getElementById("modal").classList.remove("open");
  });
}
