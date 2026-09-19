// Run with: node --test tests/susskind-reader.test.cjs
const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const html = readFileSync(path.join(__dirname, "../docs/leonardsusskind-reader.html"), "utf8");
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const rawBase = "https://raw.githubusercontent.com/lachlanchen/leonardsusskind/main/";
const catalog = [...html.matchAll(/<a data-book="([^"]+)" data-title="([^"]+)" data-subtitle="([^"]+)" href="([^"]+)"/g)]
  .map(([, book, title, subtitle, href]) => ({ dataset: { book, title, subtitle }, href }));
const expected = [
  {
    "slug": "classical-mechanics-theoretical-minimum",
    "title": "Classical Mechanics",
    "subtitle": "The Theoretical Minimum",
    "pdf": "core_classical_mechanics/2011_fall_theoretical_minimum/classical_mechanics_theoretical_minimum.pdf"
  },
  {
    "slug": "quantum-mechanics-theoretical-minimum",
    "title": "Quantum Mechanics",
    "subtitle": "The Theoretical Minimum",
    "pdf": "core_quantum_mechanics/2012_winter_theoretical_minimum/quantum_mechanics_theoretical_minimum.pdf"
  },
  {
    "slug": "special-relativity-theoretical-minimum",
    "title": "Special Relativity",
    "subtitle": "The Theoretical Minimum",
    "pdf": "core_special_relativity/2012_spring_theoretical_minimum/special_relativity_theoretical_minimum.pdf"
  },
  {
    "slug": "general-relativity-theoretical-minimum",
    "title": "General Relativity",
    "subtitle": "The Theoretical Minimum",
    "pdf": "core_general_relativity/2012_fall_theoretical_minimum/general_relativity_theoretical_minimum.pdf"
  },
  {
    "slug": "statistical-mechanics-theoretical-minimum",
    "title": "Statistical Mechanics",
    "subtitle": "The Theoretical Minimum",
    "pdf": "core_statistical_mechanics/2013_spring_theoretical_minimum/statistical_mechanics_theoretical_minimum.pdf"
  },
  {
    "slug": "cosmology-theoretical-minimum",
    "title": "Cosmology",
    "subtitle": "The Theoretical Minimum",
    "pdf": "core_cosmology/2013_winter_theoretical_minimum/cosmology_theoretical_minimum.pdf"
  },
  {
    "slug": "classical-mechanics-stanford-partial",
    "title": "Classical Mechanics",
    "subtitle": "Stanford partial run",
    "pdf": "core_classical_mechanics/2011_fall_modern_physics_stanford_partial/classical_mechanics_stanford_partial.pdf"
  },
  {
    "slug": "quantum-mechanics-modern-physics-stanford",
    "title": "Quantum Mechanics",
    "subtitle": "Modern Physics Stanford",
    "pdf": "core_quantum_mechanics/2012_winter_modern_physics_stanford/quantum_mechanics_modern_physics_stanford.pdf"
  },
  {
    "slug": "general-relativity-einstein-2008",
    "title": "General Relativity",
    "subtitle": "Einstein's General Theory of Relativity, 2008",
    "pdf": "core_general_relativity/2008_fall_einsteins_general_theory_of_relativity/general_relativity_2008_fall_einsteins_general_theory_of_relativity.pdf"
  },
  {
    "slug": "cosmology-legacy",
    "title": "Cosmology",
    "subtitle": "Legacy course",
    "pdf": "core_cosmology/2009_winter_legacy_cosmology/cosmology_legacy.pdf"
  },
  {
    "slug": "advanced-quantum-mechanics",
    "title": "Advanced Quantum Mechanics",
    "subtitle": "Supplementary course book",
    "pdf": "supplemental_advanced_quantum/advanced_quantum_mechanics.pdf"
  },
  {
    "slug": "particle-physics-1-basic-concepts",
    "title": "Particle Physics 1",
    "subtitle": "Basic Concepts",
    "pdf": "supplemental_particle_physics_1/particle_physics_1_basic_concepts.pdf"
  },
  {
    "slug": "particle-physics-2-standard-model",
    "title": "Particle Physics 2",
    "subtitle": "Standard Model",
    "pdf": "supplemental_particle_physics_2/particle_physics_2_standard_model.pdf"
  },
  {
    "slug": "particle-physics-3-supersymmetry-and-grand-unification",
    "title": "Particle Physics 3",
    "subtitle": "Supersymmetry and Grand Unification",
    "pdf": "supplemental_particle_physics_3/particle_physics_3_supersymmetry_and_grand_unification.pdf"
  },
  {
    "slug": "quantum-entanglement-part-1",
    "title": "Quantum Entanglement",
    "subtitle": "Part 1",
    "pdf": "supplemental_quantum_entanglement/quantum_entanglement_part_1.pdf"
  },
  {
    "slug": "quantum-entanglement-part-3",
    "title": "Quantum Entanglement",
    "subtitle": "Part 3",
    "pdf": "supplemental_quantum_entanglement/quantum_entanglement_part_3.pdf"
  },
  {
    "slug": "topics-in-string-theory",
    "title": "Topics in String Theory",
    "subtitle": "Cosmology and Black Holes",
    "pdf": "supplemental_cosmology_and_black_holes/topics_in_string_theory.pdf"
  },
  {
    "slug": "string-theory-and-m-theory",
    "title": "String Theory and M-Theory",
    "subtitle": "Supplementary course book",
    "pdf": "supplemental_string_theory/string_theory_and_m_theory.pdf"
  },
  {
    "slug": "demystifying-the-higgs-boson",
    "title": "Demystifying the Higgs Boson",
    "subtitle": "Single-lecture book",
    "pdf": "supplemental_higgs_boson/demystifying_the_higgs_boson.pdf"
  }
];

function render(query = "") {
  const nodes = new Map();
  for (const id of ["book-select", "book-title", "book-subtitle", "pdf-frame",
    "raw-pdf-link", "source-link", "fallback-pdf-link", "book-controls", "course-pdfs"]) {
    nodes.set(id, { hidden: true, open: true, children: [], events: {},
      appendChild(child) { this.children.push(child); },
      addEventListener(name, callback) { this.events[name] = callback; } });
  }
  const document = {
    getElementById: (id) => nodes.get(id),
    querySelectorAll: (selector) => {
      assert.equal(selector, "#course-pdfs a[data-book]");
      return catalog;
    },
    createElement: (tag) => { assert.equal(tag, "option"); return {}; },
  };
  const window = { location: { search: query, href: "https://learn.lazying.art/leonardsusskind-reader.html" + query } };
  vm.runInNewContext(script, { document, window, URL, URLSearchParams });
  return { nodes, document, window };
}

test("all 19 original courses have static direct PDF links", () => {
  assert.equal(catalog.length, expected.length);
  assert.equal(new Set(catalog.map((link) => link.dataset.book)).size, 19);
  for (const book of expected) {
    const link = catalog.find((link) => link.dataset.book === book.slug);
    assert.deepEqual(link, { dataset: { book: book.slug, title: book.title, subtitle: book.subtitle },
      href: rawBase + book.pdf });
  }
});

test("uninitialized reader offers real links and hides empty JS controls", () => {
  assert.doesNotMatch(html, /href="#"/);
  assert.match(html, /<details id="course-pdfs"[^>]* open>/);
  assert.match(html, /id="book-controls" hidden/);
  assert.match(html, /<iframe id="pdf-frame"[^>]* hidden>/);
  assert.match(html, /#book-controls\[hidden\], #pdf-frame\[hidden\] \{ display: none; \}/);
  assert.match(html, /id="raw-pdf-link" href="#course-pdfs"[^>]*>Choose a course PDF/);
  assert.match(html, /id="fallback-pdf-link" href="#course-pdfs"/);
  assert.match(html, /id="source-link" href="https:\/\/github.com\/lachlanchen\/leonardsusskind"/);
});

for (const book of expected) {
  test("selected course: " + book.slug, () => {
    const { nodes, document } = render("?book=" + book.slug);
    const raw = rawBase + book.pdf;
    assert.equal(nodes.get("book-title").textContent, book.title);
    assert.equal(nodes.get("book-subtitle").textContent, book.subtitle);
    assert.equal(document.title, book.title + " Reader · Leonard Susskind");
    assert.equal(nodes.get("book-select").value, book.slug);
    assert.deepEqual(nodes.get("book-select").children.map((option) => option.value), expected.map((item) => item.slug));
    for (const id of ["raw-pdf-link", "fallback-pdf-link"]) {
      assert.equal(nodes.get(id).href, raw);
      assert.equal(nodes.get(id).target, "_blank");
    }
    assert.equal(nodes.get("source-link").href, "https://github.com/lachlanchen/leonardsusskind/blob/main/" + book.pdf);
    assert.equal(nodes.get("pdf-frame").src, "https://mozilla.github.io/pdf.js/web/viewer.html?file=" + encodeURIComponent(raw));
    assert.equal(nodes.get("pdf-frame").hidden, false);
    assert.equal(nodes.get("book-controls").hidden, false);
    assert.equal(nodes.get("course-pdfs").open, false);
  });
}

for (const query of ["", "?book=", "?book=unknown", "?book=__proto__", "?book=constructor", "?book=toString"]) {
  test("safe default for " + (query || "missing query"), () => {
    const { nodes } = render(query);
    assert.equal(nodes.get("book-select").value, "quantum-mechanics-theoretical-minimum");
    assert.equal(nodes.get("book-title").textContent, "Quantum Mechanics");
    assert.ok(nodes.get("raw-pdf-link").href.endsWith("/quantum_mechanics_theoretical_minimum.pdf"));
  });
}

test("course selection keeps other query parameters", () => {
  const { nodes, window } = render("?book=cosmology-legacy&utm_source=test");
  nodes.get("book-select").events.change({ target: { value: "advanced-quantum-mechanics" } });
  const next = new URL(window.location.href);
  assert.equal(next.searchParams.get("book"), "advanced-quantum-mechanics");
  assert.equal(next.searchParams.get("utm_source"), "test");
});
