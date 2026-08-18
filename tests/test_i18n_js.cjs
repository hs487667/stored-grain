const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");

const catalogs = Object.fromEntries(["en", "hi", "te"].map((locale) => [
  `/static/locales/${locale}.json`,
  JSON.parse(fs.readFileSync(`src/app/static/locales/${locale}.json`, "utf8"))
]));

global.fetch = async (path) => ({
  ok: true,
  json: async () => catalogs[path]
});

const i18n = require("../src/app/static/i18n.js");

test("starts in English and does not read persisted state", async () => {
  await i18n.init();
  assert.equal(i18n.getLocale(), "en");
  assert.equal(i18n.t("status.measured", { lot: "A-4" }), "Measured A-4.");
});

test("switches locale and interpolates without changing supplied data", async () => {
  await i18n.setLocale("te");
  assert.equal(i18n.t("status.measured", { lot: "A-4" }), "A-4 కొలవబడింది.");
});

test("invalid locale and missing localized keys fall back to English", async () => {
  await i18n.setLocale("fr");
  assert.equal(i18n.getLocale(), "en");
  assert.equal(i18n.t("status.measured", { lot: "A-4" }), "Measured A-4.");
});
