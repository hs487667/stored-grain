const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");

const catalogs = Object.fromEntries(["en", "hi", "te"].map((locale) => [
  `/static/locales/${locale}.json`,
  JSON.parse(fs.readFileSync(`src/app/static/locales/${locale}.json`, "utf8"))
]));

function loadI18n(fetchImpl) {
  global.fetch = fetchImpl || (async (path) => ({
    ok: true,
    json: async () => catalogs[path]
  }));
  const modulePath = require.resolve("../src/app/static/i18n.js");
  delete require.cache[modulePath];
  return require(modulePath);
}

test("starts in English and does not read persisted state", async () => {
  const i18n = loadI18n();
  await i18n.init();
  assert.equal(i18n.getLocale(), "en");
  assert.equal(i18n.t("status.measured", { lot: "A-4" }), "Measured A-4.");
});

test("switches locale and interpolates without changing supplied data", async () => {
  const i18n = loadI18n();
  await i18n.init();
  await i18n.setLocale("te");
  assert.equal(i18n.t("status.measured", { lot: "A-4" }), "A-4 కొలవబడింది.");
});

test("invalid locale and missing localized keys fall back to English", async () => {
  const i18n = loadI18n();
  await i18n.init();
  await i18n.setLocale("fr");
  assert.equal(i18n.getLocale(), "en");
  assert.equal(i18n.t("status.measured", { lot: "A-4" }), "Measured A-4.");
});

test("keeps English controls usable when the English catalog cannot load", async () => {
  const i18n = loadI18n(async () => { throw new Error("offline"); });
  await assert.doesNotReject(i18n.init());
  assert.equal(i18n.getLocale(), "en");
  assert.equal(i18n.t("validation.lot"), "Lot ID is required.");
  assert.equal(i18n.t("error.network"), "Network error. Check your connection and try again.");
});

test("message state retranslates validation and network errors", async () => {
  const i18n = loadI18n();
  await i18n.init();
  const status = i18n.createMessageState();

  status.set({ key: "validation.lot" });
  await i18n.setLocale("hi");
  assert.equal(status.render(), "लॉट आईडी आवश्यक है।");

  status.set({ key: "error.network" });
  await i18n.setLocale("te");
  assert.equal(status.render(), "నెట్‌వర్క్ లోపం. కనెక్షన్ తనిఖీ చేసి మళ్లీ ప్రయత్నించండి.");
});
