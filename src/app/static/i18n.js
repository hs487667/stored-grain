(function (root, factory) {
  var api = factory(root && root.document);
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.GrainI18n = api;
}(typeof window !== "undefined" ? window : globalThis, function (document) {
  "use strict";

  var supported = ["en", "hi", "te"];
  var catalogs = Object.create(null);
  var locale = "en";
  var listeners = [];

  function normalize(value) {
    var code = String(value || "").trim().toLowerCase().split(/[-_]/)[0];
    return supported.indexOf(code) >= 0 ? code : "en";
  }

  function load(code) {
    if (catalogs[code]) return Promise.resolve(catalogs[code]);
    return fetch("/static/locales/" + code + ".json").then(function (response) {
      if (!response.ok) throw new Error("catalog unavailable");
      return response.json();
    }).then(function (catalog) {
      catalogs[code] = catalog;
      return catalog;
    });
  }

  function interpolate(text, values) {
    return String(text).replace(/\{([^{}]+)\}/g, function (match, name) {
      return values && Object.prototype.hasOwnProperty.call(values, name)
        ? String(values[name])
        : match;
    });
  }

  function t(key, values) {
    var english = catalogs.en || {};
    var active = catalogs[locale] || english;
    var text = Object.prototype.hasOwnProperty.call(active, key) ? active[key] : english[key];
    return text == null ? key : interpolate(text, values);
  }

  function message(descriptor) {
    if (!descriptor || !descriptor.key || !Object.prototype.hasOwnProperty.call(catalogs.en || {}, descriptor.key)) {
      return descriptor && descriptor.fallback != null ? String(descriptor.fallback) : "";
    }
    return t(descriptor.key, descriptor.values || {});
  }

  function apply(root) {
    if (!document) return;
    var scope = root || document;
    [
      ["[data-i18n]", "textContent", "data-i18n"],
      ["[data-i18n-placeholder]", "placeholder", "data-i18n-placeholder"],
      ["[data-i18n-aria-label]", "aria-label", "data-i18n-aria-label"],
      ["[data-i18n-content]", "content", "data-i18n-content"]
    ].forEach(function (rule) {
      scope.querySelectorAll(rule[0]).forEach(function (element) {
        if (rule[1] === "textContent") {
          element.textContent = t(element.getAttribute(rule[2]));
        } else {
          element.setAttribute(rule[1], t(element.getAttribute(rule[2])));
        }
      });
    });
  }

  function updateDocument() {
    if (!document) return;
    document.documentElement.lang = locale;
    apply(document);
    document.querySelectorAll("[data-locale]").forEach(function (button) {
      button.setAttribute("aria-pressed", button.getAttribute("data-locale") === locale ? "true" : "false");
    });
  }

  function notify() {
    listeners.slice().forEach(function (listener) { listener(locale); });
  }

  function init() {
    locale = "en";
    return load("en").then(function () {
      updateDocument();
      return locale;
    });
  }

  function setLocale(value) {
    var next = normalize(value);
    return load(next).then(function () {
      locale = next;
      updateDocument();
      notify();
      return locale;
    }).catch(function (error) {
      locale = "en";
      updateDocument();
      notify();
      throw error;
    });
  }

  function subscribe(listener) {
    listeners.push(listener);
    return function () {
      listeners = listeners.filter(function (candidate) { return candidate !== listener; });
    };
  }

  return {
    init: init,
    setLocale: setLocale,
    getLocale: function () { return locale; },
    t: t,
    message: message,
    apply: apply,
    subscribe: subscribe
  };
}));
