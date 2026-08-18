"use strict";

(function () {

  var SCAN_MS = 900;

  var form = document.getElementById("capture-form");
  var lotInput = document.getElementById("f-lot");
  var tempInput = document.getElementById("f-temp");
  var moistInput = document.getElementById("f-moist");
  var photoInput = document.getElementById("f-photo");
  var dropzone = document.getElementById("dropzone");
  var filenameOut = document.getElementById("photo-filename");
  var measureBtn = document.getElementById("measure");
  var statusEl = document.getElementById("status");
  var resultPanel = document.getElementById("result");
  var resultBody = document.getElementById("result-body");
  var rankingBody = document.getElementById("ranking-body");
  var resetBtn = document.getElementById("reset");

  var busy = false;
  var shownLot = null;

  // ---------------------------------------------------------------- helpers

  function reduceMotion() {
    return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  }

  function esc(value) {
    return String(value == null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function fixed(value, places) {
    return typeof value === "number" && isFinite(value)
      ? value.toFixed(places)
      : "—";
  }

  function whole(value) {
    return typeof value === "number" && isFinite(value)
      ? String(Math.round(value))
      : "—";
  }

  function setStatus(tone, text) {
    statusEl.textContent = text;
    if (tone) {
      statusEl.setAttribute("data-tone", tone);
    } else {
      statusEl.removeAttribute("data-tone");
    }
  }

  function wait(ms) {
    return new Promise(function (resolve) { window.setTimeout(resolve, ms); });
  }

  function loadImage(src) {
    return new Promise(function (resolve, reject) {
      var img = new Image();
      img.onload = function () { resolve(src); };
      img.onerror = function () { reject(new Error("overlay unavailable")); };
      img.src = src;
    });
  }

  // ------------------------------------------------------------- API errors

  // Carries a human sentence already fit for #status, so callers never have to
  // interpret a status code or a rejected fetch a second time.
  function ApiError(message) {
    this.name = "ApiError";
    this.message = message;
  }
  ApiError.prototype = Object.create(Error.prototype);

  function messageForStatus(status, detail) {
    if (detail) return detail;
    if (status === 415) return "That file is not an image. Upload a JPEG or PNG of the tray.";
    if (status === 422) return "No kernels found. Reshoot with the tray filling the frame under even light.";
    if (status === 404) return "That lot is no longer on the server.";
    if (status === 500) return "The server failed on that request. Try again with a different frame.";
    if (status === 413) return "That photograph is too large. Send a smaller image.";
    if (status === 429) return "Too many measurements. Wait a moment and try again.";
    if (status >= 500) return "The server is not answering correctly. Try again shortly.";
    return "The server rejected that request (" + status + ").";
  }

  function readDetail(response) {
    return response.json().then(
      function (body) {
        var detail = body && body.detail;
        if (typeof detail === "string" && detail.trim()) return detail.trim();
        if (Array.isArray(detail) && detail.length) {
          return detail
            .map(function (item) {
              return typeof item === "string" ? item : (item && item.msg) || "";
            })
            .filter(Boolean)
            .join("; ");
        }
        return "";
      },
      function () { return ""; }
    );
  }

  function request(path, options) {
    return fetch(path, options).then(
      function (response) {
        if (response.ok) {
          if (response.status === 204) return null;
          return response.json().catch(function () {
            throw new ApiError("The server sent a reply this page could not read.");
          });
        }
        return readDetail(response).then(function (detail) {
          throw new ApiError(messageForStatus(response.status, detail));
        });
      },
      function () {
        throw new ApiError("Cannot reach the server. It may be asleep — wait a moment and try again.");
      }
    );
  }

  function errorText(err) {
    return err && err.message ? err.message : "Something failed and the cause is unknown. Try again.";
  }

  // ------------------------------------------------------------- tie groups

  // Ties are transitive for display purposes: if the sample cannot separate A
  // from B and cannot separate B from C, it cannot put the three in an order,
  // so all three share one rank. Union-find over the symmetric closure of
  // tied_with produces those components for any group size.
  function tieComponents(entries) {
    var parent = Object.create(null);
    var present = Object.create(null);

    entries.forEach(function (entry) {
      parent[entry.lot_id] = entry.lot_id;
      present[entry.lot_id] = true;
    });

    function find(id) {
      while (parent[id] !== id) {
        parent[id] = parent[parent[id]];
        id = parent[id];
      }
      return id;
    }

    function union(a, b) {
      var ra = find(a);
      var rb = find(b);
      if (ra !== rb) parent[rb] = ra;
    }

    entries.forEach(function (entry) {
      var peers = Array.isArray(entry.tied_with) ? entry.tied_with : [];
      peers.forEach(function (peer) {
        if (present[peer]) union(entry.lot_id, peer);
      });
    });

    var groups = [];
    var byRoot = Object.create(null);

    entries.forEach(function (entry) {
      var root = find(entry.lot_id);
      var group = byRoot[root];
      if (!group) {
        group = { entries: [], rank: entry.rank, gap: 0 };
        byRoot[root] = group;
        groups.push(group);
      }
      group.entries.push(entry);
      if (typeof entry.rank === "number" && entry.rank < group.rank) {
        group.rank = entry.rank;
      }
      if (typeof entry.resolvable_gap_pct === "number" && entry.resolvable_gap_pct > group.gap) {
        group.gap = entry.resolvable_gap_pct;
      }
    });

    return groups;
  }

  function byRankThenDamage(a, b) {
    var ra = typeof a.rank === "number" ? a.rank : Infinity;
    var rb = typeof b.rank === "number" ? b.rank : Infinity;
    if (ra !== rb) return ra - rb;
    return (b.damage_mass_pct || 0) - (a.damage_mass_pct || 0);
  }

  // -------------------------------------------------------- ranking markup

  // The bar encodes degradation rate, not damage, because the rate is what
  // orders the ladder -- it carries temperature and moisture as well. Filling
  // by damage would draw rank 1 with a shorter bar than rank 2 whenever a
  // hotter, wetter lot outranks a drier, more damaged one.
  function fillPercent(entry, maxRate) {
    if (!(maxRate > 0)) return "0";
    var share = ((entry.degradation_rate || 0) / maxRate) * 100;
    return Math.max(0, Math.min(100, share)).toFixed(1);
  }

  function rungMarkup(entry, rank, tied, maxRate) {
    var lot = esc(entry.lot_id);
    return '<li class="rung" data-rank="' + esc(rank) + '" data-tied="' + (tied ? "true" : "false") + '">' +
      '<span class="rung-rank num">' + esc(rank) + "</span>" +
      '<span class="rung-lot">' + lot + "</span>" +
      '<span class="rung-bar"><span class="rung-fill" style="--fill: ' + fillPercent(entry, maxRate) + '%"></span></span>' +
      '<span class="rung-value num">' + fixed(entry.damage_mass_pct, 2) + "%</span>" +
      '<span class="rung-rate num">rate ' + fixed(entry.degradation_rate, 4) + "</span>" +
      '<button type="button" class="rung-remove" data-lot="' + lot + '" aria-label="Remove ' + lot + '">Remove</button>' +
      "</li>";
  }

  function tieGroupMarkup(group, maxRate) {
    var rungs = group.entries.map(function (entry) {
      return rungMarkup(entry, group.rank, true, maxRate);
    }).join("");

    return '<li class="tie-group">' +
      '<div class="tie-bracket" aria-hidden="true"></div>' +
      '<p class="tie-note">Cannot be separated at this sample size — the damage gap is smaller than <span class="num">' +
      fixed(group.gap, 2) + "</span> points.</p>" +
      '<ol class="tie-rungs">' + rungs + "</ol>" +
      "</li>";
  }

  function renderRanking(ranking) {
    var entries = (ranking || []).slice().sort(byRankThenDamage);

    if (!entries.length) {
      rankingBody.innerHTML = '<p class="empty">No lots yet. Photograph a tray to start the ranking.</p>';
      return;
    }

    var maxRate = entries.reduce(function (top, entry) {
      var value = entry.degradation_rate;
      return typeof value === "number" && value > top ? value : top;
    }, 0);

    var groups = tieComponents(entries).sort(function (a, b) {
      return a.rank - b.rank;
    });

    var items = groups.map(function (group) {
      group.entries.sort(byRankThenDamage);
      return group.entries.length > 1
        ? tieGroupMarkup(group, maxRate)
        : rungMarkup(group.entries[0], group.rank, false, maxRate);
    }).join("");

    rankingBody.innerHTML = '<ol class="ladder">' + items + "</ol>";
  }

  // -------------------------------------------------------- reading markup

  function biologicalSummary(biological) {
    var names = biological ? Object.keys(biological) : [];
    if (!names.length) return "none reported";
    return names.map(function (name) {
      return esc(name) + ' <span class="num">' + fixed(biological[name], 1) + "%</span>";
    }).join(", ");
  }

  function figureMarkup(state, src) {
    return '<figure class="shot" data-state="' + esc(state) + '">' +
      '<img class="shot-img" alt="Tray with each detected kernel outlined" src="' + esc(src) + '">' +
      '<div class="scanline" aria-hidden="true"></div>' +
      '<figcaption class="legend">' +
      '<span class="key key-damage">counted as damage</span>' +
      '<span class="key key-sound">not counted</span>' +
      "</figcaption></figure>";
  }

  function daysSummary(reading) {
    if (typeof reading.days_to_threshold !== "number") return "&mdash;";
    var days = fixed(reading.days_to_threshold, 1);
    if (typeof reading.days_to_threshold_error_pct !== "number") return days;
    return days + ' <span class="band">&plusmn;' + fixed(reading.days_to_threshold_error_pct, 1) + "%</span>";
  }

  function readingMarkup(reading, overlaySrc) {
    var parts = [];

    if (overlaySrc) parts.push(figureMarkup("ready", overlaySrc));

    parts.push(
      '<div class="readout">' +
      '<p class="readout-value"><span class="num">' + fixed(reading.damage_mass_pct, 2) +
      '</span><span class="pct">%</span></p>' +
      '<p class="readout-label">mechanical damage by weight</p>' +
      '<p class="readout-band">&plusmn;<span class="num">' + fixed(reading.resolvable_gap_pct, 2) +
      '</span> resolvable &middot; <span class="num">' + whole(reading.kernels_counted) +
      "</span> kernels counted</p></div>"
    );

    parts.push(
      '<dl class="facts">' +
      '<div class="fact"><dt>By count</dt><dd class="num">' + fixed(reading.damage_count_pct, 2) + "%</dd></div>" +
      '<div class="fact"><dt>Temperature</dt><dd class="num">' + fixed(reading.temperature_c, 1) + " &deg;C</dd></div>" +
      '<div class="fact"><dt>Moisture</dt><dd class="num">' + fixed(reading.moisture_pct_wb, 1) + "% wb</dd></div>" +
      '<div class="fact"><dt>Days to 0.5% loss</dt><dd class="num">' + daysSummary(reading) + "</dd></div>" +
      '<div class="fact"><dt>Mode</dt><dd>' + esc(reading.mode) + "</dd></div>" +
      '<div class="fact fact-wide">' +
      "<dt>Biological deterioration, withheld from the model</dt>" +
      "<dd>" + biologicalSummary(reading.biological_pct) + "</dd></div>" +
      "</dl>"
    );

    if (reading.calibrated !== true) {
      parts.push(
        '<p class="caveat caveat-uncorrected">Uncorrected reading. The classifier errs in both ' +
        "directions, overstating clean grain and understating heavily damaged grain. Correction " +
        "is off because it does not transfer between imaging sessions.</p>"
      );
    }

    var reasons = Array.isArray(reading.suppression_reasons) ? reading.suppression_reasons.filter(Boolean) : [];
    if (reasons.length) {
      parts.push(
        '<p class="caveat caveat-gated">Days to threshold withheld: ' +
        reasons.map(esc).join("; ") + "</p>"
      );
    }

    var modelNotes = Array.isArray(reading.model_notes) ? reading.model_notes.filter(Boolean) : [];
    if (modelNotes.length) {
      parts.push('<p class="caveat caveat-source">' + modelNotes.map(esc).join(" ") + "</p>");
    }

    var notes = Array.isArray(reading.range_notes) ? reading.range_notes.filter(Boolean) : [];
    if (reading.ranges_ok === false) {
      parts.push(
        '<p class="caveat caveat-range">Outside the published validity range: ' +
        (notes.length ? notes.map(esc).join("; ") : "the server did not say which input.") + "</p>"
      );
    }

    return parts.join("");
  }

  function overlayUrl(lotId) {
    return "/api/lots/" + encodeURIComponent(lotId) + "/overlay.png?t=" + Date.now();
  }

  function showScanning(previewSrc) {
    resultBody.innerHTML = figureMarkup("loading", previewSrc);
    resultPanel.hidden = false;
  }

  function showReading(reading, overlaySrc) {
    resultBody.innerHTML = readingMarkup(reading, overlaySrc);
    resultPanel.hidden = false;
    shownLot = reading.lot_id;
  }

  function clearReading() {
    resultBody.innerHTML = "";
    resultPanel.hidden = true;
    shownLot = null;
  }

  // ---------------------------------------------------------------- session

  function applySession(data) {
    var lots = (data && data.lots) || [];
    renderRanking((data && data.ranking) || []);
    resetBtn.hidden = lots.length === 0;

    if (shownLot && !lots.some(function (lot) { return lot.lot_id === shownLot; })) {
      clearReading();
    }
  }

  function refreshSession() {
    return request("/api/lots", { method: "GET" }).then(function (data) {
      applySession(data);
      return data;
    });
  }

  // ------------------------------------------------------------ file choice

  function showChosenFile(file) {
    if (file) {
      filenameOut.textContent = file.name;
      dropzone.setAttribute("data-empty", "false");
    } else {
      filenameOut.textContent = "";
      dropzone.setAttribute("data-empty", "true");
    }
  }

  function adoptFile(file) {
    if (!file) return;
    if (file.type && file.type.indexOf("image/") !== 0) {
      setStatus("error", "That file is not an image. Drop a JPEG or PNG of the tray.");
      return;
    }
    var transfer = new DataTransfer();
    transfer.items.add(file);
    photoInput.files = transfer.files;
    showChosenFile(file);
    setStatus(null, "");
  }

  photoInput.addEventListener("change", function () {
    showChosenFile(photoInput.files && photoInput.files[0]);
  });

  ["dragenter", "dragover"].forEach(function (name) {
    dropzone.addEventListener(name, function (event) {
      event.preventDefault();
      dropzone.classList.add("is-dragover");
    });
  });

  ["dragleave", "dragend"].forEach(function (name) {
    dropzone.addEventListener(name, function () {
      dropzone.classList.remove("is-dragover");
    });
  });

  dropzone.addEventListener("drop", function (event) {
    event.preventDefault();
    dropzone.classList.remove("is-dragover");
    var files = event.dataTransfer && event.dataTransfer.files;
    adoptFile(files && files[0]);
  });

  // A file dropped outside the dropzone would otherwise navigate the page away.
  ["dragover", "drop"].forEach(function (name) {
    window.addEventListener(name, function (event) {
      if (!dropzone.contains(event.target)) event.preventDefault();
    });
  });

  // ------------------------------------------------------------- measuring

  function firstProblem() {
    if (!lotInput.value.trim()) return "Name the lot before measuring.";
    if (!photoInput.files || !photoInput.files[0]) return "Add a photograph of the tray before measuring.";
    if (!isFinite(parseFloat(tempInput.value))) return "Enter the storage temperature in degrees Celsius.";
    if (!isFinite(parseFloat(moistInput.value))) return "Enter the moisture content as percent wet basis.";
    return null;
  }

  function setBusy(state) {
    busy = state;
    measureBtn.disabled = state;
    resetBtn.disabled = state;
  }

  form.addEventListener("submit", function (event) {
    event.preventDefault();
    if (busy) return;

    var problem = firstProblem();
    if (problem) {
      setStatus("error", problem);
      return;
    }

    var file = photoInput.files[0];
    var previewUrl = URL.createObjectURL(file);
    var minimumScan = reduceMotion() ? 0 : SCAN_MS;

    setBusy(true);
    setStatus("working", "Measuring the tray…");
    showScanning(previewUrl);

    var measured = request("/api/lots", { method: "POST", body: new FormData(form) });

    Promise.all([measured, wait(minimumScan)])
      .then(function (results) {
        var reading = results[0];
        var src = overlayUrl(reading.lot_id);
        return loadImage(src).then(
          function () {
            showReading(reading, src);
            setStatus("done", "Measured " + reading.lot_id + ".");
          },
          function () {
            showReading(reading, null);
            setStatus("error", "Measured " + reading.lot_id + ", but the outlined image did not load. Reload to see it.");
          }
        );
      })
      .then(function () {
        photoInput.value = "";
        showChosenFile(null);
        return refreshSession();
      })
      .catch(function (err) {
        clearReading();
        setStatus("error", errorText(err));
      })
      .then(function () {
        URL.revokeObjectURL(previewUrl);
        setBusy(false);
      });
  });

  // ------------------------------------------------------ delete and reset

  rankingBody.addEventListener("click", function (event) {
    var button = event.target.closest(".rung-remove");
    if (!button || busy) return;

    var lotId = button.getAttribute("data-lot");
    setBusy(true);
    setStatus("working", "Removing " + lotId + "…");

    request("/api/lots/" + encodeURIComponent(lotId), { method: "DELETE" })
      .then(function () {
        if (shownLot === lotId) clearReading();
        return refreshSession();
      })
      .then(function () {
        setStatus("done", "Removed " + lotId + ".");
      })
      .catch(function (err) {
        setStatus("error", errorText(err));
      })
      .then(function () {
        setBusy(false);
      });
  });

  resetBtn.addEventListener("click", function () {
    if (busy) return;
    if (!window.confirm("Clear every lot in this session? This cannot be undone.")) return;

    setBusy(true);
    setStatus("working", "Clearing the session…");

    request("/api/session/reset", { method: "POST" })
      .then(function () {
        clearReading();
        return refreshSession();
      })
      .then(function () {
        setStatus("done", "Session cleared.");
      })
      .catch(function (err) {
        setStatus("error", errorText(err));
      })
      .then(function () {
        setBusy(false);
      });
  });

  // ----------------------------------------------------------------- start

  showChosenFile(photoInput.files && photoInput.files[0]);
  renderRanking([]);

  refreshSession().catch(function (err) {
    rankingBody.innerHTML = '<p class="empty">' + esc(errorText(err)) + "</p>";
    setStatus("error", errorText(err));
  });

}());
