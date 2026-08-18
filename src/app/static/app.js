"use strict";

const status = document.getElementById("status");
const result = document.getElementById("result");

function show(name) {
  const capturing = name === "capture";
  document.getElementById("capture").hidden = !capturing;
  document.getElementById("ranking").hidden = capturing;
  document.getElementById("tab-capture").classList.toggle("active", capturing);
  document.getElementById("tab-ranking").classList.toggle("active", !capturing);
  if (!capturing) refreshRanking();
}

document.getElementById("tab-capture").onclick = () => show("capture");
document.getElementById("tab-ranking").onclick = () => show("ranking");

document.getElementById("capture-form").onsubmit = async (event) => {
  event.preventDefault();
  const form = event.target;
  status.textContent = "Measuring…";
  result.hidden = true;

  const response = await fetch("/api/lots", {
    method: "POST",
    body: new FormData(form),
  });

  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }));
    status.textContent = body.detail;
    return;
  }

  status.textContent = "";
  renderReading(await response.json());
  form.querySelector('input[name="photo"]').value = "";
};

function renderReading(lot) {
  // The gap travels with the number: a damage figure the sample cannot
  // support is a figure that should not be compared with another one.
  const bio = Object.entries(lot.biological_pct)
    .map(([name, pct]) => `${name.replace(/_/g, " ")} ${pct.toFixed(1)}%`)
    .join(", ") || "none detected";

  result.innerHTML = `
    <p class="headline">${lot.damage_mass_pct.toFixed(2)}%
      <small>mechanical damage by weight, from ${lot.kernels_counted} kernels</small>
    </p>
    <img class="overlay" alt="detected kernels"
         src="/api/lots/${encodeURIComponent(lot.lot_id)}/overlay.png?t=${Date.now()}">
    <p class="legend">
      <span class="damaged">counted as damage</span> &nbsp;
      <span class="sound">not counted</span>
    </p>
    <table>
      <tr><th>By count</th><td>${lot.damage_count_pct.toFixed(2)}%</td></tr>
      <tr><th>Smallest gap this sample can resolve</th>
          <td>${lot.resolvable_gap_pct.toFixed(2)} points</td></tr>
      <tr><th>Biological deterioration, withheld from the model</th><td>${bio}</td></tr>
      <tr><th>Mode</th><td>${lot.mode}</td></tr>
    </table>
    <p class="caveat">Uncorrected reading. The classifier errs in both
      directions, which overstates clean grain and understates heavily damaged
      grain; correction is off because it does not transfer between imaging
      sessions.</p>
    ${lot.ranges_ok ? "" : `<p class="caveat">Outside the published validity
      range: ${lot.range_notes.join("; ")}</p>`}
  `;
  result.hidden = false;
}

async function refreshRanking() {
  const body = document.getElementById("ranking-body");
  const data = await (await fetch("/api/lots")).json();

  if (!data.ranking.length) {
    body.innerHTML = "<p>No lots photographed yet.</p>";
    return;
  }

  const rows = data.ranking.map((entry) => `
    <tr>
      <td>${entry.rank}</td>
      <td>${entry.lot_id}
        ${entry.tied_with.length
          ? `<div class="tie">cannot be separated from
             ${entry.tied_with.join(", ")} at this sample size</div>`
          : ""}
      </td>
      <td>${entry.damage_mass_pct.toFixed(2)}%</td>
      <td>${entry.degradation_rate.toFixed(4)}</td>
    </tr>
  `).join("");

  body.innerHTML = `
    <table>
      <tr><th>Rank</th><th>Lot</th><th>Damage</th><th>Rate</th></tr>
      ${rows}
    </table>
  `;
}

document.getElementById("reset").onclick = async () => {
  await fetch("/api/session/reset", { method: "POST" });
  refreshRanking();
};
