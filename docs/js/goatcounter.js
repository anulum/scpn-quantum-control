// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — GoatCounter loader
//
// Privacy-friendly page-view analytics via GoatCounter. No cookies,
// no personal data, no IP logging (GoatCounter hashes per-day).
// Dashboard at https://anulum.goatcounter.com (Arcane Sapience
// access only). If this script fails to load, the docs site still
// works — there is no hard dependency.

(function () {
  // Skip on localhost and on preview builds.
  var host = window.location.hostname;
  if (host === "localhost" || host === "127.0.0.1" || host === "") {
    return;
  }
  // Prefix the path with the GitHub Pages subdirectory so the
  // anulum.goatcounter.com dashboard keeps scpn-quantum-control
  // page-views cleanly separated from other anulum projects.
  window.goatcounter = window.goatcounter || {};
  window.goatcounter.path = function (p) {
    return "scpn-quantum-control" + p;
  };
  var s = document.createElement("script");
  s.async = true;
  s.src = "//gc.zgo.at/count.js";
  s.setAttribute("data-goatcounter", "https://anulum.goatcounter.com/count");
  document.head.appendChild(s);
})();
