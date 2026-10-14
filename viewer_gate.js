/* Shared password gate. The digest is SHA-256 of the team password.
   The check runs in the browser. It does not encrypt the map HTML. */
var PCVS_PASSWORD_SHA256 = "31d37fe2e73c6e72e0fce885ab9a32b903a383329b89912557bfb63f531c6e9e";
var PCVS_AUTH_KEY = "pcvs-auth";

(function () {
  var waiters = [];

  function unlocked() {
    try {
      return sessionStorage.getItem(PCVS_AUTH_KEY) === PCVS_PASSWORD_SHA256;
    } catch (err) {
      return false;
    }
  }

  function sha256(text) {
    var data = new TextEncoder().encode(text);
    return crypto.subtle.digest("SHA-256", data).then(function (buffer) {
      return Array.from(new Uint8Array(buffer)).map(function (byte) {
        return byte.toString(16).padStart(2, "0");
      }).join("");
    });
  }

  function release() {
    document.documentElement.classList.remove("pcvs-locked");
    var pending = waiters.slice();
    waiters = [];
    pending.forEach(function (fn) { fn(); });
  }

  function mount() {
    if (document.getElementById("pcvs-lock")) return;
    var style = document.createElement("style");
    style.textContent = [
      "html.pcvs-locked body > *:not(#pcvs-lock){visibility:hidden}",
      "#pcvs-lock{position:fixed;inset:0;z-index:100000;display:grid;",
      "place-items:center;background:#0a1120;color:#e8edf6;",
      "font-family:ui-sans-serif,system-ui,sans-serif}",
      "#pcvs-lock form{width:min(360px,calc(100% - 32px));padding:22px;",
      "border:1px solid rgba(255,255,255,.14);border-radius:12px;",
      "background:#131c2e}",
      "#pcvs-lock h1{font-size:16px;margin:0 0 8px}",
      "#pcvs-lock p{font-size:12px;color:#93a3bd;margin:0 0 14px}",
      "#pcvs-lock input{width:100%;box-sizing:border-box;margin:0 0 10px;",
      "padding:8px 10px;border-radius:8px;border:1px solid rgba(255,255,255,.2);",
      "background:#0a1120;color:#e8edf6}",
      "#pcvs-lock button{padding:8px 12px;border:0;border-radius:8px;",
      "background:#2f6fed;color:#fff;font:inherit;cursor:pointer}",
      "#pcvs-lock .pcvs-lock-error{color:#fecaca;min-height:1.2em;font-size:12px}"
    ].join("");
    document.head.appendChild(style);
    var panel = document.createElement("form");
    panel.id = "pcvs-lock";
    panel.innerHTML = [
      "<h1>PCVS</h1>",
      "<p>This atlas is not public. Enter the team password.</p>",
      '<input id="pcvs-password" type="password" autocomplete="current-password" required>',
      '<button type="submit">Enter</button>',
      '<p class="pcvs-lock-error" id="pcvs-lock-error"></p>'
    ].join("");
    document.body.appendChild(panel);
    panel.addEventListener("submit", function (event) {
      event.preventDefault();
      var value = document.getElementById("pcvs-password").value;
      sha256(value).then(function (digest) {
        if (digest === PCVS_PASSWORD_SHA256) {
          sessionStorage.setItem(PCVS_AUTH_KEY, PCVS_PASSWORD_SHA256);
          panel.remove();
          release();
        } else {
          document.getElementById("pcvs-lock-error").textContent = "Wrong password.";
        }
      });
    });
  }

  window.PCVS_AUTH = {
    unlocked: unlocked,
    whenUnlocked: function (fn) {
      if (unlocked()) fn();
      else waiters.push(fn);
    }
  };

  if (unlocked()) return;
  document.documentElement.classList.add("pcvs-locked");
  if (document.body) mount();
  else document.addEventListener("DOMContentLoaded", mount);
})();
