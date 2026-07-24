/* Copy-to-clipboard that works on a LAN over plain HTTP.
 *
 * navigator.clipboard only exists in a SECURE CONTEXT (https, or localhost).
 * The vault is served over http on a LAN address, so on every machine except
 * the server itself it is undefined — and `undefined.writeText(...)` throws
 * synchronously, which no .catch() on the promise can rescue. The button just
 * did nothing.
 *
 * Order: the modern API when it exists, then the legacy execCommand path, and
 * if both fail the text is left SELECTED so Ctrl+C finishes the job. The
 * button always reports what actually happened.
 */
(function () {
  function legacyCopy(text) {
    var helper = document.createElement("textarea");
    helper.value = text;
    helper.setAttribute("readonly", "");
    // off-screen but focusable; display:none would not be selectable
    helper.style.position = "fixed";
    helper.style.top = "-1000px";
    helper.style.opacity = "0";
    document.body.appendChild(helper);
    helper.select();
    helper.setSelectionRange(0, helper.value.length);
    var ok = false;
    try {
      ok = document.execCommand("copy");
    } catch (err) {
      ok = false;
    }
    document.body.removeChild(helper);
    return ok;
  }

  function selectNode(node) {
    try {
      var range = document.createRange();
      range.selectNodeContents(node);
      var selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      return true;
    } catch (err) {
      return false;
    }
  }

  function done(button, label, state) {
    button.textContent = label;
    button.classList.toggle("done", state === "ok");
    button.classList.toggle("failed", state !== "ok");
  }

  document.addEventListener("click", function (event) {
    var button = event.target.closest("[data-copy-target]");
    if (!button) return;

    var source = document.getElementById(button.getAttribute("data-copy-target"));
    if (!source) return;
    var text = source.textContent.trim();

    function fallback() {
      if (legacyCopy(text)) {
        done(button, button.dataset.copied, "ok");
        return;
      }
      selectNode(source);
      done(button, button.dataset.manual, "failed");
    }

    if (window.isSecureContext && navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(function () {
        done(button, button.dataset.copied, "ok");
      }).catch(fallback);
    } else {
      fallback();
    }
  });
})();
