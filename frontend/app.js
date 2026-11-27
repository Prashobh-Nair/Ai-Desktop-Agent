/**
 * JARVIS Opening Interface Controller
 * Connects the dark/light minimal orb assistant to real backend SSE & REST actions.
 * Dynamic Island (src/dynamic_island.py) remains untouched for task execution.
 */

(function () {
  const htmlEl = document.documentElement;
  const canvas = document.getElementById("jarvis-orb");
  const orb = new window.JarvisOrb(canvas);
  orb.start();

  const orbBtn = document.getElementById("orb-btn");
  const statusLine = document.getElementById("status-line");
  const recognizedEl = document.getElementById("recognized-text");
  const replyEl = document.getElementById("reply-text");
  const errorBanner = document.getElementById("error-banner");

  const confirmBar = document.getElementById("confirm-bar");
  const confirmQuestion = document.getElementById("confirm-question");
  const confirmYes = document.getElementById("confirm-yes");
  const confirmNo = document.getElementById("confirm-no");

  const historyPanel = document.getElementById("history-panel");
  const historyList = document.getElementById("history-list");
  const historyBtn = document.getElementById("history-btn");
  const historyClear = document.getElementById("history-clear");
  const historyUndo = document.getElementById("history-undo");

  const composer = document.getElementById("composer");
  const textInput = document.getElementById("text-input");
  const sendBtn = document.getElementById("send-btn");
  const typeBtn = document.getElementById("type-btn");
  const micBtn = document.getElementById("mic-btn");
  const micLabel = document.getElementById("mic-label");
  const muteBtn = document.getElementById("mute-btn");
  const muteIcon = document.getElementById("mute-icon");
  const muteLabel = document.getElementById("mute-label");

  const themeBtn = document.getElementById("theme-btn");
  const themeIcon = document.getElementById("theme-icon");
  const collapseBtn = document.getElementById("collapse-btn");

  const svcDot = document.getElementById("svc-dot");
  const svcText = document.getElementById("svc-text");

  let lastClearedHistory = null;
  let lastState = "IDLE";

  // Theme initialization
  const savedTheme = localStorage.getItem("jarvis_theme") || "dark";
  applyTheme(savedTheme);

  function applyTheme(theme) {
    const t = theme === "light" ? "light" : "dark";
    htmlEl.setAttribute("data-theme", t);
    localStorage.setItem("jarvis_theme", t);
    themeIcon.textContent = t === "dark" ? "\u2600" : "\u263E";
  }

  themeBtn.addEventListener("click", () => {
    const cur = htmlEl.getAttribute("data-theme");
    applyTheme(cur === "dark" ? "light" : "dark");
  });

  collapseBtn.addEventListener("click", () => {
    fetch("/window/minimize", { method: "POST" }).catch(() => {});
  });

  // Orb / Mic click
  async function toggleMic() {
    await fetch("/mic/toggle", { method: "POST" }).catch(() => {});
  }
  orbBtn.addEventListener("click", toggleMic);
  micBtn.addEventListener("click", toggleMic);

  // Mute toggle
  muteBtn.addEventListener("click", async () => {
    await fetch("/tts/mute", { method: "POST" }).catch(() => {});
  });

  // Type toggle (hidden by default)
  typeBtn.addEventListener("click", () => {
    const show = composer.hidden;
    composer.hidden = !show;
    typeBtn.setAttribute("aria-expanded", String(show));
    typeBtn.classList.toggle("active", show);
    if (show) textInput.focus();
  });

  // History toggle + Clear + Undo
  historyBtn.addEventListener("click", () => {
    const show = historyPanel.hidden;
    historyPanel.hidden = !show;
    historyBtn.setAttribute("aria-expanded", String(show));
    historyBtn.classList.toggle("active", show);
  });

  historyClear.addEventListener("click", async () => {
    const items = Array.from(historyList.children).map((li) => ({
      role: li.classList.contains("user") ? "user" : "assistant",
      text: li.textContent,
    }));
    if (items.length) {
      lastClearedHistory = items;
      historyUndo.hidden = false;
    }
    await fetch("/history/clear", { method: "POST" }).catch(() => {});
  });

  historyUndo.addEventListener("click", () => {
    if (!lastClearedHistory) return;
    renderHistory(lastClearedHistory);
    lastClearedHistory = null;
    historyUndo.hidden = true;
  });

  // Confirmation buttons
  confirmYes.addEventListener("click", () => {
    fetch("/confirm", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ approved: true }),
    }).catch(() => {});
  });

  confirmNo.addEventListener("click", () => {
    fetch("/confirm", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ approved: false }),
    }).catch(() => {});
  });

  // Typed command submission (Enter submits, Shift+Enter newline)
  async function submitTypedCommand() {
    const text = textInput.value.trim();
    if (!text) return;
    sendBtn.disabled = true;
    try {
      await fetch("/command", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      textInput.value = "";
      textInput.style.height = "auto";
    } finally {
      sendBtn.disabled = false;
    }
  }

  composer.addEventListener("submit", (e) => {
    e.preventDefault();
    submitTypedCommand();
  });

  textInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submitTypedCommand();
    }
  });

  textInput.addEventListener("input", () => {
    textInput.style.height = "auto";
    textInput.style.height = Math.min(100, textInput.scrollHeight) + "px";
  });

  function renderHistory(items) {
    historyList.innerHTML = "";
    for (const item of items) {
      const li = document.createElement("li");
      li.className = `history-item ${item.role === "user" ? "user" : "assistant"}`;
      li.textContent = item.role === "user" ? `You: ${item.text}` : `Jarvis: ${item.text}`;
      historyList.appendChild(li);
    }
    historyList.scrollTop = historyList.scrollHeight;
  }

  function applyState(data) {
    if (!data) return;
    const st = (data.state || "IDLE").toUpperCase();
    lastState = st;

    orb.setState(st, Number(data.mic_level || 0), Number(data.tts_level || 0));
    statusLine.textContent = data.status_text || st;

    // Busy state for send button only when actively thinking/executing
    const busy = st === "THINKING" || st === "EXECUTING";
    sendBtn.disabled = busy;

    // Mic button active state
    const isListening = st === "LISTENING";
    micBtn.classList.toggle("active", isListening);
    micLabel.textContent = isListening ? "Stop" : "Mic";

    // Mute button state
    const muted = Boolean(data.tts_muted);
    muteBtn.setAttribute("aria-pressed", String(muted));
    muteIcon.textContent = muted ? "\uD83D\uDD07" : "\uD83D\uDD0A";
    muteLabel.textContent = muted ? "Muted" : "Voice";

    // Recognized text & reply
    if (data.recognized_text) {
      recognizedEl.textContent = `"${data.recognized_text}"`;
      recognizedEl.hidden = false;
    } else {
      recognizedEl.hidden = true;
    }

    if (data.reply_text) {
      replyEl.textContent = data.reply_text;
    }

    // Error banner
    if (st === "ERROR" && data.error_message) {
      errorBanner.textContent = data.error_message;
      errorBanner.hidden = false;
    } else {
      errorBanner.hidden = true;
    }

    // Confirmation bar
    const conf = data.confirmation || {};
    if (conf.active) {
      confirmQuestion.textContent = conf.question || "Confirm this action?";
      confirmBar.hidden = false;
    } else {
      confirmBar.hidden = true;
    }

    // History
    if (Array.isArray(data.history) && data.history.length) {
      renderHistory(data.history);
    } else if (!lastClearedHistory) {
      historyList.innerHTML = "";
    }

    // Service readiness indicator
    const svcs = data.services || {};
    const unavailable = Object.entries(svcs)
      .filter(([, v]) => v === "unavailable")
      .map(([k]) => k.toUpperCase());
    if (unavailable.length) {
      svcDot.className = "svc-dot warn";
      svcText.textContent = `Offline: ${unavailable.join(", ")}`;
    } else {
      svcDot.className = "svc-dot";
      svcText.textContent = "Services ready";
    }
  }

  function connectSSE() {
    const es = new EventSource("/events");
    es.onmessage = (ev) => {
      try {
        applyState(JSON.parse(ev.data));
      } catch (_) {}
    };
    es.onerror = () => {
      svcDot.className = "svc-dot err";
      svcText.textContent = "Reconnecting...";
      es.close();
      setTimeout(connectSSE, 1200);
    };
  }

  fetch("/state")
    .then((r) => r.json())
    .then(applyState)
    .catch(() => {})
    .finally(connectSSE);
})();
