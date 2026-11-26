/**
 * JARVIS Frontend Controller
 * Subscribes to real backend SSE events (/events) and drives:
 *  - 5 dynamic orb animation states (IDLE, LISTENING, THINKING, SPEAKING, EXECUTING)
 *  - Live mic RMS reactivity & TTS amplitude reactivity
 *  - Gemini-style floating task execution popup
 */

(function () {
  const canvas = document.getElementById("jarvis-orb");
  const orb = new window.JarvisOrb(canvas);
  orb.start();

  const stateDot = document.getElementById("state-dot");
  const stateLabel = document.getElementById("state-label");

  const popup = document.getElementById("task-popup");
  const badgeDot = document.getElementById("task-badge-dot");
  const headerTitle = document.getElementById("task-header-title");
  const taskCurrent = document.getElementById("task-current");
  const taskSteps = document.getElementById("task-steps");

  function applyState(data) {
    if (!data) return;

    const st = (data.state || "IDLE").toUpperCase();
    const mic = Number(data.mic_level || 0);
    const tts = Number(data.tts_level || 0);

    orb.setState(st, mic, tts);

    // Update minimal state pill
    stateDot.className = "state-dot " + st.toLowerCase();
    stateLabel.textContent = data.status_text || st;

    // Update Gemini-style floating task popup
    const task = data.task || {};
    if (task.active) {
      popup.classList.add("visible");
      popup.setAttribute("aria-hidden", "false");

      if (task.completed) {
        badgeDot.classList.add("done");
        headerTitle.textContent = "JARVIS \u2014 COMPLETED";
      } else {
        badgeDot.classList.remove("done");
        headerTitle.textContent = "JARVIS \u2014 EXECUTING";
      }

      taskCurrent.textContent = task.title ? `Current task: ${task.title}` : "Running task...";

      const steps = Array.isArray(task.steps) ? task.steps : [];
      taskSteps.innerHTML = "";
      for (const s of steps) {
        const li = document.createElement("li");
        const status = s.status || "pending";
        li.className = `task-step ${status}`;

        const icon = document.createElement("span");
        icon.className = "step-icon";
        if (status === "active") {
          const sp = document.createElement("span");
          sp.className = "step-spinner";
          icon.appendChild(sp);
        } else if (status === "done") {
          icon.textContent = "\u2713";
        } else if (status === "error") {
          icon.textContent = "!";
        } else {
          icon.textContent = "\u2022";
        }

        const label = document.createElement("span");
        label.textContent = s.label || "";

        li.appendChild(icon);
        li.appendChild(label);
        taskSteps.appendChild(li);
      }
    } else {
      popup.classList.remove("visible");
      popup.setAttribute("aria-hidden", "true");
    }
  }

  function connectSSE() {
    const es = new EventSource("/events");
    es.onmessage = (ev) => {
      try {
        applyState(JSON.parse(ev.data));
      } catch (_) {
        // Ignore malformed frame
      }
    };
    es.onerror = () => {
      es.close();
      setTimeout(connectSSE, 1200);
    };
  }

  // Initial snapshot + SSE stream
  fetch("/state")
    .then((r) => r.json())
    .then(applyState)
    .catch(() => {})
    .finally(connectSSE);
})();
