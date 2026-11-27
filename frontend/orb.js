/**
 * JARVIS Original Dark Glassy Sphere Orb
 * Dark obsidian glass sphere with a restrained, luminous flowing band inside
 * (teal, blue, violet, and warm amber). Respects prefers-reduced-motion.
 */

class JarvisOrb {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d", { alpha: true });

    this.dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.logicalSize = 440;
    this.canvas.width = this.logicalSize * this.dpr;
    this.canvas.height = this.logicalSize * this.dpr;
    this.ctx.scale(this.dpr, this.dpr);

    this.state = "IDLE";
    this.micLevel = 0;
    this.ttsLevel = 0;

    this._reducedMotionQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
    this.reducedMotion = this._reducedMotionQuery.matches;
    this._reducedMotionQuery.addEventListener("change", (e) => {
      this.reducedMotion = e.matches;
      if (this.reducedMotion) this.render(0);
    });

    this.anim = {
      speed: 0.35,
      scale: 1.0,
      bandAmp: 0.14,
      glow: 0.28,
      teal: 0.85,
      violet: 0.55,
      amber: 0.72,
      errorMix: 0.0,
    };

    this.phase = 0;
    this.lastTime = performance.now();
    this._raf = null;
  }

  setState(state, micLevel = 0, ttsLevel = 0) {
    this.state = (state || "IDLE").toUpperCase();
    this.micLevel = Math.max(0, Math.min(1, micLevel || 0));
    this.ttsLevel = Math.max(0, Math.min(1, ttsLevel || 0));
    if (this.reducedMotion) {
      this.render(0);
    }
  }

  start() {
    if (this._raf) return;
    const loop = (now) => {
      const dt = Math.min(0.05, (now - this.lastTime) / 1000);
      this.lastTime = now;
      if (!this.reducedMotion) {
        this.update(dt);
        this.render(now / 1000);
      }
      this._raf = requestAnimationFrame(loop);
    };
    this.render(0);
    this._raf = requestAnimationFrame(loop);
  }

  stop() {
    if (this._raf) {
      cancelAnimationFrame(this._raf);
      this._raf = null;
    }
  }

  update(dt) {
    let tSpeed = 0.35;
    let tScale = 1.0;
    let tBand = 0.14;
    let tGlow = 0.26;
    let tTeal = 0.85;
    let tViolet = 0.55;
    let tAmber = 0.72;
    let tError = 0.0;

    switch (this.state) {
      case "LISTENING": {
        const m = this.micLevel;
        tSpeed = 0.75 + m * 1.6;
        tScale = 1.02 + m * 0.07;
        tBand = 0.18 + m * 0.14;
        tGlow = 0.42 + m * 0.35;
        tTeal = 1.0;
        tViolet = 0.45;
        tAmber = 0.45 + m * 0.3;
        break;
      }
      case "THINKING":
      case "EXECUTING": {
        tSpeed = 1.05;
        tScale = 0.99;
        tBand = 0.20;
        tGlow = 0.38;
        tTeal = 0.85;
        tViolet = 0.92;
        tAmber = 0.65;
        break;
      }
      case "SPEAKING": {
        const v = this.ttsLevel;
        tSpeed = 0.85 + v * 1.9;
        tScale = 1.01 + v * 0.09;
        tBand = 0.17 + v * 0.18;
        tGlow = 0.40 + v * 0.38;
        tTeal = 0.75;
        tViolet = 0.55;
        tAmber = 1.0;
        break;
      }
      case "CLARIFICATION": {
        tSpeed = 0.55;
        tScale = 1.02;
        tBand = 0.16;
        tGlow = 0.42;
        tTeal = 0.95;
        tViolet = 0.65;
        tAmber = 0.85;
        break;
      }
      case "ERROR": {
        tSpeed = 0.25;
        tScale = 0.98;
        tBand = 0.10;
        tGlow = 0.36;
        tError = 1.0;
        break;
      }
      default:
        break;
    }

    const lerp = (a, b, k) => a + (b - a) * k;
    const k = Math.min(1, dt * 6.0);
    this.anim.speed = lerp(this.anim.speed, tSpeed, k);
    this.anim.scale = lerp(this.anim.scale, tScale, k);
    this.anim.bandAmp = lerp(this.anim.bandAmp, tBand, k);
    this.anim.glow = lerp(this.anim.glow, tGlow, k);
    this.anim.teal = lerp(this.anim.teal, tTeal, k);
    this.anim.violet = lerp(this.anim.violet, tViolet, k);
    this.anim.amber = lerp(this.anim.amber, tAmber, k);
    this.anim.errorMix = lerp(this.anim.errorMix, tError, k);

    this.phase += dt * this.anim.speed;
  }

  render(t) {
    const ctx = this.ctx;
    const S = this.logicalSize;
    ctx.clearRect(0, 0, S, S);

    const cx = S / 2;
    const cy = S / 2;
    const breath = this.reducedMotion ? 1.0 : 1.0 + Math.sin(t * 1.8) * 0.012;
    const R = 152 * this.anim.scale * breath;

    // 1. Restrained outer ambient glow
    ctx.save();
    const glowColor =
      this.anim.errorMix > 0.5
        ? `rgba(248, 113, 113, ${this.anim.glow * 0.45})`
        : `rgba(56, 189, 248, ${this.anim.glow * 0.38})`;
    const outerGrad = ctx.createRadialGradient(cx, cy, R * 0.65, cx, cy, R * 1.32);
    outerGrad.addColorStop(0, glowColor);
    outerGrad.addColorStop(0.6, `rgba(139, 92, 246, ${this.anim.glow * 0.14})`);
    outerGrad.addColorStop(1, "rgba(0, 0, 0, 0)");
    ctx.fillStyle = outerGrad;
    ctx.beginPath();
    ctx.arc(cx, cy, R * 1.32, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();

    // 2. Dark glassy sphere body
    ctx.save();
    ctx.beginPath();
    ctx.arc(cx, cy, R, 0, Math.PI * 2);
    ctx.clip();

    const coreGrad = ctx.createRadialGradient(cx - R * 0.22, cy - R * 0.25, R * 0.08, cx, cy, R);
    coreGrad.addColorStop(0, "#171f2e");
    coreGrad.addColorStop(0.55, "#0b101b");
    coreGrad.addColorStop(0.88, "#070a12");
    coreGrad.addColorStop(1, "#16243b");
    ctx.fillStyle = coreGrad;
    ctx.fillRect(cx - R - 2, cy - R - 2, R * 2 + 4, R * 2 + 4);

    // 3. Luminous flowing band inside the dark glass (teal -> blue -> violet -> warm amber)
    const p = this.reducedMotion ? 0.5 : this.phase;
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(-0.42 + (this.state === "THINKING" && !this.reducedMotion ? p * 0.35 : Math.sin(p * 0.4) * 0.12));

    const span = R * 1.38;
    const amp = R * this.anim.bandAmp;
    const y1 = Math.sin(p * 1.1) * amp;
    const y2 = Math.cos(p * 0.9 + 1.1) * amp;

    // Primary luminous band
    const bandGrad = ctx.createLinearGradient(-span, -R * 0.2, span, R * 0.2);
    if (this.anim.errorMix > 0.5) {
      bandGrad.addColorStop(0, "rgba(248, 113, 113, 0.2)");
      bandGrad.addColorStop(0.5, "rgba(245, 158, 11, 0.85)");
      bandGrad.addColorStop(1, "rgba(239, 68, 68, 0.25)");
    } else {
      bandGrad.addColorStop(0, `rgba(20, 184, 166, ${0.85 * this.anim.teal})`);
      bandGrad.addColorStop(0.35, `rgba(56, 189, 248, ${0.9 * this.anim.teal})`);
      bandGrad.addColorStop(0.68, `rgba(139, 92, 246, ${0.85 * this.anim.violet})`);
      bandGrad.addColorStop(1, `rgba(245, 158, 11, ${0.9 * this.anim.amber})`);
    }

    ctx.globalCompositeOperation = "screen";
    ctx.fillStyle = bandGrad;
    ctx.beginPath();
    ctx.moveTo(-span, -R * 0.18);
    ctx.bezierCurveTo(-span * 0.35, y1 - R * 0.24, span * 0.35, -y2 - R * 0.18, span, -R * 0.10);
    ctx.lineTo(span, R * 0.16);
    ctx.bezierCurveTo(span * 0.35, -y2 + R * 0.22, -span * 0.35, y1 + R * 0.20, -span, R * 0.12);
    ctx.closePath();
    ctx.fill();

    // Subtle secondary core filament
    ctx.strokeStyle = "rgba(255, 255, 255, 0.42)";
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    ctx.moveTo(-span, 0);
    ctx.bezierCurveTo(-span * 0.35, y1 * 0.7, span * 0.35, -y2 * 0.7, span, 0);
    ctx.stroke();

    ctx.restore();

    // 4. Glass rim refraction & specular arc
    const rimGrad = ctx.createRadialGradient(cx, cy, R * 0.76, cx, cy, R);
    rimGrad.addColorStop(0, "rgba(255, 255, 255, 0)");
    rimGrad.addColorStop(0.88, "rgba(148, 163, 184, 0.14)");
    rimGrad.addColorStop(1, "rgba(255, 255, 255, 0.38)");
    ctx.fillStyle = rimGrad;
    ctx.fillRect(cx - R, cy - R, R * 2, R * 2);

    ctx.save();
    ctx.translate(cx - R * 0.32, cy - R * 0.34);
    ctx.rotate(-0.6);
    const spec = ctx.createRadialGradient(0, 0, 2, 0, 0, R * 0.44);
    spec.addColorStop(0, "rgba(255, 255, 255, 0.42)");
    spec.addColorStop(1, "rgba(255, 255, 255, 0)");
    ctx.fillStyle = spec;
    ctx.beginPath();
    ctx.ellipse(0, 0, R * 0.42, R * 0.14, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();

    ctx.restore(); // end clip

    // Crisp outer glass edge
    ctx.save();
    ctx.strokeStyle =
      this.state === "LISTENING"
        ? "rgba(56, 189, 248, 0.75)"
        : this.state === "ERROR"
        ? "rgba(248, 113, 113, 0.75)"
        : "rgba(255, 255, 255, 0.24)";
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    ctx.arc(cx, cy, R - 0.5, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
  }
}

window.JarvisOrb = JarvisOrb;
