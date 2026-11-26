/**
 * JARVIS 3D Translucent Glass Sphere & Flowing Energy Ribbons
 * Inspired by the reference orb: icy blue, cyan, pearl white, and vibrant orange
 * silky layered ribbons inside a refractive 3D glass sphere.
 * Pure HTML5 2D Canvas — ultra-lightweight, 60fps on 8 GB RAM laptops.
 */

class JarvisOrb {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d", { alpha: true });

    // HiDPI scaling
    this.dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.logicalSize = 520;
    this.canvas.width = this.logicalSize * this.dpr;
    this.canvas.height = this.logicalSize * this.dpr;
    this.ctx.scale(this.dpr, this.dpr);

    this.state = "IDLE";
    this.micLevel = 0;
    this.ttsLevel = 0;

    // Smoothly interpolated animation parameters
    this.anim = {
      speed: 0.42,
      swirl: 0.18,
      breathAmp: 0.018,
      scale: 1.0,
      cyanBoost: 0.55,
      orangeBoost: 0.75,
      violetBoost: 0.15,
      glowAlpha: 0.32,
      ribbonSpread: 1.0,
    };

    this.phase = 0;
    this.lastTime = performance.now();
    this._raf = null;
  }

  setState(state, micLevel = 0, ttsLevel = 0) {
    this.state = (state || "IDLE").toUpperCase();
    this.micLevel = Math.max(0, Math.min(1, micLevel || 0));
    this.ttsLevel = Math.max(0, Math.min(1, ttsLevel || 0));
  }

  start() {
    if (this._raf) return;
    const loop = (now) => {
      const dt = Math.min(0.05, (now - this.lastTime) / 1000);
      this.lastTime = now;
      this.update(dt, now / 1000);
      this.render(now / 1000);
      this._raf = requestAnimationFrame(loop);
    };
    this._raf = requestAnimationFrame(loop);
  }

  stop() {
    if (this._raf) {
      cancelAnimationFrame(this._raf);
      this._raf = null;
    }
  }

  update(dt, t) {
    // Target parameters per state
    let tSpeed = 0.42;
    let tSwirl = 0.15;
    let tBreath = 0.018;
    let tScale = 1.0;
    let tCyan = 0.65;
    let tOrange = 0.78;
    let tViolet = 0.12;
    let tGlow = 0.30;
    let tSpread = 1.0;

    switch (this.state) {
      case "LISTENING": {
        const m = this.micLevel;
        tSpeed = 0.95 + m * 2.2;
        tSwirl = 0.25 + m * 0.45;
        tBreath = 0.024 + m * 0.055;
        tScale = 1.02 + m * 0.085;
        tCyan = 0.95;
        tOrange = 0.55 + m * 0.25;
        tViolet = 0.18;
        tGlow = 0.45 + m * 0.38;
        tSpread = 1.04 + m * 0.14;
        break;
      }
      case "THINKING": {
        tSpeed = 1.45;
        tSwirl = 0.85;
        tBreath = 0.025;
        tScale = 0.99;
        tCyan = 0.85;
        tOrange = 0.72;
        tViolet = 0.52;
        tGlow = 0.46;
        tSpread = 1.08;
        break;
      }
      case "SPEAKING": {
        const v = this.ttsLevel;
        tSpeed = 1.05 + v * 2.5;
        tSwirl = 0.30 + v * 0.55;
        tBreath = 0.02 + v * 0.075;
        tScale = 1.01 + v * 0.11;
        tCyan = 0.78;
        tOrange = 0.95;
        tViolet = 0.22;
        tGlow = 0.42 + v * 0.45;
        tSpread = 1.05 + v * 0.22;
        break;
      }
      case "EXECUTING": {
        tSpeed = 1.65;
        tSwirl = 0.52;
        tBreath = 0.022;
        tScale = 1.02;
        tCyan = 0.92;
        tOrange = 0.88;
        tViolet = 0.25;
        tGlow = 0.48;
        tSpread = 1.1;
        break;
      }
      default: // IDLE
        break;
    }

    const lerp = (a, b, k) => a + (b - a) * k;
    const k = Math.min(1, dt * 6.5);
    this.anim.speed = lerp(this.anim.speed, tSpeed, k);
    this.anim.swirl = lerp(this.anim.swirl, tSwirl, k);
    this.anim.breathAmp = lerp(this.anim.breathAmp, tBreath, k);
    this.anim.scale = lerp(this.anim.scale, tScale, k);
    this.anim.cyanBoost = lerp(this.anim.cyanBoost, tCyan, k);
    this.anim.orangeBoost = lerp(this.anim.orangeBoost, tOrange, k);
    this.anim.violetBoost = lerp(this.anim.violetBoost, tViolet, k);
    this.anim.glowAlpha = lerp(this.anim.glowAlpha, tGlow, k);
    this.anim.ribbonSpread = lerp(this.anim.ribbonSpread, tSpread, k);

    this.phase += dt * this.anim.speed;
  }

  render(t) {
    const ctx = this.ctx;
    const S = this.logicalSize;
    ctx.clearRect(0, 0, S, S);

    const floatY = Math.sin(t * 1.45) * 6.5;
    const breath = 1 + Math.sin(t * 2.1) * this.anim.breathAmp;
    const cx = S / 2;
    const cy = S / 2 + floatY;
    const R = 178 * this.anim.scale * breath;

    // 1. Soft floor shadow & outer ambient aura
    ctx.save();
    const shadowGrad = ctx.createRadialGradient(cx, cy + R + 32, 10, cx, cy + R + 32, R * 0.95);
    shadowGrad.addColorStop(0, "rgba(18, 42, 82, 0.14)");
    shadowGrad.addColorStop(0.55, "rgba(14, 165, 233, 0.06)");
    shadowGrad.addColorStop(1, "rgba(255, 255, 255, 0)");
    ctx.fillStyle = shadowGrad;
    ctx.beginPath();
    ctx.ellipse(cx, cy + R + 32, R * 0.92, 26, 0, 0, Math.PI * 2);
    ctx.fill();

    // Outer luminous halo (icy cyan + warm orange)
    const haloGrad = ctx.createRadialGradient(cx - R * 0.2, cy - R * 0.2, R * 0.6, cx, cy, R * 1.38);
    haloGrad.addColorStop(0, `rgba(56, 189, 248, ${this.anim.glowAlpha * 0.45})`);
    haloGrad.addColorStop(0.55, `rgba(251, 146, 60, ${this.anim.glowAlpha * 0.22})`);
    haloGrad.addColorStop(1, "rgba(255, 255, 255, 0)");
    ctx.fillStyle = haloGrad;
    ctx.beginPath();
    ctx.arc(cx, cy, R * 1.38, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();

    // 2. Clip inside the 3D Glass Sphere
    ctx.save();
    ctx.beginPath();
    ctx.arc(cx, cy, R, 0, Math.PI * 2);
    ctx.clip();

    // Deep sapphire translucent inner core (matches reference image dark-blue depth)
    const coreGrad = ctx.createRadialGradient(cx - R * 0.15, cy - R * 0.15, R * 0.12, cx, cy, R);
    coreGrad.addColorStop(0, "#0b2e59");
    coreGrad.addColorStop(0.58, "#071d3d");
    coreGrad.addColorStop(0.88, "#164e7a");
    coreGrad.addColorStop(1, "#bae6fd");
    ctx.fillStyle = coreGrad;
    ctx.fillRect(cx - R - 4, cy - R - 4, R * 2 + 8, R * 2 + 8);

    // 3. Flowing Internal Energy Ribbons (layered silky Bezier bands)
    const p = this.phase;
    const swirlAngle = Math.sin(p * 0.35) * this.anim.swirl + (this.state === "THINKING" ? p * 0.35 : -0.48);

    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(swirlAngle);

    // Back icy-blue & pearl-white sweeping layers
    this._drawRibbon(ctx, R, p, {
      yOffset: -0.34,
      amp: 0.22,
      freq: 1.5,
      thickness: 0.42,
      colors: [
        [0, `rgba(186, 230, 253, ${0.15 * this.anim.cyanBoost})`],
        [0.45, `rgba(224, 242, 254, ${0.72 * this.anim.cyanBoost})`],
        [0.85, `rgba(56, 189, 248, ${0.55 * this.anim.cyanBoost})`],
        [1, "rgba(255, 255, 255, 0.85)"],
      ],
      composite: "screen",
    });

    // Subtle violet/indigo depth ribbon (especially visible in THINKING)
    if (this.anim.violetBoost > 0.1) {
      this._drawRibbon(ctx, R, p * 1.15 + 1.4, {
        yOffset: -0.08,
        amp: 0.26,
        freq: 1.8,
        thickness: 0.30,
        colors: [
          [0, `rgba(99, 102, 241, ${0.15 * this.anim.violetBoost})`],
          [0.5, `rgba(139, 92, 246, ${0.55 * this.anim.violetBoost})`],
          [1, `rgba(56, 189, 248, ${0.45 * this.anim.violetBoost})`],
        ],
        composite: "screen",
      });
    }

    // Vibrant warm orange silky ribbon (signature reference ribbon)
    this._drawRibbon(ctx, R, p * 0.92 + 0.7, {
      yOffset: 0.16,
      amp: 0.24,
      freq: 1.35,
      thickness: 0.46,
      colors: [
        [0, `rgba(254, 215, 170, ${0.25 * this.anim.orangeBoost})`],
        [0.35, `rgba(251, 146, 60, ${0.92 * this.anim.orangeBoost})`],
        [0.72, `rgba(234, 88, 12, ${0.88 * this.anim.orangeBoost})`],
        [1, `rgba(255, 237, 213, ${0.75 * this.anim.orangeBoost})`],
      ],
      composite: "source-over",
    });

    // Secondary bright orange-gold inner fold
    this._drawRibbon(ctx, R, p * 1.28 + 2.2, {
      yOffset: 0.28,
      amp: 0.19,
      freq: 1.65,
      thickness: 0.24,
      colors: [
        [0, "rgba(255, 255, 255, 0.12)"],
        [0.5, `rgba(253, 186, 116, ${0.85 * this.anim.orangeBoost})`],
        [1, `rgba(249, 115, 22, ${0.70 * this.anim.orangeBoost})`],
      ],
      composite: "screen",
    });

    // Foreground translucent icy-cyan & pearl-white glass veil ribbon
    this._drawRibbon(ctx, R, p * 0.8 + 3.4, {
      yOffset: -0.22,
      amp: 0.25,
      freq: 1.25,
      thickness: 0.34,
      colors: [
        [0, "rgba(255, 255, 255, 0.18)"],
        [0.4, `rgba(125, 211, 252, ${0.78 * this.anim.cyanBoost})`],
        [0.78, "rgba(240, 249, 255, 0.88)"],
        [1, "rgba(255, 255, 255, 0.25)"],
      ],
      composite: "screen",
    });

    ctx.restore();

    // 4. Glass Sphere Lighting, Fresnel Rim & Specular Reflections
    // Inner Fresnel cyan/white rim glow
    const rimGrad = ctx.createRadialGradient(cx, cy, R * 0.72, cx, cy, R);
    rimGrad.addColorStop(0, "rgba(255, 255, 255, 0)");
    rimGrad.addColorStop(0.76, "rgba(186, 230, 253, 0.18)");
    rimGrad.addColorStop(0.93, "rgba(224, 242, 254, 0.58)");
    rimGrad.addColorStop(1, "rgba(255, 255, 255, 0.92)");
    ctx.fillStyle = rimGrad;
    ctx.fillRect(cx - R, cy - R, R * 2, R * 2);

    // Top-left crisp glass specular crescent
    ctx.save();
    ctx.translate(cx - R * 0.30, cy - R * 0.34);
    ctx.rotate(-0.62);
    const specGrad = ctx.createRadialGradient(0, 0, 4, 0, 0, R * 0.52);
    specGrad.addColorStop(0, "rgba(255, 255, 255, 0.75)");
    specGrad.addColorStop(0.45, "rgba(240, 249, 255, 0.26)");
    specGrad.addColorStop(1, "rgba(255, 255, 255, 0)");
    ctx.fillStyle = specGrad;
    ctx.beginPath();
    ctx.ellipse(0, 0, R * 0.48, R * 0.20, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();

    // Bottom-right warm orange/cyan caustic bounce
    const bounceGrad = ctx.createRadialGradient(cx + R * 0.35, cy + R * 0.42, 6, cx + R * 0.35, cy + R * 0.42, R * 0.58);
    bounceGrad.addColorStop(0, `rgba(254, 215, 170, ${0.42 * this.anim.orangeBoost})`);
    bounceGrad.addColorStop(0.6, "rgba(186, 230, 253, 0.14)");
    bounceGrad.addColorStop(1, "rgba(255, 255, 255, 0)");
    ctx.fillStyle = bounceGrad;
    ctx.fillRect(cx - R, cy - R, R * 2, R * 2);

    ctx.restore(); // end sphere clip

    // 5. Crisp outer glass edge ring
    ctx.save();
    ctx.strokeStyle = "rgba(255, 255, 255, 0.82)";
    ctx.lineWidth = 2.0;
    ctx.beginPath();
    ctx.arc(cx, cy, R - 0.5, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
  }

  _drawRibbon(ctx, R, phase, opts) {
    const spread = this.anim.ribbonSpread;
    const yBase = opts.yOffset * R * spread;
    const amp = opts.amp * R * spread;
    const thick = opts.thickness * R * spread;

    const w = R * 1.45;
    const x0 = -w;
    const x1 = -w * 0.35;
    const x2 = w * 0.35;
    const x3 = w;

    const s1 = Math.sin(phase * opts.freq) * amp;
    const s2 = Math.cos(phase * opts.freq * 0.85 + 1.1) * amp;
    const s3 = Math.sin(phase * opts.freq * 1.1 + 2.2) * (amp * 0.65);

    ctx.save();
    ctx.globalCompositeOperation = opts.composite || "source-over";

    const grad = ctx.createLinearGradient(x0, yBase - thick, x3, yBase + thick);
    for (const [stop, color] of opts.colors) {
      grad.addColorStop(stop, color);
    }
    ctx.fillStyle = grad;

    ctx.beginPath();
    ctx.moveTo(x0, yBase + s3 - thick * 0.45);
    ctx.bezierCurveTo(
      x1, yBase + s1 - thick * 0.55,
      x2, yBase - s2 - thick * 0.4,
      x3, yBase - s3 - thick * 0.3
    );
    ctx.lineTo(x3, yBase - s3 + thick * 0.55);
    ctx.bezierCurveTo(
      x2, yBase - s2 + thick * 0.65,
      x1, yBase + s1 + thick * 0.45,
      x0, yBase + s3 + thick * 0.4
    );
    ctx.closePath();
    ctx.fill();

    // Fine silky highlight edge along the ribbon crest
    ctx.strokeStyle = "rgba(255, 255, 255, 0.34)";
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    ctx.moveTo(x0, yBase + s3 - thick * 0.45);
    ctx.bezierCurveTo(
      x1, yBase + s1 - thick * 0.55,
      x2, yBase - s2 - thick * 0.4,
      x3, yBase - s3 - thick * 0.3
    );
    ctx.stroke();

    ctx.restore();
  }
}

window.JarvisOrb = JarvisOrb;
