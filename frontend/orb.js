/**
 * JARVIS 3D Translucent Glass Sphere & Sculpted Energy Ribbons (v2)
 * High-craft 2D Canvas 3D-illusion renderer inspired by the reference orb:
 *  - Deep sapphire-navy optical core for true volumetric depth
 *  - Multi-layered 3D-shaded sweeping crescent ribbons (icy cyan, pearl white, vibrant orange)
 *  - Double-walled Fresnel glass shell with chromatic rim dispersion & specular highlights
 *  - Subtle cursor parallax + 5 fluid backend-driven states
 */

class JarvisOrb {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d", { alpha: true });

    this.dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.logicalSize = 560;
    this.canvas.width = this.logicalSize * this.dpr;
    this.canvas.height = this.logicalSize * this.dpr;
    this.ctx.scale(this.dpr, this.dpr);

    this.state = "IDLE";
    this.micLevel = 0;
    this.ttsLevel = 0;

    // Subtle pointer parallax (-1..1)
    this.pointer = { x: 0, y: 0, tx: 0, ty: 0 };
    window.addEventListener("pointermove", (e) => {
      const nx = (e.clientX / Math.max(1, window.innerWidth)) * 2 - 1;
      const ny = (e.clientY / Math.max(1, window.innerHeight)) * 2 - 1;
      this.pointer.tx = Math.max(-1, Math.min(1, nx));
      this.pointer.ty = Math.max(-1, Math.min(1, ny));
    });

    this.anim = {
      speed: 0.45,
      swirl: 0.14,
      breathAmp: 0.016,
      scale: 1.0,
      cyanBoost: 0.85,
      orangeBoost: 0.92,
      violetBoost: 0.14,
      glowAlpha: 0.34,
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
      this.update(dt);
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

  update(dt) {
    let tSpeed = 0.45;
    let tSwirl = 0.12;
    let tBreath = 0.016;
    let tScale = 1.0;
    let tCyan = 0.85;
    let tOrange = 0.92;
    let tViolet = 0.12;
    let tGlow = 0.34;
    let tSpread = 1.0;

    switch (this.state) {
      case "LISTENING": {
        const m = this.micLevel;
        tSpeed = 0.95 + m * 2.3;
        tSwirl = 0.22 + m * 0.42;
        tBreath = 0.02 + m * 0.055;
        tScale = 1.02 + m * 0.085;
        tCyan = 1.0;
        tOrange = 0.72 + m * 0.22;
        tViolet = 0.18;
        tGlow = 0.48 + m * 0.42;
        tSpread = 1.04 + m * 0.15;
        break;
      }
      case "THINKING": {
        tSpeed = 1.55;
        tSwirl = 0.95;
        tBreath = 0.024;
        tScale = 0.99;
        tCyan = 0.92;
        tOrange = 0.85;
        tViolet = 0.58;
        tGlow = 0.48;
        tSpread = 1.08;
        break;
      }
      case "SPEAKING": {
        const v = this.ttsLevel;
        tSpeed = 1.1 + v * 2.6;
        tSwirl = 0.26 + v * 0.5;
        tBreath = 0.02 + v * 0.075;
        tScale = 1.01 + v * 0.11;
        tCyan = 0.88;
        tOrange = 1.0;
        tViolet = 0.22;
        tGlow = 0.45 + v * 0.45;
        tSpread = 1.05 + v * 0.22;
        break;
      }
      case "EXECUTING": {
        tSpeed = 1.7;
        tSwirl = 0.48;
        tBreath = 0.02;
        tScale = 1.02;
        tCyan = 0.98;
        tOrange = 0.96;
        tViolet = 0.28;
        tGlow = 0.52;
        tSpread = 1.1;
        break;
      }
      default:
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

    this.pointer.x = lerp(this.pointer.x, this.pointer.tx, Math.min(1, dt * 5));
    this.pointer.y = lerp(this.pointer.y, this.pointer.ty, Math.min(1, dt * 5));

    this.phase += dt * this.anim.speed;
  }

  render(t) {
    const ctx = this.ctx;
    const S = this.logicalSize;
    ctx.clearRect(0, 0, S, S);

    const floatY = Math.sin(t * 1.35) * 6.5;
    const breath = 1 + Math.sin(t * 2.0) * this.anim.breathAmp;
    const cx = S / 2 + this.pointer.x * 4;
    const cy = S / 2 + floatY + this.pointer.y * 4;
    const R = 192 * this.anim.scale * breath;

    // 1. Soft studio floor shadow & luminous ambient aura
    ctx.save();
    const shadowGrad = ctx.createRadialGradient(cx, cy + R + 36, 8, cx, cy + R + 36, R * 0.96);
    shadowGrad.addColorStop(0, "rgba(15, 35, 70, 0.16)");
    shadowGrad.addColorStop(0.5, "rgba(14, 165, 233, 0.08)");
    shadowGrad.addColorStop(1, "rgba(255, 255, 255, 0)");
    ctx.fillStyle = shadowGrad;
    ctx.beginPath();
    ctx.ellipse(cx, cy + R + 36, R * 0.88, 26, 0, 0, Math.PI * 2);
    ctx.fill();

    // Dual-tone outer halo (icy cyan top-right + warm orange bottom-left)
    const haloCyan = ctx.createRadialGradient(cx + R * 0.25, cy - R * 0.25, R * 0.45, cx, cy, R * 1.36);
    haloCyan.addColorStop(0, `rgba(56, 189, 248, ${this.anim.glowAlpha * 0.42})`);
    haloCyan.addColorStop(1, "rgba(255, 255, 255, 0)");
    ctx.fillStyle = haloCyan;
    ctx.beginPath();
    ctx.arc(cx, cy, R * 1.36, 0, Math.PI * 2);
    ctx.fill();

    const haloOrange = ctx.createRadialGradient(cx - R * 0.25, cy + R * 0.28, R * 0.35, cx, cy, R * 1.3);
    haloOrange.addColorStop(0, `rgba(251, 146, 60, ${this.anim.glowAlpha * 0.28})`);
    haloOrange.addColorStop(1, "rgba(255, 255, 255, 0)");
    ctx.fillStyle = haloOrange;
    ctx.beginPath();
    ctx.arc(cx, cy, R * 1.3, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();

    // 2. Clip inside the 3D Glass Sphere
    ctx.save();
    ctx.beginPath();
    ctx.arc(cx, cy, R, 0, Math.PI * 2);
    ctx.clip();

    // Deep sapphire-navy optical core (gives dramatic 3D hollow-glass depth)
    const coreGrad = ctx.createRadialGradient(
      cx - R * 0.08 + this.pointer.x * 10,
      cy - R * 0.08 + this.pointer.y * 10,
      R * 0.06,
      cx,
      cy,
      R
    );
    coreGrad.addColorStop(0, "#031026");
    coreGrad.addColorStop(0.46, "#061d40");
    coreGrad.addColorStop(0.76, "#0f3f6e");
    coreGrad.addColorStop(0.92, "#5eb8e5");
    coreGrad.addColorStop(1, "#dff2fe");
    ctx.fillStyle = coreGrad;
    ctx.fillRect(cx - R - 4, cy - R - 4, R * 2 + 8, R * 2 + 8);

    // 3. Sculpted 3D-shaded Silk Ribbons sweeping inside the sphere
    const p = this.phase;
    const baseTilt = -0.62 + Math.sin(p * 0.32) * this.anim.swirl + (this.state === "THINKING" ? p * 0.36 : 0);

    ctx.save();
    ctx.translate(cx + this.pointer.x * 6, cy + this.pointer.y * 6);
    ctx.rotate(baseTilt);

    // Layer A: Deep back icy-blue translucent veil
    this._drawSculptedRibbon(ctx, R, p * 0.85, {
      yCenter: -0.44,
      curve: 0.34,
      width: 0.44,
      freq: 1.2,
      stops: [
        [0, `rgba(14, 165, 233, ${0.18 * this.anim.cyanBoost})`],
        [0.45, `rgba(125, 211, 252, ${0.68 * this.anim.cyanBoost})`],
        [0.85, "rgba(240, 249, 255, 0.88)"],
        [1, "rgba(255, 255, 255, 0.2)"],
      ],
      edgeGlow: "rgba(255, 255, 255, 0.55)",
      composite: "screen",
    });

    // Layer B: Optional violet/indigo energy fold (prominent during THINKING)
    if (this.anim.violetBoost > 0.1) {
      this._drawSculptedRibbon(ctx, R, p * 1.15 + 1.6, {
        yCenter: -0.14,
        curve: 0.28,
        width: 0.30,
        freq: 1.55,
        stops: [
          [0, `rgba(79, 70, 229, ${0.15 * this.anim.violetBoost})`],
          [0.5, `rgba(139, 92, 246, ${0.58 * this.anim.violetBoost})`],
          [1, `rgba(56, 189, 248, ${0.42 * this.anim.violetBoost})`],
        ],
        edgeGlow: "rgba(196, 181, 253, 0.45)",
        composite: "screen",
      });
    }

    // Layer C: Deep burnt-orange shadow fold (gives orange ribbon true 3D thickness)
    this._drawSculptedRibbon(ctx, R, p * 0.92 + 0.4, {
      yCenter: 0.34,
      curve: -0.30,
      width: 0.52,
      freq: 1.15,
      stops: [
        [0, `rgba(154, 52, 18, ${0.45 * this.anim.orangeBoost})`],
        [0.45, `rgba(234, 88, 12, ${0.95 * this.anim.orangeBoost})`],
        [0.85, `rgba(249, 115, 22, ${0.90 * this.anim.orangeBoost})`],
        [1, `rgba(254, 215, 170, ${0.75 * this.anim.orangeBoost})`],
      ],
      edgeGlow: "rgba(255, 237, 213, 0.55)",
      composite: "source-over",
    });

    // Layer D: Primary luminous warm-orange silky crest
    this._drawSculptedRibbon(ctx, R, p * 1.05 + 1.1, {
      yCenter: 0.22,
      curve: -0.22,
      width: 0.34,
      freq: 1.35,
      stops: [
        [0, `rgba(255, 247, 237, ${0.35 * this.anim.orangeBoost})`],
        [0.35, `rgba(251, 146, 60, ${0.96 * this.anim.orangeBoost})`],
        [0.72, `rgba(253, 186, 116, ${0.92 * this.anim.orangeBoost})`],
        [1, "rgba(255, 255, 255, 0.85)"],
      ],
      edgeGlow: "rgba(255, 255, 255, 0.72)",
      composite: "source-over",
    });

    // Layer E: Foreground frosted icy-cyan & pearl-white glass petal
    this._drawSculptedRibbon(ctx, R, p * 0.78 + 2.8, {
      yCenter: -0.28,
      curve: 0.26,
      width: 0.36,
      freq: 1.18,
      stops: [
        [0, "rgba(255, 255, 255, 0.25)"],
        [0.42, `rgba(186, 230, 253, ${0.84 * this.anim.cyanBoost})`],
        [0.78, "rgba(240, 249, 255, 0.94)"],
        [1, "rgba(255, 255, 255, 0.35)"],
      ],
      edgeGlow: "rgba(255, 255, 255, 0.85)",
      composite: "screen",
    });

    // Layer F: Crisp thin white-gold crossing filament for intricate depth
    this._drawSculptedRibbon(ctx, R, p * 1.32 + 4.1, {
      yCenter: 0.04,
      curve: -0.14,
      width: 0.14,
      freq: 1.7,
      stops: [
        [0, "rgba(255, 255, 255, 0.08)"],
        [0.5, "rgba(255, 255, 255, 0.78)"],
        [1, "rgba(186, 230, 253, 0.25)"],
      ],
      edgeGlow: "rgba(255, 255, 255, 0.85)",
      composite: "screen",
    });

    ctx.restore();

    // 4. Double-Walled Fresnel Glass Shell & Chromatic Rim Dispersion
    const rimGrad = ctx.createRadialGradient(cx, cy, R * 0.68, cx, cy, R);
    rimGrad.addColorStop(0, "rgba(255, 255, 255, 0)");
    rimGrad.addColorStop(0.75, "rgba(186, 230, 253, 0.14)");
    rimGrad.addColorStop(0.89, "rgba(224, 242, 254, 0.52)");
    rimGrad.addColorStop(0.96, "rgba(255, 255, 255, 0.88)");
    rimGrad.addColorStop(1, "rgba(186, 230, 253, 0.95)");
    ctx.fillStyle = rimGrad;
    ctx.fillRect(cx - R, cy - R, R * 2, R * 2);

    // Subtle warm-peach internal rim reflection on lower-left
    const warmRim = ctx.createRadialGradient(cx - R * 0.55, cy + R * 0.45, 4, cx - R * 0.55, cy + R * 0.45, R * 0.62);
    warmRim.addColorStop(0, `rgba(254, 215, 170, ${0.45 * this.anim.orangeBoost})`);
    warmRim.addColorStop(1, "rgba(255, 255, 255, 0)");
    ctx.fillStyle = warmRim;
    ctx.fillRect(cx - R, cy - R, R * 2, R * 2);

    // Top-left crisp glass specular crescent (follows pointer slightly)
    ctx.save();
    ctx.translate(cx - R * 0.30 + this.pointer.x * 7, cy - R * 0.34 + this.pointer.y * 7);
    ctx.rotate(-0.62);
    const specGrad = ctx.createRadialGradient(0, 0, 2, 0, 0, R * 0.50);
    specGrad.addColorStop(0, "rgba(255, 255, 255, 0.88)");
    specGrad.addColorStop(0.38, "rgba(240, 249, 255, 0.34)");
    specGrad.addColorStop(1, "rgba(255, 255, 255, 0)");
    ctx.fillStyle = specGrad;
    ctx.beginPath();
    ctx.ellipse(0, 0, R * 0.48, R * 0.18, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();

    // Secondary crisp bottom-right specular glint
    ctx.save();
    ctx.translate(cx + R * 0.36, cy + R * 0.42);
    ctx.rotate(-0.55);
    const glintGrad = ctx.createRadialGradient(0, 0, 2, 0, 0, R * 0.34);
    glintGrad.addColorStop(0, "rgba(255, 255, 255, 0.52)");
    glintGrad.addColorStop(1, "rgba(255, 255, 255, 0)");
    ctx.fillStyle = glintGrad;
    ctx.beginPath();
    ctx.ellipse(0, 0, R * 0.32, R * 0.10, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();

    ctx.restore(); // end sphere clip

    // 5. Outer glass lens ring (crisp white + subtle inner glass thickness)
    ctx.save();
    ctx.strokeStyle = "rgba(255, 255, 255, 0.9)";
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    ctx.arc(cx, cy, R - 0.5, 0, Math.PI * 2);
    ctx.stroke();

    ctx.strokeStyle = "rgba(186, 230, 253, 0.55)";
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.arc(cx, cy, R - 4.5, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
  }

  _drawSculptedRibbon(ctx, R, phase, opts) {
    const spread = this.anim.ribbonSpread;
    const yC = opts.yCenter * R * spread;
    const curve = opts.curve * R * spread;
    const wHalf = (opts.width * R * spread) * 0.5;
    const span = R * 1.48;

    const w1 = Math.sin(phase * opts.freq) * (R * 0.14 * spread);
    const w2 = Math.cos(phase * opts.freq * 0.82 + 1.2) * (R * 0.14 * spread);
    const w3 = Math.sin(phase * opts.freq * 1.15 + 2.5) * (R * 0.09 * spread);

    const x0 = -span;
    const x1 = -span * 0.36;
    const x2 = span * 0.36;
    const x3 = span;

    const topY0 = yC + w3 - wHalf * 0.7;
    const topY1 = yC + curve + w1 - wHalf * 1.1;
    const topY2 = yC - curve * 0.6 - w2 - wHalf * 0.9;
    const topY3 = yC - w3 - wHalf * 0.5;

    const botY0 = yC + w3 + wHalf * 0.8;
    const botY1 = yC + curve + w1 + wHalf * 0.95;
    const botY2 = yC - curve * 0.6 - w2 + wHalf * 1.15;
    const botY3 = yC - w3 + wHalf * 0.9;

    ctx.save();
    ctx.globalCompositeOperation = opts.composite || "source-over";

    const grad = ctx.createLinearGradient(x0, yC - wHalf * 1.5, x3, yC + wHalf * 1.5);
    for (const [stop, color] of opts.stops) {
      grad.addColorStop(stop, color);
    }
    ctx.fillStyle = grad;

    ctx.beginPath();
    ctx.moveTo(x0, topY0);
    ctx.bezierCurveTo(x1, topY1, x2, topY2, x3, topY3);
    ctx.lineTo(x3, botY3);
    ctx.bezierCurveTo(x2, botY2, x1, botY1, x0, botY0);
    ctx.closePath();
    ctx.fill();

    // Luminous silk ridge highlight along top edge
    if (opts.edgeGlow) {
      ctx.strokeStyle = opts.edgeGlow;
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      ctx.moveTo(x0, topY0);
      ctx.bezierCurveTo(x1, topY1, x2, topY2, x3, topY3);
      ctx.stroke();
    }

    ctx.restore();
  }
}

window.JarvisOrb = JarvisOrb;
