/**
 * CanineLens - Master Effects Controller & Sensory FX Engine
 * Orchestrates 3D mascot, Web Speech narration, confetti celebrations,
 * video/canvas parallax, and master Effects On/Off switch.
 */

(function () {
  "use strict";

  class CanineFXController {
    constructor() {
      this.enabled = this.resolveEnabledState();
      this.mascot = null;
      this.speech = null;
      this.confettiCanvas = null;
      this.confettiCtx = null;
      this.confettiParticles = [];
      this.confettiAnimId = null;

      this.mascotWrapper = null;
      this.toggleBtn = null;

      // Initialize on DOM ready
      if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", () => this.init());
      } else {
        this.init();
      }
    }

    resolveEnabledState() {
      // 1. URL Query Param '?fx=off' overrides everything
      const params = new URLSearchParams(window.location.search);
      if (params.get("fx") === "off") {
        return false;
      }
      // 2. localStorage persistence (defaults to true)
      const stored = localStorage.getItem("canine_fx_enabled");
      return stored !== "false";
    }

    init() {
      this.mascotWrapper = document.getElementById("mascotWrapper");
      this.setupToggleUI();

      if (!this.enabled) {
        this.disableAllFX();
        return;
      }

      this.setupMascotAndSpeech();
      this.setupVideoBackground();
      this.setupConfetti();
    }

    setupToggleUI() {
      this.toggleBtn = document.getElementById("fxToggleBtn");
      if (!this.toggleBtn) {
        // Dynamically add to header navigation if not present in HTML
        const nav = document.querySelector(".nav-links");
        if (nav) {
          this.toggleBtn = document.createElement("button");
          this.toggleBtn.type = "button";
          this.toggleBtn.id = "fxToggleBtn";
          this.toggleBtn.className = "fx-toggle-btn";
          nav.appendChild(this.toggleBtn);
        }
      }

      if (this.toggleBtn) {
        this.updateToggleUI();
        this.toggleBtn.addEventListener("click", () => this.toggleFX());
      }
    }

    updateToggleUI() {
      if (!this.toggleBtn) return;
      if (this.enabled) {
        this.toggleBtn.className = "fx-toggle-btn";
        this.toggleBtn.innerHTML = `<span class="fx-indicator-dot"></span><span>Effects: On</span>`;
        this.toggleBtn.setAttribute("aria-label", "Toggle effects off");
        this.toggleBtn.title = "Click to turn off 3D mascot and audio effects";
      } else {
        this.toggleBtn.className = "fx-toggle-btn fx-off";
        this.toggleBtn.innerHTML = `<span class="fx-indicator-dot"></span><span>Effects: Off</span>`;
        this.toggleBtn.setAttribute("aria-label", "Toggle effects on");
        this.toggleBtn.title = "Click to turn on 3D mascot and audio effects";
      }
    }

    toggleFX() {
      this.enabled = !this.enabled;
      localStorage.setItem("canine_fx_enabled", this.enabled ? "true" : "false");
      this.updateToggleUI();

      if (this.enabled) {
        if (this.mascotWrapper) this.mascotWrapper.style.display = "";
        this.setupMascotAndSpeech();
      } else {
        this.disableAllFX();
      }
    }

    disableAllFX() {
      if (this.speech) {
        this.speech.stop();
      }
      if (this.mascot) {
        this.mascot.dispose();
        this.mascot = null;
      }
      if (this.mascotWrapper) {
        this.mascotWrapper.style.display = "none";
      }
      this.stopConfetti();
      const videoBg = document.querySelector(".bg-video-container");
      if (videoBg) videoBg.remove();
    }

    async setupMascotAndSpeech() {
      const container = document.getElementById("dog3dContainer");
      if (!container) return;

      if (!this.mascot && typeof window.Dog3DMascot !== "undefined") {
        this.mascot = new window.Dog3DMascot("dog3dContainer");
        await this.mascot.init();
      }

      if (!this.speech && typeof window.TalkingDogEngine !== "undefined") {
        this.speech = new window.TalkingDogEngine({
          bubbleElement: document.getElementById("speechBubble"),
          captionElement: document.getElementById("speechCaption"),
          replayBtn: document.getElementById("speechReplayBtn"),
          muteBtn: document.getElementById("speechMuteBtn"),
          stopBtn: document.getElementById("speechStopBtn"),
          mascot: this.mascot
        });
      }
    }

    async setupVideoBackground() {
      // Respect prefers-reduced-motion
      if (window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
        return;
      }

      // Check if optional video exists in static/assets/videos/bg.mp4
      try {
        const head = await fetch("/static/assets/videos/bg.mp4", { method: "HEAD" });
        if (head.ok) {
          let container = document.querySelector(".bg-video-container");
          if (!container) {
            container = document.createElement("div");
            container.className = "bg-video-container";
            container.innerHTML = `
              <video class="bg-video-element" autoplay loop muted playsinline>
                <source src="/static/assets/videos/bg.mp4" type="video/mp4">
              </video>
            `;
            document.body.prepend(container);
          }
        }
      } catch (_) {
        // Fallback gracefully to background canvas
      }
    }

    setupConfetti() {
      if (document.getElementById("confettiCanvas")) return;
      this.confettiCanvas = document.createElement("canvas");
      this.confettiCanvas.id = "confettiCanvas";
      document.body.appendChild(this.confettiCanvas);
      this.confettiCtx = this.confettiCanvas.getContext("2d");

      const resizeConfetti = () => {
        if (!this.confettiCanvas) return;
        this.confettiCanvas.width = window.innerWidth;
        this.confettiCanvas.height = window.innerHeight;
      };
      window.addEventListener("resize", resizeConfetti);
      resizeConfetti();
    }

    launchConfetti() {
      if (!this.enabled || !this.confettiCanvas || !this.confettiCtx) return;
      if (window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

      const colors = ["#818cf8", "#ec4899", "#f59e0b", "#10b981", "#38bdf8", "#fbbf24", "#a855f7"];
      this.confettiParticles = [];
      const particleCount = window.innerWidth < 768 ? 60 : 120;

      for (let i = 0; i < particleCount; i++) {
        this.confettiParticles.push({
          x: window.innerWidth * (0.2 + Math.random() * 0.6),
          y: window.innerHeight * 0.45,
          vx: (Math.random() - 0.5) * 16,
          vy: -Math.random() * 14 - 4,
          size: Math.random() * 8 + 4,
          color: colors[Math.floor(Math.random() * colors.length)],
          rotation: Math.random() * 360,
          rotationSpeed: (Math.random() - 0.5) * 12,
          gravity: 0.35,
          opacity: 1,
          isPaw: Math.random() > 0.85
        });
      }

      if (this.confettiAnimId) {
        cancelAnimationFrame(this.confettiAnimId);
      }
      this.animateConfetti();
    }

    animateConfetti() {
      if (!this.confettiCtx || this.confettiParticles.length === 0) {
        this.stopConfetti();
        return;
      }

      this.confettiCtx.clearRect(0, 0, this.confettiCanvas.width, this.confettiCanvas.height);

      for (let i = this.confettiParticles.length - 1; i >= 0; i--) {
        const p = this.confettiParticles[i];
        p.x += p.vx;
        p.y += p.vy;
        p.vy += p.gravity;
        p.vx *= 0.98;
        p.rotation += p.rotationSpeed;
        p.opacity -= 0.007;

        if (p.opacity <= 0 || p.y > this.confettiCanvas.height + 50) {
          this.confettiParticles.splice(i, 1);
          continue;
        }

        this.confettiCtx.save();
        this.confettiCtx.globalAlpha = Math.max(0, p.opacity);
        this.confettiCtx.translate(p.x, p.y);
        this.confettiCtx.rotate((p.rotation * Math.PI) / 180);

        if (p.isPaw) {
          this.confettiCtx.font = `${p.size * 2}px sans-serif`;
          this.confettiCtx.fillText("🐾", -p.size, p.size);
        } else {
          this.confettiCtx.fillStyle = p.color;
          this.confettiCtx.fillRect(-p.size / 2, -p.size / 2, p.size, p.size * 0.6);
        }

        this.confettiCtx.restore();
      }

      if (this.confettiParticles.length > 0) {
        this.confettiAnimId = requestAnimationFrame(() => this.animateConfetti());
      } else {
        this.stopConfetti();
      }
    }

    stopConfetti() {
      if (this.confettiAnimId) {
        cancelAnimationFrame(this.confettiAnimId);
        this.confettiAnimId = null;
      }
      if (this.confettiCtx && this.confettiCanvas) {
        this.confettiCtx.clearRect(0, 0, this.confettiCanvas.width, this.confettiCanvas.height);
      }
      this.confettiParticles = [];
    }

    // --- Hooks called from app.js ---

    primeAudio() {
      if (!this.enabled || !this.speech) return;
      this.speech.primeGesture();
    }

    onPredictStart() {
      if (!this.enabled) return;

      this.primeAudio();

      if (this.mascotWrapper) {
        this.mascotWrapper.style.display = "block";
      }

      if (this.mascot) {
        this.mascot.setState("thinking");
        this.mascot.onResize();
      }

      const caption = document.getElementById("speechCaption");
      const bubble = document.getElementById("speechBubble");
      if (caption && bubble) {
        caption.textContent = "Sniffing out canine features with AI...";
        bubble.style.display = "block";
      }
    }

    onPredictSuccess(data) {
      if (!this.enabled) return;

      if (this.mascotWrapper) {
        this.mascotWrapper.style.display = "block";
      }

      if (this.mascot) {
        this.mascot.onResize();
      }

      // Check prediction outcome
      if (!data || data.is_dog === false) {
        if (this.mascot) this.mascot.setState("confused");
      } else if (data.is_low_confidence || (data.top_breed && data.top_breed.confidence < 50)) {
        if (this.mascot) {
          this.mascot.setState("confused");
          if (data.traits) {
            this.mascot.adaptToBreed(data.traits, data.top_breed ? data.top_breed.confidence : 40);
          }
        }
      } else {
        // Confident dog detection!
        if (this.mascot) {
          this.mascot.setState("happy");
          if (data.traits) {
            this.mascot.adaptToBreed(data.traits, data.top_breed ? data.top_breed.confidence : 80);
          }
        }

        // Celebratory confetti if confidence >= 70%
        if (data.top_breed && data.top_breed.confidence >= 70) {
          this.launchConfetti();
        }
      }

      // Voice narration
      if (this.speech) {
        this.speech.speakPredictionResult(data);
      }
    }

    onPredictError(_err) {
      if (!this.enabled) return;
      if (this.mascot) {
        this.mascot.setState("idle");
      }
      if (this.speech) {
        this.speech.stop();
      }
    }

    onReset() {
      if (!this.enabled) return;
      if (this.mascot) {
        this.mascot.setState("idle");
      }
      if (this.speech) {
        this.speech.stop();
      }
      const bubble = document.getElementById("speechBubble");
      if (bubble) {
        bubble.style.display = "none";
      }
      this.stopConfetti();
    }
  }

  // Instantiate and expose globally
  window.CanineFX = new CanineFXController();
})();
