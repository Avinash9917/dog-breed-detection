/**
 * CanineLens - Talking Dog Speech Engine & WebAudio Synthesizer
 * Uses browser-native Web Speech API & Web Audio API (100% free, client-side).
 */

class TalkingDogEngine {
  constructor(options = {}) {
    this.bubbleElement = options.bubbleElement || document.getElementById("speechBubble");
    this.captionElement = options.captionElement || document.getElementById("speechCaption");
    this.replayBtn = options.replayBtn || document.getElementById("speechReplayBtn");
    this.muteBtn = options.muteBtn || document.getElementById("speechMuteBtn");
    this.stopBtn = options.stopBtn || document.getElementById("speechStopBtn");
    this.mascot = options.mascot || null;

    this.synth = window.speechSynthesis || null;
    this.audioCtx = null;
    this.selectedVoice = null;
    this.isMuted = localStorage.getItem("canine_speech_muted") === "true";
    this.lastScriptText = "";
    this.currentUtterance = null;

    this.initAudioContext();
    this.initVoice();
    this.setupUI();
  }

  initAudioContext() {
    try {
      const AudioCtxClass = window.AudioContext || window.webkitAudioContext;
      if (AudioCtxClass) {
        this.audioCtx = new AudioCtxClass();
      }
    } catch (_) {}
  }

  initVoice() {
    if (!this.synth) {
      if (this.bubbleElement) {
        // Speech API unsupported, keep speech bubble for captions only
        if (this.replayBtn) this.replayBtn.style.display = "none";
        if (this.muteBtn) this.muteBtn.style.display = "none";
        if (this.stopBtn) this.stopBtn.style.display = "none";
      }
      return;
    }

    const selectBestVoice = () => {
      const voices = this.synth.getVoices();
      if (!voices || voices.length === 0) return;

      // Priority: en-IN -> en-GB -> en-US -> generic English
      const enIN = voices.find(v => v.lang === "en-IN" || v.lang.startsWith("en-IN"));
      const enGB = voices.find(v => v.lang === "en-GB" || v.lang.startsWith("en-GB"));
      const enUS = voices.find(v => v.lang === "en-US" || v.lang.startsWith("en-US"));
      const enAny = voices.find(v => v.lang.startsWith("en"));

      this.selectedVoice = enIN || enGB || enUS || enAny || voices[0];
    };

    selectBestVoice();
    if (this.synth.onvoiceschanged !== undefined) {
      this.synth.onvoiceschanged = selectBestVoice;
    }
  }

  setupUI() {
    this.updateMuteButtonUI();

    if (this.muteBtn) {
      this.muteBtn.addEventListener("click", () => {
        this.isMuted = !this.isMuted;
        localStorage.setItem("canine_speech_muted", this.isMuted ? "true" : "false");
        this.updateMuteButtonUI();
        if (this.isMuted) {
          this.stop();
        }
      });
    }

    if (this.replayBtn) {
      this.replayBtn.addEventListener("click", () => {
        if (this.lastScriptText) {
          this.speakText(this.lastScriptText);
        }
      });
    }

    if (this.stopBtn) {
      this.stopBtn.addEventListener("click", () => {
        this.stop();
      });
    }
  }

  updateMuteButtonUI() {
    if (!this.muteBtn) return;
    if (this.isMuted) {
      this.muteBtn.innerHTML = `<span>🔇</span><span>Unmute</span>`;
      this.muteBtn.classList.add("active-muted");
    } else {
      this.muteBtn.innerHTML = `<span>🔊</span><span>Mute</span>`;
      this.muteBtn.classList.remove("active-muted");
    }
  }

  primeGesture() {
    // Resume AudioContext and trigger silent utterance on user interaction
    if (this.audioCtx && this.audioCtx.state === "suspended") {
      this.audioCtx.resume();
    }
    if (this.synth && !this.synth.speaking) {
      const silent = new SpeechSynthesisUtterance("");
      silent.volume = 0;
      this.synth.speak(silent);
    }
  }

  playPuppyBark() {
    if (this.isMuted || !this.audioCtx) return;
    try {
      if (this.audioCtx.state === "suspended") {
        this.audioCtx.resume();
      }

      const now = this.audioCtx.currentTime;
      const osc = this.audioCtx.createOscillator();
      const gain = this.audioCtx.createGain();

      osc.type = "sine";
      // Dual-tone bark pitch sweep: 440Hz -> 680Hz -> 280Hz
      osc.frequency.setValueAtTime(440, now);
      osc.frequency.exponentialRampToValueAtTime(720, now + 0.08);
      osc.frequency.exponentialRampToValueAtTime(320, now + 0.22);

      gain.gain.setValueAtTime(0.01, now);
      gain.gain.linearRampToValueAtTime(0.35, now + 0.05);
      gain.gain.exponentialRampToValueAtTime(0.001, now + 0.25);

      osc.connect(gain);
      gain.connect(this.audioCtx.destination);

      osc.start(now);
      osc.stop(now + 0.26);
    } catch (_) {}
  }

  buildSpeechScript(predictData) {
    // 1. Not A Dog Case
    if (!predictData || predictData.is_dog === false) {
      return "Hmm, that doesn't look like a dog to me! Please show me a clearer photo of a pup!";
    }

    const top = predictData.top_breed || {};
    const breedName = top.breed_display || top.breed || "Canine";
    const conf = Math.round(top.confidence || 0);
    const traits = predictData.traits || {};

    // 2. Low Confidence Case (<50%)
    if (predictData.is_low_confidence || conf < 50) {
      const p1 = (predictData.predictions && predictData.predictions[0]) ? predictData.predictions[0].breed_display : breedName;
      const p2 = (predictData.predictions && predictData.predictions[1]) ? predictData.predictions[1].breed_display : null;
      if (p2) {
        return `Woof! I'm not totally sure, this might be a mixed breed. My top guesses are ${p1} and ${p2}!`;
      }
      return `Woof! I'm not totally sure, this might be a mixed breed. My closest guess is a ${p1}!`;
    }

    // 3. High/Medium Confidence Valid Dog Case (Strict non-null fields only)
    const sentences = [];
    sentences.push(`Woof! I think this is a ${breedName}, about ${conf} percent sure!`);

    // Size & Weight
    const parts = [];
    if (traits.size) {
      parts.push(`a ${traits.size} dog`);
    }
    if (traits.weight_range_kg) {
      parts.push(`weighing around ${traits.weight_range_kg} kilograms`);
    }
    if (parts.length > 0) {
      sentences.push(`I'm ${parts.join(", ")}.`);
    }

    // Origin
    if (traits.origin) {
      sentences.push(`My breed originated in ${traits.origin}.`);
    }

    // Temperament
    if (Array.isArray(traits.temperament) && traits.temperament.length > 0) {
      const topTraits = traits.temperament.slice(0, 3).join(", ");
      sentences.push(`Known for being ${topTraits}.`);
    }

    // Lifespan & Kids
    if (traits.lifespan_years) {
      sentences.push(`We typically live for ${traits.lifespan_years} years.`);
    }

    return sentences.join(" ");
  }

  speakPredictionResult(predictData) {
    const script = this.buildSpeechScript(predictData);
    this.lastScriptText = script;

    // Show Speech Bubble with sanitized textContent
    if (this.captionElement) {
      this.captionElement.textContent = script;
    }
    if (this.bubbleElement) {
      this.bubbleElement.style.display = "block";
    }

    // Bark sound on valid dog result
    if (predictData && predictData.is_dog) {
      this.playPuppyBark();
    }

    this.speakText(script);
  }

  speakText(text) {
    if (!this.synth) return;

    // Cancel any current utterance
    this.synth.cancel();

    if (this.isMuted || !text) {
      if (this.mascot && this.mascot.setState) {
        this.mascot.setState("happy");
      }
      return;
    }

    const utterance = new SpeechSynthesisUtterance(text);
    if (this.selectedVoice) {
      utterance.voice = this.selectedVoice;
    }

    // Cute mascot pitch and friendly speaking rate
    utterance.pitch = 1.15;
    utterance.rate = 1.02;

    utterance.onstart = () => {
      if (this.mascot && this.mascot.setState) {
        this.mascot.setState("talking");
      }
    };

    utterance.onend = () => {
      if (this.mascot && this.mascot.setState) {
        this.mascot.setState("happy");
      }
    };

    utterance.onerror = (e) => {
      console.warn("Speech error:", e);
      if (this.mascot && this.mascot.setState) {
        this.mascot.setState("idle");
      }
    };

    this.currentUtterance = utterance;
    this.synth.speak(utterance);
  }

  stop() {
    if (this.synth) {
      this.synth.cancel();
    }
    if (this.mascot && this.mascot.setState) {
      this.mascot.setState("idle");
    }
  }
}

// Global instance export
window.TalkingDogEngine = TalkingDogEngine;
