/**
 * CanineLens - Interactive 3D Frontend & Live Camera Application Logic
 */

document.addEventListener("DOMContentLoaded", () => {
  // ---------------------------------------------------------------------------
  // DOM Elements
  // ---------------------------------------------------------------------------
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  const cameraInput = document.getElementById("cameraInput");
  const browseBtn = document.getElementById("browseBtn");
  const cameraBtn = document.getElementById("cameraBtn");
  
  const previewPanel = document.getElementById("previewPanel");
  const previewImg = document.getElementById("previewImg");
  const previewName = document.getElementById("previewName");
  const previewSize = document.getElementById("previewSize");
  const clearBtn = document.getElementById("clearBtn");
  const gradcamToggle = document.getElementById("gradcamToggle");
  const predictBtn = document.getElementById("predictBtn");

  const loadingIndicator = document.getElementById("loadingIndicator");
  const loadingStatusText = document.getElementById("loadingStatusText");
  const loadingSubtext = document.getElementById("loadingSubtext");

  const resultsSection = document.getElementById("resultsSection");
  const notDogAlert = document.getElementById("notDogAlert");
  const lowConfidenceAlert = document.getElementById("lowConfidenceAlert");
  const demoBanner = document.getElementById("demoBanner");
  const demoBannerText = document.getElementById("demoBannerText");

  const showcaseCard = document.getElementById("showcaseCard");
  const resultMainImage = document.getElementById("resultMainImage");
  const imageSwitchTabs = document.getElementById("imageSwitchTabs");
  const tabOriginal = document.getElementById("tabOriginal");
  const tabGradcam = document.getElementById("tabGradcam");

  const topConfidenceBadge = document.getElementById("topConfidenceBadge");
  const topBreedName = document.getElementById("topBreedName");
  const breedOriginText = document.getElementById("breedOriginText");
  const candidatesList = document.getElementById("candidatesList");

  const traitsCard = document.getElementById("traitsCard");
  const traitSize = document.getElementById("traitSize");
  const traitWeight = document.getElementById("traitWeight");
  const traitHeight = document.getElementById("traitHeight");
  const traitLifespan = document.getElementById("traitLifespan");
  const traitEnergy = document.getElementById("traitEnergy");
  const traitGrooming = document.getElementById("traitGrooming");
  const traitKids = document.getElementById("traitKids");
  const traitTemperament = document.getElementById("traitTemperament");
  const traitDescription = document.getElementById("traitDescription");

  // Camera Modal Elements
  const cameraModal = document.getElementById("cameraModal");
  const cameraBackdrop = document.getElementById("cameraBackdrop");
  const cameraVideo = document.getElementById("cameraVideo");
  const cameraCanvas = document.getElementById("cameraCanvas");
  const capturePhotoBtn = document.getElementById("capturePhotoBtn");
  const switchCameraBtn = document.getElementById("switchCameraBtn");
  const closeCameraBtn = document.getElementById("closeCameraBtn");
  const cancelCameraBtn = document.getElementById("cancelCameraBtn");
  const cameraErrorMessage = document.getElementById("cameraErrorMessage");

  let selectedFile = null;
  let originalImageSrc = null;
  let gradcamImageSrc = null;
  let statusInterval = null;
  let cameraStream = null;
  let currentFacingMode = "environment"; // default to rear camera on mobile, webcam on desktop

  const MAX_FILE_SIZE = 5 * 1024 * 1024;
  const ALLOWED_TYPES = ["image/jpeg", "image/png", "image/webp"];

  // ---------------------------------------------------------------------------
  // Check Health & Demo Mode at Startup
  // ---------------------------------------------------------------------------
  fetch("/api/health")
    .then(r => r.json())
    .then(data => {
      if (data.is_demo_mode) {
        demoBanner.style.display = "flex";
        demoBannerText.textContent = "DEMO MODE ACTIVE: Pre-trained model weights not detected. Simulated outputs will be displayed.";
      }
    })
    .catch(err => console.warn("Health check error:", err));

  // ---------------------------------------------------------------------------
  // File Picker & Drag-and-Drop Handling
  // ---------------------------------------------------------------------------
  browseBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    fileInput.click();
  });

  dropzone.addEventListener("click", () => {
    fileInput.click();
  });

  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });

  ["dragenter", "dragover"].forEach(event => {
    dropzone.addEventListener(event, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add("drag-over");
    });
  });

  ["dragleave", "drop"].forEach(event => {
    dropzone.addEventListener(event, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove("drag-over");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      handleFileSelected(files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileSelected(e.target.files[0]);
    }
  });

  cameraInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileSelected(e.target.files[0]);
    }
  });

  clearBtn.addEventListener("click", () => {
    resetSelection();
  });

  function resetSelection() {
    selectedFile = null;
    originalImageSrc = null;
    gradcamImageSrc = null;
    fileInput.value = "";
    cameraInput.value = "";
    previewPanel.style.display = "none";
    dropzone.style.display = "block";
    resultsSection.style.display = "none";
    // FX HOOK: Reset mascot and sensory effects
    window.CanineFX?.onReset?.();
  }

  function handleFileSelected(file) {
    if (!ALLOWED_TYPES.includes(file.type.toLowerCase())) {
      alert("Invalid file format. Please select a JPG, PNG, or WebP image.");
      return;
    }

    if (file.size > MAX_FILE_SIZE) {
      alert(`File is too large (${(file.size / (1024 * 1024)).toFixed(1)} MB). Maximum allowed size is 5 MB.`);
      return;
    }

    selectedFile = file;

    const reader = new FileReader();
    reader.onload = (e) => {
      originalImageSrc = e.target.result;
      previewImg.src = originalImageSrc;
      previewName.textContent = file.name;
      previewSize.innerHTML = `${(file.size / 1024).toFixed(1)} KB &bull; <span style="text-transform:uppercase">${file.type.split("/")[1]}</span>`;
      
      dropzone.style.display = "none";
      previewPanel.style.display = "block";
      resultsSection.style.display = "none";
    };
    reader.readAsDataURL(file);
  }

  // ---------------------------------------------------------------------------
  // LIVE CAMERA MODAL IMPLEMENTATION (Webcam on Desktop & Camera on Mobile)
  // ---------------------------------------------------------------------------
  cameraBtn.addEventListener("click", async (e) => {
    e.stopPropagation();
    await openLiveCamera();
  });

  async function openLiveCamera() {
    cameraErrorMessage.style.display = "none";
    cameraErrorMessage.textContent = "";

    // Check mediaDevices support
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      // Graceful fallback to file input capture
      cameraInput.click();
      return;
    }

    try {
      // Request camera stream
      const constraints = {
        video: {
          facingMode: { ideal: currentFacingMode },
          width: { ideal: 1280 },
          height: { ideal: 720 }
        },
        audio: false
      };

      cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
      cameraVideo.srcObject = cameraStream;
      await cameraVideo.play();

      cameraModal.style.display = "flex";

      // Detect multiple cameras (e.g. front and rear)
      try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(d => d.kind === "videoinput");
        if (videoDevices.length > 1) {
          switchCameraBtn.style.display = "inline-flex";
        } else {
          switchCameraBtn.style.display = "none";
        }
      } catch (_) {}

    } catch (err) {
      console.warn("Camera access failed:", err);
      // If permission denied or camera unavailable, show message with fallback
      cameraModal.style.display = "flex";
      cameraErrorMessage.style.display = "block";
      cameraErrorMessage.innerHTML = `
        Unable to open camera (${err.name || "Access Error"}). Please ensure camera permissions are allowed in your browser, or 
        <button type="button" id="fallbackFileBtn" class="btn btn-secondary btn-sm" style="margin-top:0.5rem; display:block; width:100%;">
          📁 Select Photo from Device
        </button>
      `;
      document.getElementById("fallbackFileBtn")?.addEventListener("click", () => {
        closeCamera();
        cameraInput.click();
      });
    }
  }

  function closeCamera() {
    if (cameraStream) {
      cameraStream.getTracks().forEach(track => track.stop());
      cameraStream = null;
    }
    cameraVideo.srcObject = null;
    cameraModal.style.display = "none";
  }

  closeCameraBtn.addEventListener("click", closeCamera);
  cancelCameraBtn.addEventListener("click", closeCamera);
  cameraBackdrop.addEventListener("click", closeCamera);

  // Switch / Flip Camera between front & rear
  switchCameraBtn.addEventListener("click", async () => {
    currentFacingMode = (currentFacingMode === "environment" ? "user" : "environment");
    if (cameraStream) {
      cameraStream.getTracks().forEach(track => track.stop());
    }
    await openLiveCamera();
  });

  // Capture Frame from Live Video
  capturePhotoBtn.addEventListener("click", () => {
    if (!cameraVideo.srcObject) return;

    const w = cameraVideo.videoWidth || 640;
    const h = cameraVideo.videoHeight || 480;

    cameraCanvas.width = w;
    cameraCanvas.height = h;
    const ctx = cameraCanvas.getContext("2d");
    ctx.drawImage(cameraVideo, 0, 0, w, h);

    cameraCanvas.toBlob((blob) => {
      if (!blob) {
        alert("Failed to capture image frame.");
        return;
      }
      const capturedFile = new File([blob], `dog_snap_${Date.now()}.jpg`, { type: "image/jpeg" });
      closeCamera();
      handleFileSelected(capturedFile);
    }, "image/jpeg", 0.95);
  });

  // ---------------------------------------------------------------------------
  // PREDICTION FLOW
  // ---------------------------------------------------------------------------
  predictBtn.addEventListener("click", async () => {
    if (!selectedFile) return;

    resultsSection.style.display = "none";
    notDogAlert.style.display = "none";
    lowConfidenceAlert.style.display = "none";
    showcaseCard.style.display = "none";
    traitsCard.style.display = "none";

    previewPanel.style.display = "none";
    loadingIndicator.style.display = "block";
    predictBtn.disabled = true;

    startLoadingCycle(gradcamToggle.checked);

    // FX HOOK: Notify FX controller of classification start (thinking mascot & audio prime)
    window.CanineFX?.onPredictStart?.();

    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("gradcam", gradcamToggle.checked ? "true" : "false");

    try {
      const response = await fetch("/predict", {
        method: "POST",
        body: formData
      });

      clearInterval(statusInterval);
      loadingIndicator.style.display = "none";
      predictBtn.disabled = false;
      previewPanel.style.display = "block";

      if (!response.ok) {
        let errMessage = "Failed to process image.";
        try {
          const errData = await response.json();
          errMessage = errData.detail || errMessage;
        } catch (_) {}
        alert(`Error (${response.status}): ${errMessage}`);
        return;
      }

      const data = await response.json();
      renderResults(data);

    } catch (error) {
      clearInterval(statusInterval);
      loadingIndicator.style.display = "none";
      predictBtn.disabled = false;
      previewPanel.style.display = "block";
      // FX HOOK: Notify FX controller of classification error
      window.CanineFX?.onPredictError?.(error);
      alert(`Network or Server error: ${error.message}`);
    }
  });

  function startLoadingCycle(withGradcam) {
    const steps = [
      { text: "Verifying image structure...", sub: "Performing in-memory Pillow integrity inspection" },
      { text: "Running MobileNetV2 Gatekeeper...", sub: "Evaluating canine anatomical features (ImageNet classes 151-268)" },
      { text: "Classifying 120 Canine Breeds...", sub: "Inference via fine-tuned EfficientNetB0 backbone" }
    ];
    if (withGradcam) {
      steps.push({ text: "Computing Grad-CAM Saliency Maps...", sub: "Extracting gradient activations from top_conv layer" });
    }

    let currentStep = 0;
    loadingStatusText.textContent = steps[0].text;
    loadingSubtext.textContent = steps[0].sub;

    statusInterval = setInterval(() => {
      currentStep = (currentStep + 1) % steps.length;
      loadingStatusText.textContent = steps[currentStep].text;
      loadingSubtext.textContent = steps[currentStep].sub;
    }, 1200);
  }

  function renderResults(data) {
    resultsSection.style.display = "block";

    // FX HOOK: Dispatch prediction results to 3D mascot and speech narration
    window.CanineFX?.onPredictSuccess?.(data);

    if (data.is_dog === false) {
      notDogAlert.style.display = "flex";
      resultsSection.scrollIntoView({ behavior: "smooth" });
      return;
    }

    showcaseCard.style.display = "grid";
    traitsCard.style.display = "block";

    if (data.is_low_confidence) {
      lowConfidenceAlert.style.display = "flex";
    }

    resultMainImage.src = originalImageSrc;
    gradcamImageSrc = data.gradcam_image;

    if (gradcamImageSrc) {
      imageSwitchTabs.style.display = "flex";
      tabOriginal.classList.add("active");
      tabGradcam.classList.remove("active");

      tabOriginal.onclick = () => {
        tabOriginal.classList.add("active");
        tabGradcam.classList.remove("active");
        resultMainImage.src = originalImageSrc;
      };

      tabGradcam.onclick = () => {
        tabGradcam.classList.add("active");
        tabOriginal.classList.remove("active");
        resultMainImage.src = gradcamImageSrc;
      };
    } else {
      imageSwitchTabs.style.display = "none";
    }

    const top = data.top_breed;
    topBreedName.textContent = top.breed_display;
    topConfidenceBadge.textContent = `${top.confidence.toFixed(1)}% Confidence`;

    const traits = data.traits || {};
    breedOriginText.textContent = traits.origin || "Unknown";

    traitSize.textContent = traits.size || "Not specified";
    traitWeight.textContent = traits.weight_range_kg ? `${traits.weight_range_kg} kg` : "Not specified";
    traitHeight.textContent = traits.height_range_cm ? `${traits.height_range_cm} cm` : "Not specified";
    traitLifespan.textContent = traits.lifespan_years ? `${traits.lifespan_years} years` : "Not specified";
    traitEnergy.textContent = traits.energy_level || "Not specified";
    traitGrooming.textContent = traits.grooming_needs || "Not specified";

    if (traits.good_with_kids === true) {
      traitKids.innerHTML = '<span style="color: #34d399;">✓ Yes, family friendly</span>';
    } else if (traits.good_with_kids === false) {
      traitKids.innerHTML = '<span style="color: #f87171;">⚠ Supervised / Adult homes</span>';
    } else {
      traitKids.innerHTML = '<span style="color: var(--text-dim);">Dependent on socialization</span>';
    }

    traitTemperament.innerHTML = "";
    if (Array.isArray(traits.temperament) && traits.temperament.length > 0) {
      traits.temperament.forEach(tag => {
        const span = document.createElement("span");
        span.className = "temperament-pill";
        span.textContent = tag;
        traitTemperament.appendChild(span);
      });
    } else {
      traitTemperament.innerHTML = '<span style="color: var(--text-dim);">Not documented</span>';
    }

    traitDescription.textContent = traits.description || "Detailed kennel club breed description unavailable.";

    candidatesList.innerHTML = "";
    (data.predictions || []).forEach((item, idx) => {
      const row = document.createElement("div");
      row.className = "candidate-row";

      const header = document.createElement("div");
      header.className = "candidate-header";
      header.innerHTML = `
        <span>#${idx + 1} ${item.breed_display}</span>
        <span>${item.confidence.toFixed(1)}%</span>
      `;

      const barBg = document.createElement("div");
      barBg.className = "progress-bar-bg";

      const barFill = document.createElement("div");
      barFill.className = "progress-bar-fill";
      barFill.style.width = "0%";
      setTimeout(() => {
        barFill.style.width = `${Math.min(100, Math.max(2, item.confidence))}%`;
      }, 50);

      barBg.appendChild(barFill);
      row.appendChild(header);
      row.appendChild(barBg);
      candidatesList.appendChild(row);
    });

    resultsSection.scrollIntoView({ behavior: "smooth" });
  }

  // ---------------------------------------------------------------------------
  // INTERACTIVE 3D MOUSE TILT ON CARDS
  // ---------------------------------------------------------------------------
  const tiltCards = document.querySelectorAll(".glass-card, .camera-modal-card");
  tiltCards.forEach(card => {
    card.addEventListener("mousemove", (e) => {
      const rect = card.getBoundingClientRect();
      const x = e.clientX - rect.left - rect.width / 2;
      const y = e.clientY - rect.top - rect.height / 2;
      const rotateX = -(y / (rect.height / 2)) * 6;
      const rotateY = (x / (rect.width / 2)) * 6;
      card.style.transform = `perspective(1000px) rotateX(${rotateX.toFixed(2)}deg) rotateY(${rotateY.toFixed(2)}deg) translateZ(6px)`;
    });

    card.addEventListener("mouseleave", () => {
      card.style.transform = "perspective(1000px) rotateX(0deg) rotateY(0deg) translateZ(0px)";
    });
  });

  // ---------------------------------------------------------------------------
  // 3D FLOATING CONSTELLATION & PARTICLE BACKGROUND CANVAS
  // ---------------------------------------------------------------------------
  init3DBackgroundCanvas();

  function init3DBackgroundCanvas() {
    const canvas = document.getElementById("bgCanvas3D");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    let width = canvas.width = window.innerWidth;
    let height = canvas.height = window.innerHeight;

    window.addEventListener("resize", () => {
      width = canvas.width = window.innerWidth;
      height = canvas.height = window.innerHeight;
    });

    // 3D Particles
    const particles = [];
    const count = 55;
    for (let i = 0; i < count; i++) {
      particles.push({
        x: (Math.random() - 0.5) * width * 1.5,
        y: (Math.random() - 0.5) * height * 1.5,
        z: Math.random() * 800 + 100,
        radius: Math.random() * 2 + 1.2,
        color: i % 3 === 0 ? "rgba(99, 102, 241, " : (i % 3 === 1 ? "rgba(6, 182, 212, " : "rgba(168, 85, 247, "),
        vx: (Math.random() - 0.5) * 0.4,
        vy: (Math.random() - 0.5) * 0.4,
        vz: (Math.random() - 0.5) * 0.8
      });
    }

    let mouseX = 0;
    let mouseY = 0;
    let targetRotX = 0;
    let targetRotY = 0;

    window.addEventListener("mousemove", (e) => {
      mouseX = (e.clientX - width / 2) / (width / 2);
      mouseY = (e.clientY - height / 2) / (height / 2);
      targetRotY = mouseX * 0.15;
      targetRotX = -mouseY * 0.15;
    });

    const fov = 400;

    function render3D() {
      ctx.clearRect(0, 0, width, height);

      const cx = width / 2;
      const cy = height / 2;

      // Project each 3D particle
      for (let i = 0; i < particles.length; i++) {
        const p = particles[i];
        p.x += p.vx;
        p.y += p.vy;
        p.z += p.vz;

        if (p.z < 50) p.z = 850;
        if (p.z > 900) p.z = 60;

        // Apply mouse rotation around Y and X
        const cosY = Math.cos(targetRotY);
        const sinY = Math.sin(targetRotY);
        const xRot = p.x * cosY - p.z * sinY;
        const zRot = p.x * sinY + p.z * cosY;

        const cosX = Math.cos(targetRotX);
        const sinX = Math.sin(targetRotX);
        const yRot = p.y * cosX - zRot * sinX;
        const finalZ = p.y * sinX + zRot * cosX;

        if (finalZ <= -fov + 10) continue;

        const scale = fov / (fov + finalZ);
        const screenX = cx + xRot * scale;
        const screenY = cy + yRot * scale;
        const screenRadius = Math.max(0.5, p.radius * scale);
        const alpha = Math.min(0.85, Math.max(0.1, (1 - finalZ / 900) * 0.7));

        ctx.beginPath();
        ctx.arc(screenX, screenY, screenRadius, 0, Math.PI * 2);
        ctx.fillStyle = `${p.color}${alpha})`;
        ctx.shadowColor = `${p.color}0.8)`;
        ctx.shadowBlur = 8 * scale;
        ctx.fill();

        // Connect nearby points in 3D
        for (let j = i + 1; j < particles.length; j++) {
          const p2 = particles[j];
          const dx = p.x - p2.x;
          const dy = p.y - p2.y;
          const dz = p.z - p2.z;
          const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
          if (dist < 150) {
            const scale2 = fov / (fov + (p2.y * sinX + (p2.x * sinY + p2.z * cosY) * cosX));
            const x2 = cx + (p2.x * cosY - p2.z * sinY) * scale2;
            const y2 = cy + (p2.y * cosX - (p2.x * sinY + p2.z * cosY) * sinX) * scale2;

            ctx.beginPath();
            ctx.moveTo(screenX, screenY);
            ctx.lineTo(x2, y2);
            ctx.strokeStyle = `rgba(99, 102, 241, ${(1 - dist / 150) * 0.18 * scale})`;
            ctx.lineWidth = 1;
            ctx.shadowBlur = 0;
            ctx.stroke();
          }
        }
      }

      requestAnimationFrame(render3D);
    }

    render3D();
  }
});
