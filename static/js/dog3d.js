/**
 * CanineLens - Interactive 3D Dog Mascot (Three.js)
 * Supports GLTF loading or procedural low-poly 3D dog with head tracking,
 * blinking eyes, wagging tail, breathing idle, and animated talking jaw.
 */

class Dog3DMascot {
  constructor(containerId = "dog3dContainer") {
    this.container = document.getElementById(containerId);
    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.dogGroup = null;
    this.headGroup = null;
    this.jawMesh = null;
    this.tongueMesh = null;
    this.leftEar = null;
    this.rightEar = null;
    this.leftEyelid = null;
    this.rightEyelid = null;
    this.tailGroup = null;
    this.frontLeftLeg = null;
    this.frontRightLeg = null;
    this.furMaterial = null;

    this.mixer = null;
    this.clock = null;
    this.animationFrameId = null;
    this.isVisible = true;

    // Mascot State: 'idle' | 'thinking' | 'happy' | 'talking' | 'confused'
    this.state = "idle";
    this.previousState = "idle";

    // Pointer tracking
    this.targetPointerX = 0;
    this.targetPointerY = 0;
    this.currentPointerX = 0;
    this.currentPointerY = 0;

    // Blinking
    this.lastBlinkTime = 0;
    this.isBlinking = false;
    this.blinkDuration = 0.15;

    // Target scale for breed sizing
    this.baseScale = 1.0;
    this.targetScale = 1.0;

    this.isInitialized = false;
  }

  isWebGLAvailable() {
    try {
      const canvas = document.createElement("canvas");
      return !!(window.WebGLRenderingContext && (canvas.getContext("webgl") || canvas.getContext("experimental-webgl")));
    } catch (_) {
      return false;
    }
  }

  async init() {
    if (!this.container) return;
    if (this.isInitialized) return;

    // WebGL Check
    if (!this.isWebGLAvailable() || typeof THREE === "undefined") {
      console.warn("[Dog3D] WebGL or Three.js not available; mounting 2D mascot fallback.");
      this.mount2DFallback();
      return;
    }

    this.clock = new THREE.Clock();
    const width = this.container.clientWidth || 280;
    const height = this.container.clientHeight || 280;

    // 1. Scene
    this.scene = new THREE.Scene();

    // 2. Camera
    this.camera = new THREE.PerspectiveCamera(40, width / height, 0.1, 100);
    this.camera.position.set(0, 1.2, 3.8);
    this.camera.lookAt(0, 0.4, 0);

    // 3. Renderer with antialias and soft shadows
    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      powerPreference: "high-performance"
    });
    this.renderer.setSize(width, height);
    const isMobile = window.innerWidth <= 768;
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, isMobile ? 1.5 : 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    this.container.innerHTML = "";
    this.container.appendChild(this.renderer.domElement);

    // 4. Lighting Setup
    const ambientLight = new THREE.AmbientLight(0xfff6ea, 0.85);
    this.scene.add(ambientLight);

    const keyLight = new THREE.DirectionalLight(0xffffff, 1.0);
    keyLight.position.set(2.5, 4.5, 3.5);
    keyLight.castShadow = true;
    keyLight.shadow.mapSize.width = 512;
    keyLight.shadow.mapSize.height = 512;
    keyLight.shadow.camera.near = 0.5;
    keyLight.shadow.camera.far = 10;
    keyLight.shadow.bias = -0.001;
    this.scene.add(keyLight);

    const rimLight = new THREE.DirectionalLight(0x818cf8, 0.65);
    rimLight.position.set(-3.0, 2.0, -2.0);
    this.scene.add(rimLight);

    // Soft Shadow Ground Plane
    const shadowGeo = new THREE.PlaneGeometry(6, 6);
    const shadowMat = new THREE.ShadowMaterial({ opacity: 0.22 });
    const shadowPlane = new THREE.Mesh(shadowGeo, shadowMat);
    shadowPlane.rotation.x = -Math.PI / 2;
    shadowPlane.position.y = -0.65;
    shadowPlane.receiveShadow = true;
    this.scene.add(shadowPlane);

    // 5. Load GLTF or build procedural dog
    await this.setupModel();

    // 6. Event Listeners
    this.setupEvents();

    this.isInitialized = true;
    this.animate();
  }

  async setupModel() {
    this.dogGroup = new THREE.Group();
    this.scene.add(this.dogGroup);

    // Check if optional static/assets/models/dog.glb exists
    let gltfLoaded = false;
    if (typeof THREE.GLTFLoader !== "undefined") {
      try {
        const headCheck = await fetch("static/assets/models/dog.glb", { method: "HEAD" });
        if (headCheck.ok) {
          const loader = new THREE.GLTFLoader();
          await new Promise((resolve, reject) => {
            loader.load(
              "static/assets/models/dog.glb",
              (gltf) => {
                const model = gltf.scene;
                model.traverse(node => {
                  if (node.isMesh) {
                    node.castShadow = true;
                    node.receiveShadow = true;
                  }
                });
                this.dogGroup.add(model);
                if (gltf.animations && gltf.animations.length > 0) {
                  this.mixer = new THREE.AnimationMixer(model);
                  const action = this.mixer.clipAction(gltf.animations[0]);
                  action.play();
                }
                gltfLoaded = true;
                resolve();
              },
              undefined,
              reject
            );
          });
        }
      } catch (_) {
        gltfLoaded = false;
      }
    }

    if (!gltfLoaded) {
      // Build friendly low-poly 3D dog
      this.buildProceduralDog();
    }
  }

  buildProceduralDog() {
    // Palette
    const furColor = 0xd97736;       // Warm golden-tan fur
    const chestColor = 0xfef3c7;     // Cream chest/snout
    const noseColor = 0x1e293b;      // Dark charcoal nose/eyes
    const pinkColor = 0xf472b6;      // Tongue pink
    const collarColor = 0x6366f1;    // Indigo collar
    const tagColor = 0xf59e0b;       // Gold medal tag

    this.furMaterial = new THREE.MeshLambertMaterial({ color: furColor });
    const chestMaterial = new THREE.MeshLambertMaterial({ color: chestColor });
    const darkMaterial = new THREE.MeshLambertMaterial({ color: noseColor });
    const tongueMaterial = new THREE.MeshLambertMaterial({ color: pinkColor });
    const collarMaterial = new THREE.MeshLambertMaterial({ color: collarColor });
    const tagMaterial = new THREE.MeshStandardMaterial({ color: tagColor, metalness: 0.6, roughness: 0.3 });

    // 1. Torso / Body
    const bodyGeo = new THREE.BoxGeometry(0.85, 0.75, 1.25);
    const bodyMesh = new THREE.Mesh(bodyGeo, this.furMaterial);
    bodyMesh.position.set(0, 0, 0);
    bodyMesh.castShadow = true;
    bodyMesh.receiveShadow = true;

    // Chest patch
    const chestGeo = new THREE.BoxGeometry(0.65, 0.55, 0.4);
    const chestMesh = new THREE.Mesh(chestGeo, chestMaterial);
    chestMesh.position.set(0, -0.08, 0.5);
    chestMesh.castShadow = true;

    // 2. Collar & Tag
    const collarGeo = new THREE.CylinderGeometry(0.38, 0.42, 0.12, 16);
    const collarMesh = new THREE.Mesh(collarGeo, collarMaterial);
    collarMesh.position.set(0, 0.42, 0.48);
    collarMesh.rotation.x = Math.PI * 0.15;
    collarMesh.castShadow = true;

    const tagGeo = new THREE.CylinderGeometry(0.08, 0.08, 0.03, 12);
    const tagMesh = new THREE.Mesh(tagGeo, tagMaterial);
    tagMesh.position.set(0, 0.32, 0.62);
    tagMesh.rotation.x = Math.PI * 0.5;

    // 3. Head Assembly
    this.headGroup = new THREE.Group();
    this.headGroup.position.set(0, 0.65, 0.55);

    // Cranium
    const headGeo = new THREE.BoxGeometry(0.72, 0.65, 0.68);
    const headMesh = new THREE.Mesh(headGeo, this.furMaterial);
    headMesh.castShadow = true;
    headMesh.receiveShadow = true;
    this.headGroup.add(headMesh);

    // Upper Snout
    const snoutGeo = new THREE.BoxGeometry(0.42, 0.28, 0.48);
    const snoutMesh = new THREE.Mesh(snoutGeo, chestMaterial);
    snoutMesh.position.set(0, -0.1, 0.48);
    snoutMesh.castShadow = true;
    this.headGroup.add(snoutMesh);

    // Nose
    const noseGeo = new THREE.BoxGeometry(0.16, 0.12, 0.12);
    const noseMesh = new THREE.Mesh(noseGeo, darkMaterial);
    noseMesh.position.set(0, 0.02, 0.72);
    this.headGroup.add(noseMesh);

    // Eyes
    const eyeGeo = new THREE.SphereGeometry(0.08, 12, 12);
    const leftEye = new THREE.Mesh(eyeGeo, darkMaterial);
    leftEye.position.set(0.22, 0.14, 0.34);
    this.headGroup.add(leftEye);

    const rightEye = new THREE.Mesh(eyeGeo, darkMaterial);
    rightEye.position.set(-0.22, 0.14, 0.34);
    this.headGroup.add(rightEye);

    // Eyelids (for blinking)
    const eyelidGeo = new THREE.BoxGeometry(0.18, 0.1, 0.08);
    this.leftEyelid = new THREE.Mesh(eyelidGeo, this.furMaterial);
    this.leftEyelid.position.set(0.22, 0.23, 0.35);
    this.leftEyelid.scale.set(1, 0.01, 1);
    this.headGroup.add(this.leftEyelid);

    this.rightEyelid = new THREE.Mesh(eyelidGeo, this.furMaterial);
    this.rightEyelid.position.set(-0.22, 0.23, 0.35);
    this.rightEyelid.scale.set(1, 0.01, 1);
    this.headGroup.add(this.rightEyelid);

    // Articulated Jaw (pivot at base of snout)
    const jawPivot = new THREE.Group();
    jawPivot.position.set(0, -0.22, 0.32);

    const jawGeo = new THREE.BoxGeometry(0.38, 0.12, 0.44);
    this.jawMesh = new THREE.Mesh(jawGeo, chestMaterial);
    this.jawMesh.position.set(0, -0.05, 0.18);
    this.jawMesh.castShadow = true;
    jawPivot.add(this.jawMesh);

    // Tongue
    const tongueGeo = new THREE.BoxGeometry(0.2, 0.04, 0.28);
    this.tongueMesh = new THREE.Mesh(tongueGeo, tongueMaterial);
    this.tongueMesh.position.set(0, 0.02, 0.16);
    this.jawMesh.add(this.tongueMesh);

    this.headGroup.add(jawPivot);
    this.jawPivot = jawPivot;

    // Ears (Floppy with pivots)
    const earGeo = new THREE.BoxGeometry(0.18, 0.48, 0.22);

    this.leftEar = new THREE.Mesh(earGeo, this.furMaterial);
    this.leftEar.position.set(0.42, 0.18, 0.0);
    this.leftEar.rotation.z = -Math.PI * 0.12;
    this.leftEar.castShadow = true;
    this.headGroup.add(this.leftEar);

    this.rightEar = new THREE.Mesh(earGeo, this.furMaterial);
    this.rightEar.position.set(-0.42, 0.18, 0.0);
    this.rightEar.rotation.z = Math.PI * 0.12;
    this.rightEar.castShadow = true;
    this.headGroup.add(this.rightEar);

    // 4. Legs & Paws
    const legGeo = new THREE.BoxGeometry(0.22, 0.55, 0.24);
    const pawGeo = new THREE.BoxGeometry(0.24, 0.12, 0.32);

    const makeLeg = (x, y, z) => {
      const legGroup = new THREE.Group();
      legGroup.position.set(x, y, z);
      const legMesh = new THREE.Mesh(legGeo, this.furMaterial);
      legMesh.position.set(0, -0.25, 0);
      legMesh.castShadow = true;
      legMesh.receiveShadow = true;
      legGroup.add(legMesh);

      const pawMesh = new THREE.Mesh(pawGeo, chestMaterial);
      pawMesh.position.set(0, -0.48, 0.05);
      pawMesh.castShadow = true;
      legGroup.add(pawMesh);
      return legGroup;
    };

    this.frontLeftLeg = makeLeg(0.3, -0.15, 0.4);
    this.frontRightLeg = makeLeg(-0.3, -0.15, 0.4);
    const backLeftLeg = makeLeg(0.3, -0.15, -0.4);
    const backRightLeg = makeLeg(-0.3, -0.15, -0.4);

    // 5. Tail
    this.tailGroup = new THREE.Group();
    this.tailGroup.position.set(0, 0.22, -0.6);

    const tailGeo = new THREE.CylinderGeometry(0.06, 0.1, 0.55, 8);
    const tailMesh = new THREE.Mesh(tailGeo, this.furMaterial);
    tailMesh.position.set(0, 0.24, -0.16);
    tailMesh.rotation.x = -Math.PI * 0.28;
    tailMesh.castShadow = true;
    this.tailGroup.add(tailMesh);

    // Assemble Dog Model Group
    this.dogGroup.add(bodyMesh);
    this.dogGroup.add(chestMesh);
    this.dogGroup.add(collarMesh);
    this.dogGroup.add(tagMesh);
    this.dogGroup.add(this.headGroup);
    this.dogGroup.add(this.frontLeftLeg);
    this.dogGroup.add(this.frontRightLeg);
    this.dogGroup.add(backLeftLeg);
    this.dogGroup.add(backRightLeg);
    this.dogGroup.add(this.tailGroup);

    // Subtle initial rotation to face slightly 3/4
    this.dogGroup.position.y = -0.05;
    this.dogGroup.rotation.y = Math.PI * 0.08;
  }

  setupEvents() {
    // Window Resize
    window.addEventListener("resize", () => this.onResize());

    // Pointer Tracking for 3D Head Following
    window.addEventListener("mousemove", (e) => {
      this.targetPointerX = (e.clientX / window.innerWidth) * 2 - 1;
      this.targetPointerY = -(e.clientY / window.innerHeight) * 2 + 1;
    });

    window.addEventListener("touchmove", (e) => {
      if (e.touches && e.touches.length > 0) {
        this.targetPointerX = (e.touches[0].clientX / window.innerWidth) * 2 - 1;
        this.targetPointerY = -(e.touches[0].clientY / window.innerHeight) * 2 + 1;
      }
    }, { passive: true });

    // Page Visibility Optimization
    document.addEventListener("visibilitychange", () => {
      this.isVisible = !document.hidden;
      if (this.isVisible && !this.animationFrameId) {
        this.clock.start();
        this.animate();
      }
    });

    // IntersectionObserver to pause rendering when offscreen
    if ("IntersectionObserver" in window) {
      const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
          this.isVisible = entry.isIntersecting;
        });
      }, { threshold: 0.1 });
      observer.observe(this.container);
    }
  }

  onResize() {
    if (!this.container || !this.renderer || !this.camera) return;
    const width = this.container.clientWidth || 280;
    const height = this.container.clientHeight || 280;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }

  setState(newState) {
    if (this.state === newState) return;
    this.previousState = this.state;
    this.state = newState;

    if (this.jawPivot && newState !== "talking") {
      this.jawPivot.rotation.x = 0;
    }
  }

  adaptToBreed(traits, confidence) {
    if (!traits) return;

    // 1. Scale according to breed size
    const size = (traits.size || "").toLowerCase();
    if (size.includes("toy")) {
      this.targetScale = 0.78;
    } else if (size.includes("small")) {
      this.targetScale = 0.88;
    } else if (size.includes("medium")) {
      this.targetScale = 1.0;
    } else if (size.includes("giant")) {
      this.targetScale = 1.28;
    } else if (size.includes("large")) {
      this.targetScale = 1.16;
    } else {
      this.targetScale = 1.0;
    }

    // 2. Subtle coat tint adjustment
    if (this.furMaterial) {
      const breedName = (traits.breed || "").toLowerCase();
      if (breedName.includes("golden") || breedName.includes("yellow") || breedName.includes("wheat")) {
        this.furMaterial.color.setHex(0xeab308); // Golden
      } else if (breedName.includes("black") || breedName.includes("dark") || breedName.includes("rottweiler") || breedName.includes("doberman")) {
        this.furMaterial.color.setHex(0x334155); // Slate Black
      } else if (breedName.includes("white") || breedName.includes("samoyed") || breedName.includes("maltese") || breedName.includes("west_highland")) {
        this.furMaterial.color.setHex(0xf8fafc); // Snowy White
      } else if (breedName.includes("red") || breedName.includes("irish") || breedName.includes("vizsla")) {
        this.furMaterial.color.setHex(0xb45309); // Reddish Amber
      } else {
        this.furMaterial.color.setHex(0xd97736); // Default warm tan
      }
    }
  }

  animate() {
    if (!this.isVisible) {
      this.animationFrameId = null;
      return;
    }

    this.animationFrameId = requestAnimationFrame(() => this.animate());

    const delta = this.clock.getDelta();
    const time = this.clock.getElapsedTime();

    // GLTF Mixer update
    if (this.mixer) {
      this.mixer.update(delta);
    }

    // Smooth pointer damping
    this.currentPointerX += (this.targetPointerX - this.currentPointerX) * 0.08;
    this.currentPointerY += (this.targetPointerY - this.currentPointerY) * 0.08;

    // Smooth Scale Lerp
    if (this.dogGroup) {
      this.baseScale += (this.targetScale - this.baseScale) * 0.05;
      this.dogGroup.scale.set(this.baseScale, this.baseScale, this.baseScale);
    }

    // Procedural Animation States
    if (this.headGroup && this.tailGroup) {
      this.updateProceduralAnimations(time);
    }

    if (this.renderer && this.scene && this.camera) {
      this.renderer.render(this.scene, this.camera);
    }
  }

  updateProceduralAnimations(time) {
    // 1. Natural Breathing
    const breath = Math.sin(time * 2.2) * 0.02;
    this.dogGroup.position.y = -0.05 + breath;

    // 2. Head Follows Cursor with realistic limits
    const maxHeadYaw = 0.55;
    const maxHeadPitch = 0.35;
    const targetHeadYaw = THREE.MathUtils.clamp(this.currentPointerX * 0.6, -maxHeadYaw, maxHeadYaw);
    const targetHeadPitch = THREE.MathUtils.clamp(-this.currentPointerY * 0.4, -maxHeadPitch, maxHeadPitch);

    // 3. Eye Blinking Logic
    if (time - this.lastBlinkTime > 3.8) {
      this.isBlinking = true;
      this.lastBlinkTime = time;
    }
    if (this.isBlinking) {
      const blinkProgress = (time - this.lastBlinkTime) / this.blinkDuration;
      if (blinkProgress <= 0.5) {
        const h = blinkProgress * 2;
        this.leftEyelid.scale.y = Math.max(0.01, h * 1.8);
        this.rightEyelid.scale.y = Math.max(0.01, h * 1.8);
      } else if (blinkProgress <= 1.0) {
        const h = (1.0 - blinkProgress) * 2;
        this.leftEyelid.scale.y = Math.max(0.01, h * 1.8);
        this.rightEyelid.scale.y = Math.max(0.01, h * 1.8);
      } else {
        this.isBlinking = false;
        this.leftEyelid.scale.y = 0.01;
        this.rightEyelid.scale.y = 0.01;
      }
    }

    // 4. State-Dependent Behaviors
    switch (this.state) {
      case "thinking":
        // Curious head tilt, alert ears
        this.headGroup.rotation.y = targetHeadYaw * 0.4;
        this.headGroup.rotation.x = -0.15;
        this.headGroup.rotation.z = Math.sin(time * 1.5) * 0.05 + 0.22;
        this.tailGroup.rotation.z = Math.sin(time * 4) * 0.18;
        if (this.leftEar && this.rightEar) {
          this.leftEar.rotation.z = -Math.PI * 0.18;
          this.rightEar.rotation.z = Math.PI * 0.18;
        }
        break;

      case "happy":
        // Rapid tail wagging, joyful bouncing
        this.headGroup.rotation.y = targetHeadYaw * 0.6;
        this.headGroup.rotation.x = targetHeadPitch;
        this.headGroup.rotation.z = Math.sin(time * 8) * 0.06;
        this.tailGroup.rotation.z = Math.sin(time * 24) * 0.75;
        this.dogGroup.position.y = -0.05 + Math.abs(Math.sin(time * 7)) * 0.22;
        if (this.jawPivot) {
          this.jawPivot.rotation.x = 0.22 + Math.sin(time * 6) * 0.08;
        }
        break;

      case "talking":
        // Jaw articulation synchronized with speech
        this.headGroup.rotation.y = targetHeadYaw * 0.7;
        this.headGroup.rotation.x = targetHeadPitch + Math.sin(time * 10) * 0.04;
        this.headGroup.rotation.z = Math.sin(time * 3) * 0.04;
        this.tailGroup.rotation.z = Math.sin(time * 14) * 0.45;
        if (this.jawPivot) {
          this.jawPivot.rotation.x = 0.08 + Math.abs(Math.sin(time * 12)) * 0.32;
        }
        break;

      case "confused":
        // Puzzled deep head tilt, drooped ears
        this.headGroup.rotation.y = targetHeadYaw * 0.3;
        this.headGroup.rotation.x = 0.12;
        this.headGroup.rotation.z = -0.32 + Math.sin(time * 1.5) * 0.04;
        this.tailGroup.rotation.z = Math.sin(time * 2) * 0.12;
        if (this.leftEar && this.rightEar) {
          this.leftEar.rotation.z = -Math.PI * 0.05;
          this.rightEar.rotation.z = Math.PI * 0.05;
        }
        break;

      case "idle":
      default:
        // Gentle breathing and calm tail swish
        this.headGroup.rotation.y = targetHeadYaw;
        this.headGroup.rotation.x = targetHeadPitch;
        this.headGroup.rotation.z = Math.sin(time * 1.2) * 0.02;
        this.tailGroup.rotation.z = Math.sin(time * 3.5) * 0.24;
        if (this.leftEar && this.rightEar) {
          this.leftEar.rotation.z = -Math.PI * 0.12 + Math.sin(time * 2) * 0.02;
          this.rightEar.rotation.z = Math.PI * 0.12 - Math.sin(time * 2) * 0.02;
        }
        if (this.jawPivot) {
          this.jawPivot.rotation.x = 0;
        }
        break;
    }
  }

  mount2DFallback() {
    if (!this.container) return;
    this.container.innerHTML = `
      <div class="mascot-2d-fallback" style="text-align:center; padding:1.5rem;">
        <svg viewBox="0 0 120 120" width="120" height="120" style="filter: drop-shadow(0 6px 15px rgba(99,102,241,0.4));">
          <ellipse cx="60" cy="70" rx="35" ry="30" fill="#d97736" />
          <ellipse cx="60" cy="75" rx="20" ry="18" fill="#fef3c7" />
          <circle cx="60" cy="40" r="24" fill="#d97736" />
          <ellipse cx="40" cy="30" rx="8" ry="18" fill="#b45309" transform="rotate(-20 40 30)" />
          <ellipse cx="80" cy="30" rx="8" ry="18" fill="#b45309" transform="rotate(20 80 30)" />
          <ellipse cx="60" cy="46" rx="14" ry="10" fill="#fef3c7" />
          <circle cx="60" cy="42" r="4.5" fill="#1e293b" />
          <circle cx="52" cy="34" r="3.5" fill="#1e293b" />
          <circle cx="68" cy="34" r="3.5" fill="#1e293b" />
          <path d="M56 48 Q60 52 64 48" stroke="#1e293b" stroke-width="2" fill="none" />
        </svg>
        <p style="font-size:0.8rem; color:#94a3b8; margin-top:0.5rem; font-weight:600;">Canine Mascot (2D Mode)</p>
      </div>
    `;
  }

  dispose() {
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
    if (this.scene) {
      this.scene.traverse(object => {
        if (object.geometry) object.geometry.dispose();
        if (object.material) {
          if (Array.isArray(object.material)) {
            object.material.forEach(m => m.dispose());
          } else {
            object.material.dispose();
          }
        }
      });
    }
    if (this.renderer) {
      this.renderer.dispose();
    }
    if (this.container) {
      this.container.innerHTML = "";
    }
    this.isInitialized = false;
  }
}

// Global instance export
window.Dog3DMascot = Dog3DMascot;
