/**
 * CanineLens - Model Info Dynamic Metrics Loader
 */

document.addEventListener("DOMContentLoaded", async () => {
  const statTop1 = document.getElementById("statTop1");
  const statTop5 = document.getElementById("statTop5");
  const statTestImages = document.getElementById("statTestImages");
  const statTestLoss = document.getElementById("statTestLoss");

  const annTop1 = document.getElementById("annTop1");
  const annTop5 = document.getElementById("annTop5");
  const cnnTop1 = document.getElementById("cnnTop1");
  const cnnTop5 = document.getElementById("cnnTop5");
  const effTop1 = document.getElementById("effTop1");
  const effTop5 = document.getElementById("effTop5");

  const barAnn = document.getElementById("barAnn");
  const barAnnLabel = document.getElementById("barAnnLabel");
  const barCnn = document.getElementById("barCnn");
  const barCnnLabel = document.getElementById("barCnnLabel");
  const barEff = document.getElementById("barEff");
  const barEffLabel = document.getElementById("barEffLabel");

  let data = null;
  try {
    const res = await fetch("/api/info");
    if (res.ok) {
      data = await res.json();
    }
  } catch (_) {}

  // Fallback for static hosting like GitHub Pages
  if (!data) {
    try {
      const resStatic = await fetch("models/metrics.json");
      if (resStatic.ok) {
        data = await resStatic.json();
      }
    } catch (_) {}
  }

  // Built-in benchmark fallback from Stanford Dogs test evaluation
  if (!data) {
    data = {
      model_name: "EfficientNetB0 (transfer learning)",
      dataset: "Stanford Dogs (120 breeds)",
      test_images: 8580,
      top1_accuracy: 0.8415,
      top5_accuracy: 0.9840,
      test_loss: 0.5264,
      ann_top1_accuracy: 0.0409,
      ann_top5_accuracy: 0.1703,
      cnn_top1_accuracy: 0.1089,
      cnn_top5_accuracy: 0.3260
    };
  }

  try {
    // Populate Top Statistics
    const top1Pct = (data.top1_accuracy * 100.0).toFixed(2);
    const top5Pct = (data.top5_accuracy * 100.0).toFixed(2);
    const annTop1Pct = (data.ann_top1_accuracy * 100.0).toFixed(2);
    const annTop5Pct = (data.ann_top5_accuracy * 100.0).toFixed(2);
    const cnnTop1Pct = (data.cnn_top1_accuracy * 100.0).toFixed(2);
    const cnnTop5Pct = (data.cnn_top5_accuracy * 100.0).toFixed(2);

    statTop1.textContent = `${top1Pct}%`;
    statTop5.textContent = `${top5Pct}%`;
    statTestImages.textContent = data.test_images.toLocaleString();
    statTestLoss.textContent = data.test_loss ? data.test_loss.toFixed(4) : "0.5264";

    // Populate Comparison Table
    annTop1.textContent = `${annTop1Pct}%`;
    annTop5.textContent = `${annTop5Pct}%`;

    cnnTop1.textContent = `${cnnTop1Pct}%`;
    cnnTop5.textContent = `${cnnTop5Pct}%`;

    effTop1.textContent = `${top1Pct}%`;
    effTop5.textContent = `${top5Pct}%`;

    // Animate Visual Comparison Bars
    barAnnLabel.textContent = `${annTop1Pct}%`;
    barCnnLabel.textContent = `${cnnTop1Pct}%`;
    barEffLabel.textContent = `${top1Pct}%`;

    setTimeout(() => {
      barAnn.style.width = `${annTop1Pct}%`;
      barCnn.style.width = `${cnnTop1Pct}%`;
      barEff.style.width = `${top1Pct}%`;
    }, 100);

  } catch (error) {
    console.error("Error populating model metrics:", error);
    statTop1.textContent = "84.15%";
    statTop5.textContent = "98.40%";
    statTestImages.textContent = "8,580";
    statTestLoss.textContent = "0.5264";
  }
});
