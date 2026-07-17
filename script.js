const API_BASE = window.VOICEORIGIN_API_BASE || "https://voiceorigin-api.onrender.com";
const RECORD_MS = 8000;

const recordBtn = document.getElementById("recordBtn");
const demoResult = document.getElementById("demoResult");
const demoDetail = document.getElementById("demoDetail");

const sampleOrigins = [
  { country: "United Kingdom", confidence: 82 },
  { country: "Australia", confidence: 71 },
  { country: "Nigeria", confidence: 64 },
  { country: "India", confidence: 58 },
  { country: "United States", confidence: 47 },
];

let busy = false;

recordBtn.addEventListener("click", () => {
  if (busy) return;
  runAnalysis();
});

async function runAnalysis() {
  busy = true;
  clearDetail();

  if (!navigator.mediaDevices || !window.MediaRecorder) {
    demoResult.textContent = "This browser doesn't support microphone recording. Showing a simulated result.";
    await sleep(600);
    showMockResult();
    busy = false;
    return;
  }

  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    demoResult.textContent = "Microphone permission is required. Showing a simulated result.";
    showMockResult();
    busy = false;
    return;
  }

  try {
    recordBtn.textContent = "🎙️ Recording...";
    demoResult.textContent = `Speak now for ${RECORD_MS / 1000} seconds...`;

    const wavBlob = await recordAsWav(stream, RECORD_MS);

    recordBtn.textContent = "🎧 Analyzing...";
    demoResult.textContent = "Analyzing...";

    const result = await sendForAnalysis(wavBlob);
    renderResult(result);
  } catch (err) {
    console.error(err);
    demoResult.textContent = "Couldn't reach the analysis server. Showing a simulated result.";
    showMockResult();
  } finally {
    stream.getTracks().forEach((t) => t.stop());
    recordBtn.textContent = "🎙️ Start recording";
    busy = false;
  }
}

function recordAsWav(stream, durationMs) {
  return new Promise((resolve, reject) => {
    const mimeType = MediaRecorder.isTypeSupported("audio/webm")
      ? "audio/webm"
      : "";
    const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
    const chunks = [];

    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunks.push(e.data);
    };

    recorder.onstop = async () => {
      try {
        const blob = new Blob(chunks, { type: recorder.mimeType });
        const wavBlob = await encodeBlobAsWav(blob);
        resolve(wavBlob);
      } catch (err) {
        reject(err);
      }
    };

    recorder.onerror = (e) => reject(e.error || new Error("recording failed"));

    recorder.start();
    setTimeout(() => recorder.stop(), durationMs);
  });
}

async function encodeBlobAsWav(blob) {
  const arrayBuffer = await blob.arrayBuffer();
  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  const audioCtx = new AudioCtx();
  const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
  const wav = audioBufferToWav(audioBuffer);
  await audioCtx.close();
  return new Blob([wav], { type: "audio/wav" });
}

function audioBufferToWav(audioBuffer) {
  const numChannels = 1; // downmix to mono
  const sampleRate = audioBuffer.sampleRate;
  const channelData =
    audioBuffer.numberOfChannels > 1
      ? mixDown(audioBuffer)
      : audioBuffer.getChannelData(0);

  const samples = channelData.length;
  const bytesPerSample = 2;
  const blockAlign = numChannels * bytesPerSample;
  const dataSize = samples * blockAlign;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeString(view, 8, "WAVE");
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bytesPerSample * 8, true);
  writeString(view, 36, "data");
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let i = 0; i < samples; i++) {
    const s = Math.max(-1, Math.min(1, channelData[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    offset += 2;
  }

  return buffer;
}

function mixDown(audioBuffer) {
  const length = audioBuffer.length;
  const out = new Float32Array(length);
  for (let ch = 0; ch < audioBuffer.numberOfChannels; ch++) {
    const data = audioBuffer.getChannelData(ch);
    for (let i = 0; i < length; i++) out[i] += data[i] / audioBuffer.numberOfChannels;
  }
  return out;
}

function writeString(view, offset, str) {
  for (let i = 0; i < str.length; i++) {
    view.setUint8(offset + i, str.charCodeAt(i));
  }
}

async function sendForAnalysis(wavBlob) {
  const form = new FormData();
  form.append("file", wavBlob, "clip.wav");

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 20000);

  const res = await fetch(`${API_BASE}/analyze`, {
    method: "POST",
    body: form,
    signal: controller.signal,
  });
  clearTimeout(timeout);

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`analyze failed: ${res.status} ${text}`);
  }
  return res.json();
}

function renderResult(result) {
  demoResult.textContent = result.verdict || "Analysis complete";

  const lines = [];
  const step1 = result.step1;
  if (step1) {
    lines.push(
      `Step 1 — ${step1.label === "synthetic" ? "Synthetic" : step1.label === "human" ? "Human" : "Unknown"} (confidence ${step1.confidence}%)`
    );
    if (step1.reasons?.length) {
      lines.push(`  · ${step1.reasons.join(" / ")}`);
    }
  }

  const step2 = result.step2;
  if (step2 && step2.status === "ok") {
    lines.push("Step 2 — Accent estimation");
    step2.distribution.slice(0, 4).forEach((d) => {
      lines.push(`  · ${d.label}: ${d.probability}%`);
    });
    lines.push(`  · confidence: ${step2.confidence_level} (voiced ${step2.voiced_duration_sec}s)`);
  } else if (step2 && step2.status === "insufficient_data") {
    lines.push("Step 2 — Inconclusive (insufficient speech length/voiced content)");
  }

  demoDetail.textContent = lines.join("\n");
}

function showMockResult() {
  const pick = sampleOrigins[Math.floor(Math.random() * sampleOrigins.length)];
  demoResult.textContent = `[Simulated] Likely origin: ${pick.country} (${pick.confidence}% confidence)`;
  demoDetail.textContent = "Example result shown while the backend is not connected.";
}

function clearDetail() {
  demoDetail.textContent = "";
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}
