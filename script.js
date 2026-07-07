const recordBtn = document.getElementById("recordBtn");
const demoResult = document.getElementById("demoResult");

const sampleOrigins = [
  { country: "United Kingdom", confidence: 82 },
  { country: "Australia", confidence: 71 },
  { country: "Nigeria", confidence: 64 },
  { country: "India", confidence: 58 },
  { country: "United States", confidence: 47 },
];

let analyzing = false;

recordBtn.addEventListener("click", () => {
  if (analyzing) return;
  analyzing = true;

  recordBtn.textContent = "🎧 Analyzing...";
  demoResult.textContent = "Listening to speech patterns...";

  setTimeout(() => {
    const pick = sampleOrigins[Math.floor(Math.random() * sampleOrigins.length)];
    demoResult.textContent = `Likely origin: ${pick.country} (${pick.confidence}% confidence)`;
    recordBtn.textContent = "🎙️ Start recording";
    analyzing = false;
  }, 1400);
});
