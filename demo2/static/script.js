let mediaRecorder;
let audioChunks = [];

// Elements
const recordBtn = document.getElementById("recordBtn");
const stopBtn = document.getElementById("stopBtn");
const statusEl = document.getElementById("status");
const transcriptionEl = document.getElementById("transcription");
const emotionEl = document.getElementById("emotion");
const matchesEl = document.getElementById("matches");
const dominantEl = document.getElementById("dominant");

const uploadBtn = document.getElementById("uploadBtn");
const audioFileInput = document.getElementById("audioFile");
const audioPlayer = document.getElementById("audioPlayer");

uploadBtn.addEventListener("click", async () => {
  const file = audioFileInput.files[0];
  if (!file) {
    alert("Please select a file first.");
    return;
  }

  let formData = new FormData();
  formData.append("file", file);

  statusEl.textContent = "⏳ Uploading...";
  statusEl.style.color = "orange";

  try {
    let response = await fetch("/upload_audio", { method: "POST", body: formData });
    let result = await response.json();

    transcriptionEl.innerText = result.transcription || "---";
    emotionEl.innerText = result.emotion || "---";
    dominantEl.innerText = "Dominant State: " + (result.dominant_state || "None");

    matchesEl.innerHTML = "";
    result.matched_words.forEach(m => {
      let li = document.createElement("li");
      li.textContent = `${m.token} → ${m.matched_anchor} (sim: ${m.similarity.toFixed(2)})`;
      matchesEl.appendChild(li);
    });

    stateChart.data.datasets[0].data = [
      result.state_percentages.hypo,
      result.state_percentages.hyper,
      result.state_percentages.flow
    ];
    stateChart.update();

    statusEl.textContent = "✅ File processed!";
    statusEl.style.color = "green";

  } catch (err) {
    console.error("Error uploading audio:", err);
    statusEl.textContent = "❌ Failed to process file";
    statusEl.style.color = "red";
  }
});

audioFileInput.addEventListener("change", () => {
  const file = audioFileInput.files[0];
  if (file) {
    const url = URL.createObjectURL(file);
    audioPlayer.src = url;
    audioPlayer.style.display = "block";  // show the player
  } else {
    audioPlayer.style.display = "none";
  }
});

// Chart setup
let ctx = document.getElementById("stateChart").getContext("2d");
let stateChart = new Chart(ctx, {
  type: "bar",
  data: {
    labels: ["Hypo", "Hyper", "Flow"],
    datasets: [{
      label: "State Percentage",
      data: [0, 0, 0],
      backgroundColor: ["#2196f3", "#f44336", "#4caf50"]
    }]
  },
  options: { scales: { y: { beginAtZero: true, max: 100 } } }
});

// Add a pulse animation for recording
recordBtn.style.transition = "all 0.2s ease";

recordBtn.addEventListener("click", async () => {
  try {
    // Start recording visuals
    recordBtn.textContent = "🔴 Recording...";
    recordBtn.style.backgroundColor = "#f44336";
    statusEl.textContent = "🎙️ Recording...";
    statusEl.style.color = "red";
    recordBtn.disabled = true;
    stopBtn.disabled = false;

    // Get microphone stream
    let stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    audioChunks = [];

    mediaRecorder.ondataavailable = e => audioChunks.push(e.data);

    mediaRecorder.onstop = async () => {
      // Processing visuals
      recordBtn.textContent = "⏳ Processing...";
      recordBtn.style.backgroundColor = "#ff9800";
      statusEl.textContent = "⏳ Processing...";
      statusEl.style.color = "orange";

      let blob = new Blob(audioChunks, { type: mediaRecorder.mimeType || "audio/webm" });
      let formData = new FormData();
      formData.append("audio", blob, "recording.wav");

      try {
        let response = await fetch("/analyze_audio", { method: "POST", body: formData });
        let result = await response.json();

        // Update UI
        transcriptionEl.innerText = result.transcription || "---";
        emotionEl.innerText = result.emotion || "---";
        dominantEl.innerText = "Dominant State: " + (result.dominant_state || "None");

        // Update matches
        matchesEl.innerHTML = "";
        result.matched_words.forEach(m => {
          let li = document.createElement("li");
          li.textContent = `${m.token} → ${m.matched_anchor} (sim: ${m.similarity.toFixed(2)})`;
          matchesEl.appendChild(li);
        });

        // Update chart
        stateChart.data.datasets[0].data = [
          result.state_percentages.hypo,
          result.state_percentages.hyper,
          result.state_percentages.flow
        ];
        stateChart.update();

        // Done visuals
        recordBtn.textContent = "🎙️ Start Recording";
        recordBtn.style.backgroundColor = "#4CAF50";
        statusEl.textContent = "✅ Done!";
        statusEl.style.color = "green";

      } catch (err) {
        console.error("Error sending audio:", err);
        recordBtn.textContent = "🎙️ Start Recording";
        recordBtn.style.backgroundColor = "#4CAF50";
        statusEl.textContent = "❌ Failed to process audio";
        statusEl.style.color = "red";
      }
    };

    mediaRecorder.start();

  } catch (err) {
    console.error("Microphone error:", err);
    statusEl.textContent = "❌ Microphone access denied";
    statusEl.style.color = "red";
  }
});

stopBtn.addEventListener("click", () => {
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
    stopBtn.disabled = true;
    recordBtn.disabled = false;
  }
});
