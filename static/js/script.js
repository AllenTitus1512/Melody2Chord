"use strict";
console.info("[Melody2Chord] Upload JavaScript loaded.");

const form = document.getElementById("upload-form");
const fileInput = document.getElementById("audio-file");
const dropZone = document.getElementById("drop-zone");
const selectedFile = document.getElementById("selected-file");
const analyzeButton = document.getElementById("analyze-button");
const statusMessage = document.getElementById("status-message");
const resultPanel = document.getElementById("result-panel");
const audioPreview = document.getElementById("audio-preview");
const resultFilename = document.getElementById("result-filename");
const resultDuration = document.getElementById("result-duration");
const resultRate = document.getElementById("result-rate");
const melodyPanel = document.getElementById("melody-panel");
const extractButton = document.getElementById("extract-button");
const melodyStatus = document.getElementById("melody-status");
const melodyResults = document.getElementById("melody-results");
const noteList = document.getElementById("note-list");
const noteCount = document.getElementById("note-count");
const melodyDuration = document.getElementById("melody-duration");
const timelineBody = document.getElementById("timeline-body");
const keyPanel = document.getElementById("key-panel");
const detectKeyButton = document.getElementById("detect-key-button");
const keyStatus = document.getElementById("key-status");
const keyResults = document.getElementById("key-results");
const detectedKey = document.getElementById("detected-key");
const scaleNotes = document.getElementById("scale-notes");
const chordPanel = document.getElementById("chord-panel");
const generateButton = document.getElementById("generate-sheet-button");
const chordStatus = document.getElementById("chord-status");
const chordResults = document.getElementById("chord-results");
const chordSheet = document.getElementById("lyrics-chord-sheet");

const allowedExtensions = ["wav", "mp3"];
const maxFileSize = 50 * 1024 * 1024;
let previewUrl = null;
let uploadId = null;
let completedSheet = null;
let audioRevision = 0;
let keyAnalysisForUpload = null;

function setStatus(element, message, type = "") {
  element.textContent = message;
  element.className = `status-message ${type}`;
}

function isValidAudioFile(file) {
  const extension = file.name.split(".").pop().toLowerCase();
  return allowedExtensions.includes(extension) && file.size > 0 && file.size <= maxFileSize;
}

function invalidateCurrentAudio() {
  audioRevision += 1;
  uploadId = null;
  completedSheet = null;
  keyAnalysisForUpload = null;
  resultPanel.hidden = true;
  melodyPanel.hidden = true;
  melodyResults.hidden = true;
  keyPanel.hidden = true;
  keyResults.hidden = true;
  chordPanel.hidden = true;
  chordResults.hidden = true;
  chordSheet.replaceChildren();
  setStatus(chordStatus, "Upload a vocal recording to begin.");
}

function selectAudio(file) {
  invalidateCurrentAudio();
  if (!file) return;
  if (!isValidAudioFile(file)) {
    selectedFile.textContent = "Choose a non-empty WAV or MP3 file smaller than 50 MB.";
    analyzeButton.disabled = true;
    setStatus(statusMessage, "That file is not supported. Please select a WAV or MP3 recording.", "error");
    return;
  }
  selectedFile.textContent = `Selected: ${file.name}`;
  analyzeButton.disabled = false;
  setStatus(statusMessage, "Ready to upload and analyze your recording.");
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  audioPreview.src = previewUrl;
}

fileInput.addEventListener("change", () => {
  const selectedFile = fileInput.files[0];
  console.info("[Melody2Chord] File input change event fired.");
  if (selectedFile) {
    console.info("[Melody2Chord] Selected audio file.", {
      filename: selectedFile.name,
      sizeBytes: selectedFile.size
    });
  } else {
    console.info("[Melody2Chord] File picker returned no selection.");
  }
  selectAudio(selectedFile);
});

["dragenter", "dragover"].forEach(name => dropZone.addEventListener(name, event => {
  event.preventDefault();
  dropZone.classList.add("drag-over");
}));
["dragleave", "drop"].forEach(name => dropZone.addEventListener(name, event => {
  event.preventDefault();
  dropZone.classList.remove("drag-over");
}));
dropZone.addEventListener("drop", event => {
  const [file] = event.dataTransfer.files;
  if (!file) return;
  const transfer = new DataTransfer();
  transfer.items.add(file);
  fileInput.files = transfer.files;
  selectAudio(file);
});

form.addEventListener("submit", async event => {
  event.preventDefault();
  const file = fileInput.files[0];
  if (!file || !isValidAudioFile(file)) {
    console.warn("[Melody2Chord] Upload submit blocked: no supported file is selected.");
    setStatus(statusMessage, "Please select a valid WAV or MP3 file first.", "error");
    return;
  }

  analyzeButton.disabled = true;
  analyzeButton.classList.add("loading");
  const requestRevision = audioRevision;
  console.info("[Melody2Chord] Upload request started.", {
    filename: file.name,
    sizeBytes: file.size,
    endpoint: "/upload"
  });
  setStatus(statusMessage, "Uploading and reading audio details...");
  try {
    const formData = new FormData();
    formData.append("audio", file);
    const response = await fetch("/upload", { method: "POST", body: formData });
    const contentType = response.headers.get("content-type") || "";
    console.info("[Melody2Chord] Upload response received.", {
      status: response.status,
      contentType
    });
    if (!contentType.toLowerCase().includes("application/json")) {
      throw new Error(response.ok
        ? "The server returned an unexpected response while uploading the audio."
        : `Upload failed (HTTP ${response.status}). Please try again.`);
    }
    const payload = await response.json();
    if (requestRevision !== audioRevision) return;
    if (!response.ok) throw new Error(payload.error || "Upload failed. Please try again.");

    uploadId = payload.audio.upload_id;
    console.info("[Melody2Chord] Upload succeeded.", { filename: payload.audio.filename });
    resultFilename.textContent = payload.audio.filename;
    resultDuration.textContent = `${payload.audio.duration.toFixed(2)} seconds`;
    resultRate.textContent = `${payload.audio.sampling_rate.toLocaleString()} Hz`;
    resultPanel.hidden = false;
    melodyPanel.hidden = false;
    chordPanel.hidden = false;
    setStatus(statusMessage, payload.message, "success");
    setStatus(chordStatus, "Ready to generate your chord sheet.");
  } catch (error) {
    if (requestRevision === audioRevision) {
      console.error("[Melody2Chord] Upload failed.", error);
      setStatus(statusMessage, error.message || "Something went wrong while processing the audio.", "error");
    }
  } finally {
    analyzeButton.disabled = false;
    analyzeButton.classList.remove("loading");
  }
});

console.info("[Melody2Chord] Upload UI initialized. Choose a file, then click Upload & analyze to send it.");

extractButton.addEventListener("click", async () => {
  if (!uploadId) {
    setStatus(melodyStatus, "Upload an audio file before extracting its melody.", "error");
    return;
  }
  extractButton.disabled = true;
  extractButton.classList.add("loading");
  const requestUploadId = uploadId;
  const requestRevision = audioRevision;
  melodyResults.hidden = true;
  keyPanel.hidden = true;
  keyResults.hidden = true;
  keyAnalysisForUpload = null;
  setStatus(melodyStatus, "Detecting voiced pitch and smoothing notes…");
  try {
    const response = await fetch("/api/extract-melody", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ upload_id: uploadId })
    });
    const payload = await response.json();
    if (requestRevision !== audioRevision || requestUploadId !== uploadId) return;
    if (!response.ok) throw new Error(payload.error || "Melody extraction failed.");
    renderMelody(payload.melody);
    keyPanel.hidden = false;
    setStatus(melodyStatus, "Melody extracted successfully.", "success");
  } catch (error) {
    if (requestRevision === audioRevision && requestUploadId === uploadId) {
      setStatus(melodyStatus, error.message || "Melody extraction could not be completed.", "error");
    }
  } finally {
    extractButton.disabled = false;
    extractButton.classList.remove("loading");
  }
});

function renderMelody(melody) {
  noteList.replaceChildren(...melody.notes.map(note => {
    const chip = document.createElement("span");
    chip.className = "note-chip";
    chip.textContent = note;
    return chip;
  }));
  noteCount.textContent = `${melody.note_count} ${melody.note_count === 1 ? "note" : "notes"} detected`;
  melodyDuration.textContent = `Melody duration: ${melody.melody_duration.toFixed(2)} seconds`;
  timelineBody.replaceChildren(...melody.timeline.map(item => {
    const row = document.createElement("tr");
    [`${item.start.toFixed(2)}s`, `${item.end.toFixed(2)}s`, item.note, `${item.frequency.toFixed(2)} Hz`]
      .forEach(value => {
        const cell = document.createElement("td");
        cell.textContent = value;
        row.append(cell);
      });
    return row;
  }));
  melodyResults.hidden = false;
}

detectKeyButton.addEventListener("click", async () => {
  if (!uploadId) {
    setStatus(keyStatus, "Upload and extract a melody before detecting its key.", "error");
    return;
  }
  detectKeyButton.disabled = true;
  detectKeyButton.classList.add("loading");
  const requestUploadId = uploadId;
  const requestRevision = audioRevision;
  keyResults.hidden = true;
  setStatus(keyStatus, "Comparing pitch classes with major and minor key profiles…");
  try {
    const response = await fetch("/api/detect-key", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ upload_id: uploadId })
    });
    const payload = await response.json();
    if (requestRevision !== audioRevision || requestUploadId !== uploadId) return;
    if (!response.ok) throw new Error(payload.error || "Key detection failed.");
    keyAnalysisForUpload = { uploadId, analysis: payload.key_analysis };
    renderKey(payload.key_analysis);
    setStatus(keyStatus, "Key detected from the extracted melody.", "success");
  } catch (error) {
    if (requestRevision === audioRevision && requestUploadId === uploadId) {
      keyAnalysisForUpload = null;
      setStatus(keyStatus, error.message || "Key detection could not be completed.", "error");
    }
  } finally {
    detectKeyButton.disabled = false;
    detectKeyButton.classList.remove("loading");
  }
});

function renderKey(analysis) {
  detectedKey.textContent = analysis.key;
  scaleNotes.replaceChildren(...(analysis.scale_notes || []).map(note => {
    const chip = document.createElement("span");
    chip.className = "note-chip";
    chip.textContent = note;
    return chip;
  }));
  keyResults.hidden = false;
}

generateButton.addEventListener("click", async () => {
  if (!uploadId) {
    setStatus(chordStatus, "Upload a vocal recording before generating a chord sheet.", "error");
    return;
  }
  if (generateButton.disabled) return;

  if (completedSheet?.uploadId === uploadId) {
    chordSheet.scrollIntoView({ behavior: "smooth", block: "center" });
    setStatus(chordStatus, "Chord sheet generated successfully.", "success");
    return;
  }

  generateButton.disabled = true;
  generateButton.classList.add("loading");
  const requestUploadId = uploadId;
  const requestRevision = audioRevision;
  setStatus(chordStatus, "Generating chord sheet...");
  try {
    const response = await fetch("/api/generate-chord-sheet", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        upload_id: requestUploadId,
        musical_key: keyAnalysisForUpload?.uploadId === requestUploadId
          ? keyAnalysisForUpload.analysis.key
          : null
      })
    });
    const payload = await response.json();
    if (requestRevision !== audioRevision || requestUploadId !== uploadId) return;
    if (!response.ok || !payload.success) {
      throw new Error(payload.error || "Chord sheet generation failed. Please try again.");
    }
    if (!payload.alignment?.lines?.some(line => line.chords?.length)) {
      throw new Error("No chord predictions could be aligned with the timed lyrics.");
    }

    renderChordSheet(payload.alignment);
    if (payload.key_analysis && keyAnalysisForUpload?.uploadId !== requestUploadId) {
      keyAnalysisForUpload = { uploadId: requestUploadId, analysis: payload.key_analysis };
      keyPanel.hidden = false;
      renderKey(payload.key_analysis);
      setStatus(keyStatus, "Key detected from the extracted melody.", "success");
    }
    chordResults.hidden = false;
    completedSheet = { uploadId: requestUploadId, alignment: payload.alignment };
    console.info("[Melody2Chord] Whisper language detection", {
      language: payload.detected_language,
      probability: payload.language_probability,
      confidence: payload.language_confidence
    });
    setStatus(chordStatus, "Chord sheet generated successfully.", "success");
  } catch (error) {
    if (requestRevision === audioRevision && requestUploadId === uploadId) {
      setStatus(chordStatus, error.message || "Chord sheet generation failed. Please try again.", "error");
    }
  } finally {
    generateButton.disabled = false;
    generateButton.classList.remove("loading");
  }
});

function renderChordSheet(alignment) {
  const lineNodes = (alignment.lines || []).map(line => {
    const lineNode = document.createElement("div");
    lineNode.className = "chord-sheet-line";
    const chordsByWord = new Map();
    (line.chords || []).forEach(event => {
      if (!chordsByWord.has(event.word_index)) chordsByWord.set(event.word_index, []);
      chordsByWord.get(event.word_index).push(event);
    });

    line.words.forEach((item, index) => {
      const token = document.createElement("span");
      token.className = "chord-sheet-word";
      const labels = document.createElement("span");
      labels.className = "sheet-chords";
      (chordsByWord.get(index) || []).forEach(event => {
        const label = document.createElement("span");
        label.className = "sheet-chord";
        label.textContent = event.chord;
        labels.append(label);
      });
      const lyric = document.createElement("span");
      lyric.className = "sheet-lyric";
      lyric.textContent = item.word;
      token.append(labels, lyric);
      lineNode.append(token);
    });
    return lineNode;
  });
  chordSheet.replaceChildren(...lineNodes);
}
