<script setup lang="ts">
import { ref, onMounted, onUnmounted } from "vue";
import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";

const isRecording = ref(false);
const transcriptEntries = ref<
  Array<{ text: string; timing: string; timestamp: Date }>
>([]);
const statusMessage = ref("Ready to start recording");
const isLoading = ref(false);
const lastTranscriptionTime = ref("");
const droppedSamplesCount = ref(0);
const translatedSamplesCount = ref(0);

let unlistenTranscript: UnlistenFn | null = null;
let droppedSamplesInterval: number | null = null;

onMounted(async () => {
  // Listen for real-time sentance events
  unlistenTranscript = await listen("sentance", (event) => {
    const payload = event.payload as {
      text: string;
      timing_ms: number;
      timing_display: string;
      is_final: boolean;
    };
    const text = payload?.text?.trim();
    if (!text) return;

    lastTranscriptionTime.value = payload.timing_display;
    changeLastTo({
      text: text,
      timing: payload.timing_display,
      timestamp: new Date(),
    });
    if (payload.is_final) {
      transcriptEntries.value.push({
        text: "",
        timing: "",
        timestamp: new Date(),
      });
    }

    // Auto-scroll to bottom
    setTimeout(() => {
      const transcriptList = document.getElementById("transcript-list");
      if (transcriptList) {
        transcriptList.scrollTop = transcriptList.scrollHeight;
      }
    }, 10);
  });
});

onUnmounted(() => {
  if (unlistenTranscript) {
    unlistenTranscript();
  }
  if (droppedSamplesInterval) {
    clearInterval(droppedSamplesInterval);
  }
});

async function toggleRecording() {
  if (isRecording.value) {
    await stopRecording();
  } else {
    await startRecording();
  }
}

async function startRecording() {
  try {
    isLoading.value = true;
    statusMessage.value = "Starting recording...";

    await invoke("start_recording");

    isRecording.value = true;
    statusMessage.value =
      "Recording system audio... Play any audio on your computer";
    transcriptEntries.value = []; // Clear previous transcript
    droppedSamplesCount.value = 0; // Reset dropped samples counter
    translatedSamplesCount.value = 0; // Reset translated samples counter

    // Start periodic updates for sample counts
    droppedSamplesInterval = setInterval(updateSamplesCounts, 1000);
  } catch (error) {
    console.error("Failed to start recording:", error);
    statusMessage.value = `Error: ${error}`;
  } finally {
    isLoading.value = false;
  }
}

async function stopRecording() {
  try {
    isLoading.value = true;
    statusMessage.value = "Stopping recording...";

    await invoke("stop_recording");

    isRecording.value = false;
    statusMessage.value = "Recording stopped";

    // Stop periodic updates
    if (droppedSamplesInterval) {
      clearInterval(droppedSamplesInterval);
      droppedSamplesInterval = null;
    }
  } catch (error) {
    console.error("Failed to stop recording:", error);
    statusMessage.value = `Error: ${error}`;
  } finally {
    isLoading.value = false;
  }
}

function clearTranscript() {
  transcriptEntries.value = [];
  lastTranscriptionTime.value = "";
}

async function updateDroppedSamplesCount() {
  try {
    const count = await invoke("get_dropped_samples_count");
    droppedSamplesCount.value = count as number;
  } catch (error) {
    console.error("Failed to get dropped samples count:", error);
  }
}

async function updateTranslatedSamplesCount() {
  try {
    const count = await invoke("get_translated_samples_count");
    translatedSamplesCount.value = count as number;
  } catch (error) {
    console.error("Failed to get translated samples count:", error);
  }
}

async function updateSamplesCounts() {
  await Promise.all([
    updateDroppedSamplesCount(),
    updateTranslatedSamplesCount(),
  ]);
}

function changeLastTo(entry: {
  text: string;
  timing: string;
  timestamp: Date;
}) {
  if (transcriptEntries.value.length > 0) {
    transcriptEntries.value[transcriptEntries.value.length - 1] = entry;
  } else {
    transcriptEntries.value.push(entry);
  }
}
</script>

<template>
  <main class="container">
    <h1>System Audio Transcription with Whisper</h1>

    <!-- Status Display -->
    <div class="status-section">
      <div
        class="status-indicator"
        :class="{ recording: isRecording, loading: isLoading }"
      >
        <span v-if="isLoading">⏳</span>
        <span v-else-if="isRecording">🔴</span>
        <span v-else>⚪</span>
      </div>
      <p class="status-text">{{ statusMessage }}</p>
    </div>

    <!-- Recording Controls -->
    <div class="controls-section">
      <button
        @click="toggleRecording"
        :disabled="isLoading"
        :class="{
          'btn-record': !isRecording,
          'btn-stop': isRecording,
          'btn-loading': isLoading,
        }"
        class="main-button"
      >
        {{
          isLoading
            ? "Processing..."
            : isRecording
            ? "Stop Recording"
            : "Start Recording"
        }}
      </button>

      <button
        @click="clearTranscript"
        :disabled="isLoading"
        class="secondary-button"
      >
        Clear Transcript
      </button>
    </div>

    <!-- Transcript Display -->
    <div class="transcript-section">
      <div class="transcript-header">
        <h2>Live Transcript</h2>
        <div class="metrics-info">
          <div v-if="lastTranscriptionTime" class="timing-info">
            Last transcription:
            <span class="timing-value">{{ lastTranscriptionTime }}</span>
          </div>
          <div v-if="isRecording" class="samples-counters">
            <div class="translated-samples-info">
              Translated:
              <span class="translated-samples-value">
                {{ translatedSamplesCount.toLocaleString() }}
              </span>
            </div>
            <div class="dropped-samples-info">
              Dropped:
              <span
                class="dropped-samples-value"
                :class="{ warning: droppedSamplesCount > 0 }"
              >
                {{ droppedSamplesCount.toLocaleString() }}
              </span>
            </div>
          </div>
        </div>
      </div>
      <div
        id="transcript-list"
        class="transcript-list"
        v-if="transcriptEntries.length > 0"
      >
        <div
          v-for="(entry, index) in transcriptEntries"
          :key="index"
          class="transcript-entry"
        >
          <div class="transcript-text">{{ entry.text }}</div>
          <div class="transcript-timing">{{ entry.timing }}</div>
        </div>
      </div>
      <div v-else class="transcript-placeholder">
        Your speech will appear here in real-time...
      </div>
    </div>
  </main>
</template>

<style scoped>
.container {
  margin: 0;
  padding: 20px;
  max-width: 800px;
  margin: 0 auto;
  font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
}

h1 {
  text-align: center;
  color: #2c3e50;
  margin-bottom: 30px;
}

.status-section {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 15px;
  margin-bottom: 25px;
  padding: 15px;
  background: #f8f9fa;
  border-radius: 10px;
}

.status-indicator {
  font-size: 20px;
  transition: all 0.3s ease;
}

.status-indicator.recording {
  animation: pulse 1s infinite;
}

.status-indicator.loading {
  animation: rotate 1s linear infinite;
}

@keyframes pulse {
  0%,
  100% {
    opacity: 1;
  }
  50% {
    opacity: 0.5;
  }
}

@keyframes rotate {
  from {
    transform: rotate(0deg);
  }
  to {
    transform: rotate(360deg);
  }
}

.status-text {
  margin: 0;
  font-weight: 500;
  color: #34495e;
}

.controls-section {
  display: flex;
  gap: 15px;
  justify-content: center;
  margin-bottom: 30px;
}

.main-button {
  padding: 12px 30px;
  font-size: 16px;
  font-weight: 600;
  border: none;
  border-radius: 25px;
  cursor: pointer;
  transition: all 0.3s ease;
  min-width: 150px;
}

.btn-record {
  background: linear-gradient(135deg, #27ae60, #2ecc71);
  color: white;
}

.btn-record:hover:not(:disabled) {
  background: linear-gradient(135deg, #219a52, #27ae60);
  transform: translateY(-2px);
}

.btn-stop {
  background: linear-gradient(135deg, #e74c3c, #c0392b);
  color: white;
}

.btn-stop:hover:not(:disabled) {
  background: linear-gradient(135deg, #c0392b, #a93226);
  transform: translateY(-2px);
}

.btn-loading {
  background: #bdc3c7;
  color: #7f8c8d;
  cursor: not-allowed;
}

.secondary-button {
  padding: 10px 20px;
  font-size: 14px;
  background: #ecf0f1;
  color: #2c3e50;
  border: 1px solid #bdc3c7;
  border-radius: 20px;
  cursor: pointer;
  transition: all 0.3s ease;
}

.secondary-button:hover:not(:disabled) {
  background: #d5dbdb;
  transform: translateY(-1px);
}

.secondary-button:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.transcript-section {
  margin-bottom: 30px;
}

.transcript-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 15px;
  flex-wrap: wrap;
  gap: 10px;
}

.metrics-info {
  display: flex;
  flex-direction: column;
  gap: 8px;
  align-items: flex-end;
}

.samples-counters {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}

.transcript-section h2 {
  margin: 0;
  color: #2c3e50;
}

.timing-info {
  font-size: 14px;
  color: #7f8c8d;
  background: #ecf0f1;
  padding: 5px 10px;
  border-radius: 15px;
  border: 1px solid #bdc3c7;
}

.timing-value {
  font-weight: 600;
  color: #2c3e50;
  font-family: "Courier New", monospace;
}

.translated-samples-info,
.dropped-samples-info {
  font-size: 14px;
  color: #7f8c8d;
  background: #ecf0f1;
  padding: 5px 10px;
  border-radius: 15px;
  border: 1px solid #bdc3c7;
}

.translated-samples-value {
  font-weight: 600;
  color: #27ae60;
  font-family: "Courier New", monospace;
}

.dropped-samples-value {
  font-weight: 600;
  color: #27ae60;
  font-family: "Courier New", monospace;
}

.dropped-samples-value.warning {
  color: #e74c3c;
  background: #fdf2f2;
  padding: 2px 6px;
  border-radius: 8px;
}

.transcript-list {
  width: 100%;
  max-height: 400px;
  min-height: 200px;
  padding: 15px;
  border: 2px solid #e0e0e0;
  border-radius: 10px;
  background: #fafafa;
  box-sizing: border-box;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.transcript-placeholder {
  width: 100%;
  height: 200px;
  padding: 15px;
  border: 2px solid #e0e0e0;
  border-radius: 10px;
  background: #fafafa;
  box-sizing: border-box;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #7f8c8d;
  font-style: italic;
}

.transcript-entry {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 15px;
  padding: 10px;
  background: white;
  border: 1px solid #e8e8e8;
  border-radius: 8px;
  transition: all 0.2s ease;
}

.transcript-entry:hover {
  background: #f8f9fa;
  border-color: #d0d0d0;
}

.transcript-text {
  flex: 1;
  font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
  font-size: 14px;
  line-height: 1.4;
  color: #2c3e50;
  word-wrap: break-word;
}

.transcript-timing {
  flex-shrink: 0;
  font-family: "Courier New", monospace;
  font-size: 12px;
  font-weight: 600;
  color: #7f8c8d;
  background: #ecf0f1;
  padding: 3px 8px;
  border-radius: 12px;
  border: 1px solid #bdc3c7;
  white-space: nowrap;
}

.devices-section {
  margin-bottom: 20px;
  padding: 15px;
  background: #fff3cd;
  border: 1px solid #ffeaa7;
  border-radius: 10px;
}

.devices-section h3 {
  margin-top: 0;
  margin-bottom: 10px;
  color: #856404;
}

.devices-section p {
  margin-bottom: 15px;
  color: #856404;
}

.devices-button {
  padding: 8px 16px;
  background: #f39c12;
  color: white;
  border: none;
  border-radius: 5px;
  cursor: pointer;
  transition: background 0.3s ease;
  margin-bottom: 15px;
}

.devices-button:hover {
  background: #e67e22;
}

.device-list {
  background: #f8f9fa;
  border: 1px solid #dee2e6;
  border-radius: 5px;
  padding: 10px;
  margin: 0;
  font-family: "Courier New", monospace;
  font-size: 12px;
  line-height: 1.4;
  max-height: 300px;
  overflow-y: auto;
  white-space: pre-wrap;
}

.setup-warning {
  margin-top: 15px;
  padding: 15px;
  background: #fff3cd;
  border: 1px solid #ffeaa7;
  border-radius: 8px;
  border-left: 4px solid #f39c12;
}

.setup-warning h4 {
  margin-top: 0;
  color: #856404;
}

.setup-warning p {
  margin-bottom: 10px;
  color: #856404;
}

.setup-warning ol {
  color: #856404;
  margin-bottom: 0;
}

.setup-warning a {
  color: #856404;
  text-decoration: underline;
}

.info-section {
  background: #e8f4fd;
  padding: 20px;
  border-radius: 10px;
  border-left: 4px solid #3498db;
}

.info-section p {
  margin-top: 0;
  color: #2c3e50;
  font-weight: 600;
}

.info-section ul {
  color: #34495e;
  line-height: 1.6;
}

.info-section li {
  margin-bottom: 5px;
}

/* Responsive design */
@media (max-width: 600px) {
  .container {
    padding: 15px;
  }

  .controls-section {
    flex-direction: column;
    align-items: center;
  }

  .main-button,
  .secondary-button {
    width: 100%;
    max-width: 250px;
  }

  .transcript-header {
    flex-direction: column;
    align-items: flex-start;
  }

  .metrics-info {
    align-items: stretch;
  }

  .samples-counters {
    justify-content: center;
  }

  .timing-info,
  .translated-samples-info,
  .dropped-samples-info {
    text-align: center;
  }
}
</style>
