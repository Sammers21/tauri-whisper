<script setup lang="ts">
import { ref, onMounted, onUnmounted } from "vue";
import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";

const isRecording = ref(false);
const transcriptText = ref("");
const statusMessage = ref("Ready to start recording");
const isLoading = ref(false);
const deviceList = ref("");

let unlistenTranscript: UnlistenFn | null = null;

onMounted(async () => {
  // Listen for real-time transcript events
  unlistenTranscript = await listen("audio-transcript", (event) => {
    const newText = event.payload as string;
    if (newText && newText.trim()) {
      transcriptText.value += newText + " ";
      // Auto-scroll to bottom
      setTimeout(() => {
        const textarea = document.getElementById(
          "transcript-area"
        ) as HTMLTextAreaElement;
        if (textarea) {
          textarea.scrollTop = textarea.scrollHeight;
        }
      }, 10);
    }
  });
});

onUnmounted(() => {
  if (unlistenTranscript) {
    unlistenTranscript();
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
    transcriptText.value = ""; // Clear previous transcript
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
  } catch (error) {
    console.error("Failed to stop recording:", error);
    statusMessage.value = `Error: ${error}`;
  } finally {
    isLoading.value = false;
  }
}

function clearTranscript() {
  transcriptText.value = "";
}

async function listAudioDevices() {
  try {
    const devices = await invoke("list_audio_devices");
    deviceList.value = devices as string;
  } catch (error) {
    deviceList.value = `Error: ${error}`;
  }
}

function getAudioSetupStatus() {
  if (!deviceList.value) {
    return "Click 'List Audio Devices' to check setup";
  }

  if (deviceList.value.includes("✅ System audio capture available")) {
    return "✅ System audio capture ready - no setup required";
  }

  if (
    deviceList.value.includes("BlackHole") ||
    deviceList.value.includes("loopback") ||
    deviceList.value.includes("monitor")
  ) {
    return "✅ System audio device detected";
  }

  return "⚠️ Using fallback device - may capture microphone instead";
}

async function greet() {
  // Keep the original greet function for testing
  try {
    const message = await invoke("greet", { name: "Vue" });
    statusMessage.value = message as string;
  } catch (error) {
    statusMessage.value = `Error: ${error}`;
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
      <h2>Live Transcript</h2>
      <textarea
        id="transcript-area"
        v-model="transcriptText"
        readonly
        placeholder="Your speech will appear here in real-time..."
        class="transcript-area"
      ></textarea>
    </div>

    <!-- Audio Devices Section -->
    <div class="devices-section">
      <h3>Audio Setup</h3>
      <p><strong>Current Status:</strong> {{ getAudioSetupStatus() }}</p>
      <button @click="listAudioDevices" class="devices-button">
        List Audio Devices
      </button>
      <pre v-if="deviceList" class="device-list">{{ deviceList }}</pre>

      <div
        class="setup-warning"
        v-if="
          deviceList &&
          !deviceList.includes('✅ System audio capture available') &&
          !deviceList.includes('BlackHole') &&
          !deviceList.includes('loopback') &&
          !deviceList.includes('monitor')
        "
      >
        <h4>⚠️ System Audio Capture Not Available</h4>
        <p>
          The app is using a fallback device which may capture microphone audio
          instead of system audio.
        </p>
        <p><strong>To improve system audio capture:</strong></p>
        <ul>
          <li>
            <strong>macOS:</strong> App should automatically use
            ScreenCaptureKit (requires permission)
          </li>
          <li>
            <strong>Windows:</strong> App should automatically use WASAPI
            loopback
          </li>
          <li><strong>Linux:</strong> Uses PulseAudio monitor sources</li>
          <li>
            <strong>Alternative:</strong> Install a virtual audio device like
            BlackHole (macOS)
          </li>
        </ul>
      </div>
    </div>

    <!-- Test Section -->
    <div class="test-section">
      <h3>Test Connection</h3>
      <button @click="greet" class="test-button">Test Greet</button>
    </div>

    <!-- Info Section -->
    <div class="info-section">
      <p><strong>Instructions:</strong></p>
      <ul>
        <li>
          <strong>System Audio:</strong> Click "Start Recording" to capture all
          computer audio (like OBS)
        </li>
        <li>
          <strong>Permission:</strong> May prompt for screen recording
          permission on macOS
        </li>
        <li>
          <strong>Testing:</strong> Play music, videos, or any audio while
          recording
        </li>
        <li>Click "List Audio Devices" to see available capture methods</li>
        <li>Transcription appears in real-time every 500ms</li>
        <li>Check the browser console for detailed audio capture logs</li>
        <li>Click "Stop Recording" when finished</li>
      </ul>
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

.transcript-section h2 {
  margin-bottom: 15px;
  color: #2c3e50;
}

.transcript-area {
  width: 100%;
  height: 200px;
  padding: 15px;
  border: 2px solid #e0e0e0;
  border-radius: 10px;
  font-family: "Courier New", monospace;
  font-size: 14px;
  line-height: 1.5;
  resize: vertical;
  background: #fafafa;
  box-sizing: border-box;
}

.transcript-area:focus {
  outline: none;
  border-color: #3498db;
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

.test-section {
  margin-bottom: 30px;
  padding: 15px;
  background: #ecf0f1;
  border-radius: 10px;
}

.test-section h3 {
  margin-top: 0;
  margin-bottom: 10px;
  color: #2c3e50;
}

.test-button {
  padding: 8px 16px;
  background: #3498db;
  color: white;
  border: none;
  border-radius: 5px;
  cursor: pointer;
  transition: background 0.3s ease;
}

.test-button:hover {
  background: #2980b9;
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
}
</style>
