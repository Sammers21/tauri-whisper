// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/

// mod audio; // Commented out - code moved inline to avoid conflicts

use core_foundation::error::CFError;
use core_media_rs::cm_sample_buffer::CMSampleBuffer;

use screencapturekit::{
    shareable_content::SCShareableContent,
    stream::{
        configuration::SCStreamConfiguration, content_filter::SCContentFilter,
        output_trait::SCStreamOutputTrait, output_type::SCStreamOutputType, SCStream,
    },
};
use serde::Serialize;
use std::sync::{Arc, LazyLock, Mutex, OnceLock};
use std::{
    sync::mpsc::{channel, Sender},
    time,
};
use tauri::{AppHandle, Emitter};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

#[derive(Serialize)]
struct TranscriptEvent {
    text: String,
    timing_ms: f64,
    timing_display: String,
}

fn format_timing(seconds: f64) -> String {
    if seconds >= 1.0 {
        format!("{:.3}s", seconds)
    } else {
        format!("{:.0}ms", seconds * 1000.0)
    }
}

// Audio resampling constants
const SOURCE_SAMPLE_RATE: u32 = 48000;
const TARGET_SAMPLE_RATE: u32 = 16000;
const RESAMPLE_RATIO: f32 = TARGET_SAMPLE_RATE as f32 / SOURCE_SAMPLE_RATE as f32;

struct AudioStreamOutput {
    sender: Sender<CMSampleBuffer>,
}

impl SCStreamOutputTrait for AudioStreamOutput {
    fn did_output_sample_buffer(
        &self,
        sample_buffer: CMSampleBuffer,
        _of_type: SCStreamOutputType,
    ) {
        self.sender
            .send(sample_buffer)
            .expect("could not send to output_buffer");
    }
}

static MODEL: OnceLock<Mutex<Option<WhisperContext>>> = OnceLock::new();

// Simple linear interpolation resampling from 48kHz to 16kHz
fn resample_audio(input: &[f32]) -> Vec<f32> {
    let output_len = (input.len() as f32 * RESAMPLE_RATIO) as usize;
    let mut output = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let src_idx = i as f32 / RESAMPLE_RATIO;
        let idx = src_idx as usize;

        if idx + 1 < input.len() {
            // Linear interpolation between samples
            let frac = src_idx - idx as f32;
            let sample = input[idx] * (1.0 - frac) + input[idx + 1] * frac;
            output.push(sample);
        } else if idx < input.len() {
            output.push(input[idx]);
        }
    }

    output
}

// Audio recording control state
struct AudioRecordingState {
    audio_buffer: Vec<f32>, // Rolling buffer of audio samples
    last_process_time: time::Instant,
    dropped_samples_count: u64, // Track dropped samples due to buffer overflow
    translated_samples_count: u64, // Track samples that were successfully processed

    is_recording: bool,
    stream: Option<SCStream>,
}

// Constants for sliding window
const WINDOW_SIZE_SECONDS: f32 = 15.0; // Process 3 seconds of audio at a time
const WINDOW_STEP_SECONDS: f32 = 10.0; // Process every 0.5 seconds
const MAX_BUFFER_SECONDS: f32 = 20.0; // Keep maximum 10 seconds of audi0

static RECORDING_STATE: LazyLock<Arc<Mutex<AudioRecordingState>>> = LazyLock::new(|| {
    Arc::new(Mutex::new(AudioRecordingState {
        audio_buffer: Vec::new(),
        last_process_time: time::Instant::now(),
        dropped_samples_count: 0,
        translated_samples_count: 0,

        is_recording: false,
        stream: None,
    }))
});

fn init() {
    let mut context_param = WhisperContextParameters::default();
    context_param.dtw_parameters.mode = whisper_rs::DtwMode::ModelPreset {
        model_preset: whisper_rs::DtwModelPreset::LargeV3,
    };
    // Enable GPU if available for better performance
    context_param.use_gpu = true;

    let model_path = "/Users/sammers/Git/my/tauri-whisper/ggml-large-v3.bin"; // Path to your Whisper model file
    let ctx =
        WhisperContext::new_with_params(&model_path, context_param).expect("failed to load model");
    MODEL.get_or_init(|| Mutex::new(Some(ctx)));
}

fn get_stream(tx: Sender<CMSampleBuffer>) -> Result<SCStream, CFError> {
    let config = SCStreamConfiguration::new().set_captures_audio(true)?;
    let display = SCShareableContent::get().unwrap().displays().remove(0);
    let filter = SCContentFilter::new().with_display_excluding_windows(&display, &[]);
    let mut stream = SCStream::new(&filter, &config);
    stream.add_output_handler(AudioStreamOutput { sender: tx }, SCStreamOutputType::Audio);
    Ok(stream)
}

fn transcribe_audio_chunk(audio_data: &[f32]) -> Option<(String, f64)> {
    let start_time = time::Instant::now();
    // Resample audio from 48kHz to 16kHz
    let resampled_audio = resample_audio(audio_data);
    let model_guard = MODEL.get()?.lock().ok()?;
    let ctx = model_guard.as_ref()?;
    // Create a state
    let mut state = ctx.create_state().ok()?;
    // Create params optimized for real-time transcription
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 0 });
    params.set_translate(false); // Don't translate, just transcribe
    params.set_language(Some("en"));
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    // Optimize for speed
    params.set_single_segment(true);
    params.set_suppress_blank(false);
    if state.full(params, &resampled_audio).is_err() {
        return None;
    }
    // Get the transcript
    let num_segments = state.full_n_segments().ok()?;
    let mut result = String::new();
    for i in 0..num_segments {
        if let Ok(segment) = state.full_get_segment_text(i) {
            result.push_str(&segment);
            result.push(' ');
        }
    }

    let elapsed_time = start_time.elapsed().as_secs_f64();
    let timing_display = format_timing(elapsed_time);
    println!("Transcription took {}", timing_display);

    Some((result.trim().to_string(), elapsed_time))
}

#[tauri::command]
async fn start_recording(app_handle: AppHandle) -> Result<String, String> {
    if let Ok(mut state) = RECORDING_STATE.lock() {
        let (tx, rx) = channel();
        let stream = get_stream(tx).unwrap();
        stream.start_capture().unwrap();
        state.stream = Some(stream);
        state.is_recording = true;
        state.last_process_time = time::Instant::now();
        state.audio_buffer.clear();
        state.dropped_samples_count = 0; // Reset counter on new recording
        state.translated_samples_count = 0; // Reset translated samples counter
                                            // Clone the app_handle to move into the spawned task
        let app_handle_clone = app_handle.clone();
        tokio::spawn(async move {
            while let Ok(sample_buffer) = rx.recv() {
                if let Ok(mut tokio_state) = RECORDING_STATE.lock() {
                    let mut all_samples = Vec::new();
                    let buffer_list = sample_buffer.get_audio_buffer_list().expect("should work");

                    // Process all audio buffers and mix them properly
                    for buffer_index in 0..buffer_list.num_buffers() {
                        let buffer = buffer_list.get(buffer_index).expect("should work");
                        let data_slice = buffer.data();
                        let sample_count = data_slice.len() / 4;
                        let channels = buffer.number_channels as usize;

                        // Process samples from this buffer
                        for i in 0..sample_count / channels {
                            let mut mixed_sample = 0.0f32;

                            // Mix all channels together (mono downmix)
                            for ch in 0..channels {
                                let idx = (i * channels + ch) * 4;
                                if idx + 3 < data_slice.len() {
                                    let sample_bytes = [
                                        data_slice[idx],
                                        data_slice[idx + 1],
                                        data_slice[idx + 2],
                                        data_slice[idx + 3],
                                    ];
                                    let sample = f32::from_le_bytes(sample_bytes);
                                    mixed_sample += sample / channels as f32;
                                }
                            }

                            all_samples.push(mixed_sample);
                        }
                    }
                    // Add samples to the rolling buffer
                    tokio_state.audio_buffer.extend(all_samples);
                    // Keep buffer size under control
                    let max_samples = (SOURCE_SAMPLE_RATE as f32 * MAX_BUFFER_SECONDS) as usize;
                    if tokio_state.audio_buffer.len() > max_samples {
                        let excess = tokio_state.audio_buffer.len() - max_samples;
                        tokio_state.dropped_samples_count += excess as u64;
                        println!(
                            "ERROR: Dropping {} samples (total dropped: {})",
                            excess, tokio_state.dropped_samples_count
                        );
                        tokio_state.audio_buffer.drain(0..excess);
                    }
                    // Process with sliding window at regular intervals
                    if tokio_state.last_process_time.elapsed().as_secs_f32() >= WINDOW_STEP_SECONDS
                    {
                        let window_samples =
                            (SOURCE_SAMPLE_RATE as f32 * WINDOW_SIZE_SECONDS) as usize;
                        // Only process if we have enough audio
                        if tokio_state.audio_buffer.len() >= window_samples {
                            // Take the last WINDOW_SIZE_SECONDS of audio
                            let start_idx = tokio_state
                                .audio_buffer
                                .len()
                                .saturating_sub(window_samples);
                            let audio_window = &tokio_state.audio_buffer[start_idx..];
                            println!("Processing window of {} samples", audio_window.len());
                            if let Some((transcript, timing_seconds)) =
                                transcribe_audio_chunk(audio_window)
                            {
                                // Count translated samples (successful processing)
                                tokio_state.translated_samples_count += audio_window.len() as u64;

                                if !transcript.trim().is_empty() {
                                    let transcript_event = TranscriptEvent {
                                        text: transcript,
                                        timing_ms: timing_seconds * 1000.0,
                                        timing_display: format_timing(timing_seconds),
                                    };
                                    if let Err(e) =
                                        app_handle_clone.emit("audio-transcript", &transcript_event)
                                    {
                                        eprintln!("Failed to emit transcript event: {:?}", e);
                                    } else {
                                        println!(
                                            "Transcript emitted: {} (took {})",
                                            transcript_event.text, transcript_event.timing_display
                                        );
                                    }
                                }
                            }
                            tokio_state.last_process_time = time::Instant::now();
                        }
                    }
                }
            }
        });
    } else {
        return Err("Failed to acquire recording state lock".to_string());
    }
    Ok("System audio recording started successfully".to_string())
}

#[tauri::command]
async fn stop_recording() -> Result<String, String> {
    if let Ok(mut state) = RECORDING_STATE.lock() {
        if let Some(stream) = state.stream.take() {
            stream.stop_capture().unwrap();
        }
        state.is_recording = false;
        state.stream = None;
        state.audio_buffer.clear();
    } else {
        return Err("Failed to acquire recording state lock".to_string());
    }
    Ok("Recording stopped".to_string())
}

#[tauri::command]
async fn get_dropped_samples_count() -> Result<u64, String> {
    if let Ok(state) = RECORDING_STATE.lock() {
        Ok(state.dropped_samples_count)
    } else {
        Err("Failed to acquire recording state lock".to_string())
    }
}

#[tauri::command]
async fn get_translated_samples_count() -> Result<u64, String> {
    if let Ok(state) = RECORDING_STATE.lock() {
        Ok(state.translated_samples_count)
    } else {
        Err("Failed to acquire recording state lock".to_string())
    }
}

#[tauri::command]
async fn list_audio_devices() -> Result<String, String> {
    // For macOS with ScreenCaptureKit, system audio capture is built-in
    Ok(
        "✅ System audio capture available via ScreenCaptureKit\nNo additional setup required."
            .to_string(),
    )
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    init();
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            start_recording,
            stop_recording,
            get_dropped_samples_count,
            get_translated_samples_count,
            list_audio_devices,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
