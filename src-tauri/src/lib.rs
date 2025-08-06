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
use std::sync::{Arc, LazyLock, Mutex, OnceLock};
use std::{
    sync::mpsc::{channel, Sender},
    time,
};
use tauri::{AppHandle, Emitter};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

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
    audio_buffer: Vec<f32>,  // Rolling buffer of audio samples
    last_transcript: String, // Keep track of last transcript to detect changes
    last_process_time: time::Instant,

    is_recording: bool,
    stream: Option<SCStream>,
    sentences_buffer: String, // Buffer for accumulating complete sentences
}

// Constants for sliding window
const WINDOW_SIZE_SECONDS: f32 = 10.0; // Process 3 seconds of audio at a time
const WINDOW_STEP_SECONDS: f32 = 1.0; // Process every 0.5 seconds
const MAX_BUFFER_SECONDS: f32 = 30.0; // Keep maximum 10 seconds of audio

static RECORDING_STATE: LazyLock<Arc<Mutex<AudioRecordingState>>> = LazyLock::new(|| {
    Arc::new(Mutex::new(AudioRecordingState {
        audio_buffer: Vec::new(),
        last_transcript: String::new(),
        last_process_time: time::Instant::now(),

        is_recording: false,
        stream: None,
        sentences_buffer: String::new(),
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

// Helper function to detect if text ends with a sentence boundary
fn ends_with_sentence_boundary(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return false;
    }

    // Check for common sentence endings
    trimmed.ends_with('.')
        || trimmed.ends_with('!')
        || trimmed.ends_with('?')
        || trimmed.ends_with("...")
        || trimmed.ends_with(';')
}

// Extract new content from the transcript
fn extract_new_content(current_transcript: &str, last_transcript: &str) -> Option<String> {
    if current_transcript.len() > last_transcript.len() {
        // Find the common prefix
        let common_len = last_transcript
            .chars()
            .zip(current_transcript.chars())
            .take_while(|(a, b)| a == b)
            .count();

        let new_content = &current_transcript[common_len..];
        if !new_content.trim().is_empty() {
            return Some(new_content.to_string());
        }
    }
    None
}

fn transcribe_audio_chunk(audio_data: &[f32]) -> Option<String> {
    // Resample audio from 48kHz to 16kHz
    let resampled_audio = resample_audio(audio_data);

    let model_guard = MODEL.get()?.lock().ok()?;
    let ctx = model_guard.as_ref()?;
    // Create a state
    let mut state = ctx.create_state().ok()?;
    // Create params optimized for real-time transcription
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 0 });
    params.set_n_threads(10);
    params.set_translate(false); // Don't translate, just transcribe
    params.set_language(Some("en"));
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    // Optimize for speed
    params.set_single_segment(false);
    params.set_no_speech_thold(0.6); // Higher threshold to filter out noise
    params.set_suppress_blank(true); // Suppress blank outputs
                                     // Run the model
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

    Some(result.trim().to_string())
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
        state.last_transcript.clear();
        state.sentences_buffer.clear();
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
                            if let Some(transcript) = transcribe_audio_chunk(audio_window) {
                                if !transcript.trim().is_empty() {
                                    // Extract only the new content
                                    if let Some(new_content) = extract_new_content(
                                        &transcript,
                                        &tokio_state.last_transcript,
                                    ) {
                                        println!("New content: {}", new_content);
                                        // Add new content to sentence buffer
                                        tokio_state.sentences_buffer.push_str(&new_content);
                                        // Check if we have complete sentences
                                        if ends_with_sentence_boundary(
                                            &tokio_state.sentences_buffer,
                                        ) {
                                            // Emit the complete sentences
                                            let sentences =
                                                tokio_state.sentences_buffer.trim().to_string();
                                            if !sentences.is_empty() {
                                                if let Err(e) = app_handle_clone
                                                    .emit("audio-transcript", &sentences)
                                                {
                                                    eprintln!(
                                                        "Failed to emit transcript event: {:?}",
                                                        e
                                                    );
                                                } else {
                                                    println!("Sentences emitted: {}", sentences);
                                                }
                                                tokio_state.sentences_buffer.clear();
                                            }
                                        } else if tokio_state.sentences_buffer.len() > 200 {
                                            // Emit long incomplete sentences to avoid too much buffering
                                            let partial =
                                                tokio_state.sentences_buffer.trim().to_string();
                                            if !partial.is_empty() {
                                                if let Err(e) = app_handle_clone
                                                    .emit("audio-transcript", &partial)
                                                {
                                                    eprintln!(
                                                        "Failed to emit transcript event: {:?}",
                                                        e
                                                    );
                                                }
                                                tokio_state.sentences_buffer.clear();
                                            }
                                        }
                                    }

                                    // Update last transcript
                                    tokio_state.last_transcript = transcript;
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
        state.last_transcript.clear();
        state.sentences_buffer.clear();
    } else {
        return Err("Failed to acquire recording state lock".to_string());
    }
    Ok("Recording stopped".to_string())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    init();
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![start_recording, stop_recording,])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
