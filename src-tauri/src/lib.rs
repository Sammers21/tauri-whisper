// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/

// mod audio; // Commented out - code moved inline to avoid conflicts

use core_foundation::error::CFError;
use core_media_rs::cm_sample_buffer::CMSampleBuffer;
use regex::Regex;
use ringbuf::{HeapRb, Rb};
use strsim::normalized_levenshtein;

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

// Constants for audio processing
const SOURCE_SAMPLE_RATE: u32 = 48000;
const TARGET_SAMPLE_RATE: u32 = 16000;
const SLIDING_WINDOW_SIZE: f32 = 30.0;
const RESAMPLE_RATIO: f32 = TARGET_SAMPLE_RATE as f32 / SOURCE_SAMPLE_RATE as f32;
const WINDOW_SAMPLES: usize = (SLIDING_WINDOW_SIZE * TARGET_SAMPLE_RATE as f32) as usize;

// Whisper model
static MODEL: OnceLock<Mutex<Option<WhisperContext>>> = OnceLock::new();

// Audio recording state
static RECORDING_STATE: LazyLock<Arc<Mutex<AudioRecordingState>>> = LazyLock::new(|| {
    // Calculate ring buffer capacity for resampled audio (16kHz)
    let buffer_capacity = (TARGET_SAMPLE_RATE as f32 * SLIDING_WINDOW_SIZE) as usize;
    Arc::new(Mutex::new(AudioRecordingState {
        audio_buffer: AudioRingBuffer::new(buffer_capacity),
        dropped_samples_count: 0,
        translated_samples_count: 0,
        all_sentances: Vec::new(),
        is_recording: false,
        stream: None,
    }))
});

fn perform_transcription_update_ui(app_handle: &AppHandle) {
    let (audio_window, mut current_sentances) = {
        let mut window = Vec::with_capacity(WINDOW_SAMPLES);
        let mut current_sentances: Vec<String> = Vec::new();
        if let Ok(state) = RECORDING_STATE.lock() {
            window = state.audio_buffer.get_latest_samples(WINDOW_SAMPLES);
            current_sentances = state.all_sentances.clone();
        }
        (window, current_sentances)
    };
    if audio_window.is_empty() {
        return;
    }
    let Some((text, timing_seconds)) = transcribe_audio_chunk(&audio_window) else {
        return;
    };
    if text.trim().is_empty() {
        return;
    }
    let events = generate_new_sentance_events(text.clone(), &mut current_sentances, timing_seconds);
    if events.is_empty() {
        return;
    }
    for event in &events {
        println!("Emitting sentance event: {:?}", event);
        if let Err(e) = app_handle.emit("sentance", event) {
            eprintln!("Failed to emit sentance event: {:?}", e);
        }
    }
    if let Ok(mut state) = RECORDING_STATE.lock() {
        state.all_sentances = current_sentances;
    }
}

fn process_sample_buffer(sample_buffer: &CMSampleBuffer) {
    let all_samples = extract_audio_samples_from_buffer(sample_buffer);
    if let Ok(mut state) = RECORDING_STATE.lock() {
        state.audio_buffer.extend_resampled(&all_samples);
    }
}

fn build_sentance_event(text: String, timing_seconds: f64, index: usize) -> SentanceEvent {
    SentanceEvent {
        text,
        timing_ms: timing_seconds * 1000.0,
        timing_display: format_timing(timing_seconds),
        index: index,
    }
}

fn generate_new_sentance_events(
    newer_transcript: String,
    current_sentances: &mut Vec<String>,
    timing_seconds: f64,
) -> Vec<SentanceEvent> {
    let newer_sentances = split_transcript_into_sentances(newer_transcript);
    let last_sentance_opt = newer_sentances.last().cloned();
    let old_last_opt = current_sentances.last().cloned();
    // Safely get the pre-last element only if there are at least 2 items
    let old_pre_last_opt = current_sentances
        .len()
        .checked_sub(2)
        .and_then(|idx| current_sentances.get(idx))
        .cloned();
    if let Some(last_sentance) = last_sentance_opt {
        if let Some(old_last) = old_last_opt {
            let (is_common_middle, common_part_direct) =
                common_middle_part(last_sentance.clone(), old_last.clone());
            let pre_last_plus_last_match = old_pre_last_opt.as_ref().map(|old_pre_last| {
                common_middle_part(
                    format!("{} {}", old_pre_last, old_last),
                    last_sentance.clone(),
                )
            });
            if is_common_middle {
                current_sentances.pop();
                current_sentances.push(last_sentance.clone());
                let event = build_sentance_event(
                    last_sentance.clone(),
                    timing_seconds,
                    current_sentances.len() - 1,
                );
                println!("Case 1: new last + old last have common middle '{}', so we replace the old last with the new one", common_part_direct);
                return vec![event];
            } else if let Some((true, common_part_combo)) = pre_last_plus_last_match {
                current_sentances.pop();
                current_sentances.pop();
                current_sentances.push(last_sentance.clone());
                let event = build_sentance_event(
                    last_sentance.clone(),
                    timing_seconds,
                    current_sentances.len() - 1,
                );
                println!("Case 2: new last + old last+pre_last have common middle '{}', so we replace old last+pre_last with the new one", common_part_combo);
                return vec![event];
            } else {
                current_sentances.push(last_sentance.clone());
                let event = build_sentance_event(
                    last_sentance.clone(),
                    timing_seconds,
                    current_sentances.len() - 1,
                );
                println!("Case 3: new last + old last and old last+pre_last have no common middle, so we emit the new last");
                return vec![event];
            }
        } else {
            current_sentances.push(last_sentance.clone());
            let event = build_sentance_event(
                last_sentance.clone(),
                timing_seconds,
                current_sentances.len() - 1,
            );
            println!("Case 4: no old last, so we emit the new last");
            return vec![event];
        }
    }
    Vec::new()
}

// returns if the two strings have the same middle part that is more than one character(not including spaces)
fn common_middle_part(new: String, old: String) -> (bool, String) {
    let threshold = 0.75;
    let mut explanation = String::new();
    let score = normalized_levenshtein(&new, &old);
    explanation.push_str(&format!(
        "CASE 1: The normalized levenshtein score between '{}' and '{}' is {}\n",
        new, old, score
    ));
    if score < threshold {
        // Compare only up to the shorter length using char boundaries to avoid UTF-8 slicing issues
        let new_chars_count = new.chars().count();
        let old_chars_count = old.chars().count();
        let min_chars = new_chars_count.min(old_chars_count);
        let new_cut: String = new.chars().take(min_chars).collect();
        let old_cut: String = old.chars().take(min_chars).collect();
        let score = normalized_levenshtein(&new_cut, &old_cut);
        explanation.push_str(&format!(
            "CASE 2: The normalized levenshtein score between '{}' and '{}' is {}\n",
            new_cut, old_cut, score
        ));
        if score < threshold {
            // Compare suffixes of equal char length using char-safe slicing
            let new_cut_from_end: String = new
                .chars()
                .skip(new_chars_count.saturating_sub(min_chars))
                .collect();
            let old_cut_from_end: String = old
                .chars()
                .skip(old_chars_count.saturating_sub(min_chars))
                .collect();
            let score = normalized_levenshtein(&new_cut_from_end, &old_cut_from_end);
            explanation.push_str(&format!(
                "CASE 3: The normalized levenshtein score between '{}' and '{}' is {}\n",
                new_cut_from_end, old_cut_from_end, score
            ));
            if score < threshold {
                return (false, explanation);
            } else {
                return (true, explanation);
            }
        } else {
            return (true, explanation);
        }
    } else {
        return (true, explanation);
    }
}

// Example: "Hello, how are you? I'm doing well, thank you." -> ["Hello, how are you?", "I'm doing well, thank you."]
// Example: "'...open other facilities, rare earth buildings,' was split into ["...open other facilities, rare earth buildings,"]
// Just a '.' or '?' or '!' or '…' is enough to split the sentance
fn split_transcript_into_sentances(transcript: String) -> Vec<String> {
    let mut result: Vec<String> = Vec::new();
    // Match minimally up to a sentence terminator (. ? ! …), then include any trailing closing quotes/brackets
    // Use DOTALL so we can span across newlines if present
    let re = Regex::new(r#"(?s)\s*(.+?[\.!\?…]+(?:[\"'”’»)\]}]+)?)"#).expect("invalid regex");

    let mut last_end: usize = 0;
    for caps in re.captures_iter(&transcript) {
        if let Some(m) = caps.get(0) {
            last_end = m.end();
        }
        if let Some(m) = caps.get(1) {
            let mut s = m.as_str().trim().to_string();
            if (s.starts_with('"') && s.ends_with('"'))
                || (s.starts_with('\'') && s.ends_with('\''))
            {
                if s.len() >= 2 {
                    s = s[1..s.len() - 1].to_string();
                }
            }
            if !s.is_empty() {
                result.push(s);
            }
        }
    }

    let tail = transcript[last_end..].trim();
    if !tail.is_empty() {
        let mut s = tail.to_string();
        if (s.starts_with('"') && s.ends_with('"')) || (s.starts_with('\'') && s.ends_with('\'')) {
            if s.len() >= 2 {
                s = s[1..s.len() - 1].to_string();
            }
        }
        result.push(s);
    }

    println!("Transcript: '{}' was split into {:?}", transcript, result);
    result
}

#[derive(Serialize, Clone, Debug)]
struct SentanceEvent {
    text: String,
    timing_ms: f64,
    timing_display: String,
    index: usize,
}

fn format_timing(seconds: f64) -> String {
    if seconds >= 1.0 {
        format!("{:.3}s", seconds)
    } else {
        format!("{:.0}ms", seconds * 1000.0)
    }
}

// Audio ring buffer wrapper for convenience
struct AudioRingBuffer {
    buffer: HeapRb<f32>,
}

impl AudioRingBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            buffer: HeapRb::new(capacity),
        }
    }

    fn extend_resampled(&mut self, samples: &[f32]) {
        // Resample on-the-go while adding to ring buffer
        let resampled = resample_audio(samples);

        for sample in resampled {
            // Use push_overwrite which automatically handles buffer overflow
            // We don't track drops here - drops should be tracked when samples
            // don't make it to the model, not during resampling
            self.buffer.push_overwrite(sample);
        }
    }

    fn len(&self) -> usize {
        self.buffer.len()
    }

    fn clear(&mut self) {
        self.buffer.clear();
    }

    fn get_latest_samples(&self, count: usize) -> Vec<f32> {
        let total_len = self.buffer.len();
        if total_len == 0 {
            return Vec::new();
        }
        let samples_to_take = count.min(total_len);
        let start_idx = total_len.saturating_sub(samples_to_take);
        self.buffer.iter().skip(start_idx).copied().collect()
    }
}

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
    audio_buffer: AudioRingBuffer, // Ring buffer for efficient audio storage (already resampled to 16kHz)
    dropped_samples_count: u64,    // Track dropped samples due to buffer overflow
    translated_samples_count: u64, // Track samples that were successfully processed
    all_sentances: Vec<String>,
    is_recording: bool,
    stream: Option<SCStream>,
}

fn init() {
    // Redirect whisper.cpp & GGML logs into Rust logging hooks; with no backend features enabled,
    // this effectively silences them from stdout/stderr.
    whisper_rs::install_logging_hooks();

    let mut context_param = WhisperContextParameters::default();
    // Disable DTW token-level timestamps for streaming to avoid median filter assertions on small chunks
    context_param.dtw_parameters.mode = whisper_rs::DtwMode::None;
    // Enable GPU if available for better performance
    context_param.use_gpu = true;
    let gpu_enabled = context_param.use_gpu;
    let model_path = "/Users/sammers/Git/my/tauri-whisper/ggml-large-v3-turbo.bin"; // Path to your Whisper model file
    let ctx =
        WhisperContext::new_with_params(&model_path, context_param).expect("failed to load model");
    println!("✅ Whisper model loaded with GPU enabled: {}", gpu_enabled);
    MODEL.get_or_init(|| Mutex::new(Some(ctx)));
}

fn start_mac_os_stream(tx: Sender<CMSampleBuffer>) -> Result<SCStream, CFError> {
    let config = SCStreamConfiguration::new().set_captures_audio(true)?;
    let display = SCShareableContent::get().unwrap().displays().remove(0);
    let filter = SCContentFilter::new().with_display_excluding_windows(&display, &[]);
    let mut stream = SCStream::new(&filter, &config);
    stream.add_output_handler(AudioStreamOutput { sender: tx }, SCStreamOutputType::Audio);
    Ok(stream)
}

fn extract_audio_samples_from_buffer(sample_buffer: &CMSampleBuffer) -> Vec<f32> {
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
    all_samples
}

/// Transcribe a 16 kHz mono PCM chunk using the global Whisper context.
///
/// - Expects `audio_data` at 16_000 Hz (mono)
/// - Returns `Some((transcript, elapsed_seconds))` on success, `None` on failure
/// - Synchronous; run off the main/UI thread
fn transcribe_audio_chunk(audio_data: &[f32]) -> Option<(String, f64)> {
    let start_time = time::Instant::now();
    // Audio data is already resampled to 16kHz in the ring buffer
    let model_guard = MODEL.get()?.lock().ok()?;
    let ctx = model_guard.as_ref()?;
    // Create a state
    let mut state = ctx.create_state().ok()?;
    // Create params optimized for real-time transcription
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 0 });
    params.set_translate(false);
    params.set_language(Some("ru"));
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    // Optimize for speed and avoid DTW on small streaming windows
    params.set_single_segment(true);
    params.set_suppress_blank(false);
    params.set_token_timestamps(false);
    println!("Transcribing audio chunk of length {}", audio_data.len());
    if state.full(params, audio_data).is_err() {
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
        let stream = start_mac_os_stream(tx).unwrap();
        stream.start_capture().unwrap();
        state.stream = Some(stream);
        state.is_recording = true;
        state.audio_buffer.clear();
        state.dropped_samples_count = 0; // Reset counter on new recording
        state.translated_samples_count = 0; // Reset translated samples counter
        let app_handle_clone = app_handle.clone();
        tokio::spawn(async move {
            loop {
                if !is_recording() {
                    println!("Stopping audio processing loop because recording is stopped");
                    break;
                }
                match rx.recv() {
                    Ok(sample_buffer) => {
                        process_sample_buffer(&sample_buffer);
                    }
                    Err(_) => {
                        println!("Error receiving sample buffer, stopping audio processing loop");
                        break;
                    }
                }
            }
        });
        tokio::spawn(async move {
            loop {
                if !is_recording() {
                    println!("Stopping transcription loop because recording is stopped");
                    break;
                }
                perform_transcription_update_ui(&app_handle_clone);
            }
        });
    } else {
        return Err("Failed to acquire recording state lock".to_string());
    }
    Ok("System audio recording started successfully".to_string())
}

fn is_recording() -> bool {
    if let Ok(state) = RECORDING_STATE.lock() {
        state.is_recording
    } else {
        false
    }
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
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_common_middle_part_today_theyre_producing_vs_today_there() {
        let new_text = "Today they're producing".to_string();
        let old_text = "Today there".to_string();
        let (is_common, common_part) = common_middle_part(new_text, old_text);
        assert!(
            is_common,
            "expected common middle part to be false; got true with common_part='{}'",
            common_part
        );
    }

    #[test]
    fn test_common_middle_part_we_re_thrilled_to_work_together_on_a_cutting_edge_rare_earth_recycling_log_vs_we_re_thrilled_to_work_together_on_a_cutting_edge_rare_earth_recycling(
    ) {
        let new_text =
            "We're thrilled to work together on a cutting-edge, rare-earth recycling log"
                .to_string();
        let old_text =
            "We're thrilled to work together on a cutting edge rare earth recycling".to_string();
        let (is_common, common_part) = common_middle_part(new_text, old_text);
        assert!(
            is_common,
            "expected common middle part to be true; got false with common_part='{}'",
            common_part
        );
    }
}
