// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/

// mod audio; // Commented out - code moved inline to avoid conflicts

use core_foundation::error::CFError;
use core_media_rs::cm_sample_buffer::CMSampleBuffer;
use hound::{WavSpec, WavWriter};
use screencapturekit::{
    shareable_content::SCShareableContent,
    stream::{
        configuration::SCStreamConfiguration, content_filter::SCContentFilter,
        output_trait::SCStreamOutputTrait, output_type::SCStreamOutputType, SCStream,
    },
};
use std::sync::{Arc, LazyLock, Mutex, OnceLock};
use std::{
    collections::HashMap,
    sync::mpsc::{channel, Sender},
    time,
};
use tauri::{AppHandle, Emitter};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

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

// Audio recording control state
struct AudioRecordingState {
    current_chunk: Vec<f32>,
    counter: u32,
    wav_writer: Option<HashMap<usize, WavWriter<std::io::BufWriter<std::fs::File>>>>,
    is_recording: bool,
    stream: Option<SCStream>,
    tick: time::Instant,
}

static RECORDING_STATE: LazyLock<Arc<Mutex<AudioRecordingState>>> = LazyLock::new(|| {
    Arc::new(Mutex::new(AudioRecordingState {
        current_chunk: Vec::new(),
        counter: 0,
        wav_writer: None,
        is_recording: false,
        stream: None,
        tick: time::Instant::now(),
    }))
});

fn init() {
    let mut context_param =  WhisperContextParameters::default();
    context_param.dtw_parameters.mode = whisper_rs::DtwMode::ModelPreset {
        model_preset: whisper_rs::DtwModelPreset::LargeV3,
    };
    let model_path = "/Users/sammers/Git/my/tauri-whisper/ggml-large-v3.bin"; // Path to your Whisper model file
    let ctx = WhisperContext::new_with_params(&model_path, context_param)
        .expect("failed to load model");
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

fn transcribe_audio_chunk(audio_data: &[f32]) -> Option<String> {
    let model_guard = MODEL.get()?.lock().ok()?;
    let ctx = model_guard.as_ref()?;
    // Create a state
    let mut state = ctx.create_state().ok()?;
    // Create params
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 0 });
    params.set_n_threads(10);
    params.set_translate(true);
    params.set_language(Some("en"));
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    // Run the model
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
        state.tick = time::Instant::now();
        state.wav_writer = Some(HashMap::new());

        // Clone the app_handle to move into the spawned task
        let app_handle_clone = app_handle.clone();
        tokio::spawn(async move {
            while let Ok(sample_buffer) = rx.recv() {
                if let Ok(mut tokio_state) = RECORDING_STATE.lock() {
                    let tick = tokio_state.tick;
                    if tick.elapsed().as_secs() > 5 {
                        println!("Transcribing audio chunk of length {}", tokio_state.current_chunk.len());
                        if let Some(transcript) = transcribe_audio_chunk(&tokio_state.current_chunk)
                        {
                            // Emit the transcript to the frontend
                            if let Err(e) = app_handle_clone.emit("audio-transcript", &transcript) {
                                eprintln!("Failed to emit transcript event: {:?}", e);
                            } else {
                                println!("Transcript emitted: {}", transcript);
                            }
                        }
                        tokio_state.current_chunk.clear();
                        tokio_state.tick = time::Instant::now();
                    }
                    let mut all_samples = Vec::new();
                    let wav_writers = tokio_state.wav_writer.as_mut().unwrap();
                    let buffer_list = sample_buffer.get_audio_buffer_list().expect("should work");
                    for buffer_index in 0..buffer_list.num_buffers() {
                        let buffer = buffer_list.get(buffer_index).expect("should work");
                        if !wav_writers.contains_key(&buffer_index) {
                            let spec = WavSpec {
                                channels: buffer.number_channels as u16,
                                sample_rate: 48000,
                                bits_per_sample: 32,
                                sample_format: hound::SampleFormat::Float,
                            };
                            let writer =
                                WavWriter::create(format!("../out_{buffer_index}.wav"), spec)
                                    .expect("failed to create WAV writer");
                            wav_writers.insert(buffer_index, writer);
                        }
                        let wav_writer = wav_writers.get_mut(&buffer_index).unwrap();
                        let data_slice = buffer.data();
                        let sample_count = data_slice.len() / 4;
                        for i in 0..sample_count {
                            let sample_bytes = [
                                data_slice[i * 4],
                                data_slice[i * 4 + 1],
                                data_slice[i * 4 + 2],
                                data_slice[i * 4 + 3],
                            ];
                            let sample_f32 = f32::from_le_bytes(sample_bytes);
                            if let Err(e) = wav_writer.write_sample(sample_f32) {
                                eprintln!("failed to write sample to WAV file: {:?}", e);
                            }
                            if buffer_index == 0 {
                                all_samples.push(sample_f32);
                            }
                        }
                    }
                    tokio_state.current_chunk.extend(all_samples);
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
        state.wav_writer = None;
        state.current_chunk.clear();
        state.counter = 0;
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
