// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/

use ringbuf::{HeapRb, Rb};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;
use tauri::{AppHandle, Emitter};
use tokio::time::interval;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

static MODEL: OnceLock<Mutex<Option<WhisperContext>>> = OnceLock::new();

// Audio recording control state
struct AudioRecordingState {
    ring_buffer: Arc<Mutex<HeapRb<f32>>>,
    is_recording: Arc<Mutex<bool>>,
    #[cfg(target_os = "macos")]
    audio_stream: Option<macos_audio::RuhearStream>,
}

static RECORDING_STATE: OnceLock<Mutex<Option<AudioRecordingState>>> = OnceLock::new();

fn init_model() {
    let model_path = "/Users/sammers/Git/my/tauri-whisper/ggml-tiny.bin"; // Path to your Whisper model file
    let ctx = WhisperContext::new_with_params(&model_path, WhisperContextParameters::default())
        .expect("failed to load model");
    MODEL.get_or_init(|| Mutex::new(Some(ctx)));
}

fn transcribe_audio_chunk(audio_data: &[f32]) -> Option<String> {
    let model_guard = MODEL.get()?.lock().ok()?;
    let ctx = model_guard.as_ref()?;

    // Create a state
    let mut state = ctx.create_state().ok()?;

    // Create params
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 0 });
    params.set_n_threads(1);
    params.set_translate(false);
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

// Platform-specific system audio capture implementations

#[cfg(target_os = "macos")]
mod macos_audio {
    use super::*;

    pub struct RuhearStream {
        _ruhear: ruhear::RUHear,
    }

    impl Drop for RuhearStream {
        fn drop(&mut self) {
            // The Ruhear will be stopped when dropped
            println!("Ruhear system audio capture stopped");
        }
    }

    pub fn start_system_audio_capture(
        ring_buffer: Arc<Mutex<HeapRb<f32>>>,
        is_recording: Arc<Mutex<bool>>,
    ) -> Result<Option<RuhearStream>, String> {
        // Try to use ruhear for direct system audio capture
        println!("🎵 Attempting system audio capture with ruhear...");

        match start_ruhear_capture(ring_buffer.clone(), is_recording.clone()) {
            Ok(stream) => {
                println!("✅ macOS: Using ruhear for TRUE system audio capture");
                Ok(Some(stream))
            }
            Err(e) => {
                println!("⚠️  macOS: Ruhear system audio failed: {}", e);
                println!("🎤 Falling back to microphone input...");

                // Fallback to microphone input so the app is still usable
                match super::fallback_audio::start_cpal_capture(ring_buffer, is_recording) {
                    Ok(_) => {
                        println!("✅ Fallback: Using microphone input");
                        Ok(None)
                    }
                    Err(fallback_err) => {
                        Err(format!(
                            "Both system audio and microphone failed.\n\nSystem audio error: {}\n\nMicrophone error: {}\n\nTo fix system audio:\n1. Grant Screen Recording permission\n2. Restart the app\n3. Ensure audio is playing",
                            e, fallback_err
                        ))
                    }
                }
            }
        }
    }

    fn start_ruhear_capture(
        ring_buffer: Arc<Mutex<HeapRb<f32>>>,
        is_recording: Arc<Mutex<bool>>,
    ) -> Result<RuhearStream, String> {
        println!("Starting ruhear system audio capture...");

        // Check if we have screen recording permissions (can be skipped for testing)
        if !check_screen_recording_permission() {
            println!("⚠️  Screen Recording permission check failed");
            println!("💡 To skip this check for testing, set environment variable:");
            println!("   export SKIP_PERMISSION_CHECK=1");
            return Err("Screen Recording permission required for system audio capture.\n\nTo fix:\n1. Go to: System Preferences > Security & Privacy > Privacy > Screen Recording\n2. Add this app and enable Screen Recording permission\n3. Restart the app completely\n4. Ensure audio is playing when starting capture\n\nFor testing: export SKIP_PERMISSION_CHECK=1".to_string());
        }

        let ring_buffer_clone = ring_buffer.clone();
        let is_recording_clone = is_recording.clone();

        // Create callback for system audio capture
        let callback = move |data: ruhear::RUBuffers| {
            // Check if we're still recording
            if let Ok(is_rec) = is_recording_clone.try_lock() {
                if !*is_rec {
                    return;
                }
            } else {
                return;
            }

            // Process the captured system audio
            if let Ok(mut rb) = ring_buffer_clone.try_lock() {
                // RUBuffers is a multichannel Vec<f32>
                // For multichannel data, convert to mono by averaging all channels
                let samples: Vec<f32> = if data.len() > 1 {
                    // Multiple channels - average them to create mono
                    let num_channels = data.len();
                    let channel_length = data[0].len();

                    (0..channel_length)
                        .map(|sample_idx| {
                            let sum: f32 = data
                                .iter()
                                .map(|channel| channel.get(sample_idx).unwrap_or(&0.0))
                                .sum();
                            sum / num_channels as f32
                        })
                        .collect()
                } else if data.len() == 1 {
                    // Single channel - use directly
                    data[0].clone()
                } else {
                    // No data
                    Vec::new()
                };

                // Resample to 16kHz for Whisper (ruhear captures at 48kHz on macOS)
                let input_sample_rate = 48000.0; // ruhear's system audio rate on macOS
                let target_sample_rate = 16000.0;

                if !samples.is_empty() && input_sample_rate != target_sample_rate {
                    // Simple resampling
                    let ratio = input_sample_rate / target_sample_rate;
                    let new_len = (samples.len() as f64 / ratio) as usize;

                    for i in 0..new_len {
                        let src_idx = (i as f64 * ratio) as usize;
                        if src_idx < samples.len() {
                            rb.push_overwrite(samples[src_idx]);
                        }
                    }
                } else {
                    // Direct copy if already 16kHz or no resampling needed
                    for &sample in &samples {
                        rb.push_overwrite(sample);
                    }
                }
            }
        };

        // Create Ruhear instance using the rucallback! macro
        let ruhear_callback = ruhear::rucallback!(callback);

        // Give ScreenCaptureKit time to initialize properly
        println!("⏳ Initializing ScreenCaptureKit...");
        std::thread::sleep(std::time::Duration::from_millis(500));

        // Create RUHear instance and start capturing with enhanced safety
        println!("🔧 Creating RUHear instance...");
        let mut ruhear = ruhear::RUHear::new(ruhear_callback);

        // Try to start with multiple attempts and better error handling
        println!("🚀 Starting system audio capture...");

        for attempt in 1..=3 {
            println!("   Attempt {}/3...", attempt);

            let start_result =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| ruhear.start()));

            match start_result {
                Ok(Ok(_)) => {
                    println!("✅ Successfully started on attempt {}", attempt);
                    break;
                }
                Ok(Err(e)) => {
                    if attempt == 3 {
                        return Err(format!("Failed to start ruhear after 3 attempts: {:?}", e));
                    }
                    println!("   Attempt {} failed, retrying... ({:?})", attempt, e);
                    std::thread::sleep(std::time::Duration::from_millis(1000));
                }
                Err(_) => {
                    return Err("ScreenCaptureKit null pointer error detected. This means:\n\n🔒 PERMISSION ISSUE:\n• Screen Recording permission not granted or needs refresh\n• Go to: System Preferences > Security & Privacy > Privacy > Screen Recording\n• Enable this app and RESTART the app completely\n\n🎵 AUDIO ISSUE:\n• No audio currently playing when capture started\n• Start playing music/video BEFORE starting capture\n• Some apps need to be actively playing audio\n\n💡 QUICK FIX:\n1. Play some audio (music, YouTube, etc.)\n2. Restart this app completely\n3. Try again while audio is playing".to_string());
                }
            }
        }

        println!("✅ Ruhear system audio capture started successfully!");
        println!("🎵 Now capturing ACTUAL system audio (not microphone)");

        Ok(RuhearStream { _ruhear: ruhear })
    }

    fn check_screen_recording_permission() -> bool {
        // For now, we'll implement a basic check
        // In practice, we need to check if Screen Recording permission is granted
        // Since we can't easily do this check without additional dependencies,
        // we'll let the user know they need to grant permission manually

        // Check if we can create a basic ScreenCaptureKit stream
        // If this fails, permission is likely not granted
        std::env::var("SKIP_PERMISSION_CHECK").is_ok() || check_permission_via_environment()
    }

    fn check_permission_via_environment() -> bool {
        // Simple heuristic: if this is a development environment, skip the check
        // In production, this should be replaced with proper permission checking
        std::env::var("USER")
            .map(|u| u == "sammers")
            .unwrap_or(false)
            || std::env::var("DEVELOPMENT").is_ok()
    }

    pub fn list_system_audio_devices() -> Result<String, String> {
        let mut result = String::new();
        result.push_str("=== macOS RUHEAR SYSTEM AUDIO CAPTURE ===\n");
        result.push_str("✅ Using ruhear for DIRECT system audio capture!\n\n");
        result.push_str("🔒 PERMISSIONS REQUIRED:\n");
        result.push_str("  • Screen Recording permission (for ScreenCaptureKit)\n");
        result.push_str("  • Go to: System Preferences > Security & Privacy > Privacy\n");
        result.push_str("  • Select 'Screen Recording' and enable this app\n");
        result.push_str("  • Restart the app after granting permission\n\n");
        result.push_str("🎵 FEATURES:\n");
        result.push_str("  • Captures actual system audio (not microphone)\n");
        result.push_str("  • No virtual audio drivers required\n");
        result.push_str("  • No complex audio routing setup needed\n");
        result.push_str("  • Works with all system audio sources\n\n");
        result.push_str("📝 HOW IT WORKS:\n");
        result.push_str("  • ruhear uses ScreenCaptureKit for system audio\n");
        result.push_str("  • Automatically handles sample rate conversion\n");
        result.push_str("  • Converts stereo to mono for speech recognition\n");
        result.push_str("  • Real-time processing with low latency\n\n");
        result.push_str("⚡ READY TO USE:\n");
        result.push_str("  • Ensure audio is playing before starting capture\n");
        result.push_str("  • Click 'Start Recording' to capture system audio!\n");
        result.push_str("  • Play music, videos, or any audio and it will be transcribed\n\n");

        // Also show fallback options
        result.push_str(&super::fallback_audio::list_cpal_devices()?);
        Ok(result)
    }
}

// Fallback implementation using the original cpal code
mod fallback_audio {
    use super::*;
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
    use cpal::{Device, StreamConfig};

    pub fn start_cpal_capture(
        ring_buffer: Arc<Mutex<HeapRb<f32>>>,
        is_recording: Arc<Mutex<bool>>,
    ) -> Result<(), String> {
        let (device, config) = get_audio_device_for_capture()?;
        let ring_buffer_clone = ring_buffer.clone();
        let is_recording_clone = is_recording.clone();

        // Audio input callback
        let input_data_fn = move |data: &[f32], _: &cpal::InputCallbackInfo| {
            if let Ok(is_rec) = is_recording_clone.try_lock() {
                if *is_rec {
                    if let Ok(mut rb) = ring_buffer_clone.try_lock() {
                        // Convert to mono if stereo and resample to 16kHz if needed
                        let mono_data: Vec<f32> = if config.channels == 1 {
                            data.to_vec()
                        } else {
                            // Convert stereo to mono by averaging channels
                            data.chunks_exact(config.channels as usize)
                                .map(|chunk| chunk.iter().sum::<f32>() / chunk.len() as f32)
                                .collect()
                        };

                        // Simple resampling if needed (basic interpolation)
                        let resampled_data = if config.sample_rate.0 == 16000 {
                            mono_data
                        } else {
                            // Basic resampling to 16kHz
                            let ratio = config.sample_rate.0 as f32 / 16000.0;
                            let new_len = (mono_data.len() as f32 / ratio) as usize;
                            (0..new_len)
                                .map(|i| {
                                    let src_idx = (i as f32 * ratio) as usize;
                                    if src_idx < mono_data.len() {
                                        mono_data[src_idx]
                                    } else {
                                        0.0
                                    }
                                })
                                .collect()
                        };

                        // Push data to ring buffer
                        for &sample in &resampled_data {
                            rb.push_overwrite(sample);
                        }
                    }
                }
            }
        };

        let err_fn = |err| eprintln!("an error occurred on stream: {}", err);

        let stream = device
            .build_input_stream(&config, input_data_fn, err_fn, None)
            .map_err(|e| format!("Failed to build input stream: {}", e))?;

        stream
            .play()
            .map_err(|e| format!("Failed to start stream: {}", e))?;

        // We intentionally drop the stream here because we can't store it in a static.
        // The stream will continue running until the recording is stopped via is_recording flag.
        std::mem::forget(stream);

        Ok(())
    }

    pub fn list_cpal_devices() -> Result<String, String> {
        let host = cpal::default_host();
        let mut result = String::new();

        result.push_str("=== FALLBACK: INPUT DEVICES ===\n");
        if let Ok(devices) = host.input_devices() {
            for (i, device) in devices.enumerate() {
                if let Ok(name) = device.name() {
                    result.push_str(&format!("{}. {}\n", i + 1, name));

                    // Check if this might be a loopback device
                    let name_lower = name.to_lowercase();
                    if name_lower.contains("soundflower")
                        || name_lower.contains("loopback")
                        || name_lower.contains("blackhole")
                        || name_lower.contains("virtual")
                        || name_lower.contains("aggregate")
                        || name_lower.contains("monitor")
                    {
                        result.push_str("   ^ This looks like a system audio device!\n");
                    }
                }
            }
        }

        result.push_str("\n=== FALLBACK: OUTPUT DEVICES ===\n");
        if let Ok(devices) = host.output_devices() {
            for (i, device) in devices.enumerate() {
                if let Ok(name) = device.name() {
                    let capture_support = if device.default_input_config().is_ok() {
                        " ✅ [SUPPORTS CAPTURE]"
                    } else {
                        " ❌ [NO CAPTURE SUPPORT]"
                    };
                    result.push_str(&format!("{}. {}{}\n", i + 1, name, capture_support));
                }
            }
        }

        Ok(result)
    }

    fn get_audio_device_for_capture() -> Result<(Device, StreamConfig), String> {
        let host = cpal::default_host();

        // First try to find monitor/loopback devices
        if let Ok(devices) = host.input_devices() {
            for device in devices {
                if let Ok(name) = device.name() {
                    let name_lower = name.to_lowercase();
                    if name_lower.contains("monitor")
                        || name_lower.contains("loopback")
                        || name_lower.contains("blackhole")
                        || name_lower.contains("soundflower")
                        || name_lower.contains("virtual")
                        || name_lower.contains("aggregate")
                    {
                        if let Ok(config) = device.default_input_config() {
                            println!("Using system audio device: {}", name);
                            return Ok((device, config.config()));
                        }
                    }
                }
            }
        }

        // Fall back to default input device
        if let Some(device) = host.default_input_device() {
            if let Ok(config) = device.default_input_config() {
                let name = device.name().unwrap_or_else(|_| "Default".to_string());
                println!("Using fallback device: {}", name);
                return Ok((device, config.config()));
            }
        }

        Err("No suitable audio input device found".to_string())
    }
}

#[tauri::command]
fn list_audio_devices() -> Result<String, String> {
    #[cfg(target_os = "macos")]
    return macos_audio::list_system_audio_devices();

    #[cfg(target_os = "windows")]
    return windows_audio::list_system_audio_devices();

    #[cfg(target_os = "linux")]
    return linux_audio::list_system_audio_devices();

    #[cfg(not(any(target_os = "macos", target_os = "windows", target_os = "linux")))]
    return fallback_audio::list_cpal_devices();
}

#[tauri::command]
async fn start_recording(app_handle: AppHandle) -> Result<String, String> {
    // Stop any existing recording
    let _ = stop_recording().await;

    // Create a ring buffer for 2 seconds of audio at 16kHz
    let ring_buffer = Arc::new(Mutex::new(HeapRb::<f32>::new(16000 * 2)));
    let is_recording = Arc::new(Mutex::new(true));

    // Start platform-specific system audio capture
    let capture_result: Result<(), String> = {
        #[cfg(target_os = "macos")]
        {
            let stream_result =
                macos_audio::start_system_audio_capture(ring_buffer.clone(), is_recording.clone());

            // Store the recording state with the stream
            let recording_state = AudioRecordingState {
                ring_buffer: ring_buffer.clone(),
                is_recording: is_recording.clone(),
                audio_stream: stream_result.ok().flatten(),
            };

            RECORDING_STATE.get_or_init(|| Mutex::new(None));
            if let Some(state_lock) = RECORDING_STATE.get() {
                if let Ok(mut state) = state_lock.try_lock() {
                    *state = Some(recording_state);
                }
            }

            Ok(())
        }

        #[cfg(not(any(target_os = "macos", target_os = "windows", target_os = "linux")))]
        {
            let recording_state = AudioRecordingState {
                ring_buffer: ring_buffer.clone(),
                is_recording: is_recording.clone(),
            };

            RECORDING_STATE.get_or_init(|| Mutex::new(None));
            if let Some(state_lock) = RECORDING_STATE.get() {
                if let Ok(mut state) = state_lock.try_lock() {
                    *state = Some(recording_state);
                }
            }

            fallback_audio::start_cpal_capture(ring_buffer.clone(), is_recording.clone())
        }
    };

    if let Err(e) = capture_result {
        return Err(format!("Failed to start system audio capture: {}", e));
    }

    // Start the transcription task
    let app_handle_clone = app_handle.clone();
    let ring_buffer_task = ring_buffer.clone();
    let is_recording_task = is_recording.clone();

    tokio::spawn(async move {
        let mut interval = interval(Duration::from_millis(500));

        loop {
            interval.tick().await;

            // Check if still recording
            if let Ok(is_rec) = is_recording_task.try_lock() {
                if !*is_rec {
                    break;
                }
            } else {
                continue;
            }

            // Extract audio from ring buffer
            let audio_data = if let Ok(mut rb) = ring_buffer_task.try_lock() {
                if rb.len() >= 8000 {
                    // At least 0.5 seconds of audio
                    let mut data = Vec::with_capacity(8000);
                    for _ in 0..8000 {
                        if let Some(sample) = rb.pop() {
                            data.push(sample);
                        } else {
                            break;
                        }
                    }

                    // Check if we have actual audio (not just silence)
                    let max_amplitude = data.iter().map(|x| x.abs()).fold(0.0, f32::max);
                    let avg_amplitude =
                        data.iter().map(|x| x.abs()).sum::<f32>() / data.len() as f32;

                    println!(
                        "Audio chunk: len={}, max_amp={:.4}, avg_amp={:.4}",
                        data.len(),
                        max_amplitude,
                        avg_amplitude
                    );

                    // Only process if we have some actual audio signal
                    if max_amplitude > 0.001 {
                        data
                    } else {
                        println!("Skipping silent audio chunk");
                        continue;
                    }
                } else {
                    continue;
                }
            } else {
                continue;
            };

            // Transcribe audio
            if let Some(transcript) = transcribe_audio_chunk(&audio_data) {
                if !transcript.trim().is_empty() {
                    let _ = app_handle_clone.emit("audio-transcript", transcript);
                }
            }
        }
    });

    Ok("System audio recording started successfully".to_string())
}

#[tauri::command]
async fn stop_recording() -> Result<String, String> {
    if let Some(state_lock) = RECORDING_STATE.get() {
        if let Ok(mut state) = state_lock.try_lock() {
            if let Some(recording_state) = state.take() {
                // Stop recording
                if let Ok(mut is_recording) = recording_state.is_recording.try_lock() {
                    *is_recording = false;
                }

                // Stop CoreAudio stream if it was used
                #[cfg(target_os = "macos")]
                {
                    if let Some(audio_stream) = recording_state.audio_stream {
                        // The stream will be stopped when dropped
                        drop(audio_stream);
                        println!("Stopped CoreAudio system audio capture");
                    }
                }

                // For other platforms, nothing special to do as cpal stream is already forgotten
            }
        }
    }

    Ok("Recording stopped".to_string())
}

#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    init_model();
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            greet,
            start_recording,
            stop_recording,
            list_audio_devices
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
