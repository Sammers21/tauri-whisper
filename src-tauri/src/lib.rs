// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/

use hound;
use std::fs::File;
use std::io::Write;
use std::sync::{Mutex, OnceLock};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

static MODEL: OnceLock<Mutex<WhisperContext>> = OnceLock::new();

fn init_model() {
    let model_path = "/Users/sammers/Git/my/tauri-whisper/ggml-tiny.bin"; // Path to your Whisper model file
    let ctx = WhisperContext::new_with_params(&model_path, WhisperContextParameters::default())
        .expect("failed to load model");
    MODEL.get_or_init(|| Mutex::new(ctx));
}

#[tauri::command]
fn greet(name: &str) -> String {
    let mut segemnts: Vec<String> = Vec::new();
    if MODEL.get().is_none() {
        init_model();
    }
    let tick = std::time::Instant::now();
    // Lock the model context for thread safety
    let ctx = MODEL
        .get()
        .expect("Model not initialized")
        .lock()
        .expect("Failed to lock model context");

    // Create a state
    let mut state = ctx.create_state().expect("failed to create key");

    // Create a params object for running the model.
    // The number of past samples to consider defaults to 0.
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 0 });

    // Edit params as needed.
    // Set the number of threads to use to 1.
    params.set_n_threads(1);
    // Enable translation.
    params.set_translate(true);
    // Set the language to translate to to English.
    params.set_language(Some("en"));
    // Disable anything that prints to stdout.
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    // Enable token level timestamps
    params.set_token_timestamps(true);

    // Open the audio file.
    let reader = hound::WavReader::open(
        "/Users/sammers/Git/my/tauri-whisper/whisper-rs/examples/full_usage/2830-3980-0043.wav",
    )
    .expect("failed to open file");
    #[allow(unused_variables)]
    let hound::WavSpec {
        channels,
        sample_rate,
        bits_per_sample,
        ..
    } = reader.spec();

    // Convert the audio to floating point samples.
    let samples: Vec<i16> = reader
        .into_samples::<i16>()
        .map(|x| x.expect("Invalid sample"))
        .collect();
    let mut audio = vec![0.0f32; samples.len().try_into().unwrap()];
    whisper_rs::convert_integer_to_float_audio(&samples, &mut audio).expect("Conversion error");

    // Convert audio to 16KHz mono f32 samples, as required by the model.
    // These utilities are provided for convenience, but can be replaced with custom conversion logic.
    // SIMD variants of these functions are also available on nightly Rust (see the docs).
    if channels == 2 {
        audio = whisper_rs::convert_stereo_to_mono_audio(&audio).expect("Conversion error");
    } else if channels != 1 {
        panic!(">2 channels unsupported");
    }

    if sample_rate != 16000 {
        panic!("sample rate must be 16KHz");
    }

    // Run the model.
    state.full(params, &audio[..]).expect("failed to run model");

    // Create a file to write the transcript to.
    let mut file = File::create("transcript.txt").expect("failed to create file");

    // Iterate through the segments of the transcript.
    let num_segments = state
        .full_n_segments()
        .expect("failed to get number of segments");
    for i in 0..num_segments {
        // Get the transcribed text and timestamps for the current segment.
        let segment = state
            .full_get_segment_text(i)
            .expect("failed to get segment");
        let start_timestamp = state
            .full_get_segment_t0(i)
            .expect("failed to get start timestamp");
        let end_timestamp = state
            .full_get_segment_t1(i)
            .expect("failed to get end timestamp");

        let first_token_dtw_ts = if let Ok(token_count) = state.full_n_tokens(i) {
            if token_count > 0 {
                if let Ok(token_data) = state.full_get_token_data(i, 0) {
                    token_data.t_dtw
                } else {
                    -1i64
                }
            } else {
                -1i64
            }
        } else {
            -1i64
        };
        // Print the segment to stdout.
        println!(
            "[{} - {} ({})]: {}",
            start_timestamp, end_timestamp, first_token_dtw_ts, segment
        );

        // Format the segment information as a string.
        let line = format!("[{} - {}]: {}\n", start_timestamp, end_timestamp, segment);
        segemnts.push(line.clone());
        // Write the segment information to the file.
        file.write_all(line.as_bytes())
            .expect("failed to write to file");
    }
    let elapsed = tick.elapsed();
    let res = segemnts.join("\n");
    format!("Hello, {}! You've been greeted from Rust!. Transcript: \n{}, Elapsed time: {:?}", name, res, elapsed)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Initialize the model if it hasn't been initialized yet
    init_model();
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![greet])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
