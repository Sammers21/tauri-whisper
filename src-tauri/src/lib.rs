// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/

use hound;
use std::fs::File;
use std::io::Write;
use std::sync::{Mutex, OnceLock};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

static MODEL: OnceLock<Mutex<WhisperContext>> = OnceLock::new();

fn init_model() {
    let model_path = "/Users/sammers/Git/my/tauri-whisper/ggml-tiny.bin"; // Path to your Whisper model file
    let ctx = WhisperContext::new_with_params(&model_path, WhisperContextParameters::default())
        .expect("failed to load model");
    MODEL.get_or_init(|| Mutex::new(ctx));
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
