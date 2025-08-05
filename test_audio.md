# System Audio Recording Test Guide

Your Tauri Whisper app has been enhanced to capture system audio (like OBS does). Here's how to test it:

## What Changed

1. **Enhanced Audio Capture**: The app now prioritizes system audio devices
2. **Cross-Platform Support**: Different implementations for macOS, Windows, and Linux
3. **Improved Device Detection**: Better detection of loopback and monitor devices
4. **Updated UI**: Clear messaging about system audio capabilities

## Testing Steps

1. **Start the app**: Run `npm run tauri dev`
2. **Check device list**: Click "List Audio Devices" to see available capture methods
3. **Start recording**: Click "Start Recording"
4. **Play test audio**: Play music, videos, or any audio on your computer
5. **Watch transcription**: You should see real-time transcription of the audio

## Expected Behavior

### macOS

- Should detect BlackHole, Aggregate devices, or other virtual audio devices
- Falls back to enhanced device detection if no system audio devices found
- UI shows system audio status

### Windows

- Enhanced device detection for loopback devices
- Will use WASAPI loopback when fully implemented
- Falls back to microphone if no system audio available

### Linux

- Enhanced detection for PulseAudio monitor sources
- Looks for devices with "monitor" in the name
- Falls back to microphone if no monitor devices found

## Troubleshooting

- **No system audio detected**: Install BlackHole (macOS) or equivalent virtual audio device
- **Permission errors**: Grant screen recording permission on macOS when prompted
- **No transcription**: Check browser console for audio capture logs
- **Microphone instead of system audio**: Check that virtual audio devices are properly configured

## Future Enhancements

The current implementation uses enhanced device detection. Future versions will include:

- Native ScreenCaptureKit support for macOS
- Native WASAPI loopback for Windows
- Native PulseAudio/PipeWire integration for Linux

This provides a solid foundation for system audio capture while maintaining compatibility with existing setups.
