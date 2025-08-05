#!/bin/bash

echo "🎵 Testing macOS System Audio Capture with Ruhear"
echo "=================================================="
echo

# Set environment for testing
export MACOSX_DEPLOYMENT_TARGET=10.15
export SKIP_PERMISSION_CHECK=1  # Skip permission check for initial testing

echo "🔧 Environment setup:"
echo "   MACOSX_DEPLOYMENT_TARGET=$MACOSX_DEPLOYMENT_TARGET"
echo "   SKIP_PERMISSION_CHECK=$SKIP_PERMISSION_CHECK"
echo

echo "📋 Test Steps:"
echo "1. Start playing audio (music, video, etc.)"
echo "2. Run: cargo run (or your Tauri app)"
echo "3. Click 'Start Recording' in the app"
echo "4. Observe the console output for debugging"
echo

echo "🔍 Expected Behavior:"
echo "✅ Success: 'Ruhear system audio capture started successfully!'"
echo "⚠️  Fallback: 'Fallback: Using microphone input'"
echo "❌ Error: Clear error message with fix instructions"
echo

echo "💡 If you get the null pointer error again:"
echo "1. Grant Screen Recording permission first:"
echo "   System Preferences > Security & Privacy > Privacy > Screen Recording"
echo "2. Remove the SKIP_PERMISSION_CHECK=1 line above"
echo "3. Restart the app completely"
echo "4. Ensure audio is playing BEFORE starting capture"
echo

echo "🚀 Ready to test! Start some audio and run your app."