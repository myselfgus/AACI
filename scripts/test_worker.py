#!/usr/bin/env python3
"""
Script to test the transcription worker.
"""
import argparse
import requests
from pathlib import Path


def test_transcription(audio_file: str, worker_url: str = "http://localhost:8000"):
    """
    Test transcription with an audio file.
    
    Args:
        audio_file: Path to audio file
        worker_url: URL of the worker API
    """
    audio_path = Path(audio_file)
    
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_file}")
        return
    
    print(f"Testing transcription with: {audio_file}")
    print(f"Worker URL: {worker_url}")
    
    # Health check
    try:
        response = requests.get(f"{worker_url}/health")
        response.raise_for_status()
        health = response.json()
        print(f"\nWorker Status: {health['status']}")
        print(f"Model: {health['model']}")
        print(f"Device: {health['device']}")
    except Exception as e:
        print(f"Error checking worker health: {e}")
        return
    
    # Transcribe
    print("\nTranscribing...")
    try:
        with open(audio_file, 'rb') as f:
            files = {'file': (audio_path.name, f, 'audio/wav')}
            response = requests.post(f"{worker_url}/transcribe", files=files)
            response.raise_for_status()
            
        result = response.json()
        print("\n=== Transcription Result ===")
        print(f"Text: {result['text']}")
        print(f"Language: {result['language']}")
        print(f"Duration: {result['duration']:.2f}s")
        if 'confidence' in result and result['confidence']:
            print(f"Confidence: {result['confidence']:.2%}")
        print("=" * 50)
        
    except Exception as e:
        print(f"Error during transcription: {e}")


def main():
    parser = argparse.ArgumentParser(description="Test AACI transcription worker")
    parser.add_argument("audio_file", help="Path to audio file to transcribe")
    parser.add_argument("--url", default="http://localhost:8000", help="Worker URL")
    
    args = parser.parse_args()
    
    test_transcription(args.audio_file, args.url)


if __name__ == "__main__":
    main()
