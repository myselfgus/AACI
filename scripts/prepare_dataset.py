#!/usr/bin/env python3
"""
Script to prepare audio dataset for fine-tuning.
Creates metadata.csv from audio files and transcription text files.
"""
import os
import argparse
from pathlib import Path
import csv
import librosa
from tqdm import tqdm


def prepare_dataset(data_dir: str, output_file: str = "metadata.csv"):
    """
    Prepare dataset by creating metadata CSV.
    
    Args:
        data_dir: Directory containing audio and text files
        output_file: Output CSV filename
    """
    data_path = Path(data_dir)
    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
    
    # Find all audio files
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(data_path.glob(f"*{ext}"))
    
    print(f"Found {len(audio_files)} audio files")
    
    # Prepare metadata
    metadata = []
    total_duration = 0
    
    for audio_file in tqdm(audio_files, desc="Processing audio files"):
        # Get corresponding text file
        text_file = audio_file.with_suffix('.txt')
        
        if not text_file.exists():
            print(f"Warning: No transcription found for {audio_file}")
            continue
        
        # Read transcription
        with open(text_file, 'r', encoding='utf-8') as f:
            transcription = f.read().strip()
        
        if not transcription:
            print(f"Warning: Empty transcription for {audio_file}")
            continue
        
        try:
            # Get audio duration
            duration = librosa.get_duration(filename=str(audio_file))
            total_duration += duration
            
            # Add to metadata
            metadata.append({
                'file_name': audio_file.name,
                'transcription': transcription,
                'duration': duration,
            })
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            continue
    
    # Write metadata CSV
    output_path = data_path / output_file
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['file_name', 'transcription', 'duration'])
        writer.writeheader()
        writer.writerows(metadata)
    
    print(f"\nDataset prepared:")
    print(f"  Total files: {len(metadata)}")
    print(f"  Total duration: {total_duration / 3600:.2f} hours")
    print(f"  Metadata saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare audio dataset for fine-tuning")
    parser.add_argument("data_dir", help="Directory containing audio and text files")
    parser.add_argument("--output", default="metadata.csv", help="Output metadata filename")
    
    args = parser.parse_args()
    
    prepare_dataset(args.data_dir, args.output)


if __name__ == "__main__":
    main()
