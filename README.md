# Whisper Transcription Server for HealthOS

Medical audio transcription service with speaker diarization, optimized for Portuguese (Brazil) medical consultations.

## Features

- **High Accuracy**: Uses Whisper Large v3 model for best transcription quality
- **Portuguese Optimized**: Configured for pt-BR with medical context support
- **Speaker Diarization**: Identifies who's speaking (doctor vs patient)
- **Multi-Format Support**: MP3, M4A, WAV, FLAC, OGG
- **Medical Context**: Optimized for medical terminology and consultation scenarios
- **Scalable**: Runs as Cloudflare Container Worker

## Architecture

```
┌─────────────────┐
│  TypeScript     │  - Request validation
│  Worker Wrapper │  - Auth/logging
│  (Edge)         │  - R2/KV storage
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Python         │  - Whisper model
│  Container      │  - Transcription
│  (Port 9998)    │  - Diarization
└─────────────────┘
```

## API Endpoints

### POST /transcribe

Transcribe audio file with optional speaker diarization.

**Request:**
```bash
curl -X POST http://localhost:9998/transcribe \
  -F "file=@consultation.mp3" \
  -F "language=pt" \
  -F "enable_diarization=true" \
  -F "medical_context=true"
```

**Response:**
```json
{
  "success": true,
  "transcript": "Doutor: Bom dia, como você está se sentindo hoje?...",
  "language": "pt",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 3.5,
      "text": "Bom dia, como você está se sentindo hoje?",
      "speaker": "SPEAKER_1",
      "confidence": 0.9
    }
  ],
  "metadata": {
    "model": "large-v3",
    "duration": 180.5,
    "transcription_time": 45.2,
    "diarization_enabled": true,
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

### GET /health

Health check endpoint.

### GET /

Service information.

## Configuration

### Environment Variables

- `PORT`: Server port (default: 9998)
- `MODEL_NAME`: Whisper model (default: large-v3)
- `DEFAULT_LANGUAGE`: Default language (default: pt)
- `ENABLE_DIARIZATION`: Enable speaker diarization (default: true)
- `MAX_FILE_SIZE_MB`: Max audio file size (default: 500)
- `SUPPORTED_FORMATS`: Audio formats (default: mp3,m4a,wav,flac,ogg)
- `HUGGINGFACE_TOKEN`: Token for advanced diarization (optional)

## Deployment

### Prerequisites

1. Install Wrangler CLI:
```bash
npm install -g wrangler
```

2. Login to Cloudflare:
```bash
wrangler login
```

3. Create R2 buckets:
```bash
wrangler r2 bucket create healthos-audio
wrangler r2 bucket create healthos-models
```

4. Create KV namespaces:
```bash
wrangler kv:namespace create "METADATA"
wrangler kv:namespace create "AUDIT_LOG"
```

### Deploy

```bash
# Install dependencies
npm install
pip install -r requirements.txt

# Build
npm run build

# Deploy to production
wrangler deploy
```

### Local Development

```bash
# Run locally
wrangler dev

# Or run Python container directly
docker build -t whisper-server .
docker run -p 9998:9998 whisper-server
```

## Speaker Diarization

Two modes available:

### Simple Diarization (Default)
- Uses audio features (pitch, MFCC)
- No external dependencies
- Lower accuracy but faster
- Good for basic use cases

### Advanced Diarization (Optional)
- Uses pyannote.audio
- Requires HuggingFace token
- Higher accuracy
- Better for production

To enable advanced diarization:

1. Uncomment pyannote.audio in requirements.txt
2. Get HuggingFace token from https://huggingface.co/settings/tokens
3. Accept pyannote model license at https://huggingface.co/pyannote/speaker-diarization-3.1
4. Set environment variable: `HUGGINGFACE_TOKEN=your_token`

## Medical Context Optimization

The service is optimized for medical consultations:

- Longer audio segments (consultations can be 30+ minutes)
- Medical terminology recognition
- Background noise handling (clinic environment)
- Portuguese medical vocabulary
- Doctor-patient conversation patterns

## File Formats

Supported audio formats:
- **MP3**: Compressed, good for storage
- **M4A**: Apple devices, good quality
- **WAV**: Uncompressed, highest quality
- **FLAC**: Lossless compression
- **OGG**: Open format, good compression

## Storage

### R2 Buckets
- `healthos-audio`: Stores uploaded audio files
- `healthos-models`: Stores Whisper models (optional)

### KV Namespaces
- `METADATA`: Transcription metadata (30 day TTL)
- `AUDIT_LOG`: Audit trail (90 day TTL)

## Performance

- **Transcription Speed**: ~4x real-time (15 min audio in ~4 min)
- **Model Size**: ~3GB (large-v3)
- **Memory Required**: 8GB RAM recommended
- **CPU**: 4 cores recommended

## Troubleshooting

### Model Download Issues
If model download fails, pre-download during build:
```dockerfile
RUN python -c "import whisper; whisper.load_model('large-v3')"
```

### Out of Memory
Reduce model size:
```bash
MODEL_NAME=medium  # or small, base, tiny
```

### Slow Transcription
Use GPU if available:
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

## Security

- Audio files stored in R2 with encryption
- Metadata stored in KV with TTL
- CORS enabled for web clients
- Request size limits enforced
- Audit logging for all transcriptions

## License

MIT

## Support

For issues and questions, contact HealthOS team.
