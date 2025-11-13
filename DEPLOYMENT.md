# Whisper Container Worker - Deployment Guide

## Pre-Deployment Checklist

### 1. Prerequisites

- [ ] Cloudflare account with Workers enabled
- [ ] Wrangler CLI installed (`npm install -g wrangler`)
- [ ] Docker installed (for local testing)
- [ ] Node.js >= 18.0.0
- [ ] Python 3.13+ (for local testing)

### 2. R2 Buckets Creation

Create the required R2 buckets:

```bash
# Production buckets
wrangler r2 bucket create healthos-audio
wrangler r2 bucket create healthos-models

# Preview buckets (for testing)
wrangler r2 bucket create healthos-audio-preview
wrangler r2 bucket create healthos-models-preview
```

Verify buckets:
```bash
wrangler r2 bucket list
```

### 3. KV Namespaces Creation

Create the required KV namespaces:

```bash
# Create METADATA namespace
wrangler kv:namespace create "METADATA"
# Note the ID output, update wrangler.jsonc

# Create METADATA preview namespace
wrangler kv:namespace create "METADATA" --preview
# Note the preview ID, update wrangler.jsonc

# Create AUDIT_LOG namespace
wrangler kv:namespace create "AUDIT_LOG"
# Note the ID output, update wrangler.jsonc

# Create AUDIT_LOG preview namespace
wrangler kv:namespace create "AUDIT_LOG" --preview
# Note the preview ID, update wrangler.jsonc
```

Update the IDs in `wrangler.jsonc`:
```jsonc
"kv_namespaces": [
  {
    "binding": "METADATA",
    "id": "YOUR_METADATA_NAMESPACE_ID",
    "preview_id": "YOUR_METADATA_PREVIEW_ID"
  },
  {
    "binding": "AUDIT_LOG",
    "id": "YOUR_AUDIT_LOG_NAMESPACE_ID",
    "preview_id": "YOUR_AUDIT_LOG_PREVIEW_ID"
  }
]
```

### 4. Optional: HuggingFace Token (for Advanced Diarization)

If you want advanced speaker diarization:

1. Create account at https://huggingface.co
2. Generate token at https://huggingface.co/settings/tokens
3. Accept license at https://huggingface.co/pyannote/speaker-diarization-3.1
4. Add to wrangler.jsonc:
```jsonc
"vars": {
  "HUGGINGFACE_TOKEN": "hf_xxxxxxxxxxxxx"
}
```

5. Uncomment in requirements.txt:
```txt
pyannote.audio==3.1.1
pyannote.core==5.0.0
speechbrain==0.5.16
```

## Local Testing

### Option 1: Docker (Recommended)

Build and run the container locally:

```bash
# Build image
docker build -t whisper-server .

# Run container
docker run -p 9998:9998 whisper-server

# Test health endpoint
curl http://localhost:9998/health

# Test transcription (with sample audio)
curl -X POST http://localhost:9998/transcribe \
  -F "file=@test_audio.mp3" \
  -F "language=pt" \
  -F "enable_diarization=true"
```

### Option 2: Python Direct

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run server
python app.py

# Test in another terminal
curl http://localhost:9998/health
```

### Option 3: Wrangler Dev

```bash
# Install Node dependencies
npm install

# Build TypeScript
npm run build

# Run with Wrangler (dev mode)
wrangler dev
```

## Deployment Steps

### 1. Install Dependencies

```bash
# Node dependencies
npm install

# Verify package.json
cat package.json
```

### 2. Update Configuration

Review and update `wrangler.jsonc`:
- [ ] KV namespace IDs are correct
- [ ] R2 bucket names match your account
- [ ] Environment variables are set
- [ ] Route pattern matches your domain

### 3. Build TypeScript

```bash
npm run build
```

### 4. Test Container Locally

```bash
docker build -t whisper-server .
docker run -p 9998:9998 whisper-server

# In another terminal
curl http://localhost:9998/health
```

### 5. Deploy to Cloudflare

```bash
# Login to Cloudflare (if not already)
wrangler login

# Deploy
wrangler deploy

# Watch logs
wrangler tail
```

### 6. Verify Deployment

```bash
# Check deployment status
wrangler deployments list

# Test health endpoint
curl https://whisper.healthos.ai/health

# Test transcription with sample audio
curl -X POST https://whisper.healthos.ai/transcribe \
  -F "file=@consultation.mp3" \
  -F "language=pt" \
  -F "enable_diarization=true" \
  -F "medical_context=true"
```

## Post-Deployment

### 1. Monitor Performance

```bash
# View real-time logs
wrangler tail

# Check metrics in Cloudflare dashboard
# Workers > whisper-server > Metrics
```

### 2. Set Up Alerts

Configure alerts in Cloudflare dashboard:
- Error rate > 5%
- Response time > 60s
- CPU usage > 80%

### 3. Test Different Scenarios

Test with various audio formats and sizes:
- [ ] MP3 files
- [ ] M4A files (iOS recordings)
- [ ] WAV files
- [ ] Long consultations (30+ minutes)
- [ ] Multiple speakers
- [ ] Background noise

### 4. Backup Strategy

- [ ] R2 bucket versioning enabled
- [ ] Regular KV namespace backups
- [ ] Model files backed up in R2

## Troubleshooting

### Container Build Fails

```bash
# Check Dockerfile syntax
docker build --no-cache -t whisper-server .

# View build logs
docker build -t whisper-server . 2>&1 | tee build.log
```

### Model Download Issues

Pre-download model in Dockerfile:
```dockerfile
RUN python -c "import whisper; whisper.load_model('large-v3')"
```

Or upload to R2 and load from there.

### Out of Memory

Reduce model size in wrangler.jsonc:
```jsonc
"vars": {
  "MODEL_NAME": "medium"  // or "small", "base"
}
```

Increase container resources:
```jsonc
"container": {
  "resources": {
    "memory": "16GB"
  }
}
```

### Slow Transcription

Enable GPU if available (requires GPU container):
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

Or use smaller model for faster processing.

### CORS Issues

Verify CORS headers in both app.py and src/index.ts.

### R2 Access Issues

Verify bucket bindings:
```bash
wrangler r2 bucket list
```

Test bucket access:
```typescript
const test = await env.AUDIO_BUCKET.put('test.txt', 'test');
```

## Rollback

If deployment fails:

```bash
# List deployments
wrangler deployments list

# Rollback to previous version
wrangler rollback [DEPLOYMENT_ID]
```

## Updates

To update the worker:

```bash
# Pull latest code
git pull

# Rebuild
npm run build

# Deploy
wrangler deploy
```

## Scaling

Monitor usage and scale resources:

1. Container resources (CPU, memory)
2. R2 storage limits
3. KV read/write limits
4. Worker invocation limits

Adjust in wrangler.jsonc as needed.

## Security

- [ ] Enable R2 bucket encryption
- [ ] Set up access control for sensitive data
- [ ] Implement API key authentication (if needed)
- [ ] Enable audit logging
- [ ] Set up WAF rules for DDoS protection

## Support

For issues:
1. Check logs: `wrangler tail`
2. Review Cloudflare dashboard metrics
3. Check container health: `curl /health`
4. Review R2 and KV usage

## Cost Estimation

Estimated costs (Cloudflare Workers):
- Container Worker: $0.15 per million requests
- R2 Storage: $0.015 per GB/month
- R2 Operations: Class A/B pricing
- KV Operations: Free tier available

Monitor costs in Cloudflare Billing dashboard.
