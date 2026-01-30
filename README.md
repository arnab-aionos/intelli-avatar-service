# IntelliAvatar Service

Production-ready real-time streaming avatar system using Pipecat, MuseTalk, and WebRTC.

## Features

- **Real-time Streaming**: Sub-second latency via WebRTC
- **REST API**: Asynchronous video generation endpoints
- **Client SDKs**: Python and JavaScript libraries
- **Production Ready**: Docker deployment, authentication, session management

## Architecture

```
Client → WebRTC/REST → Pipecat Server
                         ├── ChatGPT (LLM)
                         ├── EdgeTTS (Audio)
                         └── MuseTalk (Avatar)
```

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA GPU (for real-time performance)
- Redis (for session management)

### Installation

```bash
pip install -r requirements.txt
```

### Running the Server

```bash
# Start Pipecat WebRTC server
python server/main.py

# Start REST API (separate terminal)
python api/main.py
```

### Using the Client SDK

**Python:**
```python
from sdks.python import AvatarClient

client = AvatarClient(api_key="your-api-key")
session = client.create_session(avatar_id="default")
client.connect_webrtc(session.webrtc_url)
```

**JavaScript:**
```javascript
import { AvatarClient } from './sdks/javascript/avatar-client.js';

const client = new AvatarClient({ apiKey: 'your-api-key' });
const session = await client.createSession({ avatarId: 'default' });
await client.connectWebRTC(session.webrtcUrl);
```

## Directory Structure

```
intelli-avatar-service/
├── server/           # Pipecat WebRTC server
├── api/              # FastAPI REST endpoints
├── processors/       # Custom Pipecat processors
├── sdks/             # Client libraries
├── examples/         # Usage examples
└── tests/            # Test suite
```

## Documentation

- [API Reference](docs/API.md)
- [Deployment Guide](docs/DEPLOYMENT.md)
- [Development Guide](docs/DEVELOPMENT.md)

## License

MIT
