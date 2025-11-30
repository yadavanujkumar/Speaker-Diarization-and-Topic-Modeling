# Speaker Diarization and Topic Modeling Pipeline

A complete, integrated Python pipeline for performing **Speaker Diarization**, **Transcription**, and **Topic Modeling** on audio files (e.g., simulated call recordings).

## Features

- **Speaker Diarization**: Identifies different speakers in an audio file with timestamps
- **Transcription**: Converts speech to text using OpenAI's Whisper model
- **Topic Modeling**: Identifies main topics using LDA (scikit-learn) or BERTopic
- **Structured Output**: Generates clean JSON logs showing who said what and the main topic

## Installation

### Prerequisites

1. Python 3.8 or higher
2. ffmpeg (required for audio processing)

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows: Download from https://ffmpeg.org/download.html
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Optional Dependencies

For advanced speaker diarization with pyannote.audio:
```bash
pip install pyannote.audio
```

For BERTopic-based topic modeling:
```bash
pip install bertopic
```

## Usage

### Demo Mode (Simulated Data)

Run the pipeline with simulated data to see how it works:

```bash
python speaker_diarization_pipeline.py --demo
```

### Analyze an Audio File

```bash
python speaker_diarization_pipeline.py --audio_path /path/to/audio.wav
```

### Full Options

```bash
python speaker_diarization_pipeline.py \
    --audio_path /path/to/audio.wav \
    --whisper_model base \
    --topic_method lda \
    --n_topics 5 \
    --output results.json
```

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--audio_path` | Path to the audio file | None |
| `--demo` | Run in demo mode with simulated data | False |
| `--whisper_model` | Whisper model size: tiny, base, small, medium, large | base |
| `--topic_method` | Topic modeling method: lda, bertopic | lda |
| `--n_topics` | Number of topics to extract (3-5 recommended) | 5 |
| `--use_real_diarization` | Use real pyannote.audio model | False |
| `--hf_token` | HuggingFace token for pyannote.audio | None |
| `--output` | Output JSON file path | None |

## Pipeline Structure

The pipeline follows a sequential process:

### Step 1: Speaker Diarization
Generates speaker boundaries and labels (e.g., `SPEAKER_A, 0.0s - 8.5s`)

**Options:**
- Simulated diarization (default, for demo)
- Real diarization using pyannote.audio (requires HuggingFace token)

### Step 2: Transcription
Each audio segment is passed to the Whisper model to get the text.

**Whisper Models:**
- `tiny`: Fastest, least accurate
- `base`: Good balance (default)
- `small`: Better accuracy
- `medium`: High accuracy
- `large`: Best accuracy, slowest

### Step 3: Topic Modeling
Applies a topic model to the entire conversation to identify 3-5 main subjects.

**Methods:**
- LDA (Latent Dirichlet Allocation) - scikit-learn
- BERTopic - transformer-based topic modeling

## Output Format

The pipeline generates a structured JSON output:

```json
{
  "analysis_summary": {
    "total_segments": 8,
    "unique_speakers": 3,
    "topics_identified": 5
  },
  "topics": [
    {
      "id": 0,
      "label": "Topic 1: business, strategy, marketing",
      "keywords": ["business", "strategy", "marketing", "growth", "revenue"]
    }
  ],
  "speaker_summary": {
    "SPEAKER_A": {
      "total_speaking_time": 23.5,
      "segment_count": 3,
      "topics_discussed": ["Topic 1: business, strategy, marketing"]
    }
  },
  "conversation_log": [
    {
      "speaker": "SPEAKER_A",
      "start_time": 0.0,
      "end_time": 8.5,
      "duration": 8.5,
      "transcription": "I think we need to focus on the quarterly results...",
      "topic": "Topic 1: business, strategy, marketing"
    }
  ]
}
```

## Using Real Speaker Diarization

For production use with real speaker diarization:

1. Install pyannote.audio:
   ```bash
   pip install pyannote.audio
   ```

2. Get a HuggingFace token from https://huggingface.co/settings/tokens

3. Accept model terms at https://huggingface.co/pyannote/speaker-diarization

4. Run with the token:
   ```bash
   python speaker_diarization_pipeline.py \
       --audio_path audio.wav \
       --use_real_diarization \
       --hf_token YOUR_HF_TOKEN
   ```

## License

MIT License - see LICENSE file for details