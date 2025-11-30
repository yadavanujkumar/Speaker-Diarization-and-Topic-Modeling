#!/usr/bin/env python3
"""
Speaker Diarization, Transcription, and Topic Modeling Pipeline

This script performs:
1. Speaker Diarization - Identifies different speakers in an audio file
2. Transcription - Converts speech to text using Whisper
3. Topic Modeling - Identifies main topics using LDA or BERTopic

Usage:
    python speaker_diarization_pipeline.py --audio_path <path_to_audio_file>
    python speaker_diarization_pipeline.py --demo  # Run with simulated data

Output: JSON log showing who said what and the main topic of each speech segment.
"""

import argparse
import json
import os
import sys
import warnings
from dataclasses import dataclass
from typing import Optional

warnings.filterwarnings("ignore")


# =============================================================================
# Data Classes for Structured Output
# =============================================================================

@dataclass
class SpeakerSegment:
    """Represents a single speaker segment with timing and transcription."""
    speaker: str
    start_time: float
    end_time: float
    text: str = ""
    topic: str = ""


# =============================================================================
# Audio Loading Module (using pydub)
# =============================================================================

def load_audio(audio_path: str):
    """
    Load an audio file using pydub.
    
    Args:
        audio_path: Path to the audio file (supports mp3, wav, etc.)
        
    Returns:
        AudioSegment object or None if loading fails
    """
    try:
        from pydub import AudioSegment
        
        if not os.path.exists(audio_path):
            print(f"Error: Audio file not found at {audio_path}")
            return None
            
        # Detect format from extension
        file_ext = os.path.splitext(audio_path)[1].lower().replace('.', '')
        
        # Load audio based on format
        if file_ext == 'mp3':
            audio = AudioSegment.from_mp3(audio_path)
        elif file_ext == 'wav':
            audio = AudioSegment.from_wav(audio_path)
        elif file_ext == 'ogg':
            audio = AudioSegment.from_ogg(audio_path)
        elif file_ext == 'flac':
            audio = AudioSegment.from_file(audio_path, format='flac')
        else:
            audio = AudioSegment.from_file(audio_path)
            
        print(f"Loaded audio: {audio_path}")
        print(f"  Duration: {len(audio) / 1000:.2f} seconds")
        print(f"  Channels: {audio.channels}")
        print(f"  Sample Rate: {audio.frame_rate} Hz")
        
        return audio
        
    except ImportError:
        print("Error: pydub is not installed. Install it with: pip install pydub")
        return None
    except Exception as e:
        print(f"Error loading audio: {e}")
        return None


def extract_segment(audio, start_time: float, end_time: float):
    """
    Extract a segment from the audio.
    
    Args:
        audio: AudioSegment object
        start_time: Start time in seconds
        end_time: End time in seconds
        
    Returns:
        AudioSegment of the extracted portion
    """
    start_ms = int(start_time * 1000)
    end_ms = int(end_time * 1000)
    return audio[start_ms:end_ms]


# =============================================================================
# Speaker Diarization Module
# =============================================================================

class SpeakerDiarizer:
    """
    Speaker Diarization class.
    
    This provides two modes:
    1. Simulated diarization for demo purposes
    2. Real diarization using pyannote.audio (requires HuggingFace token)
    
    For production use with a real diarization model:
    - Install: pip install pyannote.audio
    - Get a HuggingFace token and accept model terms at:
      https://huggingface.co/pyannote/speaker-diarization
    """
    
    def __init__(self, use_real_model: bool = False, hf_token: Optional[str] = None):
        """
        Initialize the diarizer.
        
        Args:
            use_real_model: If True, attempts to load pyannote.audio model
            hf_token: HuggingFace token for pyannote.audio
        """
        self.use_real_model = use_real_model
        self.pipeline = None
        
        if use_real_model:
            self._load_real_model(hf_token)
    
    def _load_real_model(self, hf_token: Optional[str]):
        """
        Load the real pyannote.audio diarization model.
        
        PLACEHOLDER INSTRUCTIONS:
        To use a real diarization model:
        1. Install pyannote.audio: pip install pyannote.audio
        2. Get a HuggingFace token from https://huggingface.co/settings/tokens
        3. Accept terms at https://huggingface.co/pyannote/speaker-diarization
        4. Pass the token to this class
        """
        try:
            from pyannote.audio import Pipeline
            
            if not hf_token:
                print("Warning: No HuggingFace token provided for pyannote.audio")
                print("Using simulated diarization instead.")
                self.use_real_model = False
                return
                
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization@2.1",
                use_auth_token=hf_token
            )
            print("Loaded pyannote.audio speaker diarization model")
            
        except ImportError:
            print("Warning: pyannote.audio not installed.")
            print("Install with: pip install pyannote.audio")
            print("Using simulated diarization instead.")
            self.use_real_model = False
            
        except Exception as e:
            print(f"Warning: Could not load diarization model: {e}")
            print("Using simulated diarization instead.")
            self.use_real_model = False
    
    def diarize(self, audio_path: str = None, audio_duration: float = None) -> list:
        """
        Perform speaker diarization.
        
        Args:
            audio_path: Path to audio file (for real model)
            audio_duration: Duration in seconds (for simulation)
            
        Returns:
            List of SpeakerSegment objects with speaker labels and timing
        """
        if self.use_real_model and self.pipeline:
            return self._real_diarization(audio_path)
        else:
            return self._simulated_diarization(audio_duration or 60.0)
    
    def _real_diarization(self, audio_path: str) -> list:
        """Perform real diarization using pyannote.audio."""
        diarization = self.pipeline(audio_path)
        
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segment = SpeakerSegment(
                speaker=speaker,
                start_time=turn.start,
                end_time=turn.end
            )
            segments.append(segment)
            
        return segments
    
    def _simulated_diarization(self, duration: float) -> list:
        """
        Generate simulated speaker diarization for demo purposes.
        
        This simulates a conversation between multiple speakers with
        realistic turn-taking patterns.
        """
        print("\n=== Using Simulated Diarization ===")
        print("(For real diarization, install pyannote.audio and provide HF token)")
        
        # Simulate a multi-speaker conversation
        segments = [
            SpeakerSegment("SPEAKER_A", 0.0, 8.5),
            SpeakerSegment("SPEAKER_B", 8.7, 15.2),
            SpeakerSegment("SPEAKER_A", 15.5, 22.0),
            SpeakerSegment("SPEAKER_C", 22.3, 30.1),
            SpeakerSegment("SPEAKER_B", 30.4, 38.0),
            SpeakerSegment("SPEAKER_A", 38.2, 45.5),
            SpeakerSegment("SPEAKER_C", 45.8, 52.0),
            SpeakerSegment("SPEAKER_B", 52.3, 60.0),
        ]
        
        # Adjust segments to fit actual audio duration if provided
        if duration < 60.0:
            scale = duration / 60.0
            for seg in segments:
                seg.start_time *= scale
                seg.end_time *= scale
                if seg.end_time > duration:
                    seg.end_time = duration
                    
        # Filter out segments that exceed duration
        segments = [s for s in segments if s.start_time < duration]
        
        return segments


# =============================================================================
# Transcription Module (using Whisper)
# =============================================================================

class Transcriber:
    """
    Transcription class using OpenAI's Whisper model.
    
    Whisper is a robust speech recognition model that can:
    - Transcribe audio in multiple languages
    - Handle various audio qualities
    - Detect language automatically
    """
    
    def __init__(self, model_name: str = "base"):
        """
        Initialize the Whisper transcriber.
        
        Args:
            model_name: Whisper model size - one of:
                        'tiny', 'base', 'small', 'medium', 'large'
                        Larger models are more accurate but slower.
        """
        self.model = None
        self.model_name = model_name
        self._load_model()
    
    def _load_model(self):
        """Load the Whisper model."""
        try:
            import whisper
            
            print(f"\nLoading Whisper model: {self.model_name}")
            self.model = whisper.load_model(self.model_name)
            print(f"Whisper model '{self.model_name}' loaded successfully")
            
        except ImportError:
            print("Error: whisper is not installed.")
            print("Install with: pip install openai-whisper")
            self.model = None
            
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            self.model = None
    
    def transcribe(self, audio_path: str = None, audio_segment=None) -> str:
        """
        Transcribe audio to text.
        
        Args:
            audio_path: Path to audio file
            audio_segment: pydub AudioSegment (will be saved temporarily)
            
        Returns:
            Transcribed text
        """
        if self.model is None:
            return self._simulated_transcription()
        
        try:
            # If audio segment provided, save it temporarily
            temp_path = None
            if audio_segment is not None:
                temp_path = "/tmp/temp_segment.wav"
                audio_segment.export(temp_path, format="wav")
                audio_path = temp_path
            
            if not audio_path:
                return ""
                
            # Perform transcription
            result = self.model.transcribe(audio_path)
            
            # Clean up temp file
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
                
            return result["text"].strip()
            
        except Exception as e:
            print(f"Transcription error: {e}")
            return self._simulated_transcription()
    
    def _simulated_transcription(self) -> str:
        """Return simulated transcription for demo purposes."""
        import random
        
        simulated_texts = [
            "I think we need to focus on the quarterly results and improve our marketing strategy.",
            "The customer feedback has been overwhelmingly positive about the new product features.",
            "We should schedule a follow-up meeting to discuss the technical implementation details.",
            "The budget allocation for next year needs to be finalized by the end of this month.",
            "I agree with the proposed timeline, but we need more resources for the development team.",
            "Let me summarize the key action items from today's discussion.",
            "The project milestone is on track, and we expect to deliver on schedule.",
            "We need to address the security concerns raised in the recent audit report.",
        ]
        
        return random.choice(simulated_texts)
    
    def transcribe_segments(self, segments: list, audio=None) -> list:
        """
        Transcribe multiple speaker segments.
        
        Args:
            segments: List of SpeakerSegment objects
            audio: pydub AudioSegment for extracting portions
            
        Returns:
            List of SpeakerSegment objects with transcribed text
        """
        print("\n=== Transcribing Speaker Segments ===")
        
        for i, segment in enumerate(segments):
            print(f"Transcribing segment {i+1}/{len(segments)}: "
                  f"{segment.speaker} ({segment.start_time:.1f}s - {segment.end_time:.1f}s)")
            
            if audio is not None and self.model is not None:
                # Extract and transcribe the specific segment
                audio_portion = extract_segment(audio, segment.start_time, segment.end_time)
                segment.text = self.transcribe(audio_segment=audio_portion)
            else:
                # Use simulated transcription
                segment.text = self._simulated_transcription()
                
        return segments


# =============================================================================
# Topic Modeling Module
# =============================================================================

class TopicModeler:
    """
    Topic Modeling class using LDA (Latent Dirichlet Allocation) or BERTopic.
    
    Identifies main topics/subjects from the conversation transcript.
    """
    
    def __init__(self, method: str = "lda", n_topics: int = 5):
        """
        Initialize the topic modeler.
        
        Args:
            method: 'lda' for sklearn LDA, 'bertopic' for BERTopic
            n_topics: Number of topics to extract (3-5 recommended)
        """
        self.method = method.lower()
        self.n_topics = n_topics
        self.model = None
        self.vectorizer = None
        
    def fit_and_extract(self, texts: list) -> dict:
        """
        Fit topic model and extract topics from texts.
        
        Args:
            texts: List of text strings (one per speaker segment)
            
        Returns:
            Dictionary with topic information
        """
        if len(texts) == 0:
            return {"topics": [], "document_topics": []}
            
        if self.method == "bertopic":
            return self._bertopic_modeling(texts)
        else:
            return self._lda_modeling(texts)
    
    def _lda_modeling(self, texts: list) -> dict:
        """Perform LDA topic modeling using scikit-learn."""
        try:
            from sklearn.feature_extraction.text import CountVectorizer
            from sklearn.decomposition import LatentDirichletAllocation
            
            print("\n=== Performing Topic Modeling (LDA) ===")
            
            # Filter out empty texts
            valid_texts = [t for t in texts if t and len(t.strip()) > 0]
            
            if len(valid_texts) < 2:
                print("Warning: Not enough text for topic modeling")
                return self._simulated_topics(texts)
            
            # Vectorize the texts
            self.vectorizer = CountVectorizer(
                max_df=0.95,
                min_df=1,
                max_features=1000,
                stop_words='english'
            )
            
            try:
                doc_term_matrix = self.vectorizer.fit_transform(valid_texts)
            except ValueError:
                # If vocabulary is empty, return simulated topics
                return self._simulated_topics(texts)
            
            # Adjust number of topics based on document count
            n_topics = min(self.n_topics, max(1, len(valid_texts) - 1))
            
            # Fit LDA model
            self.model = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=10
            )
            self.model.fit(doc_term_matrix)
            
            # Extract topic keywords
            feature_names = self.vectorizer.get_feature_names_out()
            topics = []
            
            for topic_idx, topic in enumerate(self.model.components_):
                top_word_indices = topic.argsort()[:-6:-1]
                top_words = [feature_names[i] for i in top_word_indices]
                topic_label = f"Topic {topic_idx + 1}: {', '.join(top_words[:3])}"
                topics.append({
                    "id": topic_idx,
                    "label": topic_label,
                    "keywords": top_words
                })
            
            # Get topic assignments for each document
            doc_topics = self.model.transform(doc_term_matrix)
            document_topics = [int(topics.argmax()) for topics in doc_topics]
            
            # Map back to original texts (including empty ones)
            full_document_topics = []
            valid_idx = 0
            for text in texts:
                if text and len(text.strip()) > 0:
                    full_document_topics.append(document_topics[valid_idx])
                    valid_idx += 1
                else:
                    full_document_topics.append(0)  # Default topic
            
            return {
                "topics": topics,
                "document_topics": full_document_topics
            }
            
        except ImportError:
            print("Warning: scikit-learn not installed.")
            print("Install with: pip install scikit-learn")
            return self._simulated_topics(texts)
            
        except Exception as e:
            print(f"Warning: LDA modeling failed: {e}")
            return self._simulated_topics(texts)
    
    def _bertopic_modeling(self, texts: list) -> dict:
        """Perform topic modeling using BERTopic."""
        try:
            from bertopic import BERTopic
            
            print("\n=== Performing Topic Modeling (BERTopic) ===")
            
            # Filter out empty texts
            valid_texts = [t for t in texts if t and len(t.strip()) > 0]
            
            if len(valid_texts) < 2:
                print("Warning: Not enough text for topic modeling")
                return self._simulated_topics(texts)
            
            # Create and fit BERTopic model
            self.model = BERTopic(
                nr_topics=self.n_topics,
                verbose=False
            )
            
            topics, _ = self.model.fit_transform(valid_texts)
            
            # Extract topic information
            topic_info = self.model.get_topic_info()
            topics_list = []
            
            for _, row in topic_info.iterrows():
                if row['Topic'] != -1:  # Skip outlier topic
                    topic_keywords = self.model.get_topic(row['Topic'])
                    keywords = [word for word, _ in topic_keywords[:5]]
                    topics_list.append({
                        "id": row['Topic'],
                        "label": f"Topic {row['Topic'] + 1}: {', '.join(keywords[:3])}",
                        "keywords": keywords
                    })
            
            # Map back to original texts
            full_document_topics = []
            valid_idx = 0
            for text in texts:
                if text and len(text.strip()) > 0:
                    full_document_topics.append(max(0, topics[valid_idx]))
                    valid_idx += 1
                else:
                    full_document_topics.append(0)
            
            return {
                "topics": topics_list,
                "document_topics": full_document_topics
            }
            
        except ImportError:
            print("Warning: BERTopic not installed.")
            print("Install with: pip install bertopic")
            print("Falling back to LDA...")
            return self._lda_modeling(texts)
            
        except Exception as e:
            print(f"Warning: BERTopic modeling failed: {e}")
            return self._simulated_topics(texts)
    
    def _simulated_topics(self, texts: list) -> dict:
        """Return simulated topics for demo purposes."""
        print("Using simulated topic modeling")
        
        simulated_topics = [
            {
                "id": 0,
                "label": "Topic 1: Business, Strategy, Marketing",
                "keywords": ["business", "strategy", "marketing", "growth", "revenue"]
            },
            {
                "id": 1,
                "label": "Topic 2: Product, Development, Features",
                "keywords": ["product", "development", "features", "design", "launch"]
            },
            {
                "id": 2,
                "label": "Topic 3: Team, Resources, Timeline",
                "keywords": ["team", "resources", "timeline", "deadline", "milestone"]
            },
            {
                "id": 3,
                "label": "Topic 4: Customer, Feedback, Support",
                "keywords": ["customer", "feedback", "support", "satisfaction", "service"]
            },
            {
                "id": 4,
                "label": "Topic 5: Budget, Finance, Planning",
                "keywords": ["budget", "finance", "planning", "allocation", "costs"]
            }
        ]
        
        # Assign random topics to documents
        import random
        document_topics = [random.randint(0, min(4, len(texts) - 1)) for _ in texts]
        
        return {
            "topics": simulated_topics[:self.n_topics],
            "document_topics": document_topics
        }


# =============================================================================
# Main Pipeline
# =============================================================================

class SpeechAnalysisPipeline:
    """
    Main pipeline for Speaker Diarization, Transcription, and Topic Modeling.
    
    This class orchestrates the complete analysis workflow:
    1. Load audio file
    2. Perform speaker diarization
    3. Transcribe each speaker's segments
    4. Apply topic modeling to the conversation
    5. Generate structured JSON output
    """
    
    def __init__(
        self,
        whisper_model: str = "base",
        topic_method: str = "lda",
        n_topics: int = 5,
        use_real_diarization: bool = False,
        hf_token: Optional[str] = None
    ):
        """
        Initialize the pipeline.
        
        Args:
            whisper_model: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
            topic_method: Topic modeling method ('lda' or 'bertopic')
            n_topics: Number of topics to extract
            use_real_diarization: Whether to use real pyannote.audio model
            hf_token: HuggingFace token for pyannote.audio
        """
        self.diarizer = SpeakerDiarizer(use_real_diarization, hf_token)
        self.transcriber = Transcriber(whisper_model)
        self.topic_modeler = TopicModeler(topic_method, n_topics)
        
        self.audio = None
        self.segments = []
        self.topic_results = {}
    
    def run(self, audio_path: str = None, demo_mode: bool = False) -> dict:
        """
        Run the complete analysis pipeline.
        
        Args:
            audio_path: Path to the audio file
            demo_mode: If True, run with simulated data
            
        Returns:
            Dictionary with complete analysis results
        """
        print("=" * 60)
        print("SPEAKER DIARIZATION, TRANSCRIPTION & TOPIC MODELING PIPELINE")
        print("=" * 60)
        
        # Step 0: Load audio (if provided)
        audio_duration = 60.0  # Default for simulation
        
        if audio_path and os.path.exists(audio_path):
            self.audio = load_audio(audio_path)
            if self.audio:
                audio_duration = len(self.audio) / 1000.0
        elif not demo_mode:
            print("\nNo audio file provided or file not found.")
            print("Running in demo mode with simulated data.")
            demo_mode = True
        
        # Step 1: Speaker Diarization
        print("\n" + "=" * 40)
        print("STEP 1: SPEAKER DIARIZATION")
        print("=" * 40)
        
        self.segments = self.diarizer.diarize(
            audio_path=audio_path,
            audio_duration=audio_duration
        )
        
        print(f"\nIdentified {len(self.segments)} speaker segments:")
        for seg in self.segments:
            print(f"  {seg.speaker}: {seg.start_time:.1f}s - {seg.end_time:.1f}s")
        
        # Step 2: Transcription
        print("\n" + "=" * 40)
        print("STEP 2: TRANSCRIPTION (Whisper)")
        print("=" * 40)
        
        self.segments = self.transcriber.transcribe_segments(
            self.segments,
            audio=self.audio
        )
        
        # Step 3: Topic Modeling
        print("\n" + "=" * 40)
        print("STEP 3: TOPIC MODELING")
        print("=" * 40)
        
        texts = [seg.text for seg in self.segments]
        self.topic_results = self.topic_modeler.fit_and_extract(texts)
        
        # Assign topics to segments
        for i, seg in enumerate(self.segments):
            if i < len(self.topic_results.get("document_topics", [])):
                topic_idx = self.topic_results["document_topics"][i]
                if topic_idx < len(self.topic_results.get("topics", [])):
                    seg.topic = self.topic_results["topics"][topic_idx]["label"]
                else:
                    seg.topic = "Unknown Topic"
            else:
                seg.topic = "Unknown Topic"
        
        # Generate output
        return self._generate_output()
    
    def _generate_output(self) -> dict:
        """Generate structured JSON output."""
        
        # Build conversation log
        conversation_log = []
        for seg in self.segments:
            conversation_log.append({
                "speaker": seg.speaker,
                "start_time": round(seg.start_time, 2),
                "end_time": round(seg.end_time, 2),
                "duration": round(seg.end_time - seg.start_time, 2),
                "transcription": seg.text,
                "topic": seg.topic
            })
        
        # Build summary by speaker
        speaker_summary = {}
        for seg in self.segments:
            if seg.speaker not in speaker_summary:
                speaker_summary[seg.speaker] = {
                    "total_speaking_time": 0,
                    "segment_count": 0,
                    "topics_discussed": set()
                }
            speaker_summary[seg.speaker]["total_speaking_time"] += seg.end_time - seg.start_time
            speaker_summary[seg.speaker]["segment_count"] += 1
            speaker_summary[seg.speaker]["topics_discussed"].add(seg.topic)
        
        # Convert sets to lists for JSON serialization
        for speaker in speaker_summary:
            speaker_summary[speaker]["topics_discussed"] = list(
                speaker_summary[speaker]["topics_discussed"]
            )
            speaker_summary[speaker]["total_speaking_time"] = round(
                speaker_summary[speaker]["total_speaking_time"], 2
            )
        
        output = {
            "analysis_summary": {
                "total_segments": len(self.segments),
                "unique_speakers": len(speaker_summary),
                "topics_identified": len(self.topic_results.get("topics", [])),
            },
            "topics": self.topic_results.get("topics", []),
            "speaker_summary": speaker_summary,
            "conversation_log": conversation_log
        }
        
        return output
    
    def print_formatted_output(self, output: dict):
        """Print the output in a human-readable format."""
        
        print("\n" + "=" * 60)
        print("FINAL OUTPUT: STRUCTURED CONVERSATION ANALYSIS")
        print("=" * 60)
        
        print("\n--- ANALYSIS SUMMARY ---")
        summary = output["analysis_summary"]
        print(f"Total Segments: {summary['total_segments']}")
        print(f"Unique Speakers: {summary['unique_speakers']}")
        print(f"Topics Identified: {summary['topics_identified']}")
        
        print("\n--- IDENTIFIED TOPICS ---")
        for topic in output["topics"]:
            print(f"  {topic['label']}")
            print(f"    Keywords: {', '.join(topic['keywords'])}")
        
        print("\n--- SPEAKER SUMMARY ---")
        for speaker, info in output["speaker_summary"].items():
            print(f"  {speaker}:")
            print(f"    Speaking Time: {info['total_speaking_time']:.1f}s")
            print(f"    Segments: {info['segment_count']}")
            print(f"    Topics: {', '.join(info['topics_discussed'])}")
        
        print("\n--- CONVERSATION LOG ---")
        for entry in output["conversation_log"]:
            print(f"\n[{entry['speaker']}] ({entry['start_time']}s - {entry['end_time']}s)")
            print(f"  Topic: {entry['topic']}")
            print(f"  \"{entry['transcription']}\"")
        
        print("\n" + "=" * 60)


# =============================================================================
# Entry Point
# =============================================================================

def main():
    """Main entry point for the script."""
    
    parser = argparse.ArgumentParser(
        description="Speaker Diarization, Transcription, and Topic Modeling Pipeline"
    )
    parser.add_argument(
        "--audio_path",
        type=str,
        help="Path to the audio file to analyze"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run in demo mode with simulated data"
    )
    parser.add_argument(
        "--whisper_model",
        type=str,
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)"
    )
    parser.add_argument(
        "--topic_method",
        type=str,
        default="lda",
        choices=["lda", "bertopic"],
        help="Topic modeling method (default: lda)"
    )
    parser.add_argument(
        "--n_topics",
        type=int,
        default=5,
        help="Number of topics to extract (default: 5)"
    )
    parser.add_argument(
        "--use_real_diarization",
        action="store_true",
        help="Use real pyannote.audio diarization (requires HF token)"
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        help="HuggingFace token for pyannote.audio"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file path (optional)"
    )
    
    args = parser.parse_args()
    
    # Determine run mode
    demo_mode = args.demo or (not args.audio_path)
    
    # Initialize and run pipeline
    pipeline = SpeechAnalysisPipeline(
        whisper_model=args.whisper_model,
        topic_method=args.topic_method,
        n_topics=args.n_topics,
        use_real_diarization=args.use_real_diarization,
        hf_token=args.hf_token
    )
    
    # Run the pipeline
    output = pipeline.run(
        audio_path=args.audio_path,
        demo_mode=demo_mode
    )
    
    # Print formatted output
    pipeline.print_formatted_output(output)
    
    # Save JSON output if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nJSON output saved to: {args.output}")
    
    # Always print JSON to stdout as well
    print("\n--- JSON OUTPUT ---")
    print(json.dumps(output, indent=2))
    
    return output


if __name__ == "__main__":
    main()
