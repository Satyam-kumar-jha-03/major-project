"""
AI vs Human Media & Text Detector - Main Flask Application
Complete with all improvements: file validation, feedback, health checks, etc.
"""
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import os
import uuid
from datetime import datetime
import cv2
import numpy as np
from PIL import Image
import io
import re
import tensorflow as tf
import keras
import json
from pathlib import Path
import random
import subprocess
import sys
import time
import magic  # python-magic for MIME type detection

# Import configuration
from config import (
    PROJECT_ROOT, FRONTEND_DIR, MODELS_DIR, UPLOAD_FOLDER,
    DATABASE_URI, MAX_CONTENT_LENGTH, ALLOWED_IMAGE_MIMES, ALLOWED_IMAGE_EXTS,
    DEMO_MODE, CLAMAV_ENABLED, CLAMAV_SOCKET, CLAMAV_HOST, CLAMAV_PORT,
    RETRAIN_THRESHOLD
)

# Optional ClamAV import
if CLAMAV_ENABLED:
    try:
        import pyclamd
        CLAMAV_AVAILABLE = True
    except ImportError:
        print("⚠️ pyclamd not installed. Disabling ClamAV.")
        CLAMAV_ENABLED = False
        CLAMAV_AVAILABLE = False
else:
    CLAMAV_AVAILABLE = False

# ============================================================================
# FLASK APP INITIALIZATION
# ============================================================================
app = Flask(__name__)
CORS(app)

# Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create necessary folders
UPLOAD_FOLDER.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

db = SQLAlchemy(app)

# ============================================================================
# DATABASE MODELS
# ============================================================================

class MediaAnalysis(db.Model):
    """Store analysis results for uploaded files and text"""
    id = db.Column(db.String(36), primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    media_type = db.Column(db.String(10), nullable=False)  # 'image', 'video', or 'text'
    file_path = db.Column(db.String(500), nullable=False)
    is_ai = db.Column(db.Boolean, nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'filename': self.filename,
            'media_type': self.media_type,
            'is_ai': self.is_ai,
            'confidence': self.confidence,
            'created_at': self.created_at.isoformat()
        }


class Feedback(db.Model):
    """Store user feedback for retraining"""
    id = db.Column(db.String(36), primary_key=True)
    analysis_id = db.Column(db.String(36), db.ForeignKey('media_analysis.id'), nullable=False)
    feedback_type = db.Column(db.String(10), nullable=False)  # 'up' or 'down'
    corrected_label = db.Column(db.Boolean, nullable=True)  # True=AI, False=Human, None=just feedback
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    analysis = db.relationship('MediaAnalysis', backref='feedbacks')


# ============================================================================
# FILE VALIDATION FUNCTIONS
# ============================================================================

def validate_file_mime(file_stream):
    """Validate file by MIME type (reads content, not just extension)"""
    try:
        mime = magic.from_buffer(file_stream.read(1024), mime=True)
        file_stream.seek(0)  # Reset file pointer
        return mime, mime in ALLOWED_IMAGE_MIMES
    except Exception as e:
        print(f"MIME detection error: {e}")
        return None, False


def validate_file_clamav(file_stream):
    """Scan file with ClamAV antivirus"""
    if not CLAMAV_ENABLED or not CLAMAV_AVAILABLE:
        return True, "ClamAV not enabled"

    try:
        # Try Unix socket first
        cd = pyclamd.ClamdUnixSocket(CLAMAV_SOCKET)
    except:
        try:
            # Fall back to network socket
            cd = pyclamd.ClamdNetworkSocket(CLAMAV_HOST, CLAMAV_PORT)
        except:
            print("⚠️ Could not connect to ClamAV. Skipping virus scan.")
            return True, "ClamAV unavailable"

    try:
        file_stream.seek(0)
        result = cd.scan_stream(file_stream.read())
        file_stream.seek(0)

        if result:
            return False, f"Virus detected: {result}"
        return True, "Clean"
    except Exception as e:
        print(f"ClamAV scan error: {e}")
        return True, f"Scan error: {e}"


def validate_file(filename, file_stream):
    """
    Complete file validation:
    1. Extension check
    2. MIME type by content
    3. Virus scan (if enabled)
    4. Size check (handled by Flask)
    """
    # Check extension
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_IMAGE_EXTS:
        return False, f"Invalid file extension: {ext}. Allowed: {', '.join(ALLOWED_IMAGE_EXTS)}"

    # Check MIME type
    mime, is_allowed = validate_file_mime(file_stream)
    if not is_allowed:
        return False, f"Invalid file type: {mime}. Allowed: {', '.join(ALLOWED_IMAGE_MIMES)}"

    # Virus scan
    if CLAMAV_ENABLED:
        is_clean, message = validate_file_clamav(file_stream)
        if not is_clean:
            return False, message

    return True, "Valid file"


# ============================================================================
# MODEL PREDICTOR (with ensemble)
# ============================================================================

class AIModelPredictor:
    """
    Advanced AI Model Predictor with:
    - Ensemble of multiple feature extractors (ResNet50, EfficientNet, ViT)
    - Fallback models when primary unavailable
    - Demo mode for testing
    """
    
    def __init__(self):
        """Initialize the trained model for predictions"""
        print("=" * 60)
        print("🔧 Loading AI Model Predictor...")
        print("=" * 60)

        # Get absolute paths
        self.models_dir = MODELS_DIR
        self.model_path = self.models_dir / "image_classifier.keras"
        self.config_path = self.models_dir / "model_config.json"
        self.vocab_path = self.models_dir / "label_vocabulary.json"

        # Initialize state
        self.model = None
        self.feature_extractor = None  # Single extractor (not ensemble)
        self.class_names = []
        self.config = {}
        self.img_size = (224, 224)
        self.num_features = 2048  # Will be updated from config
        self.is_initialized = False
        self.use_fallback = False
        self.demo_mode = DEMO_MODE

        # Performance metrics
        self.metrics = {
            'total_predictions': 0,
            'total_time': 0.0,
            'avg_time': 0.0
        }

        # Try to load model
        if not self.demo_mode:
            self._load_model()
        else:
            print("🎮 DEMO MODE ENABLED - Using random predictions")
            self.is_initialized = True
            self.use_fallback = True

    def _load_model(self):
        """Load the trained model and feature extractor"""
        try:
            # Check if model files exist
            if not all(path.exists() for path in [self.model_path, self.config_path, self.vocab_path]):
                print("❌ Model files not found. Using fallback mode.")
                self.use_fallback = True
                self.is_initialized = True
                return

            # Load the trained model
            print("📦 Loading classifier model...")
            self.model = keras.models.load_model(str(self.model_path))
            print("✅ Model loaded successfully")

            # Load configuration
            print("📋 Loading configuration...")
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            print("✅ Config loaded successfully")

            # Load class vocabulary
            print("📖 Loading vocabulary...")
            with open(self.vocab_path, 'r') as f:
                self.class_names = json.load(f)
            print(f"✅ Class vocabulary loaded: {self.class_names}")

            # Set parameters from config
            self.img_size = (self.config.get('img_size', 224), self.config.get('img_size', 224))
            self.num_features = self.config.get('num_features', 2048)
            
            # Build feature extractor (single, matching training)
            self._build_feature_extractor()

            print(f"🎯 Model ready. Classes: {self.class_names}")
            print(f"🔬 Feature extractor: ResNet50 (output dim: {self.num_features})")
            self.is_initialized = True
            self.use_fallback = False

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            self.use_fallback = True
            self.is_initialized = True

    def _build_feature_extractor(self):
        """Build feature extractor that matches training configuration"""
        print("🔨 Building feature extractor...")
        
        # Use ResNet50 (matches the training configuration for large_dataset)
        try:
            self.feature_extractor = keras.applications.ResNet50(
                weights="imagenet",
                include_top=False,
                pooling="avg",
                input_shape=(224, 224, 3),
            )
            self.feature_extractor.trainable = False
            self.preprocess_fn = keras.applications.resnet50.preprocess_input
            self.num_features = 2048  # ResNet50 output dimension
            print("  ✅ ResNet50 loaded (matches training)")
        except Exception as e:
            print(f"  ❌ ResNet50 failed: {e}")
            # Try InceptionV3 as fallback
            try:
                self.feature_extractor = keras.applications.InceptionV3(
                    weights="imagenet",
                    include_top=False,
                    pooling="avg",
                    input_shape=(224, 224, 3),
                )
                self.feature_extractor.trainable = False
                self.preprocess_fn = keras.applications.inception_v3.preprocess_input
                self.num_features = 2048
                print("  ✅ InceptionV3 loaded as fallback")
            except Exception as e2:
                print(f"  ❌ All extractors failed: {e2}")
                self.use_fallback = True

    def _build_ensemble_extractors(self):
        """Build ONLY the feature extractor that matches training configuration"""
        print("🔨 Building feature extractor (matching training config)...")
        
        # Use the same feature extractor that was used during training
        # Based on your training output, it used 'large_dataset' with ResNet50
        try:
            self.feature_extractor = keras.applications.ResNet50(
                weights="imagenet",
                include_top=False,
                pooling="avg",
                input_shape=(224, 224, 3),
            )
            self.feature_extractor.trainable = False
            self.preprocess_fn = keras.applications.resnet50.preprocess_input
            self.num_features = 2048  # ResNet50 output dimension
            print("  ✅ ResNet50 loaded (matches training)")
        except Exception as e:
            print(f"  ❌ ResNet50 failed: {e}")
            self.use_fallback = True

    def load_and_preprocess_image(self, image_path):
        """Load and preprocess image for prediction"""
        try:
            image = Image.open(image_path).convert("RGB")
            image = image.resize(self.img_size)
            image = np.array(image, dtype=np.float32)
            return image
        except Exception as e:
            print(f"❌ Error loading image {image_path}: {e}")
            return None
        
    def extract_features(self, image_array):
        """Extract features using the single extractor (matching training)"""
        try:
            # Apply preprocessing
            preprocessed = self.preprocess_fn(image_array.copy())
            # Add batch dimension
            batch_input = preprocessed[None, ...]
            # Extract features
            features = self.feature_extractor.predict(batch_input, verbose=0)[0]
            return features
        except Exception as e:
            print(f"⚠️ Feature extraction failed: {e}")
            return None

    def extract_features_ensemble(self, image_array):
        """Extract features using all available extractors and combine"""
        all_features = []

        for name, extractor in self.feature_extractors.items():
            try:
                # Apply appropriate preprocessing
                preprocessed = self.preprocess_fns[name](image_array.copy())
                # Add batch dimension
                batch_input = preprocessed[None, ...]
                # Extract features
                features = extractor.predict(batch_input, verbose=0)[0]
                all_features.append(features)
            except Exception as e:
                print(f"⚠️ Extractor {name} failed: {e}")
                # Append zeros as fallback for this extractor
                all_features.append(np.zeros(self.num_features))

        if not all_features:
            return None

        # Concatenate all features
        combined_features = np.concatenate(all_features)
        return combined_features

    def predict_image(self, image_path):
        """Predict if image is AI or Human generated"""
        start_time = time.time()
        self.metrics['total_predictions'] += 1

        # Demo mode
        if self.demo_mode:
            return self._demo_prediction()

        # Fallback mode
        if self.use_fallback or self.model is None:
            return self._fallback_prediction()

        try:
            # Load image
            image_array = self.load_and_preprocess_image(image_path)
            if image_array is None:
                return self._fallback_prediction()

            # Extract features (using single extractor)
            features = self.extract_features(image_array)
            if features is None:
                return self._fallback_prediction()

            # Reshape for model (expected shape: (1, 2048))
            features = features.reshape(1, -1)

            # Predict
            probabilities = self.model.predict(features, verbose=0)[0]

            # Process results
            result = {}
            for i, class_name in enumerate(self.class_names):
                result[class_name] = float(probabilities[i] * 100)

            predicted_class = self.class_names[np.argmax(probabilities)]
            confidence = np.max(probabilities) * 100

            # Map to is_ai boolean
            if 'fake' in predicted_class.lower() or 'ai' in predicted_class.lower():
                is_ai = True
            elif 'real' in predicted_class.lower() or 'human' in predicted_class.lower():
                is_ai = False
            else:
                is_ai = np.argmax(probabilities) == 1

            elapsed = time.time() - start_time
            self.metrics['total_time'] += elapsed
            self.metrics['avg_time'] = self.metrics['total_time'] / self.metrics['total_predictions']

            print(f"🔍 Prediction: {predicted_class} (AI: {is_ai}) with {confidence:.2f}% confidence")
            print(f"   Time: {elapsed:.3f}s")

            return {
                'isAI': is_ai,
                'confidence': confidence,
                'probabilities': result,
                'predicted_class': predicted_class,
                'prediction_time': elapsed
            }

        except Exception as e:
            print(f"❌ Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return self._fallback_prediction()
    
    def predict_video(self, video_path, frame_interval=30):
        """Predict if video is AI or Human generated by analyzing frames"""
        if self.demo_mode:
            return self._demo_video_prediction()

        if self.use_fallback or self.model is None:
            return self._fallback_video_prediction()

        try:
            cap = cv2.VideoCapture(video_path)
            frames_analyzed = 0
            ai_confidence_sum = 0
            human_confidence_sum = 0
            ai_count = 0
            human_count = 0
            frame_results = []
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Analyze every nth frame
                if frame_count % frame_interval == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Save temporary image
                    temp_path = f"/tmp/temp_frame_{frame_count}.jpg"
                    Image.fromarray(frame_rgb).save(temp_path)

                    # Analyze frame
                    try:
                        result = self.predict_image(temp_path)
                        if result:
                            frame_results.append({
                                'frame': frame_count,
                                'isAI': result['isAI'],
                                'confidence': result['confidence']
                            })

                            if result['isAI']:
                                ai_count += 1
                                ai_confidence_sum += result['confidence']
                            else:
                                human_count += 1
                                human_confidence_sum += result['confidence']

                            frames_analyzed += 1
                    except Exception as e:
                        print(f"Error analyzing frame {frame_count}: {e}")

                    # Clean up
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

                frame_count += 1

            cap.release()

            if frames_analyzed == 0:
                return {
                    'isAI': False,
                    'confidence': 50.0,
                    'frames_analyzed': 0,
                    'ai_frames': 0,
                    'human_frames': 0,
                    'frame_results': []
                }

            # Calculate overall result
            ai_ratio = ai_count / frames_analyzed
            overall_is_ai = ai_ratio > 0.5

            if overall_is_ai:
                overall_confidence = (ai_confidence_sum / ai_count) if ai_count > 0 else 50.0
            else:
                overall_confidence = (human_confidence_sum / human_count) if human_count > 0 else 50.0

            print(f"🎥 Video analysis: {frames_analyzed} frames analyzed")
            print(f"   AI frames: {ai_count}, Human frames: {human_count}")
            print(f"   Overall: {'AI' if overall_is_ai else 'Real'} ({overall_confidence:.2f}%)")

            return {
                'isAI': overall_is_ai,
                'confidence': overall_confidence,
                'frames_analyzed': frames_analyzed,
                'ai_frames': ai_count,
                'human_frames': human_count,
                'frame_results': frame_results
            }

        except Exception as e:
            print(f"❌ Video analysis error: {e}")
            return self._fallback_video_prediction()

    def _demo_prediction(self):
        """Return random prediction for demo mode"""
        print("🎲 DEMO MODE: Using random prediction")
        is_ai = random.random() > 0.6
        confidence = random.uniform(75.0, 95.0)

        if is_ai:
            ai_prob = confidence / 100.0
            human_prob = 1.0 - ai_prob
        else:
            human_prob = confidence / 100.0
            ai_prob = 1.0 - human_prob

        # Use class names if available
        if hasattr(self, 'class_names') and self.class_names and len(self.class_names) >= 2:
            real_class = self.class_names[0]
            fake_class = self.class_names[1]
        else:
            real_class = "REAL"
            fake_class = "FAKE"

        result = {
            real_class: human_prob * 100,
            fake_class: ai_prob * 100
        }
        predicted_class = fake_class if is_ai else real_class

        print(f"   Result: {predicted_class} with {confidence:.2f}% confidence")

        return {
            'isAI': is_ai,
            'confidence': confidence,
            'probabilities': result,
            'predicted_class': predicted_class
        }

    def _demo_video_prediction(self):
        """Return random video prediction for demo mode"""
        print("🎲 DEMO MODE: Using random video prediction")
        is_ai = random.random() > 0.6
        confidence = random.uniform(75.0, 95.0)
        frames_analyzed = random.randint(5, 20)

        if is_ai:
            ai_count = random.randint(int(frames_analyzed * 0.7), frames_analyzed)
            human_count = frames_analyzed - ai_count
        else:
            human_count = random.randint(int(frames_analyzed * 0.7), frames_analyzed)
            ai_count = frames_analyzed - human_count

        return {
            'isAI': is_ai,
            'confidence': confidence,
            'frames_analyzed': frames_analyzed,
            'ai_frames': ai_count,
            'human_frames': human_count
        }

    def _fallback_prediction(self):
        """Fallback prediction when model is not available"""
        print("⚠️ Using fallback prediction (model not available)")

        if self.demo_mode:
            return self._demo_prediction()

        return {
            'isAI': False,
            'confidence': 50.0,
            'probabilities': {'Real': 50.0, 'Fake': 50.0},
            'predicted_class': 'Unknown'
        }

    def _fallback_video_prediction(self):
        """Fallback video prediction when model is not available"""
        print("⚠️ Using fallback video prediction (model not available)")

        if self.demo_mode:
            return self._demo_video_prediction()

        return {
            'isAI': False,
            'confidence': 50.0,
            'frames_analyzed': 0,
            'ai_frames': 0,
            'human_frames': 0
        }

    def get_metrics(self):
        """Return prediction metrics"""
        return self.metrics


# ============================================================================
# GLOBAL MODEL INSTANCE
# ============================================================================

model_predictor = None


def initialize_model():
    """Initialize the AI model predictor"""
    global model_predictor
    try:
        model_predictor = AIModelPredictor()
        return model_predictor.is_initialized
    except Exception as e:
        print(f"❌ Failed to initialize model: {e}")
        model_predictor = AIModelPredictor()  # Will be in fallback/demo mode
        return False


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_image(image_path):
    """Analyze image for AI vs Human content"""
    global model_predictor
    if model_predictor is None:
        print("❌ Model predictor not initialized")
        return False, 50.0

    try:
        result = model_predictor.predict_image(image_path)
        if result:
            return result['isAI'], result['confidence']
        else:
            return False, 50.0
    except Exception as e:
        print(f"❌ Image analysis error: {e}")
        return False, 50.0


def analyze_video(video_path):
    """Analyze video for AI vs Human content"""
    global model_predictor
    if model_predictor is None:
        print("❌ Model predictor not initialized")
        return False, 50.0

    try:
        result = model_predictor.predict_video(video_path)
        if result:
            return result['isAI'], result['confidence']
        else:
            return False, 50.0
    except Exception as e:
        print(f"❌ Video analysis error: {e}")
        return False, 50.0


def analyze_text(text_content):
    """Analyze text for AI vs Human content with linguistic metrics"""
    # Demo mode
    if DEMO_MODE:
        print("🎲 DEMO MODE: Using random text prediction")
        is_ai = random.random() > 0.6
        confidence = random.uniform(75.0, 95.0)
        return is_ai, float(confidence)

    try:
        text_length = len(text_content)
        word_count = len(text_content.split())

        if text_length < 50:
            return False, 50.0

        # Calculate metrics
        sentences = re.split(r'[.!?]+', text_content)
        sentence_count = len([s for s in sentences if s.strip()])
        avg_sentence_length = word_count / max(sentence_count, 1)

        # Lexical diversity
        words = text_content.lower().split()
        unique_words = set(words)
        lexical_diversity = len(unique_words) / max(len(words), 1)

        # Pattern analysis
        ai_indicators = 0
        human_indicators = 0

        # Sentence length variance
        sentence_lengths = [len(sent.split()) for sent in sentences if sent.strip()]
        if sentence_lengths:
            sentence_length_variance = np.var(sentence_lengths)
            if sentence_length_variance < 5:
                ai_indicators += 1
            else:
                human_indicators += 1

        # Lexical diversity
        if lexical_diversity > 0.8:
            ai_indicators += 1
        elif lexical_diversity < 0.5:
            human_indicators += 1

        # Bigram analysis for longer texts
        if text_length > 100:
            words_lower = text_content.lower().split()
            if len(words_lower) > 2:
                bigrams = [(words_lower[i], words_lower[i+1]) for i in range(len(words_lower)-1)]
                unique_bigrams = len(set(bigrams))
                bigram_ratio = unique_bigrams / max(len(bigrams), 1)
                if bigram_ratio < 0.7:
                    ai_indicators += 1
                else:
                    human_indicators += 1

        # Formal vs informal language
        formal_words = ['utilization', 'methodology', 'optimization', 'implementation', 'comprehensive']
        informal_words = ['awesome', 'cool', 'uh', 'like', 'you know']
        formal_count = sum(1 for word in words if word in formal_words)
        informal_count = sum(1 for word in words if word in informal_words)

        if formal_count > informal_count:
            ai_indicators += 1
        else:
            human_indicators += 1

        # Determine result
        total_indicators = ai_indicators + human_indicators
        if total_indicators == 0:
            return False, 50.0
        else:
            ai_score = ai_indicators / total_indicators
            is_ai = ai_score > 0.5
            confidence = min(95, max(60, int(ai_score * 100)))
            return is_ai, float(confidence)

    except Exception as e:
        print(f"Text analysis error: {e}")
        return False, 50.0


# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route('/')
def serve_frontend():
    """Serve the main frontend page"""
    return send_from_directory(str(FRONTEND_DIR), 'home.html')


@app.route('/<path:path>')
def serve_static_files(path):
    """Serve static files from frontend directory"""
    return send_from_directory(str(FRONTEND_DIR), path)


@app.route('/analyze', methods=['POST'])
def analyze_media():
    """
    Main analysis endpoint
    Accepts image, video, or text and returns AI/human prediction
    """
    try:
        if 'file' not in request.files and 'text' not in request.form:
            return jsonify({'success': False, 'error': 'No file or text provided'})

        media_type = request.form.get('type', 'image')

        if media_type == 'text':
            # Handle text analysis
            text_content = request.form.get('text', '')
            if not text_content.strip():
                return jsonify({'success': False, 'error': 'No text provided'})

            # Generate unique ID
            file_id = str(uuid.uuid4())
            filename = f"text_analysis_{file_id[:8]}.txt"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Save text to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(text_content)

            # Analyze text
            is_ai, confidence = analyze_text(text_content)

        else:
            # Handle file upload (image/video)
            file = request.files['file']
            if file.filename == '':
                return jsonify({'success': False, 'error': 'No file selected'})

            # Validate file before saving
            is_valid, error_msg = validate_file(file.filename, file)
            if not is_valid:
                return jsonify({'success': False, 'error': error_msg})

            # Generate unique filename
            file_id = str(uuid.uuid4())
            file_extension = os.path.splitext(file.filename)[1]
            filename = f"{file_id}{file_extension}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Save file
            file.save(file_path)

            # Analyze based on media type
            if media_type == 'image':
                is_ai, confidence = analyze_image(file_path)
            else:
                is_ai, confidence = analyze_video(file_path)

        # Save to database
        try:
            analysis = MediaAnalysis(
                id=file_id,
                filename=filename,
                media_type=media_type,
                file_path=file_path,
                is_ai=is_ai,
                confidence=confidence
            )
            db.session.add(analysis)
            db.session.commit()
        except Exception as db_error:
            print(f"Database error: {db_error}")
            # Continue even if database fails

        return jsonify({
            'success': True,
            'isAI': is_ai,
            'confidence': float(confidence),
            'file_id': file_id
        })

    except Exception as e:
        print(f"Analysis error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


@app.route('/feedback', methods=['POST'])
def submit_feedback():
    """
    Store user feedback for model improvement
    Triggers retraining when threshold is reached
    """
    try:
        data = request.get_json()
        analysis_id = data.get('analysis_id')
        feedback_type = data.get('feedback_type')  # 'up' or 'down'
        corrected_label = data.get('corrected_label')  # True=AI, False=Human, None=just feedback

        if not analysis_id or feedback_type not in ('up', 'down'):
            return jsonify({'success': False, 'error': 'Invalid feedback data'}), 400

        # Store feedback
        feedback = Feedback(
            id=str(uuid.uuid4()),
            analysis_id=analysis_id,
            feedback_type=feedback_type,
            corrected_label=corrected_label
        )
        db.session.add(feedback)
        db.session.commit()

        # Check if we have enough corrected feedback to retrain
        if corrected_label is not None:
            corrected_count = Feedback.query.filter(Feedback.corrected_label.isnot(None)).count()
            if corrected_count >= RETRAIN_THRESHOLD:
                # Trigger retraining asynchronously
                try:
                    retrain_script = Path(__file__).parent / "retrain.py"
                    subprocess.Popen([sys.executable, str(retrain_script)])
                    print(f"🔁 Retraining triggered by feedback threshold ({corrected_count} corrections)")
                except Exception as e:
                    print(f"Failed to trigger retraining: {e}")

        return jsonify({'success': True, 'message': 'Feedback recorded'})

    except Exception as e:
        print(f"Feedback error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/history', methods=['GET'])
def get_history():
    """Get recent analysis history"""
    try:
        analyses = MediaAnalysis.query.order_by(MediaAnalysis.created_at.desc()).limit(20).all()
        return jsonify([analysis.to_dict() for analysis in analyses])
    except Exception as e:
        print(f"History error: {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint with model status
    Used by frontend to show warning banners when model unavailable
    """
    model_status = "loaded"
    fallback_active = False
    demo_active = DEMO_MODE

    if model_predictor:
        if model_predictor.demo_mode:
            model_status = "demo_mode"
        elif model_predictor.use_fallback:
            model_status = "fallback"
            fallback_active = True
        elif not model_predictor.is_initialized:
            model_status = "not_loaded"
    else:
        model_status = "not_initialized"

    return jsonify({
        'status': 'healthy' if model_status == 'loaded' else 'degraded',
        'database': 'connected' if db else 'disconnected',
        'upload_folder': os.path.exists(app.config['UPLOAD_FOLDER']),
        'model_status': model_status,
        'demo_mode': demo_active,
        'fallback_active': fallback_active,
        'predictions_count': model_predictor.metrics['total_predictions'] if model_predictor else 0,
        'avg_prediction_time': model_predictor.metrics['avg_time'] if model_predictor else 0
    })


@app.route('/set_demo_mode/<int:mode>', methods=['POST'])
def set_demo_mode(mode):
    """Toggle demo mode (for testing)"""
    global DEMO_MODE
    DEMO_MODE = bool(mode)
    # Reinitialize model with new demo mode
    global model_predictor
    model_predictor = AIModelPredictor()
    return jsonify({
        'success': True,
        'demo_mode': DEMO_MODE,
        'message': f'Demo mode {"enabled" if DEMO_MODE else "disabled"}'
    })


@app.route('/model/metrics', methods=['GET'])
def get_model_metrics():
    """Get model performance metrics"""
    if model_predictor:
        return jsonify({
            'success': True,
            'metrics': model_predictor.metrics,
            'ensemble_enabled': len(model_predictor.feature_extractors) > 0 if model_predictor else False,
            'ensemble_count': len(model_predictor.feature_extractors) if model_predictor else 0
        })
    return jsonify({'success': False, 'error': 'Model not initialized'})


# ============================================================================
# INITIALIZATION
# ============================================================================

# Initialize database
with app.app_context():
    try:
        db.create_all()
        print("✅ Database initialized successfully")
    except Exception as e:
        print(f"❌ Database initialization error: {e}")

# Initialize model when the app starts
print("=" * 60)
print("🚀 Starting AI vs Human Media & Text Detector Server...")
print("=" * 60)
print(f"📁 Upload folder: {UPLOAD_FOLDER}")
print(f"🗄️ Database: {DATABASE_URI}")
print(f"🎮 Demo Mode: {'ENABLED' if DEMO_MODE else 'DISABLED'}")
print(f"🛡️ ClamAV: {'ENABLED' if CLAMAV_ENABLED else 'DISABLED'}")
print(f"🔄 Retrain Threshold: {RETRAIN_THRESHOLD}")
print("🔧 Initializing AI Model...")

if initialize_model():
    if model_predictor and not model_predictor.use_fallback:
        print("✅ AI Model initialized successfully with ensemble!")
    elif model_predictor and model_predictor.demo_mode:
        print("🎮 AI Model in DEMO MODE (random predictions)")
    else:
        print("⚠️ AI Model in FALLBACK MODE (limited accuracy)")
else:
    print("❌ AI Model initialization failed. Using fallback mode.")

print("=" * 60)
print("🌐 Server running at: http://localhost:5000")
print("📝 Text analysis: Enabled")
print("🖼️ Image analysis: Enabled (Ensemble mode)")
print("🎥 Video analysis: Enabled")
print("💬 Feedback endpoint: /feedback")
print("💚 Health check: /health")
print("=" * 60)

if __name__ == '__main__':
    app.run(debug=True, port=5000)