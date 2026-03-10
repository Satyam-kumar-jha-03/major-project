from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS # type: ignore
from flask_sqlalchemy import SQLAlchemy # type: ignore
import os
import uuid
from datetime import datetime
import cv2
import keras
import numpy as np
from PIL import Image
import io
import re
import tensorflow as tf
import json
from pathlib import Path
import random  # Keep random for optional demo mode

app = Flask(__name__)
CORS(app)

# Configuration - add DEMO_MODE flag
DEMO_MODE = os.environ.get('DEMO_MODE', 'false').lower() == 'true'  # Set to True for random predictions

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///analysis.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Create necessary folders
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('../models', exist_ok=True)

db = SQLAlchemy(app)

# Database Model
class MediaAnalysis(db.Model):
    id = db.Column(db.String(36), primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    media_type = db.Column(db.String(10), nullable=False)
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

# Initialize database
with app.app_context():
    try:
        db.create_all()
        print("✅ Database initialized successfully")
    except Exception as e:
        print(f"❌ Database initialization error: {e}")

class AIModelPredictor:
    def __init__(self):
        """Initialize the trained model for predictions"""
        print("🔧 Loading AI Model...")
        
        # Get absolute paths - FIXED: More robust path handling
        current_dir = Path(__file__).parent.absolute()
        project_root = current_dir.parent.absolute()
        models_dir = project_root / "models"
        
        print(f"📁 Looking for models in: {models_dir}")
        
        # Check if models directory exists
        if not models_dir.exists():
            print(f"❌ Models directory not found at: {models_dir}")
            print("Creating models directory...")
            models_dir.mkdir(parents=True, exist_ok=True)
        
        # Define model paths
        self.model_path = models_dir / "image_classifier.keras"
        self.config_path = models_dir / "model_config.json"
        self.vocab_path = models_dir / "label_vocabulary.json"
        
        # List all files in models directory for debugging
        if models_dir.exists():
            print(f"📂 Files in {models_dir}:")
            for f in models_dir.glob("*"):
                print(f"   - {f.name}")
        
        try:
            # Check if model files exist
            if not self.model_path.exists():
                print(f"❌ Model file not found at: {self.model_path}")
                print("Please train the model first by running train.py")
                print("\nTo train the model:")
                print("1. Make sure you have dataset in ../dataset/train/")
                print("2. Run: python train.py")
                print("3. Wait for training to complete")
                print(f"4. Model will be saved to: {self.model_path}")
                self.model = None
                self.feature_extractor = None
                return
            
            # Load the trained model
            print(f"📥 Loading model from: {self.model_path}")
            self.model = keras.models.load_model(str(self.model_path))
            print("✅ Model loaded successfully")
            
            # Load configuration
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
                print("✅ Config loaded successfully")
                
                # Get model type from config
                self.model_type = self.config.get('model_type', 'medium_dataset')
                print(f"📊 Model type: {self.model_type}")
            else:
                print("⚠️ Config file not found, using defaults")
                self.config = {
                    'img_size': 224,
                    'num_features': 2048,
                    'max_seq_length': 1
                }
                self.model_type = 'medium_dataset'
            
            # Load class vocabulary
            if self.vocab_path.exists():
                with open(self.vocab_path, 'r') as f:
                    self.class_names = json.load(f)
                print(f"✅ Class vocabulary loaded: {self.class_names}")
            else:
                print("⚠️ Vocabulary file not found, using defaults")
                self.class_names = ['REAL', 'FAKE']
            
            # Build feature extractor based on model type
            self.feature_extractor = self.build_feature_extractor()
            
            if self.feature_extractor is None:
                print("❌ Failed to build feature extractor")
                self.model = None
                return
            
            self.img_size = (self.config.get('img_size', 224), self.config.get('img_size', 224))
            self.num_features = self.config.get('num_features', 2048)
            self.max_seq_length = self.config.get('max_seq_length', 1)
            
            print(f"🎯 Model ready! Can detect: {self.class_names}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
            self.feature_extractor = None
    
    def build_feature_extractor(self):
        """Build the appropriate feature extractor based on model type"""
        try:
            print(f"🔨 Building feature extractor for {self.model_type}...")
            
            # Choose feature extractor based on model type (matching train.py)
            if self.model_type in ['small_dataset', 'video_sequence']:
                # Use InceptionV3 for smaller datasets
                feature_extractor = keras.applications.InceptionV3(
                    weights="imagenet",
                    include_top=False,
                    pooling="avg",
                    input_shape=(224, 224, 3),
                )
                self.preprocess_input = keras.applications.inception_v3.preprocess_input
                print("   Using InceptionV3 feature extractor")
            else:
                # Use ResNet50 for medium/large datasets
                feature_extractor = keras.applications.ResNet50(
                    weights="imagenet",
                    include_top=False,
                    pooling="avg",
                    input_shape=(224, 224, 3),
                )
                self.preprocess_input = keras.applications.resnet50.preprocess_input
                print("   Using ResNet50 feature extractor")
            
            feature_extractor.trainable = False
            
            inputs = keras.Input((224, 224, 3))
            preprocessed = self.preprocess_input(inputs)
            outputs = feature_extractor(preprocessed)
            
            return keras.Model(inputs, outputs, name="feature_extractor")
            
        except Exception as e:
            print(f"❌ Error building feature extractor: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def load_and_preprocess_image(self, image_path):
        """Load and preprocess image for prediction"""
        try:
            image = Image.open(image_path).convert("RGB")
            image = image.resize(self.img_size)
            image = np.array(image, dtype=np.float32)
            
            # Use the correct preprocessing function
            image = self.preprocess_input(image)
            return image
        except Exception as e:
            print(f"❌ Error loading image {image_path}: {e}")
            return None
    
    def extract_features(self, image_path):
        """Extract features using the feature extractor"""
        try:
            if self.feature_extractor is None:
                print("❌ Feature extractor not available")
                return None
                
            image = self.load_and_preprocess_image(image_path)
            if image is None:
                return None
            
            image = image[None, ...]  # Add batch dimension
            
            # Extract features
            features = self.feature_extractor.predict(image, verbose=0)
            return features.squeeze()  # Remove batch dimension
            
        except Exception as e:
            print(f"❌ Feature extraction error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def predict_image(self, image_path):
        """Predict if image is AI or Human generated"""
        # If in DEMO MODE, return random results
        if DEMO_MODE:
            return self.demo_prediction()
        
        try:
            # Check if model is loaded
            if self.model is None or self.feature_extractor is None:
                print("⚠️ Model not loaded, using fallback")
                return self.fallback_prediction()
            
            # Extract features
            features = self.extract_features(image_path)
            if features is None:
                print("⚠️ Feature extraction failed, using fallback")
                return self.fallback_prediction()
            
            # Reshape features to match model input
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
            
            # For video sequence model, we need to add sequence dimension
            if self.model_type == 'video_sequence':
                features = features[:, np.newaxis, :]
                mask = np.ones((1, self.max_seq_length), dtype="bool")
                probabilities = self.model.predict([features, mask], verbose=0)[0]
            else:
                # Predict using the trained model
                probabilities = self.model.predict(features, verbose=0)[0]
            
            # Get results
            result = {}
            for i, class_name in enumerate(self.class_names):
                result[class_name] = float(probabilities[i] * 100)
            
            # Determine final prediction
            predicted_class = self.class_names[np.argmax(probabilities)]
            confidence = np.max(probabilities) * 100
            
            # Map class names to is_ai boolean
            if 'fake' in predicted_class.lower() or 'ai' in predicted_class.lower():
                is_ai = True
            elif 'real' in predicted_class.lower() or 'human' in predicted_class.lower():
                is_ai = False
            else:
                # Default: assume second class (index 1) is FAKE/AI
                is_ai = np.argmax(probabilities) == 1
            
            print(f"🔍 Prediction: {predicted_class} (AI: {is_ai}) with {confidence:.2f}% confidence")
            print(f"   Probabilities: {result}")
            
            return {
                'isAI': is_ai,
                'confidence': confidence,
                'probabilities': result,
                'predicted_class': predicted_class
            }
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return self.fallback_prediction()
    
    def predict_video(self, video_path, frame_interval=30):
        """Predict if video is AI or Human generated by analyzing frames"""
        if DEMO_MODE:
            return self.demo_video_prediction()
        
        try:
            if self.model is None or self.feature_extractor is None:
                return self.fallback_video_prediction()
            
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
                    temp_path = f"temp_frame_{frame_count}.jpg"
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
                return {'isAI': False, 'confidence': 50.0, 'frames_analyzed': 0}
            
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
            return self.fallback_video_prediction()
    
    def demo_prediction(self):
        """Return random prediction for demo mode"""
        print("🎲 DEMO MODE: Using random prediction")
        is_ai = random.random() > 0.6
        confidence = random.uniform(75.0, 95.0)
        
        real_class = self.class_names[0] if self.class_names and len(self.class_names) > 0 else "REAL"
        fake_class = self.class_names[1] if self.class_names and len(self.class_names) > 1 else "FAKE"
        
        if is_ai:
            ai_prob = confidence / 100.0
            human_prob = 1.0 - ai_prob
            predicted_class = fake_class
        else:
            human_prob = confidence / 100.0
            ai_prob = 1.0 - human_prob
            predicted_class = real_class
        
        result = {
            real_class: human_prob * 100,
            fake_class: ai_prob * 100
        }
        
        print(f"   Result: {predicted_class} with {confidence:.2f}% confidence")
        
        return {
            'isAI': is_ai,
            'confidence': confidence,
            'probabilities': result,
            'predicted_class': predicted_class
        }
    
    def demo_video_prediction(self):
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
    
    def fallback_prediction(self):
        """Fallback prediction when model is not available"""
        print("⚠️ Using fallback prediction (model not available)")
        if DEMO_MODE:
            return self.demo_prediction()
        
        return {
            'isAI': False,
            'confidence': 50.0,
            'probabilities': {'Real': 50.0, 'Fake': 50.0},
            'predicted_class': 'Unknown'
        }
    
    def fallback_video_prediction(self):
        """Fallback video prediction when model is not available"""
        print("⚠️ Using fallback video prediction (model not available)")
        if DEMO_MODE:
            return self.demo_video_prediction()
        
        return {
            'isAI': False,
            'confidence': 50.0,
            'frames_analyzed': 0,
            'ai_frames': 0,
            'human_frames': 0
        }

# Global model instance
model_predictor = None

def initialize_model():
    """Initialize the AI model predictor"""
    global model_predictor
    try:
        model_predictor = AIModelPredictor()
        # Return True if model is loaded OR we're in demo mode
        if DEMO_MODE:
            print("🎮 Demo mode enabled - using random predictions")
            return True
        return model_predictor.model is not None
    except Exception as e:
        print(f"❌ Failed to initialize model: {e}")
        import traceback
        traceback.print_exc()
        model_predictor = AIModelPredictor()
        return False

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
    """Analyze text for AI vs Human content"""
    if DEMO_MODE:
        print("🎲 DEMO MODE: Using random text prediction")
        is_ai = random.random() > 0.6
        confidence = random.uniform(75.0, 95.0)
        return is_ai, float(confidence)
    
    try:
        # Simple text analysis without external dependencies
        text_length = len(text_content)
        word_count = len(text_content.split())
        
        if text_length < 50:
            return False, 50.0
        
        sentences = re.split(r'[.!?]+', text_content)
        sentence_count = len([s for s in sentences if s.strip()])
        
        avg_sentence_length = word_count / max(sentence_count, 1)
        
        words = text_content.lower().split()
        unique_words = set(words)
        lexical_diversity = len(unique_words) / max(len(words), 1)
        
        ai_indicators = 0
        human_indicators = 0
        
        sentence_lengths = [len(sent.split()) for sent in sentences if sent.strip()]
        if sentence_lengths:
            sentence_length_variance = np.var(sentence_lengths)
            if sentence_length_variance < 5:
                ai_indicators += 1
            else:
                human_indicators += 1
        
        if lexical_diversity > 0.8:
            ai_indicators += 1
        elif lexical_diversity < 0.5:
            human_indicators += 1
        
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
        
        formal_words = ['utilization', 'methodology', 'optimization', 'implementation', 'comprehensive']
        informal_words = ['awesome', 'cool', 'uh', 'like', 'you know']
        
        formal_count = sum(1 for word in words if word in formal_words)
        informal_count = sum(1 for word in words if word in informal_words)
        
        if formal_count > informal_count:
            ai_indicators += 1
        else:
            human_indicators += 1
        
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

@app.route('/')
def serve_frontend():
    return send_from_directory('../frontend', 'home.html')

@app.route('/<path:path>')
def serve_static_files(path):
    """Serve static files from frontend directory"""
    return send_from_directory('../frontend', path)

@app.route('/analyze', methods=['POST'])
def analyze_media():
    try:
        if 'file' not in request.files and 'text' not in request.form:
            return jsonify({'success': False, 'error': 'No file or text provided'})
        
        media_type = request.form.get('type', 'image')
        
        if media_type == 'text':
            text_content = request.form.get('text', '')
            if not text_content.strip():
                return jsonify({'success': False, 'error': 'No text provided'})
            
            file_id = str(uuid.uuid4())
            filename = f"text_analysis_{file_id[:8]}.txt"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(text_content)
            
            is_ai, confidence = analyze_text(text_content)
            
        else:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'success': False, 'error': 'No file selected'})
            
            file_id = str(uuid.uuid4())
            file_extension = os.path.splitext(file.filename)[1]
            filename = f"{file_id}{file_extension}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            file.save(file_path)
            
            if media_type == 'image':
                is_ai, confidence = analyze_image(file_path)
            else:  # video
                is_ai, confidence = analyze_video(file_path)
        
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
        
        return jsonify({
            'success': True,
            'isAI': is_ai,
            'confidence': float(confidence),
            'file_id': file_id
        })
        
    except Exception as e:
        print(f"Analysis error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/history', methods=['GET'])
def get_history():
    try:
        analyses = MediaAnalysis.query.order_by(MediaAnalysis.created_at.desc()).limit(20).all()
        return jsonify([analysis.to_dict() for analysis in analyses])
    except Exception as e:
        print(f"History error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/health', methods=['GET'])
def health_check():
    model_status = "loaded" if model_predictor and model_predictor.model is not None else "not loaded"
    return jsonify({
        'status': 'healthy',
        'database': 'connected' if db else 'disconnected',
        'upload_folder': os.path.exists(app.config['UPLOAD_FOLDER']),
        'model_status': model_status,
        'demo_mode': DEMO_MODE
    })

@app.route('/set_demo_mode/<int:mode>', methods=['POST'])
def set_demo_mode(mode):
    global DEMO_MODE
    DEMO_MODE = bool(mode)
    return jsonify({
        'success': True,
        'demo_mode': DEMO_MODE,
        'message': f'Demo mode {"enabled" if DEMO_MODE else "disabled"}'
    })

# Initialize model when the app starts
print("🚀 Starting AI vs Human Media & Text Detector Server...")
print("📁 Upload folder:", app.config['UPLOAD_FOLDER'])
print("🗄️  Database:", app.config['SQLALCHEMY_DATABASE_URI'])
print("🎮 Demo Mode:", "ENABLED" if DEMO_MODE else "DISABLED")
print("🔧 Initializing AI Model...")

if initialize_model():
    print("✅ AI Model initialized successfully!")
else:
    print("❌ AI Model initialization failed. Using fallback mode.")

print("🌐 Server running at: http://localhost:5000")
print("📝 Text analysis:", "Demo Mode" if DEMO_MODE else "Enabled")
print("🖼️  Image analysis:", "Demo Mode" if DEMO_MODE else "Enabled")
print("🎥 Video analysis:", "Demo Mode" if DEMO_MODE else "Enabled")

if __name__ == '__main__':
    app.run(debug=True, port=5000)