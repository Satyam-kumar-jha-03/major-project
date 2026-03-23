import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from tensorflow import keras # type: ignore
import tensorflow as tf
import os
from PIL import Image
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
import time
import psutil
import gc

# Set up GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 128  # Default, will be adjusted
EPOCHS = 50  # Maximum epochs, will use early stopping
MAX_SEQ_LENGTH = 1
NUM_FEATURES = 2048

# Thresholds for algorithm selection
SMALL_DATASET_THRESHOLD = 5000  # Less than 5k images = small dataset
MEDIUM_DATASET_THRESHOLD = 20000  # 5k-20k = medium, >20k = large
VIDEO_MODE_THRESHOLD = 100  # If analyzing video frames, use sequence model

class HybridTrainer:
    def __init__(self):
        self.model_type = None
        self.feature_extractor = None
        self.label_processor = None
        self.models_dir = Path("../models")
        self.models_dir.mkdir(exist_ok=True)
        
    def analyze_dataset(self, train_df):
        """Analyze dataset characteristics to choose best algorithm"""
        num_samples = len(train_df)
        num_classes = train_df['tag'].nunique()
        class_counts = train_df['tag'].value_counts()
        
        # Check for class imbalance
        max_count = class_counts.max()
        min_count = class_counts.min()
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        # Check if it's likely video frames (by filename patterns)
        is_video_frames = any('frame' in name.lower() or 'video' in name.lower() 
                              for name in train_df['img_name'].iloc[:100])
        
        print("\n📊 Dataset Analysis:")
        print(f"   - Total samples: {num_samples}")
        print(f"   - Number of classes: {num_classes}")
        print(f"   - Class distribution: {dict(class_counts)}")
        print(f"   - Imbalance ratio: {imbalance_ratio:.2f}:1")
        print(f"   - Video frames detected: {is_video_frames}")
        
        # Decide model type based on dataset characteristics
        if is_video_frames and num_samples < VIDEO_MODE_THRESHOLD * num_classes:
            self.model_type = 'video_sequence'
            print("   📽️ Selected: VIDEO SEQUENCE model (video frames detected)")
        elif num_samples < SMALL_DATASET_THRESHOLD:
            self.model_type = 'small_dataset'
            print("   🐣 Selected: SMALL DATASET model (data augmentation + transfer learning)")
        elif num_samples < MEDIUM_DATASET_THRESHOLD:
            self.model_type = 'medium_dataset'
            print("   📏 Selected: MEDIUM DATASET model (balanced approach)")
        else:
            self.model_type = 'large_dataset'
            print("   🐘 Selected: LARGE DATASET model (complex architecture)")
        
        # Adjust batch size based on dataset size
        global BATCH_SIZE
        if num_samples < 1000:
            BATCH_SIZE = 32
        elif num_samples < 5000:
            BATCH_SIZE = 64
        elif num_samples < 20000:
            BATCH_SIZE = 128
        else:
            BATCH_SIZE = 256
            
        print(f"   🔧 Adjusted batch size: {BATCH_SIZE}")
        
        return {
            'num_samples': num_samples,
            'num_classes': num_classes,
            'imbalance_ratio': imbalance_ratio,
            'is_video_frames': is_video_frames
        }
    
    def build_feature_extractor(self, model_type):
        """Build appropriate feature extractor based on model type"""
        print(f"🔨 Building feature extractor for {model_type}...")
        
        if model_type in ['small_dataset', 'video_sequence']:
            # Use InceptionV3 for smaller datasets (better transfer learning)
            feature_extractor = keras.applications.InceptionV3(
                weights="imagenet",
                include_top=False,
                pooling="avg",
                input_shape=(IMG_SIZE, IMG_SIZE, 3),
            )
            self.preprocess_input = keras.applications.inception_v3.preprocess_input
        else:
            # Use ResNet50 for medium/large datasets (more stable)
            feature_extractor = keras.applications.ResNet50(
                weights="imagenet",
                include_top=False,
                pooling="avg",
                input_shape=(IMG_SIZE, IMG_SIZE, 3),
            )
            self.preprocess_input = keras.applications.resnet50.preprocess_input
        
        # For small datasets, we might fine-tune some layers
        if model_type == 'small_dataset':
            feature_extractor.trainable = True
            # Freeze early layers, fine-tune later layers
            for layer in feature_extractor.layers[:100]:
                layer.trainable = False
            print("   🔓 Fine-tuning enabled for later layers")
        else:
            feature_extractor.trainable = False
            
        return feature_extractor
    
    def load_and_preprocess_image(self, image_path, augment=False):
        """Load and preprocess image with optional augmentation"""
        try:
            image = Image.open(image_path)
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image = image.resize((IMG_SIZE, IMG_SIZE))
            image = np.array(image, dtype=np.float32)
            
            # Data augmentation for small datasets
            if augment and self.model_type == 'small_dataset':
                # Random horizontal flip
                if np.random.random() > 0.5:
                    image = np.fliplr(image)
                
                # Random brightness adjustment
                if np.random.random() > 0.5:
                    brightness_factor = np.random.uniform(0.8, 1.2)
                    image = image * brightness_factor
                    image = np.clip(image, 0, 255)
                
                # Random rotation (small angle)
                if np.random.random() > 0.7:
                    angle = np.random.uniform(-10, 10)
                    from scipy.ndimage import rotate
                    image = rotate(image, angle, reshape=False, mode='nearest')
            
            # Apply appropriate preprocessing
            image = self.preprocess_input(image)
            return image
            
        except Exception as e:
            print(f"❌ Error loading image {image_path}: {e}")
            return None
    
    def prepare_features(self, df, label_processor, augment=False):
        """Prepare features based on dataset size"""
        num_samples = len(df)
        image_paths = df["img_name"].values.tolist()
        labels = df["tag"].values

        # Convert class labels
        labels = label_processor(labels[..., None]).numpy()

        # Initialize arrays
        frame_features = np.zeros(shape=(num_samples, NUM_FEATURES), dtype="float32")
        successful_samples = 0
        
        self.feature_extractor = self.build_feature_extractor(self.model_type)

        # Process in batches
        batch_size = 32
        total_batches = (num_samples + batch_size - 1) // batch_size
        
        start_time = time.time()
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_paths = image_paths[start_idx:end_idx]
            
            if (batch_idx + 1) % 20 == 0:
                print(f"🔄 Processing batch {batch_idx + 1}/{total_batches}...")
            
            # Load and preprocess batch
            batch_images = []
            valid_indices = []
            
            for i, path in enumerate(batch_paths):
                image = self.load_and_preprocess_image(path, augment)
                if image is not None:
                    batch_images.append(image)
                    valid_indices.append(start_idx + i)
            
            if not batch_images:
                continue
                
            batch_images = np.array(batch_images)
            
            # Extract features
            try:
                batch_features = self.feature_extractor.predict(batch_images, verbose=0)
                
                for i, features in enumerate(batch_features):
                    if successful_samples < num_samples:
                        frame_features[successful_samples] = features
                        if successful_samples < valid_indices[i]:
                            labels[successful_samples] = labels[valid_indices[i]]
                        successful_samples += 1
                        
            except Exception as e:
                print(f"❌ Error in batch {batch_idx + 1}: {e}")
                continue
        
        # Trim arrays
        frame_features = frame_features[:successful_samples]
        labels = labels[:successful_samples]
        
        elapsed = time.time() - start_time
        print(f"✅ Processed {successful_samples}/{num_samples} images in {elapsed:.2f}s")
        
        return frame_features, labels
    
    def build_model(self, num_classes, dataset_info):
        """Build model based on dataset characteristics"""
        
        if self.model_type == 'video_sequence':
            # Video sequence model (from train_model.py)
            print("🎬 Building VIDEO SEQUENCE model...")
            
            frame_features_input = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
            mask_input = keras.Input((MAX_SEQ_LENGTH,), dtype="bool")

            x = keras.layers.LSTM(128, return_sequences=True)(frame_features_input, mask=mask_input)
            x = keras.layers.Dropout(0.3)(x)
            x = keras.layers.LSTM(64)(x)
            x = keras.layers.Dropout(0.3)(x)
            x = keras.layers.Dense(128, activation="relu")(x)
            x = keras.layers.Dropout(0.4)(x)
            output = keras.layers.Dense(num_classes, activation="softmax")(x)

            model = keras.Model([frame_features_input, mask_input], output)
            
        elif self.model_type == 'small_dataset':
            # Small dataset model - more regularization and dropout
            print("🐣 Building SMALL DATASET model...")
            
            inputs = keras.Input(shape=(NUM_FEATURES,))
            
            x = keras.layers.Dense(256, activation="relu", 
                                  kernel_regularizer=keras.regularizers.l2(0.01))(inputs)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dropout(0.6)(x)
            
            x = keras.layers.Dense(128, activation="relu",
                                  kernel_regularizer=keras.regularizers.l2(0.01))(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dropout(0.5)(x)
            
            x = keras.layers.Dense(64, activation="relu",
                                  kernel_regularizer=keras.regularizers.l2(0.01))(x)
            x = keras.layers.Dropout(0.4)(x)
            
            output = keras.layers.Dense(num_classes, activation="softmax")(x)
            
            model = keras.Model(inputs, output)
            
        elif self.model_type == 'medium_dataset':
            # Medium dataset model - balanced approach
            print("📏 Building MEDIUM DATASET model...")
            
            inputs = keras.Input(shape=(NUM_FEATURES,))
            
            x = keras.layers.Dense(512, activation="relu",
                                  kernel_regularizer=keras.regularizers.l2(0.005))(inputs)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dropout(0.5)(x)
            
            x = keras.layers.Dense(256, activation="relu",
                                  kernel_regularizer=keras.regularizers.l2(0.005))(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dropout(0.4)(x)
            
            x = keras.layers.Dense(128, activation="relu")(x)
            x = keras.layers.Dropout(0.3)(x)
            
            output = keras.layers.Dense(num_classes, activation="softmax")(x)
            
            model = keras.Model(inputs, output)
            
        else:  # large_dataset
            # Large dataset model - more complex architecture
            print("🐘 Building LARGE DATASET model...")
            
            inputs = keras.Input(shape=(NUM_FEATURES,))
            
            x = keras.layers.Dense(1024, activation="relu")(inputs)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dropout(0.4)(x)
            
            x = keras.layers.Dense(512, activation="relu")(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dropout(0.3)(x)
            
            x = keras.layers.Dense(256, activation="relu")(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dropout(0.2)(x)
            
            x = keras.layers.Dense(128, activation="relu")(x)
            output = keras.layers.Dense(num_classes, activation="softmax")(x)
            
            model = keras.Model(inputs, output)
        
        # Adjust learning rate based on dataset size
        if dataset_info['num_samples'] < 1000:
            lr = 0.0005
        elif dataset_info['num_samples'] < 5000:
            lr = 0.001
        else:
            lr = 0.001
            
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            metrics=["accuracy"]
        )
        
        return model
    
    def train(self):
        """Main training function with hybrid approach"""
        print("🚀 Starting HYBRID Training System...")
        start_time = time.time()
        
        # Step 1: Create/load dataset
        if not self.create_dataset_csv():
            print("❌ Failed to create dataset CSVs.")
            return
        
        # Step 2: Load data
        train_df = pd.read_csv("train.csv")
        test_df = pd.read_csv("test.csv")
        
        print(f"\n📊 Dataset sizes: Train={len(train_df)}, Test={len(test_df)}")
        
        # Step 3: Analyze dataset and choose algorithm
        dataset_info = self.analyze_dataset(train_df)
        
        # Step 4: Prepare label processor
        self.label_processor = keras.layers.StringLookup(
            num_oov_indices=0, 
            vocabulary=np.unique(train_df["tag"])
        )
        vocab = self.label_processor.get_vocabulary()
        print(f"\n🏷️ Classes: {vocab}")
        
        # Step 5: Prepare features with appropriate augmentation
        print("\n🔄 Preparing training features...")
        train_features, train_labels = self.prepare_features(
            train_df, self.label_processor, 
            augment=(self.model_type == 'small_dataset')
        )
        
        print("\n🔄 Preparing test features...")
        test_features, test_labels = self.prepare_features(
            test_df, self.label_processor, augment=False
        )
        
        # Step 6: Build model
        print("\n🏗️ Building model...")
        model = self.build_model(len(vocab), dataset_info)
        print(model.summary())
        
        # Step 7: Prepare data based on model type
        if self.model_type == 'video_sequence':
            # Add sequence dimension for video model
            train_data = [train_features[:, np.newaxis, :], 
                         np.ones((len(train_features), MAX_SEQ_LENGTH), dtype="bool")]
            test_data = [test_features[:, np.newaxis, :], 
                        np.ones((len(test_features), MAX_SEQ_LENGTH), dtype="bool")]
        else:
            # Simple features for dense models
            train_data = train_features
            test_data = test_features
        
        # Step 8: Callbacks
        model_path = self.models_dir / "image_classifier.keras"
        
        checkpoint = keras.callbacks.ModelCheckpoint(
            str(model_path),
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        )
        
        # Adjust patience based on dataset size
        patience = 15 if dataset_info['num_samples'] < 5000 else 10
        
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=patience,
            restore_best_weights=True,
            mode='max',
            verbose=1
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience//2,
            min_lr=0.000001,
            verbose=1
        )
        
        # Step 9: Train
        print("\n🎯 Starting training...")
        history = model.fit(
            train_data,
            train_labels,
            validation_data=(test_data, test_labels),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[checkpoint, early_stopping, reduce_lr],
            verbose=1
        )
        
        # Step 10: Evaluate
        print("\n📈 Evaluating model...")
        test_loss, test_accuracy = model.evaluate(test_data, test_labels, verbose=0)
        print(f"✅ Test accuracy: {test_accuracy:.4f}")
        print(f"✅ Test loss: {test_loss:.4f}")
        
        # Step 11: Save results
        self.plot_training_history(history)
        
        # Save vocabulary
        with open(self.models_dir / 'label_vocabulary.json', 'w') as f:
            json.dump([str(v) for v in vocab], f)
        
        # Save configuration
        config = {
            'img_size': IMG_SIZE,
            'num_features': NUM_FEATURES,
            'batch_size': BATCH_SIZE,
            'test_accuracy': float(test_accuracy),
            'test_loss': float(test_loss),
            'class_names': [str(v) for v in vocab],
            'model_type': self.model_type,
            'dataset_info': dataset_info,
            'total_training_time': time.time() - start_time
        }
        
        with open(self.models_dir / 'model_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        total_time = time.time() - start_time
        print("\n🎉 Training completed successfully!")
        print(f"💾 Model saved to: {model_path}")
        print(f"📊 Test Accuracy: {test_accuracy:.2%}")
        print(f"🤖 Model Type: {self.model_type}")
        print(f"⏱️ Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    
    def create_dataset_csv(self):
        """Create dataset CSV files (from original)"""
        print("\n📁 Creating dataset CSV files...")
        
        current_dir = Path(__file__).parent
        project_root = current_dir.parent
        dataset_path = project_root / "dataset"
        train_path = dataset_path / "train"
        test_path = dataset_path / "test"
        
        if not train_path.exists():
            print(f"❌ Training path {train_path} does not exist!")
            return False
        
        # Process training data
        train_rooms = []
        label_types = [d for d in os.listdir(train_path) if os.path.isdir(train_path / d)]
        
        for item in label_types:
            class_path = train_path / item
            if os.path.isdir(class_path):
                image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
                all_images = []
                for ext in image_extensions:
                    all_images.extend([f.name for f in class_path.glob(ext)])
                    all_images.extend([f.name for f in class_path.glob(ext.upper())])
                
                for image in all_images:
                    train_rooms.append((item, str(class_path / image)))
        
        if not train_rooms:
            print("❌ No training images found!")
            return False
        
        train_df = pd.DataFrame(data=train_rooms, columns=['tag', 'img_name'])
        train_df = train_df.loc[:, ['img_name', 'tag']]
        train_df.to_csv('train.csv', index=False)
        
        # Process test data
        test_rooms = []
        if test_path.exists():
            for item in label_types:
                class_path = test_path / item
                if os.path.isdir(class_path):
                    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
                    all_images = []
                    for ext in image_extensions:
                        all_images.extend([f.name for f in class_path.glob(ext)])
                        all_images.extend([f.name for f in class_path.glob(ext.upper())])
                    
                    for image in all_images:
                        test_rooms.append((item, str(class_path / image)))
        
        if not test_rooms:
            # Create test split from training
            train_df, test_df = train_test_split(train_df, test_size=0.2, 
                                                random_state=42, stratify=train_df['tag'])
            train_df.to_csv('train.csv', index=False)
            test_df.to_csv('test.csv', index=False)
        else:
            test_df = pd.DataFrame(data=test_rooms, columns=['tag', 'img_name'])
            test_df = test_df.loc[:, ['img_name', 'tag']]
            test_df.to_csv('test.csv', index=False)
        
        return True
    
    def plot_training_history(self, history):
        """Plot training history"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title(f'Model Accuracy ({self.model_type})')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'Model Loss ({self.model_type})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.models_dir / 'training_history.png')
        plt.show()

def main():
    # Clear session to free memory
    keras.backend.clear_session()
    gc.collect()
    
    # Run hybrid training
    trainer = HybridTrainer()
    trainer.train()

if __name__ == "__main__":
    main()