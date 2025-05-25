import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report
import joblib
import cv2

# === CONFIGURATION ===
ONLY_TITLE_DATASET_PATH = "data/raw/dataset2/clickbait_data.csv"
RICH_CLICKBAIT_PATH = "data/raw/dataset1/clickbait.csv"
RICH_NONCLICKBAIT_PATH = "data/raw/dataset1/notClickbait.csv"
THUMBNAIL_DIR = "thumbnails/"
MODEL_TITLE_ONLY_PATH = "models/clickbait_model_title_only.pkl"
MODEL_RICH_PATH = "models/clickbait_model_rich.pkl"

# === IMAGE FEATURE EXTRACTOR ===
class ThumbnailFeatureExtractor(BaseEstimator, TransformerMixin):
	def __init__(self, image_dir, img_size=(64, 64)):
		self.image_dir = image_dir
		self.img_size = img_size

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		img_features = []
		for video_id in X["ID"]:
			img_path = os.path.join(self.image_dir, f"{video_id}.jpg")
			if not os.path.exists(img_path):
				img_features.append(np.zeros(self.img_size[0] * self.img_size[1]))
				continue
			img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
			img = cv2.resize(img, self.img_size)
			flat = img.flatten() / 255.0
			img_features.append(flat)
		return np.array(img_features)

# === NUMERIC FEATURE EXTRACTOR ===
class NumericFeatureExtractor(BaseEstimator, TransformerMixin):
	def fit(self, X, y=None):
		return self

	def transform(self, X):
		ratio = X['Likes'] / (X['Dislikes'] + 1)
		return np.c_[X['Views'], X['Likes'], X['Dislikes'], ratio]

# === TEXT PIPELINE DEFINITION (Tfidf) ===
def get_text_pipeline():
	return Pipeline([
		('tfidf', TfidfVectorizer(max_features=100, stop_words='english'))
	])

def get_numeric_pipeline():
	return Pipeline([
		('num_features', NumericFeatureExtractor()),
		('scaler', StandardScaler())
	])

def get_image_pipeline():
	return Pipeline([
		('img_features', ThumbnailFeatureExtractor(image_dir=THUMBNAIL_DIR))
	])

# === FEATURE EXTRACTOR CLASSES ===
class TextOnlyFeatures(BaseEstimator, TransformerMixin):
	def __init__(self):
		self.text_pipeline = get_text_pipeline()

	def fit(self, X, y=None):
		self.text_pipeline.fit(X['Video Title'], y)
		return self

	def transform(self, X):
		return self.text_pipeline.transform(X['Video Title']).toarray()

class CombinedFeatures(BaseEstimator, TransformerMixin):
	def __init__(self):
		self.text_pipeline = get_text_pipeline()
		self.numeric_pipeline = get_numeric_pipeline()
		self.image_pipeline = get_image_pipeline()

	def fit(self, X, y=None):
		self.text_pipeline.fit(X['Video Title'], y)
		self.numeric_pipeline.fit(X, y)
		self.image_pipeline.fit(X, y)
		return self

	def transform(self, X):
		t = self.text_pipeline.transform(X['Video Title']).toarray()
		n = self.numeric_pipeline.transform(X)
		i = self.image_pipeline.transform(X)
		return np.hstack([t, n, i])

# === TRAINER CLASS ===
class Trainer:
	def __init__(self, feature_extractor, model=None):
		self.feature_extractor = feature_extractor
		self.model = model or RandomForestClassifier(n_estimators=100, random_state=42)

	def fit(self, X, y):
		X_transformed = self.feature_extractor.fit_transform(X, y)
		self.model.fit(X_transformed, y)

	def predict(self, X):
		X_transformed = self.feature_extractor.transform(X)
		return self.model.predict(X_transformed)

	def predict_proba(self, X):
		X_transformed = self.feature_extractor.transform(X)
		return self.model.predict_proba(X_transformed)

	def save(self, path):
		joblib.dump((self.model, self.feature_extractor), path)

	@staticmethod
	def load(path):
		model, feature_extractor = joblib.load(path)
		return Trainer(feature_extractor, model)

# === LOAD DATA FUNCTIONS ===
def load_title_only_dataset():
	df = pd.read_csv(ONLY_TITLE_DATASET_PATH)
	df = df.rename(columns={'Title': 'Video Title'})  # adapt if needed
	df['Label'] = df['Label'].astype(int)
	return df

def load_rich_dataset():
	clickbait = pd.read_csv(RICH_CLICKBAIT_PATH)
	clickbait['Label'] = 1
	nonclickbait = pd.read_csv(RICH_NONCLICKBAIT_PATH)
	nonclickbait['Label'] = 0
	df = pd.concat([clickbait, nonclickbait], ignore_index=True)
	return df

# === ENSEMBLE CLASS ===
class EnsembleModel:
	def __init__(self, trainer_old, trainer_new, weight_old=0.5, weight_new=0.5):
		self.trainer_old = trainer_old
		self.trainer_new = trainer_new
		self.weight_old = weight_old
		self.weight_new = weight_new

	def predict(self, X_old, X_new):
		proba_old = self.trainer_old.predict_proba(X_old)
		proba_new = self.trainer_new.predict_proba(X_new)

		# Weighted average of probabilities
		proba = self.weight_old * proba_old + self.weight_new * proba_new
		preds = np.argmax(proba, axis=1)
		return preds, proba

# === TRAINING BOTH MODELS AND SAVING ===
def train_and_save_models():
	# Train old model on old dataset (titles only)
	if os.path.exists(ONLY_TITLE_DATASET_PATH):
		old_data = load_title_only_dataset()
		X_old = old_data[['Video Title']]
		y_old = old_data['Label']

		trainer_old = Trainer(TextOnlyFeatures())
		trainer_old.fit(X_old, y_old)
		trainer_old.save(MODEL_TITLE_ONLY_PATH)
		print("Old model trained and saved.")
	else:
		trainer_old = None
		print("Old dataset not found, skipping old model training.")

	# Train new model on new dataset (full features)
	if os.path.exists(RICH_CLICKBAIT_PATH) and os.path.exists(RICH_NONCLICKBAIT_PATH):
		new_data = load_rich_dataset()
		X_new = new_data[['ID', 'Video Title', 'Views', 'Likes', 'Dislikes']]
		y_new = new_data['Label']

		trainer_new = Trainer(CombinedFeatures())
		trainer_new.fit(X_new, y_new)
		trainer_new.save(MODEL_RICH_PATH)
		print("New model trained and saved.")
	else:
		trainer_new = None
		print("New dataset not found, skipping new model training.")

	return trainer_old, trainer_new

# === PREDICTION USING ENSEMBLE ===
def predict_ensemble(video_id, title, views=None, likes=None, dislikes=None):
	# Load models
	trainer_old = Trainer.load(MODEL_TITLE_ONLY_PATH) if os.path.exists(MODEL_TITLE_ONLY_PATH) else None
	trainer_new = Trainer.load(MODEL_RICH_PATH) if os.path.exists(MODEL_RICH_PATH) else None

	if trainer_old is None and trainer_new is None:
		raise RuntimeError("No models are available for prediction.")

	# Prepare inputs for title_only and rich models
	X_old = pd.DataFrame([{'Video Title': title}])
	if trainer_new is not None:
		X_new = pd.DataFrame([{
			'ID': video_id,
			'Video Title': title,
			'Views': views if views is not None else 0,
			'Likes': likes if likes is not None else 0,
			'Dislikes': dislikes if dislikes is not None else 0
		}])
	else:
		X_new = None

	# Predict probabilities and combine
	if trainer_old and trainer_new:
		ensemble = EnsembleModel(trainer_old, trainer_new)
		preds, proba = ensemble.predict(X_old, X_new)
		label = "Clickbait" if preds[0] == 1 else "Not Clickbait"
		confidence = proba[0][preds[0]]
	elif trainer_old:
		preds = trainer_old.predict(X_old)
		proba = trainer_old.predict_proba(X_old)
		label = "Clickbait" if preds[0] == 1 else "Not Clickbait"
		confidence = proba[0][preds[0]]
	else:
		preds = trainer_new.predict(X_new)
		proba = trainer_new.predict_proba(X_new)
		label = "Clickbait" if preds[0] == 1 else "Not Clickbait"
		confidence = proba[0][preds[0]]

	print(f"Prediction: {label} ({confidence*100:.2f}% confidence)")
	return preds[0], confidence

# === MAIN ===
if __name__ == "__main__":
	# train_and_save_models()

	# Example prediction:
	predict_ensemble("hfFGTVZNjis", "MrBeast HATES Chandler after this..", views=17166, likes=1207, dislikes=554)