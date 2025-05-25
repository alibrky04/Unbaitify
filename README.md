# Unbaitify

Unbaitify is a clickbait detection system for YouTube videos that leverages multiple data types — video titles, engagement metrics (views, likes, dislikes), and thumbnail images. It trains separate machine learning models for datasets with rich metadata and those with only titles, then ensembles their predictions for improved accuracy and generalization.

Additionally, Unbaitify includes a YouTube browser extension that highlights potential clickbait videos in real time to help users make informed viewing choices.

---

## Features

- Multi-modal clickbait detection using text, numeric stats, and image features
- Separate models for datasets with/without thumbnails and metadata
- Ensemble model to combine predictions for better performance
- YouTube browser extension to flag clickbait videos on the fly
