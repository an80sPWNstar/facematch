"""facematch -- local ArcFace (InsightFace buffalo_l) face-similarity toolkit.

Three ways in:

    from facematch.core import analyzer, faces, reference_bank, score_image, \\
        score_folder, score_video

    python -m facematch score --refs ref_folder --targets folder_or_video_or_glob

    the Gradio app in app/app.py

See facematch.core for the single scoring implementation shared by all three.
"""
from .core import (
    IMAGE_EXT,
    OUTLIER_CUT,
    VIDEO_EXT,
    analyzer,
    bank_from_paths,
    faces,
    gather,
    imread,
    reference_bank,
    score_folder,
    score_image,
    score_video,
)

__all__ = [
    "IMAGE_EXT",
    "OUTLIER_CUT",
    "VIDEO_EXT",
    "analyzer",
    "bank_from_paths",
    "faces",
    "gather",
    "imread",
    "reference_bank",
    "score_folder",
    "score_image",
    "score_video",
]
