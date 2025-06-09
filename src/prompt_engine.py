import os
from typing import Optional

import google.generativeai as genai
from PIL import Image


def _caption_with_gemini(image_path: str) -> Optional[str]:
    """Return a caption for *image_path* using Gemini if available."""

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-pro-vision")
        img = Image.open(image_path)
        response = model.generate_content([img, "Describe the image in one sentence."])
        return response.text.strip()
    except Exception:
        # Any failure falls back to placeholder caption
        return None


def generate_caption(image_path: str) -> str:
    """Return a caption for *image_path* using Gemini if configured."""

    caption = _caption_with_gemini(image_path)
    if caption:
        return caption

    image_name = os.path.basename(image_path)
    return f"This is a placeholder caption for {image_name}"


def generate_image(text: str, width: int = 512, height: int = 512) -> Image.Image:
    """
    Fake image generator: returns a blank white image as a PIL Image.
    """
    return Image.new("RGB", (width, height), color="white")
