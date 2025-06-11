import os
from typing import Optional

from google import genai
from PIL import Image


def generate_caption(image_path: str, prompt: str) -> str:
    """Return a caption for image_path"""

    caption = _caption_with_gemini(image_path, prompt)
    if caption:
        return caption

    image_name = os.path.basename(image_path)
    return f"This is a placeholder caption for {image_name}"


def _caption_with_gemini(image_path: str, prompt: str) -> Optional[str]:
    """Return a caption for *image_path* using Gemini if available."""

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None

    client = genai.Client(api_key=api_key)
    image = Image.open(image_path)

    response = client.models.generate_content(
        model="gemini-2.0-flash", contents=[image, prompt]
    )
    return response.text if response and hasattr(response, "text") else None


def generate_image(prompt: str, width: int = 512, height: int = 512) -> Image.Image:
    """
    Fake image generator: returns a blank white image as a PIL Image.
    """
    return Image.new("RGB", (width, height), color="white")
