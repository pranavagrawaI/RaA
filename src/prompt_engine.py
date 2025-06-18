import os
from io import BytesIO
from typing import Optional

from google import genai
from google.genai import types
from PIL import Image


def generate_caption(image_path: str, prompt: str) -> str:
    """Return a caption for ``image_path``."""

    caption = _caption_with_gemini(image_path, prompt)
    if caption:
        return caption

    image_name = os.path.basename(image_path)
    return f"This is a placeholder caption for {image_name}"


def _caption_with_gemini(image_path: str, prompt: str) -> Optional[str]:
    """Return a caption for ``image_path`` using Gemini if possible."""

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None

    client = genai.Client(api_key=api_key)

    with Image.open(image_path) as img:
        image_data = img.copy()

    response = client.models.generate_content(
        model="gemini-2.0-flash", contents=[image_data, prompt]
    )
    return response.text if response and hasattr(response, "text") else None


def generate_image(prompt: str, text: str) -> Image.Image:
    """Generate an image from ``text`` using Gemini."""

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return Image.new("RGB", (32, 32), color="gray")

    client = genai.Client(api_key=api_key)

    response = client.models.generate_images(
        model="imagen-3.0-generate-002",
        prompt=prompt + text,
        config=types.GenerateImagesConfig(
            number_of_images=1,
        ),
    )
    if not response or not response.generated_images:
        raise RuntimeError(
            "Image generation failed: API response missing or no images returned."
        )

    api_image_object = response.generated_images[0].image
    if api_image_object is None:
        raise RuntimeError(
            "Image generation failed: The image object in the API response is null."
        )

    if not hasattr(api_image_object, "image_bytes"):
        raise RuntimeError(
            "Image generation failed: Image object from API response lacks 'image_bytes' attribute."
        )

    image_data = api_image_object.image_bytes
    if image_data is None:
        raise RuntimeError(
            "Image generation failed: The 'image_bytes' data in API response is null."
        )

    try:
        image = Image.open(BytesIO(image_data))
    except Exception as e:
        raise RuntimeError(f"Failed to decode or open generated image data: {e}") from e

    return image
