import os

from PIL import Image


def generate_caption(image_path: str) -> str:
    """
    Fake caption generator that returns a placeholder string.
    """
    image_name = os.path.basename(image_path)
    return f"This is a placeholder caption for {image_name}"


def generate_image(text: str, width: int = 512, height: int = 512) -> Image.Image:
    """
    Fake image generator: returns a blank white image as a PIL Image.
    """
    return Image.new("RGB", (width, height), color="white")
