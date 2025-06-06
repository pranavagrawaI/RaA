from prompt_engine import generate_caption, generate_image


def test_generate_caption_contains_filename(tmp_path):
    fake_path = str(tmp_path / "some_image.jpg")
    (tmp_path / "some_image.jpg").write_text("dummy")
    caption = generate_caption(fake_path)
    assert "some_image.jpg" in caption


def test_generate_image_dimensions():
    img = generate_image("anything", width=32, height=16)
    assert img.size == (32, 16)
    assert img.mode == "RGB"
