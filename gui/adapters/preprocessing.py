"""Optional letterboxing helpers. Match your actual training preprocessing."""

from PIL import Image


def letterbox(image, size=640, fill=(114, 114, 114)):
    """Return a padded RGB image and the transform for mapping boxes back."""
    width, height = image.size
    scale = min(size / width, size / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    left, top = (size - resized_width) // 2, (size - resized_height) // 2
    canvas = Image.new("RGB", (size, size), fill)
    canvas.paste(image.resize((resized_width, resized_height), Image.Resampling.BILINEAR), (left, top))
    transform = {"scale_x": resized_width / width, "scale_y": resized_height / height,
                 "left": left, "top": top, "width": width, "height": height}
    return canvas, transform


def restore_box(box, transform):
    """Convert a 640-space xyxy box to original-image pixels, clipping padding."""
    x1, y1, x2, y2 = box
    return [max(0, min(transform["width"], (x - transform["left"]) / transform["scale_x"]))
            if axis == "x" else
            max(0, min(transform["height"], (x - transform["top"]) / transform["scale_y"]))
            for x, axis in [(x1, "x"), (y1, "y"), (x2, "x"), (y2, "y")]]
