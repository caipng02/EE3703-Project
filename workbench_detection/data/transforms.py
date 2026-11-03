"""Letterboxing for photos of any size, matching how the dataset canvases were made.

The dataset was built with this exact rule (bilinear resize of the long side to 640, centred, grey 114
padding), so new photos reach the model the same way the training images did.
"""

from PIL import Image

from ..config import IMAGE_SIZE, LETTERBOX_FILL


# %% letterbox
def letterbox(image, size=IMAGE_SIZE, fill=LETTERBOX_FILL):
    """Return a padded RGB image and the transform needed to map boxes back to the original pixels."""
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
    """Map a 640-canvas xyxy box back to original-image pixels, clipping away the padding."""
    x1, y1, x2, y2 = box

    def restore_x(x):
        return max(0, min(transform["width"], (x - transform["left"]) / transform["scale_x"]))

    def restore_y(y):
        return max(0, min(transform["height"], (y - transform["top"]) / transform["scale_y"]))

    return [restore_x(x1), restore_y(y1), restore_x(x2), restore_y(y2)]
