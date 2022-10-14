import numpy as np
from PIL import Image
import io
from base64 import b64encode


def convert_bytes_to_image(image_bytes):
    pil_image = Image.open(io.BytesIO(image_bytes))
    return np.array(pil_image)[:, :, ::-1]


def convert_image_to_bytes(image):
    file_object = io.BytesIO()
    img= Image.fromarray(image[:, :, ::-1])
    img.save(file_object, 'PNG')
    base64img = "data:image/png;base64," + b64encode(file_object.getvalue()).decode('ascii')
    return base64img