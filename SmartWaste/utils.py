from PIL import Image
import numpy as np

def preprocess_image(image_file):
    img = Image.open(image_file).convert('RGB')
    img = img.resize((64, 64))
    img_arr = np.array(img).flatten() / 255.0
    return img_arr.tolist()
