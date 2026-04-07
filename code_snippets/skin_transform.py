from PIL import Image
import numpy as np
class SkinTransform:
    def __call__(self, image, skin_detector):
        image = image.convert('RGB')
        numpy_image = np.array(image)
        image = numpy_image[:,:,::-1].copy()
        image = skin_detector(image)
        image = Image.fromarray(image.astype('uint8'))
        return image