from PIL import ImageFilter
class Erosion:
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size

    def __call__(self, img):
        # Apply erosion to the image
        eroded_img = img.filter(ImageFilter.MinFilter(size=self.kernel_size))
        return eroded_img