from PIL import ImageFilter
class Dilation:
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size

    def __call__(self, img):
        # Apply dilation to the image
        dilated_img = img.filter(ImageFilter.MaxFilter(size=self.kernel_size))
        return dilated_img