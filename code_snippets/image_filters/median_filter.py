from PIL import ImageFilter
class MedianFilter:
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size

    def __call__(self, img):
        filtered_img = img.filter(ImageFilter.MedianFilter(self.kernel_size))
        return filtered_img