import cv2 as cv

def image_resize(img, width=None, height=None, inter=cv.INTER_AREA):
    """ Resizes image keeping aspect ratio. """
    
    
    [h, w] = img.shape[:2]
    dim = (h, w)
    # print(h, w)
    if width is None and height is None:
        return img
    if height is not None and height < h:
        r = height / float(h)
        dim = (int(w * r), height)
        [w, h] = dim
    if width is not None and width < w:
        r = width / float(w)
        dim = (width, int(h * r))

    print(dim, height, width)
    resized = cv.resize(img, dim, interpolation=inter)
    return resized