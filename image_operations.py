from PIL import Image, ImageFilter

def bitmap():
    img = Image.open("./black_cat.jpg")
    bitmap = img.convert('RGB')
    
    return bitmap
    
def to_binary(bitmap):
    for row in range(bitmap.size[1]):
        for col in range(bitmap.size[0]):
            if(bitmap.getpixel((col, row)) > (120, 120, 120)):
                bitmap.putpixel((col, row), (255, 255, 255))
            else:
                bitmap.putpixel((col, row), (0, 0, 0))
    return bitmap

