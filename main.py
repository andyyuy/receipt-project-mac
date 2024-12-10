import pytesseract
from PIL import Image

receipt_text = pytesseract.image_to_string(Image.open('2.jpg'))
print(receipt_text)
