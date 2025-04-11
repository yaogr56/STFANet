from PIL import Image as ImagePIL
import os

infile = r'D:\pythonProject1\cam\000050.jpg'
im = ImagePIL.open(infile)
im.save(infile, dpi=(1200.0, 1200.0))