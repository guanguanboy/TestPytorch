from PIL import Image
import matplotlib.pyplot as plt

origin = Image.open('D:/DataSets/PennFudanPed/PNGImages/FudanPed00001.png')
plt.subplot(1,2,1)
plt.imshow(origin)
mask = Image.open('D:/DataSets/PennFudanPed/PedMasks/FudanPed00001_mask.png')

mask.putpalette([
    0, 0, 0,  # black background
    255, 0, 0,  # index 1 is red
    255, 255, 0,  # index 2 is yellow
    255, 153, 0,  # index 3 is orange
])
plt.subplot(1,2,2)
plt.imshow(mask)
plt.show()
mask