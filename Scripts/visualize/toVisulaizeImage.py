import pydicom
import numpy as np
import matplotlib.pyplot as plt

# 1. Load DICOM image
dcm_path = '../../Data/siim-original/dicom-images-train/1.2.276.0.7230010.3.1.2.8323329.300.1517875162.258080/1.2.276.0.7230010.3.1.3.8323329.300.1517875162.258079/1.2.276.0.7230010.3.1.4.8323329.300.1517875162.258081.dcm'
dcm = pydicom.dcmread(dcm_path)
img = dcm.pixel_array

# 2. RLE decoding function (from your script)
def rle2mask(rle, width, height):
    mask= np.zeros(width* height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position+lengths[index]] = 255
        current_position += lengths[index]

    return mask.reshape(width, height)

# Example RLE (from CSV)
rle = "735441 12 1011 16 1007 18 1006 18 1006 18 1006 18 1006 19 1005 19 1005 20 1004 20 1004 21 1003 21 1003 22 1002 22 1002 23 1001 24 1000 24 1000 25 999 26 998 26 998 27 997 27 997 28 996 28 996 28 996 29 995 29 996 28 996 28 996 29 995 29 995 29 995 30 995 29 995 30 994 30 994 31 994 30 994 30 994 31 994 30 994 31 993 31 994 31 994 30 994 30 995 30 994 30 995 30 995 29 996 29 995 29 996 29 996 28 997 28 997 27 998 27 998 26 999 26 999 25 1000 25 1000 25 1000 24 1001 24 1001 23 1001 24 1001 23 1002 23 1002 23 1001 23 1002 23 1002 23 1001 23 1002 23 1002 23 1002 23 1001 23 1002 23 1001 24 1001 23 1002 23 1001 24 1001 23 1002 23 1001 24 1001 24 1001 24 1001 23 1002 23 1002 22 1002 23 1002 23 1002 23 1002 23 1002 23 1001 24 1001 23 1002 23 1002 23 1002 23 1002 23 1002 23 1002 23 1002 22 1003 22 1003 22 1003 22 1003 22 1004 21 1004 20 1005 20 1005 20 1006 18 1007 18 1007 17 1008 16 1010 13 1013 11 1015 9 1016 6"  # Use the actual mask string
H, W = img.shape
mask = rle2mask(rle, W, H)

# 3. Visualize overlay
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title('Chest X-ray')
plt.subplot(1,2,2)
plt.imshow(img, cmap='gray')
plt.imshow(mask, alpha=0.4, cmap='Reds')
plt.title('With Pneumothorax Mask')
plt.show()
