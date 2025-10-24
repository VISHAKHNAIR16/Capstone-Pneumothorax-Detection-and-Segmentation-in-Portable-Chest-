import pydicom
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1. Load DICOM image
dcm_path = '../../Data/siim-original/dicom-images-train/1.2.276.0.7230010.3.1.2.8323329.32748.1517875162.147878/1.2.276.0.7230010.3.1.3.8323329.32748.1517875162.147877/1.2.276.0.7230010.3.1.4.8323329.32748.1517875162.147879.dcm'
dcm = pydicom.dcmread(dcm_path)
img = dcm.pixel_array

def rle2mask(rle, width, height):
    """
    Official SIIM-ACR RLE decoder
    Uses relative positioning (column-major order)
    """
    mask = np.zeros(width * height)
    
    # Handle no-pneumothorax case
    if rle == '-1' or pd.isna(rle):
        return mask.reshape(width, height)
    
    # Parse RLE: alternating offset and length
    array = np.asarray([int(x) for x in str(rle).split()])
    starts = array[0::2]      # Offset positions
    lengths = array[1::2]     # Run lengths
    
    # Reconstruct mask with relative positioning
    current_position = 0
    for index, start in enumerate(starts):
        current_position += start  # KEY: Relative positioning!
        mask[current_position:current_position+lengths[index]] = 255
        current_position += lengths[index]
    
    return mask.reshape(width, height)

# Example RLE (from CSV)
rle = "157966 5 1016 8 1014 9 1014 10 1012 11 1012 11 1011 13 1009 14 1009 14 1008 15 1008 15 1008 15 1007 16 1007 16 1007 16 1006 17 1006 17 1006 16 1006 17 1006 17 1006 17 1005 18 1005 18 1005 18 1004 19 1004 19 1004 19 1003 21 1002 21 1002 21 1001 22 1001 22 1001 22 1000 23 1000 23 999 25 998 25 997 26 997 26 996 27 996 27 996 27 995 28 995 28 995 28 994 29 994 30 993 30 992 31 992 31 992 30 992 31 992 31 992 32 991 32 991 32 991 32 990 34 989 34 989 34 989 34 989 33 990 33 990 33 989 35 988 35 988 35 988 35 988 36 987 36 987 36 987 36 987 36 987 36 987 36 987 36 987 37 986 37 986 37 986 37 986 38 986 37 986 37 986 37 986 38 986 37 986 37 986 37 986 38 986 37 986 37 986 38 985 38 986 38 985 38 985 38 985 39 985 38 985 38 985 39 984 39 985 39 984 40 983 40 983 40 984 40 983 40 983 40 984 40 983 40 984 40 984 39 984 40 984 39 984 40 984 40 983 40 983 41 983 40 983 41 982 41 983 41 982 41 983 41 983 41 982 41 983 41 982 41 983 41 983 41 982 41 983 41 982 42 982 42 981 42 982 42 982 42 982 41 982 42 982 41 983 41 982 42 982 42 982 41 982 42 982 42 981 43 981 43 980 43 981 43 981 43 981 43 980 44 980 44 980 44 980 44 980 44 979 45 979 44 980 44 979 45 979 45 979 45 978 46 978 46 978 46 978 46 977 47 977 46 978 46 978 46 978 46 977 47 977 47 977 47 977 47 976 48 976 48 976 48 976 48 976 48 976 48 976 48 976 48 976 48 976 48 976 48 976 48 977 47 977 47 977 47 977 47 977 47 978 46 978 46 978 46 978 46 978 46 979 45 979 45 979 45 979 45 979 45 980 44 980 44 981 43 981 43 981 43 982 42 982 42 982 42 982 42 982 42 983 41 983 41 984 41 983 41 984 40 984 40 984 40 984 41 983 41 984 40 984 40 985 39 985 40 984 40 985 39 985 39 985 39 985 40 984 40 985 40 984 40 985 40 984 40 985 39 985 39 986 38 986 38 986 38 987 37 988 36 989 35 989 35 990 34 991 33 992 32 993 31 994 30 995 29 997 27 998 26 999 25 1001 20 1006 16 1012 10"  # Use the actual mask string
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
