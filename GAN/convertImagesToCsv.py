import numpy as np
import cv2
import io
import glob, os
os.chdir("./data/")

# info
#   black : 0
#   white : 255

# size
imgWidth = 28
imgHeight = 28

# delim character 
delimiter = ","

# create or delete all current content
open('mnist - 1 channel - 8 bit.csv', 'w').close()

# open for appending
f = open('mnist - 1 channel - 8 bit.csv','a')

header = "".join(["Pixel_{0}{1}".format(index, delimiter) for index in range(imgWidth * imgHeight)])
header = header[:-1]

f.write("Label" + delimiter + header + "\n")

tempString = ""

for file in glob.glob("./mnist/testing/**/*.png"):  # ./newspaper/segments/raw/*.png
    image = cv2.imread(file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # imageAsInt = image.astype(int)

    s = io.BytesIO()
    np.savetxt(s, image, fmt='%i', delimiter=delimiter)
    rows = s.getvalue().decode().splitlines() 

    result = file + delimiter

    for row in rows: 
        result += row + delimiter

    result = result[:-1]  # remove last ','
    
    result += "\n"
    tempString += result   

f.write(tempString)
f.close()