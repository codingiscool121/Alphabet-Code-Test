import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import seaborn as sb
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os, ssl, time
import cv2 as cv
import warnings
warnings.filterwarnings("ignore")


# for i in range(0,60):
#     ret,bg = cap.read()
X= np.load("image (1).npz")["arr_0"]
Y= pd.read_csv("labels.csv")["labels"]

print(pd.Series(Y).value_counts())

if(not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
nclasses = len(classes)
samples_perclass = 5
# figure = plt.figure(figsize=(nclasses*2, (1+samples_perclass*2)))

# iclass=0
# for cls in classes:
#   id= np.flatnonzero(Y==cls)
#   id= np.random.choice(id, samples_perclass, replace=False)
#   i=0
#   for j in id:
#       plt_j=i*nclasses+iclass+1
#       p=plt.subplot(samples_perclass, nclasses, plt_j)
#       p= sb.heatmap(np.reshape(X[j], (22,30)), cmap=plt.cm.Blues, xticklabels=False, yticklabels = False, cbar=False)
#       i=i+1
#   iclass= iclass+1

  #Train the data and get the total number of images along with the pixel size of each image
print("The total number of letters is ", len(X))
print("Pixel Size of each image: ", len(X[1]))
#What are we training it for? Is it to recognize letters?
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y, test_size=0.25, random_state=0)
#I think we are scaling here. Need this part explained again. 
Xtrainscale = Xtrain/255.0
#What is X train scale? Why is the number 255?
Xtestscale= Xtest/255.0
classifier = LogisticRegression(solver="saga", multi_class="multinomial").fit(Xtrainscale, Ytrain)
Ypredict= classifier.predict(Xtestscale)
accuracy=accuracy_score(Ytest, Ypredict)
print("The accuracy of this model was", accuracy)

cap= cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH,640) 
cap.set(cv.CAP_PROP_FRAME_HEIGHT,480)
print("Camera turned on")
while(True):
  #Capturing frames one by one
    try:
        ret, frame = cap.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        height, width = gray.shape
        upper_left= (int(width/2)-56, int(height/2)-56)
        bottom_right= (int(width/2)+56, int(height/2)+56)
        cv.rectangle(gray, upper_left, bottom_right, (0,255,0), 2)
        roi= gray[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]

        #Converting into PIL format. It can understand the image in a better way using this format.
        im_pil= Image.fromarray(roi)
        #converting to gray scale image. The 'L' format means each pixel is represented by a single value from 0 to 255.
        image_bw = im_pil.convert('L')
        #Resize the image. Anti Alias (AA) is used to make the image look better.
        image_bw_resized = image_bw.resize((28,28), Image.ANTIALIAS)
        #We are now going to invert the image. We are then going to give it a value between 0 and 255.
        image_bw_resized_inverted= PIL.ImageOps.invert(image_bw_resized)
        #Pixel filter is the way to reduce the sharpness of the image.
        pixel_filter=20
        #We are now going to convert this to a scaler quantity.
        min_pixel= np.percentile(image_bw_resized_inverted, pixel_filter)
        #We are now going to limit the values between 0 and 255.
        image_bw_resized_inverted_scaled= np.clip(min_pixel, 0, 255) 
        max_pixel= np.max(image_bw_resized_inverted)

        #Converting this data into an array, and giving it to a model so it can predict the letter.
        image_bw_resized_inverted_scaled= np.asarray(image_bw_resized_inverted_scaled)/max_pixel

        #We are inverting the image, and then we are predicting using the sample. 
        test_sample= np.array(image_bw_resized_inverted_scaled).reshape(1, 784)
        # prediction= classifier.predict(test_sample)
    
        #This is the letter that the computer recognized.
        cv.imshow("frame", gray)
        # print("Predicted letter is", prediction)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        pass
cap.release()
cv.destroyAllWindows()
