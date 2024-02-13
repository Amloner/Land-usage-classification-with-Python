import seaborn as sns
from matplotlib import pyplot as plt 
import numpy as np
import pandas as pd
import os
from glob import glob
import json
from PIL import Image
from colormap import rgb2hex, hex2rgb

#Classifier2
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import ConfusionMatrixDisplay
print('hi2')

def get_data(folder, file):
    # download json
    f = open(folder + "/" + file,)
    data = json.load(f)
    f.close()
    cl = {}
    # Create a dictionary with classes
    for i, c in enumerate(data['classes']):
        cl[i] = dict(c)
        
    for k, v in cl.items():
        print('Class', k)
        for k2, v2 in v.items():
            print("   ", k2, v2)
    data = []
    
    # download images
    sd = [item for item in os.listdir(folder) if os.path.isdir(folder + '/' + item)] # a list of subdirectories
    print("Subdirectories: ", sd)
    for f in sd[1:2]: #choose one of the subdirectories to download
        print("Downloading: ", f)
        images = glob(folder + "/" + f + "/images" + "/*.jpg") # create a list of image files
        for im in images:
            mask_f = im.replace("images", "masks").replace("jpg", "png") # create a list of mask files
            image = Image.open(im) 
            mask = Image.open(mask_f)


            if len(np.array(mask).shape) > 2:
                data.append([image, mask])


    return (data)

def create_DataSet(data):
    DS = pd.DataFrame()
    for image, mask in data:
        # transform image to matrix
        im = np.asarray(image) 
        mk = np.asarray(mask)
        # transform a one-dimension array of r, g, b colors
        red = im[:,:,0].flatten()
        green = im[:,:,1].flatten()
        blue = im[:,:,2].flatten()
        im_f = np.array([red, green, blue])
        red = mk[:,:,0].flatten()
        green = mk[:,:,1].flatten()
        blue = mk[:,:,2].flatten()
        # calculate hex classes
        h = np.array([rgb2hex(*m) for m in zip(red, green, blue)])
        mk_f = np.array([red, green, blue, h])      
        d = np.concatenate((im_f, mk_f), axis=0)
        # create a DataSet
        DS_new = pd.DataFrame(np.transpose(d), columns = ['Im_Red', 'Im_Green', 'Im_Blue', 'Mk_Red', 'Mk_Green', 'Mk_Blue', 'HEX'])
        if len(DS) == 0:
            DS = DS_new
        else:
            DS = DS._append(DS_new)
    return DS


d = "Semantic segmentation dataset"
f = "classes.json"
data = get_data(d, f)

print("Create a training DataSet")
train = create_DataSet(data[:8])
print(train)
print("Create a test DataSet")
test = create_DataSet(data[4:])
print(test)

train.loc[:, 'HEX'] = train['HEX'].astype('category')
train['HEX']

test.loc[:, 'HEX'] = test['HEX'].astype('category')
test['HEX']

cl = ['Im_Red', 'Im_Green', 'Im_Blue', 'Mk_Red', 'Mk_Green', 'Mk_Blue']
train[cl] = train[cl].astype('int64')
test[cl] = test[cl].astype('int64')
print (train.info())
print (test.info())



clf = LogisticRegression(max_iter=100, n_jobs=-1)
c = train.columns
clf.fit(train[c[0:3]], train[c[-1:]].values.ravel())


scores_train = clf.score(train[c[0:3]], train[c[-1:]].values.ravel())
scores_test = clf.score(test[c[0:3]], test[c[-1:]].values.ravel())
print('Accuracy train DataSet: {: .1%}'.format(scores_train), 'Accuracy test DataSet: {: .1%}'.format(scores_test))


test_image = 4 # choose the number of images from the data list
mask_test = data[test_image:test_image+1] # Test Image + Mask
mask_test_DataSet = create_DataSet(mask_test) #Build a DataSet
print(mask_test_DataSet)

c = mask_test_DataSet.columns
mask_test_predict = clf.predict(mask_test_DataSet[c[0:3]])
print(mask_test_predict)

size = mask_test[0][1].size #get original image size
print(size)

predict_img = np.array(mask_test_predict).reshape((size[1], size[0])) #reshaping array of HEX colour
print(predict_img)

rgb_size = np.array(mask_test[0][0]).shape
print("Image size: ", rgb_size)
predict_img_rgb = np.zeros(rgb_size)
for i, r in enumerate(predict_img):
    for j, c in enumerate(r):
        predict_img_rgb[i, j, 0], predict_img_rgb[i, j, 1], predict_img_rgb[i, j, 2] = hex2rgb(c)

        
        
fig = plt.figure(figsize = (10,10)) #display the last image + mask
predict_img_rgb = predict_img_rgb.astype('int')

plt.subplot(1,2,1)
plt.title("Model mask")
plt.imshow(predict_img_rgb)

plt.subplot(1,2,2)
plt.title("Real mask")
plt.imshow(mask_test[0][1])
plt.show()
