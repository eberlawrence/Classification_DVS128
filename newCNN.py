import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as mimg
plt.rcParams["figure.figsize"] = (10,7)
from PIL import Image
from scipy import misc
import sys
import os
import multiprocessing
from sklearn.preprocessing import LabelBinarizer as lb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf

# DEEP LEARNING IMPORTS
from keras.models import Sequential, load_model
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Conv2D, Activation, Dropout, Flatten, MaxPooling2D

from keras.callbacks import ModelCheckpoint, EarlyStopping



sys.path.append('/home/user/GitHub/openDVS/Online')
import utilsDVS128
sys.path.append('/home/user/GitHub/HandStuff/Detection')
from segmentationUtils import segmentationUtils


if tf.__version__ == '2.0.0':
	physical_devices = tf.config.experimental.list_physical_devices('GPU')
	tf.config.experimental.set_memory_growth(physical_devices[0], True)
elif tf.__version__ == '1.14.0':
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	session = tf.Session(config=config)



def cnn_model():
	model = Sequential([Conv2D(8, 3, input_shape=(128, 128, 1)),
						MaxPooling2D(pool_size=2),
						Flatten(),
						Dense(8, activation='softmax')])
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	return model


#define a function to fit the model
def fit_and_evaluate(t_x, val_x, t_y, val_y, EPOCHS=10, BATCH_SIZE=128, val_split=0.1):
	model = None
	model = cnn_model(IMAGE_SIZE, n_classes)
	results = model.fit(t_x, t_y, epochs=EPOCHS, batch_size=BATCH_SIZE,	callbacks=[early_stopping, model_checkpoint], verbose=1, validation_split=val_split)
	print("Val Score: ", model.evaluate(val_x, val_y))
	return results



#function to draw confusion matrix
def draw_confusion_matrix(true,preds):
	conf_matx = confusion_matrix(true, preds)
	sns.heatmap(conf_matx, annot=True, annot_kws={"size": 12},fmt='g', cbar=False, cmap="Blues")
	plt.show()
	#return conf_matx



#function for converting predictions to labels
def prep_submissions(preds_array, file_name='abc.csv'):
	preds_df = pd.DataFrame(preds_array)
	predicted_labels = preds_df.idxmax(axis=1) #convert back one hot encoding to categorical variabless
	return predicted_labels


def saveModelAndWeights(model):
	model_json = model.to_json()
	with open("model.json", "w") as json_file:
		json_file.write(model_json)
	model.save_weights("model.h5")
	print("Saved model to disk")


n_classes = 8
IMAGE_SIZE = (128, 128, 1)
img, labels = utilsDVS128.createDataset()

imgROI_train, imgROI_test = [], []
for i, m in enumerate(img[0]):
	watershedImage, mask, detection, opening, sure_fg, sure_bg, markers = segmentationUtils.watershed(m,'--neuromorphic',minimumSizeBox=0.5,smallBBFilter=True,centroidDistanceFilter = True, mergeOverlapingDetectionsFilter = True,flagCloserToCenter=True)
	_, interp_img_train = segmentationUtils.getROI(detection, img[0][i])
	imgROI_train.append(interp_img_train)


for i, m in enumerate(img[1]):
	watershedImage, mask, detection, opening, sure_fg, sure_bg, markers = segmentationUtils.watershed(m,'--neuromorphic',minimumSizeBox=0.5,smallBBFilter=True,centroidDistanceFilter = True, mergeOverlapingDetectionsFilter = True,flagCloserToCenter=True)
	_, interp_img_test = segmentationUtils.getROI(detection, img[1][i])
	imgROI_test.append(interp_img_test)

img = [[], []]
img[0], img[1] = np.array(imgROI_train), np.array(imgROI_test)

# for i in img[1]:
# 	plt.imshow(i, cmap='gray')
# 	plt.show()

##########################################################################################
##########################################################################################
##########################################################################################

train_img, test_img, train_labels, test_labels = img[0], img[1], labels[0], labels[1]
train_img = train_img.reshape(train_img.shape[0], 128, 128, 1)
train_img = (train_img / 255).astype('float32')
l = np.zeros(len(train_labels) * n_classes).reshape(len(train_labels), n_classes).astype('int')
for i, v in enumerate(train_labels):
	l[i][v] = 1


train_labels = l


test_img = test_img.reshape(test_img.shape[0], 128, 128, 1)
test_img = (test_img / 255).astype('float32')
l = np.zeros(len(test_labels) * n_classes).reshape(len(test_labels), n_classes).astype('int')
for i, v in enumerate(test_labels):
	l[i][v] = 1


test_labels = l


n_folds = 5
epochs = 20
batch_size = 32
model_history = []
cv_train, cv_val = [], []

#define the model checkpoint callback -> this will keep on saving the model as a physical file
model_checkpoint = ModelCheckpoint('fas_mnist_1.h5', verbose=1, save_best_only=True)

kfold = StratifiedKFold(n_splits=n_folds, shuffle=False, random_state=0)
for train, test in kfold.split(train_img, labels[0]):
	cv_train.append(train_img[train])
	cv_val.append(train_labels[train])


cv_train = np.array(cv_train)
cv_val = np.array(cv_val)

def cross_validation_CNN(cv_train, cv_val, i):
	model = cnn_model()
	model_history.append(model.fit(cv_train[i], cv_val[i], batch_size=batch_size, epochs=epochs, callbacks=[model_checkpoint], verbose=1, validation_split=0.2))
	print("Val Score: ", model.evaluate(test_img, test_labels))


for i in range(len(cv_val)):
	cross_validation_CNN(cv_train, cv_val, i)



plt.title('Accuracies vs Epochs')
for i in range(n_folds):
	plt.plot(model_history[i].history['accuracy'], label='accuracy Training Fold ' + str(i + 1))
	plt.plot(model_history[i].history['val_accuracy'], label='val_accuracy Training Fold ' + str(i + 1), linestyle = "dashdot")


plt.legend()
plt.show()


plt.title('Train Accuracy vs Val Accuracy')
for i in range(n_folds):
	# plt.plot(model_history[i].history['accuracy'], label='Train Accuracy Fold ' + str(i + 1))
	plt.plot(model_history[i].history['loss'], label='loss Fold ' + str(i + 1), linestyle = "dashdot")
	# plt.plot(model_history[i].history['loss'], label='Training Fold ' + str(i + 1))
	plt.plot(model_history[i].history['val_loss'], label='val_loss Fold ' + str(i + 1))

plt.legend()
plt.show()


#Load the model that was saved by ModelCheckpoint
model = load_model('fas_mnist_1.h5')
model.evaluate(test_img, test_labels)


test_preds = model.predict(test_img)
test_preds_labels = prep_submissions(test_preds)

print(classification_report(labels[1], test_preds_labels))

draw_confusion_matrix(labels[1], test_preds_labels)

saveModelAndWeights(model)



# Val Score:  [0.1520575923304183, 0.9696078300476074]
# Val Score:  [0.1659652554822684, 0.966911792755127]
# Val Score:  [0.14400349408753363, 0.9671568870544434]
# Val Score:  [0.14077089210132174, 0.9698529243469238]
# Val Score:  [0.15472410473351678, 0.9666666388511658]
#
# a =[0.9696078300476074, 0.966911792755127, 0.9671568870544434, 0.9698529243469238, 0.9666666388511658]

# knife/knife_1, knife/knife_2, knife/knife_3, knife/knife_4, knife/knife_5, knife/knife_6, knife/knife_7, knife/knife_8
# key/key_1, key/key_2, key/key_3, key/key_4, key/key_5, key/key_6, key/key_7, key/key_8
# pencil/pencil_1, pencil/pencil_2, pencil/pencil_3, pencil/pencil_4, pencil/pencil_5, pencil/pencil_6, pencil/pencil_7, pencil/pencil_8
# spoon/spoon_1, spoon/spoon_2, spoon/spoon_3, spoon/spoon_4, spoon/spoon_5, spoon/spoon_6, spoon/spoon_7, spoon/spoon_8
#
#
# apple/apple_1, apple/apple_2, apple/apple_3, apple/apple_4, apple/apple_5, apple/apple_6, apple/apple_7, apple/apple_8
# banana/banana_1, banana/banana_2, banana/banana_3, banana/banana_4, banana/banana_5, banana/banana_6, banana/banana_7, banana/banana_8
# mug/mug_1, mug/mug_2, mug/mug_3, mug/mug_4, mug/mug_5, mug/mug_6, mug/mug_7, mug/mug_8
# phone/phone_1, phone/phone_2, phone/phone_3, phone/phone_4, phone/phone_5, phone/phone_6, phone/phone_7, phone/phone_8

#knife/knife_1, knife/knife_2, knife/knife_3, knife/knife_4, knife/knife_5, knife/knife_6, knife/knife_7, knife/knife_8, key/key_1, key/key_2, key/key_3, key/key_4, key/key_5, key/key_6, key/key_7, key/key_8, pencil/pencil_1, pencil/pencil_2, pencil/pencil_3, pencil/pencil_4, pencil/pencil_5, pencil/pencil_6, pencil/pencil_7, pencil/pencil_8, spoon/spoon_1, spoon/spoon_2, spoon/spoon_3, spoon/spoon_4, spoon/spoon_5, spoon/spoon_6, spoon/spoon_7, spoon/spoon_8
#anything, anything, anything, anything, anything
#apple/apple_1, apple/apple_2, apple/apple_3, apple/apple_4, apple/apple_5, apple/apple_6, apple/apple_7, apple/apple_8, banana/banana_1, banana/banana_2, banana/banana_3, banana/banana_4, banana/banana_5, banana/banana_6, banana/banana_7, banana/banana_8, mug/mug_1, mug/mug_2, mug/mug_3, mug/mug_4, mug/mug_5, mug/mug_6, mug/mug_7, mug/mug_8, phone/phone_1, phone/phone_2, phone/phone_3, phone/phone_4, phone/phone_5, phone/phone_6, phone/phone_7, phone/phone_8


#knife/knife_1, knife/knife_2, knife/knife_3, knife/knife_4, knife/knife_5, knife/knife_6, knife/knife_7, key/key_1, key/key_2, key/key_3, key/key_4, key/key_5, key/key_6, key/key_7, pencil/pencil_1, pencil/pencil_2, pencil/pencil_3, pencil/pencil_4, pencil/pencil_5, pencil/pencil_6, pencil/pencil_7, spoon/spoon_1, spoon/spoon_2, spoon/spoon_3, spoon/spoon_4, spoon/spoon_5, spoon/spoon_6, spoon/spoon_7
#anything, anything, anything, anything
#apple/apple_1, apple/apple_2, apple/apple_3, apple/apple_4, apple/apple_5, apple/apple_6, apple/apple_7, banana/banana_1, banana/banana_2, banana/banana_3, banana/banana_4, banana/banana_5, banana/banana_6, banana/banana_7, mug/mug_1, mug/mug_2, mug/mug_3, mug/mug_4, mug/mug_5, mug/mug_6, mug/mug_7, phone/phone_1, phone/phone_2, phone/phone_3, phone/phone_4, phone/phone_5, phone/phone_6, phone/phone_7




# for i in range(n_folds):
#     print("Training on Fold: ", i + 1)
#     #X_train, X_test, y_train, y_test = train_test_split(train_img, train_labels, test_size=0.1, random_state=0)
#     model_history.append(fit_and_evaluate(train_img, test_img, train_labels, test_labels, epochs, batch_size))
#     print("======="*12, end="\n\n\n")



# for i in range(50):
# 	test_labels[i+50]
# 	plt.imshow(test_img[i+50].reshape(128, 128))
# 	plt.show()
