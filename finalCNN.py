import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# DEEP LEARNING IMPORTS
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Activation, Dropout, Flatten, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

sys.path.append('/home/user/GitHub/openDVS/Online')
import utilsDVS128
sys.path.append('/home/user/GitHub/HandStuff/Detection')
from segmentationUtils import segmentationUtils


#######################################################################################################################################
#######################################################################################################################################


if tf.__version__ == '2.0.0':
	physical_devices = tf.config.experimental.list_physical_devices('GPU')
	tf.config.experimental.set_memory_growth(physical_devices[0], True)
elif tf.__version__ == '1.14.0':
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	session = tf.Session(config=config)


#######################################################################################################################################
#######################################################################################################################################


def saveModelAndWeights(model, name="model"):
	model_json = model.to_json()
	with open(name + ".json", "w") as json_file:
		json_file.write(model_json)
	model.save_weights(name + ".h5")
	print("Saved model to disk")


#function to draw confusion matrix
def draw_confusion_matrix(true,preds):
	conf_matx = confusion_matrix(true, preds)
	sns.heatmap(conf_matx, annot=True, annot_kws={"size": 12}, fmt='g', cbar=False, cmap="Blues")
	plt.show()


#function for converting predictions to labels
def prep_submissions(preds_array):
	preds_df = pd.DataFrame(preds_array)
	predicted_labels = preds_df.idxmax(axis=1) #convert back one hot encoding to categorical variabless
	return predicted_labels


class finalNetwork:
	@staticmethod
	def build(width, height, depth, classes):
		# initialize the model
		model = Sequential()
		inputShape = (height, width, depth)
		# first set of CONV => RELU => POOL layers
		model.add(Conv2D(16, (3, 3), input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		# second set of CONV => RELU => POOL layers
		model.add(Conv2D(32, (3, 3)))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		# third set of CONV => RELU => POOL layers
		model.add(Conv2D(64, (3, 3)))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		# first (and only) set of FC => RELU layers
		model.add(Flatten())
		model.add(Dense(128))
		model.add(Activation("relu"))
		model.add(Dense(128))
		model.add(Activation("relu"))
		# softmax classifier
		model.add(Dense(classes))
		model.add(Activation("softmax"))
		# return the constructed network architecture
		return model


#######################################################################################################################################
#######################################################################################################################################


'allendriver/allendriver', 'battery/battery', 'brush/brush', 'fork/fork', 'key/key', 'knife/knife', 'lighter/lighter', 'mechanical_pencil/mechanical_pencil', 'pen/pen', 'pendrive/pendrive', r'azor/razor', 'screw/screw', 'screwdriver/screwdriver', 'spoon/spoon', 'toothbrush/toothbrush', 'wrench/wrench'
['ball/ball'], ['bottle/bottle'], ['box/box'], ['calculator/calculator'], ['camera/camera'], ['can/can'], ['case/case'], ['container/container'], ['cup/cup'], ['hd/hd'], ['lamp/lamp'], ['orange/orange'], ['pear/pear'], ['phone/phone'], ['wallet/wallet']


n_classes = 2
image_shape = (128, 128, 1)
img, labels = utilsDVS128.createDataset(path='/home/user/GitHub/aedatFiles/new_dataset/',
										objClass=[['allendriver/allendriver'],
											  	  ['battery/battery'],
										   		  ['brush/brush'],
												  ['fork/fork'],
												  ['key/key'],
												  ['knife/knife'],
												  ['lighter/lighter'],
												  ['mechanical_pencil/mechanical_pencil'],
												  ['pen/pen'],
												  ['pendrive/pendrive'],
												  ['razor/razor'],
												  ['screw/screw'],
												  ['screwdriver/screwdriver'],
												  ['spoon/spoon'],
												  ['toothbrush/toothbrush'],
												  ['wrench/wrench'],
												  ['ball/ball'],
												  ['bottle/bottle'],
												  ['box/box'],
												  ['calculator/calculator'],
												  ['camera/camera'],
												  ['can/can'],
												  ['case/case'],
												  ['container/container'],
												  ['cup/cup'],
												  ['hd/hd'],
												  ['lamp/lamp'],
												  ['orange/orange'],
												  ['pear/pear'],
												  ['phone/phone'], 
												  ['wallet/wallet']],
										setUp=False,
										tI=40000)

imgROI_train, imgROI_test = [], []
rem = []
for i, m in enumerate(img[0]):
	watershedImage, mask, detection, opening, sure_fg, sure_bg, markers = segmentationUtils.watershed(m,'--neuromorphic',minimumSizeBox=0.5,smallBBFilter=True,centroidDistanceFilter = True, mergeOverlapingDetectionsFilter = True,flagCloserToCenter=True)
	if len(detection) > 0 and (detection[0][2] > 50 or detection[0][3] > 50):
		_, interp_img_train = segmentationUtils.getROI(detection, img[0][i])
		imgROI_train.append(interp_img_train)
	else:
		rem.append(i)


lab1 = np.delete(labels[0], rem)

rem = []
for i, m in enumerate(img[1]):
	watershedImage, mask, detection, opening, sure_fg, sure_bg, markers = segmentationUtils.watershed(m,'--neuromorphic',minimumSizeBox=0.5,smallBBFilter=True,centroidDistanceFilter = True, mergeOverlapingDetectionsFilter = True,flagCloserToCenter=True)
	if len(detection) > 0 and (detection[0][2] > 50 or detection[0][3] > 50):
		_, interp_img_test = segmentationUtils.getROI(detection, img[1][i])
		imgROI_test.append(interp_img_test)
	else:
		rem.append(i)

lab2 = np.delete(labels[1], rem)
img = [[], []]
img[0], img[1] = np.array(imgROI_train), np.array(imgROI_test)N


##########################################################################################
##########################################################################################
##########################################################################################


train_img, test_img, train_labels, test_labels = img[0], img[1], lab1, lab2
train_img = train_img.reshape(train_img.shape[0], image_shape[0], image_shape[1], 1)
train_img = (train_img / 255).astype('float32')
test_img = test_img.reshape(test_img.shape[0], image_shape[0], image_shape[1], 1)
test_img = (test_img / 255).astype('float32')

(train_img, test_img, train_labels, test_labels) = train_test_split(train_img, train_labels, test_size=0.1)

aux_test_labels = train_labels
aux_train_labels = test_labels

train_labels = to_categorical(train_labels, num_classes=n_classes)
test_labels = to_categorical(test_labels, num_classes=n_classes)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")


##########################################################################################
##########################################################################################


# initialize the number of epochs to train for, initia learning rate, and batch size
EPOCHS = 20
INIT_LR = 1e-3
BATCH_SIZE = 500

# initialize the model
print("[INFO] compiling model...")
model = finalNetwork.build(width=image_shape[0], height=image_shape[1], depth=1, classes=n_classes)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
# train the network
print("[INFO] training network...")
H = model.fit(x=aug.flow(train_img, train_labels, batch_size=BATCH_SIZE), validation_data=(test_img, test_labels), steps_per_epoch=len(train_img) // BATCH_SIZE,	epochs=EPOCHS, verbose=1)


plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, EPOCHS), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, EPOCHS), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Santa/Not Santa")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.show()



model.evaluate(testX, testY)

test_preds = model.predict(testX)
test_preds_labels = prep_submissions(test_preds)

print(classification_report(aux_test_labels, test_preds_labels))

draw_confusion_matrix(aux_test_labels, test_preds_labels)


test_img, test_labels
aux = to_categorical(test_labels, num_classes=2)

model.evaluate(test_img, aux)


test_preds = model.predict(test_img)

test_preds_labels = prep_submissions(test_preds)

print(classification_report(test_labels, test_preds_labels))

draw_confusion_matrix(test_labels, test_preds_labels)




                  objClass=[['spoon/spoon_1', 'spoon/spoon_2', 'spoon/spoon_3', 'spoon/spoon_4', 'spoon/spoon_5'], ['pencil/pencil_1', 'pencil/pencil_2', 'pencil/pencil_3', 'pencil/pencil_4', 'pencil/pencil_5'], ['apple/apple_1', 'apple/apple_2', 'apple/apple_3', 'apple/apple_4', 'apple/apple_5']],
                  setUp=True,
allendriver/allendriver, battery/battery, brush/brush, fork/fork, key/key, knife/knife, lighter/lighter, mechanical_pencil/mechanical_pencil, pen/pen, pendrive/pendrive, razor/razor, screw/screw, screwdriver/screwdriver, spoon/spoon, toothbrush/toothbrush, wrench/wrench
ball/ball, bottle/bottle, box/box, calculator/calculator, camera/camera, can/can, case/case, container/container, cup/cup, hd/hd, lamp/lamp, orange/orange, pear/pear, phone/phone, wallet/wallet



# knife/knife_1, knife/knife_2, knife/knife_3, knife/knife_4, knife/knife_5, knife/knife_6, knife/knife_7, knife/knife_8
# key/key_1, key/key_2, key/key_3, key/key_4, key/key_5, key/key_6, key/key_7, key/key_8
# pencil/pencil_1, pencil/pencil_2, pencil/pencil_3, pencil/pencil_4, pencil/pencil_5, pencil/pencil_6, pencil/pencil_7, pencil/pencil_8
# spoon/spoon_1, spoon/spoon_2, spoon/spoon_3, spoon/spoon_4, spoon/spoon_5, spoon/spoon_6, spoon/spoon_7, spoon/spoon_8


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

a = 0
for i, v in enumerate(img):
	a += 50
	print(i + a)
	plt.imshow(img[i+a], cmap='gray')
	plt.show()


#
# def cnn_model():
# 	# Creating model
# 	model = Sequential()
# 	# This is the first convolution
# 	model.add(Conv2D(filters=16, kernel_size=(3, 3), padding="same", activation='relu', input_shape=(128, 128, 1)))
# 	# model.add(MaxPooling2D((2, 2)))
# 	# # The second convolution
# 	# model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation='relu'))
# 	# model.add(MaxPooling2D((2, 2)))
# 	# # The third convolution
# 	# model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation='relu'))
# 	# model.add(MaxPooling2D((2, 2)))
# 	# Flatten the results to feed into a DNN
# 	model.add(Flatten())
# 	# 512 neuron hidden layer
# 	model.add(Dense(128, activation='relu'))
# 	# Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('dandelions') and 1 for the other ('grass')
# 	model.add(Dense(1, activation='sigmoid'))
# 	# Compiling the CNN
# 	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# 	return model
#
