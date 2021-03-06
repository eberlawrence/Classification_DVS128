import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import random
from time import time
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold

# DEEP LEARNING IMPORTS
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Activation, Dropout, Flatten, MaxPooling2D, AveragePooling2D
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
from keras.losses import categorical_crossentropy

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



def lenet_model(width, height, depth, classes):
	inputShape = (height, width, depth)
	model = Sequential()
	model.add(Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=inputShape))
	model.add(AveragePooling2D())
	model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
	model.add(AveragePooling2D())
	model.add(Flatten())
	model.add(Dense(units=120, activation='relu'))
	model.add(Dense(units=84, activation='relu'))
	model.add(Dense(units=classes, activation = 'softmax'))
	return model


#######################################################################################################################################
#######################################################################################################################################


n_classes = 2
image_shape = (128, 128, 1)
img, lab = utilsDVS128.createDataset(path='/home/user/GitHub/aedatFiles/new_dataset/',
										objClass=[['allendriver/allendriver'],
											  	  ['battery/battery'],
										   		  ['brush/brush'],
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
												  ['mug/mug'],
												  ['orange/orange'],
												  ['phone/phone'],
												  ['wallet/wallet']],
										setUp=False,
										tI=40000)


# a = 7000
# for i, v in enumerate(img):
# 	a += 30
# 	print(i + a)
# 	plt.imshow(img[i+a], cmap='gray')
# 	plt.show()
#




imgROI = []
rem = []
for i, m in enumerate(img):
	watershedImage, mask, detection, opening, sure_fg, sure_bg, markers = segmentationUtils.watershed(m,'--neuromorphic',minimumSizeBox=0.5,smallBBFilter=True,centroidDistanceFilter = True, mergeOverlapingDetectionsFilter = True,flagCloserToCenter=True)
	if len(detection) > 0:
		if lab[i] < 15:
			if detection[0][2]*detection[0][3] > 200:
				_, interp_img_train = segmentationUtils.getROI(detection, img[i])
				imgROI.append(interp_img_train)
			else:
				rem.append(i)
		else:
			if detection[0][2]*detection[0][3] > 800:
				_, interp_img_train = segmentationUtils.getROI(detection, img[i])
				imgROI.append(interp_img_train)
			else:
				rem.append(i)
	else:
		rem.append(i)

labels = np.delete(lab, rem)


for i in range(30):
	print(i, ": ", len(labels[labels == i]))


images = np.array(imgROI)

for i in range(len(images)):
	images[i][images[i] == 0] = 255
	images[i][images[i] < 200] = 0


size = 300
final_images = []
for i in range(30):
	aux_images = images[labels == i]
	delta = len(aux_images) - size
	to_remove = random.sample(range(len(aux_images)), delta)
	del_images = np.delete(aux_images, to_remove, 0)
	final_images.append(del_images)


final_images = np.array(final_images)

images = []


a = 0
for i, v in enumerate(train_img):
	a += 50
	print(train_labels[i + a])
	plt.imshow(train_img[i + a], cmap="gray")
	plt.show()



##########################################################################################
##########################################################################################
##########################################################################################

images_Tripod = np.concatenate((final_images[0 : 10]))
images_Power = np.concatenate((final_images[15 : 25]))
total_images = np.concatenate((images_Tripod, images_Power))
total_labels = np.concatenate((np.zeros(len(images_Tripod)), np.ones(len(images_Power))))

images_Tripod, images_Tripod = [], []

extra_images_Tripod = np.concatenate((final_images[10 : 15]))
extra_images_Power = np.concatenate((final_images[25 : 30]))
extra_images = np.concatenate((extra_images_Tripod, extra_images_Power))
extra_labels = np.concatenate((np.zeros(len(extra_images_Tripod)), np.ones(len(extra_images_Power))))

extra_images_Tripod, extra_images_Power = [], []

extra_images = extra_images.reshape(extra_images.shape[0], image_shape[0], image_shape[1], 1)
extra_images = (extra_images / 255).astype('float32')

final_images = []

total_images = total_images.reshape(total_images.shape[0], image_shape[0], image_shape[1], 1)
total_images = (total_images / 255).astype('float32')



shuffler = np.random.permutation(len(total_images))
total_images = total_images[shuffler]
total_labels = total_labels[shuffler]

total_labels_encoded = to_categorical(total_labels, num_classes=n_classes)
extra_labels_encoded = to_categorical(extra_labels, num_classes=n_classes)

##########################################################################################
##########################################################################################


# initialize the number of epochs to train for, initia learning rate, and batch size

n_folds = 5
EPOCHS = 15
BATCH_SIZE = 96
model_history = []
cv_train, cv_val = [], []
model = []
acc_per_fold, loss_per_fold = [], []


kfold = StratifiedKFold(n_splits=n_folds)

# K-fold Cross Validation model evaluation
fold_no = 1
for train, test in kfold.split(total_images, total_labels):
	# model architecture
	model = lenet_model(width=image_shape[0], height=image_shape[1], depth=1, classes=n_classes)
	# Compile the model
	model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['accuracy'])
	# Generate a print
	print('------------------------------------------------------------------------')
	print(f'Training for fold {fold_no} ...')
	# Fit data to model
	model_history.append(model.fit(total_images[train], total_labels_encoded[train], validation_data=(total_images[test], total_labels_encoded[test]), batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1))
	# Generate generalization metrics
	scores = model.evaluate(total_images[test], total_labels_encoded[test], verbose=0)
	print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
	acc_per_fold.append(scores[1] * 100)
	loss_per_fold.append(scores[0])
	# Increase fold number
	fold_no = fold_no + 1




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



##########################################################################################
##########################################################################################


EPOCHS = 15
BATCH_SIZE = 96
model_history = []
model = []

model = lenet_model(width=image_shape[0], height=image_shape[1], depth=1, classes=n_classes)
model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['accuracy'])
model_history.append(model.fit(total_images, total_labels_encoded, validation_data=(extra_images, extra_labels_encoded), batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1))


plt.title('Accuracies vs Epochs')
plt.plot(model_history[0].history['accuracy'], label='Accuracy Training')
plt.plot(model_history[0].history['val_accuracy'], label='validation_accuracy', linestyle = "dashed")
plt.legend()
plt.show()


plt.title('Loss vs Epochs')
plt.plot(model_history[0].history['loss'], label='Loss Training')
plt.plot(model_history[0].history['val_loss'], label='validation_loss', linestyle = "dashed")
plt.legend()
plt.show()



model.evaluate(extra_images, extra_labels_encoded)



saveModelAndWeights(model, name="lenet_model")























extra_images_aux = extra_images.reshape(extra_images.shape[0], image_shape[0], image_shape[1])
while True:
	index = np.random.randint(len(extra_images_aux))
	print(extra_labels[index])
	plt.imshow(extra_images_aux[index], cmap="gray")
	plt.show()


# while True:
# 	index = np.random.randint(len(total_images))
# 	print(total_labels[index])
# 	plt.imshow(total_images[index].reshape(128, 128), cmap="gray")
# 	plt.show()
#


test_preds = model.predict(extra_images)
test_preds_labels = prep_submissions(test_preds)
print(classification_report(test_labels, test_preds_labels))
draw_confusion_matrix(test_labels, test_preds_labels)


saveModelAndWeights(model, name="lenet_model")


# allendriver/allendriver, battery/battery, brush/brush, fork/fork, key/key, knife/knife, lighter/lighter, mechanical_pencil/mechanical_pencil, pen/pen, pendrive/pendrive, razor/razor, screw/screw, screwdriver/screwdriver, spoon/spoon, toothbrush/toothbrush, wrench/wrench
# ball/ball, bottle/bottle, box/box, calculator/calculator, camera/camera, can/can, case/case, container/container, cup/cup, hd/hd, lamp/lamp, orange/orange, pear/pear, phone/phone, wallet/wallet



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
	a += 20
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

kfold = StratifiedKFold(n_splits=n_folds)
for train, test in kfold.split(total_images, total_labels):
	print(test)
	cv_train.append(total_images[train])
	cv_val.append(total_labels_encoded[train])


cv_train = np.array(cv_train)
cv_val = np.array(cv_val)

for i in range(len(cv_val)):
	model = lenet_model(width=image_shape[0], height=image_shape[1], depth=1, classes=n_classes)
	model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['accuracy'])
	model_history.append(model.fit(cv_train[i], cv_val[i], batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1))
	print("Val Score: ", model.evaluate(test_img, cv_val[i], num_classes=n_classes)))










Angola
Argentina
Austria
Bolivia
Botswana
Brazil
Burundi
Cape Verde
Chile
Colombia
Democratic Republic of the Congo
Ecuador
Eswatini
French Guiana
Guyana
Lesotho
Malawi
Mauritius
Mozambique
Namibia
Paraguay
Panama
Peru
Rwanda
Seychelles
South Africa
Suriname
Tanzania
United Arab Emirates
Uruguay
Venezuela
Zambia
Zimbabwe

*Africa*

Morocco
Algeria
Tunisia
Libya
Egypt



*America*

Mexico
Puerto Rico
Honduras
Costa Rica
Belize
El Salvador
Honduras
Nicaragua
Cuba
Dominican Republic
United States
Canada
Jamaica
Haiti
