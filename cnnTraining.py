import sys
import keras
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping

sys.path.append('../openDVS/Online')
import utilsDVS128

if tf.__version__ == '2.0.0':
	physical_devices = tf.config.experimental.list_physical_devices('GPU')
	tf.config.experimental.set_memory_growth(physical_devices[0], True)
elif tf.__version__ == '1.14.0':
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	session = tf.Session(config=config)


class BiolabCNN:
	#"Grampeador, Estilete, Cubo, Celular, Tesoura, Caixa, Caneca, Mouse, Lapiseira, Nada"
	def __init__(self):
		self.model = Sequential()
		self.epochs = 20
		self.batch_size = 80
		self.num_classes = 0
		self.n_folds = 3
		self.model_history = []


	def adjustDataSet(self, size=0.30):
		self.img, self.labels = utilsDVS128.createDataset(maxSamples=800)

		X_train, X_test, y_train, y_test = train_test_split(self.img, self.labels, test_size=size, random_state=0, stratify=Target)

		self.num_classes = y_train.max() + 1
		self.X_train = X_train.reshape(X_train.shape[0], 128, 128, 1)
		self.X_test = X_test.reshape(X_test.shape[0], 128, 128, 1)

		self.y_train = keras.utils.to_categorical(y_train, self.num_classes)
		self.y_test = keras.utils.to_categorical(y_test, self.num_classes)



	def myModel(self):

		# 1st Convolution layer
		self.model.add(Conv2D(64,(3,3), padding='same', input_shape=(128, 128,1)))
		self.model.add(BatchNormalization())
		self.model.add(Activation('relu'))
		self.model.add(MaxPooling2D(pool_size=(2, 2)))
		self.model.add(Dropout(0.25))

		# 2nd Convolution layer
		self.model.add(Conv2D(128,(5,5), padding='same'))
		self.model.add(BatchNormalization())
		self.model.add(Activation('relu'))
		self.model.add(MaxPooling2D(pool_size=(2, 2)))
		self.model.add(Dropout(0.25))

		# 3rd Convolution layer
		self.model.add(Conv2D(512,(3,3), padding='same'))
		self.model.add(BatchNormalization())
		self.model.add(Activation('relu'))
		self.model.add(MaxPooling2D(pool_size=(2, 2)))
		self.model.add(Dropout(0.25))

		# 4th Convolution layer
		self.model.add(Conv2D(512,(3,3), padding='same'))
		self.model.add(BatchNormalization())
		self.model.add(Activation('relu'))
		self.model.add(MaxPooling2D(pool_size=(2, 2)))
		self.model.add(Dropout(0.25))

		# Flattening
		self.model.add(Flatten())

		# Fully connected layer 1st layer
		self.model.add(Dense(256))
		self.model.add(BatchNormalization())
		self.model.add(Activation('relu'))
		self.model.add(Dropout(0.25))

		# Fully connected layer 2nd layer
		self.model.add(Dense(512))
		self.model.add(BatchNormalization())
		self.model.add(Activation('relu'))
		self.model.add(Dropout(0.25))

		self.model.add(Dense(self.num_classes, activation='softmax'))

		self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



	def dataGeneration(self):
		gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3, height_shift_range=0.08, zoom_range=0.08)
		test_gen = ImageDataGenerator()

		self.train_generator = gen.flow(self.X_train, self.y_train, batch_size=self.batch_size)
		self.test_generator = test_gen.flow(self.X_test, self.y_test, batch_size=self.batch_size)





	#define a function to fit the model
	def fit_and_evaluate(self, t_x, val_x, t_y, val_y, EPOCHS=20, BATCH_SIZE=128):
		#set early stopping criteria
		pat = 5 #this is the number of epochs with no improvment after which the training will stop
		early_stopping = EarlyStopping(monitor='val_loss', patience=pat, verbose=1)

		#define the model checkpoint callback -> this will keep on saving the model as a physical file
		model_checkpoint = ModelCheckpoint('fas_mnist_1.h5', verbose=1, save_best_only=True)
		results = self.model.fit(t_x, t_y, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[early_stopping, model_checkpoint], verbose=1, validation_split=0.1)
		print("Val Score: ", self.model.evaluate(val_x, val_y))
		return results




	def trainModel(self, cross_val=False):
		flag = input("Type 'S' for data generation and 'N' for not (Defaut is 'S'): ")

		if flag == 'N':
			self.model.fit(self.X_train, self.y_train, batch_size=self.batch_size, epochs=self.epochs, validation_data=(self.X_test, self.y_test), shuffle=True)


		if flag == "S":
			self.model.fit_generator(self.train_generator, steps_per_epoch=len(self.X_train)//64, epochs=5, validation_data=self.test_generator, validation_steps=len(self.X_test)//64)


		if cross_val == True:
			for i in range(self.n_folds):
			    print("Training on Fold: ",i+1)
			    t_x, val_x, t_y, val_y = train_test_split(self.X_train, self.y_train, test_size=0.1, random_state = np.random.randint(1,1000, 1)[0])
			    self.model_history.append(self.fit_and_evaluate(t_x, val_x, t_y, val_y, self.epochs, self.batch_size))
			    print("======="*12, end="\n\n\n")




	def testModel(self):
		scores = self.model.evaluate(self.X_test, self.y_test, verbose=1)
		print('Test loss:', scores[0])
		print('Test accuracy:', scores[1])



	def draw_confusion_matrix(true, preds):
		conf_matx = confusion_matrix(true, preds)
		sns.heatmap(conf_matx, annot=True,annot_kws={"size": 12},fmt='g', cbar=False, cmap="navy")
		plt.show()



	def saveModelAndWeights(self):
		model_json = self.model.to_json()
		with open("model.json", "w") as json_file:
			json_file.write(model_json)


		self.model.save_weights("model.h5")
		print("Saved model to disk")


def main():
	myCNN = BiolabCNN()
	myCNN.adjustDataSet()
	myCNN.myModel()
	# print(myCNN.model.summary())
	myCNN.dataGeneration()
	myCNN.trainModel()
	myCNN.testModel()
	myCNN.saveModelAndWeights()


if __name__ == "__main__":
	main()




#y_test = np.argmax(myCNN.y_test, axis=1)
#y_pred= myCNN.model.predict_classes(myCNN.X_test)
#a = confusion_matrix(y_test, y_pred)


b = 77
plt.imshow(img[b])
labels[b]
plt.show()



#knife/knife_1, knife/knife_2, knife/knife_3, knife/knife_4, knife/knife_5, knife/knife_6, knife/knife_7, knife/knife_8, key/key_1, key/key_2, key/key_3, key/key_4, key/key_5, key/key_6, key/key_7, key/key_8, pencil/pencil_1, pencil/pencil_2, pencil/pencil_3, pencil/pencil_4, pencil/pencil_5, pencil/pencil_6, pencil/pencil_7, pencil/pencil_8, spoon/spoon_1, spoon/spoon_2, spoon/spoon_3, spoon/spoon_4, spoon/spoon_5, spoon/spoon_6, spoon/spoon_7, spoon/spoon_8


#apple/apple_1, apple/apple_2, apple/apple_3, apple/apple_4, apple/apple_5, apple/apple_6, apple/apple_7, apple/apple_8, banana/banana_1, banana/banana_2, banana/banana_3, banana/banana_4, banana/banana_5, banana/banana_6, banana/banana_7, banana/banana_8, mug/mug_1, mug/mug_2, mug/mug_3, mug/mug_4, mug/mug_5, mug/mug_6, mug/mug_7, mug/mug_8, phone/phone_1, phone/phone_2, phone/phone_3, phone/phone_4, phone/phone_5, phone/phone_6, phone/phone_7, phone/phone_8


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


#knife/knife_1, knife/knife_2, knife/knife_3, knife/knife_4, knife/knife_5, knife/knife_6, key/key_1, key/key_2, key/key_3, key/key_4, key/key_5, key/key_6, pencil/pencil_1, pencil/pencil_2, pencil/pencil_3, pencil/pencil_4, pencil/pencil_5, pencil/pencil_6, spoon/spoon_1, spoon/spoon_2, spoon/spoon_3, spoon/spoon_4, spoon/spoon_5, spoon/spoon_6


#apple/apple_1, apple/apple_2, apple/apple_3, apple/apple_4, apple/apple_5, apple/apple_6, banana/banana_1, banana/banana_2, banana/banana_3, banana/banana_4, banana/banana_5, banana/banana_6, mug/mug_1, mug/mug_2, mug/mug_3, mug/mug_4, mug/mug_5, mug/mug_6, phone/phone_1, phone/phone_2, phone/phone_3, phone/phone_4, phone/phone_5, phone/phone_6
