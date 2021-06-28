from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import preprocess_input
import tensorflow as tf
import os
import pickle
from tensorflow.keras.callbacks import ModelCheckpoint
IMG_SIZE=250

core_idg = ImageDataGenerator(samplewise_center=False,
                              samplewise_std_normalization=False,
                              horizontal_flip = True,
                              vertical_flip = False,
                              height_shift_range = 0.15,
                              width_shift_range = 0.15,
                              rotation_range = 5,
                              shear_range = 0.01,
                              fill_mode = 'nearest',
                              zoom_range=0.25,
                             preprocessing_function = preprocess_input)

test_idg = ImageDataGenerator(preprocessing_function = preprocess_input)

train_generator = core_idg.flow_from_directory('Breeds', target_size=(IMG_SIZE,IMG_SIZE), batch_size=16, class_mode='categorical')
test_generator = test_idg.flow_from_directory('Breeds_Test', target_size=(IMG_SIZE,IMG_SIZE), batch_size=16, class_mode='categorical')

with open('class_dict.pkl', 'wb') as handle:
    pickle.dump(test_generator.class_indices, handle, protocol=pickle.HIGHEST_PROTOCOL)

base_model = tf.keras.applications.Xception(weights='imagenet',input_shape=(IMG_SIZE, IMG_SIZE, 3),include_top=False)
base_model.trainable = False
inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(120)(x)
model = tf.keras.Model(inputs, outputs)

if not os.path.isdir('checkpoints'):
    os.mkdir('checkpoints')
filepath=os.path.join('checkpoints',"model-{epoch:02d}-{categorical_accuracy:.2f}.hdf5")
checkpoint = ModelCheckpoint(filepath, monitor='categorical_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.compile(optimizer=tf.keras.optimizers.Adam(),loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),metrics=[tf.keras.metrics.CategoricalAccuracy()])
model.fit(train_generator,steps_per_epoch=400, epochs=10, validation_data=test_generator,validation_steps=80)

base_model.trainable = True
model.fit(train_generator,steps_per_epoch=400, epochs=8, validation_data=test_generator,validation_steps=80, callbacks=callbacks_list)




