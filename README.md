# cifar-10_project
My first Convolutional Neural Network (CNN) project built using Python and TensorFlow/Keras to classify images from the CIFAR-10 dataset.

# CIFAR-10 Image Classification using CNN (v3)

## Project Overview
I built this project as a way to familiarize myself with Convolutional Neural Networks (CNNs) and the process of training and evaluating AI models. Throughout the project, I learned how different model components—such as optimization functions, data augmentation, and callbacks—affect performance, and how to interpret trends during training to achieve better accuracy. Overall, this was a valuable hands-on experience that helped me better understand practical deep learning techniques and the workflow of developing and improving AI models.

---

##  Model Information
- **Model Name:** `cifar10_cnn_model_v3.keras`
- **Framework:** TensorFlow / Keras
- **Architecture Type:** Custom Sequential CNN
- **Dataset:** CIFAR-10 (60,000 32×32 color images, 10 classes)

---

##  Training Configuration
| Setting | Value |
|---------|-------|
| Optimizer | Adam (learning rate = 0.001) |
| Loss Function | SparseCategoricalCrossentropy |
| Batch Size | 64 |
| Epochs | 20 (EarlyStopping used) |
| Data Augmentation | Yes (rotation, shifting, flipping, zoom) |
| Callbacks | EarlyStopping, ReduceLROnPlateau |

---

## Code Snippet (Key Training Block)

```python
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

optimizer = Adam(learning_rate=0.001)

model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)
datagen.fit(x_train)

history = model.fit(
    datagen.flow(x_train, y_train, batch_size=64),
    epochs=20,
    validation_data=(x_test, y_test),
    callbacks=[early_stopping, reduce_lr]
)

