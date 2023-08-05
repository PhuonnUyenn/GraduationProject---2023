import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
# Khởi tạo các tham số cho mô hình
num_classes = 4
batch_size = 32
epochs = 300
input_shape = (240, 240, 3)

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)
# Tạo generator để đọc dữ liệu từ thư mục
train_generator = train_datagen.flow_from_directory(
    'Training',
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical',
    subset='training')
validation_generator = train_datagen.flow_from_directory(
    'Training',
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation')
# Tạo generator cho tập dữ liệu kiểm tra
test_generator = test_datagen.flow_from_directory(
    'Testing',
    target_size=(240, 240),
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)
# Định nghĩa callback ModelCheckpoint
checkpoint_path = '/content/drive/MyDrive/save/checkpoints/weights.{epoch:02d}-{val_loss:.2f}.h5'
checkpoint_callback = ModelCheckpoint(checkpoint_path,
                                      monitor='val_loss',
                                      verbose=1,
                                      save_best_only=True,
                                      save_weights_only=False,
                                      mode='min',
                                      save_freq='epoch')

# Xây dựng mô hình CNN
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])


# Tiếp tục huấn luyện mô hình


# Compile mô hình với hàm loss, optimizer và metrics
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=100,
    restore_best_weights=True)

# Tạo callback History để lưu giá trị loss và accuracy
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[early_stopping, checkpoint_callback]
)

# Lưu mô hình cuối cùng
model.save('/content/drive/MyDrive/save/Class4_model_t300.h5')