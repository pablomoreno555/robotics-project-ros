from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Dense,
    Flatten,
    Conv2D,
    MaxPooling2D,
    AveragePooling2D,
    ReLU,
    Add,
    BatchNormalization,
    Input,
)
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from PIL import Image
import numpy as np

classes = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
]


def conv_block(X, blocknum, repetitions):
    num_filters = 64 * (2 ** blocknum)

    stride = 2
    for i in range(repetitions):
        # only the first block changes the size of X
        if i == 1:
            stride = 1

        X_residual = X

        X = Conv2D(filters=num_filters, kernel_size=1, strides=stride, padding="same")(
            X
        )
        X = BatchNormalization(axis=3)(X)
        X = ReLU()(X)

        X = Conv2D(filters=num_filters, kernel_size=3, strides=1, padding="same")(X)
        X = BatchNormalization(axis=3)(X)
        X = ReLU()(X)

        X = Conv2D(filters=num_filters * 4, kernel_size=1, strides=1, padding="same")(X)
        X = BatchNormalization(axis=3)(X)
        X = ReLU()(X)

        if stride > 1:
            X_residual = Conv2D(
                filters=num_filters * 4, kernel_size=1, strides=stride, padding="same"
            )(X_residual)
            X_residual = BatchNormalization(axis=3)(X_residual)

        X = Add()([X, X_residual])
        X = ReLU()(X)
    return X


def build_classifier(imsize, num_classes):
    sizeX, sizeY = imsize
    X_init = Input((sizeX, sizeY, 3))
    X = Conv2D(filters=64, strides=2, kernel_size=7, padding="same")(X_init)
    X = BatchNormalization(axis=3)(X)
    X = ReLU()(X)
    X = MaxPooling2D(pool_size=(3, 3), strides=2)(X)

    # block 1
    X = conv_block(X, blocknum=0, repetitions=3)

    # block 2
    X = conv_block(X, blocknum=1, repetitions=4)

    # block 3
    X = conv_block(X, blocknum=2, repetitions=6)

    # block 4
    X = conv_block(X, blocknum=3, repetitions=3)

    X = AveragePooling2D(pool_size=(2, 2), strides=1, padding="same")(X)
    X = Flatten()(X)

    X = Dense(units=num_classes, activation="softmax")(X)

    model = Model(inputs=X_init, outputs=X)
    return model


def load_ASL(dataDir, batch_size, imsize):
    data_generator = ImageDataGenerator()
    dataloader = data_generator.flow_from_directory(
        dataDir, target_size=imsize, batch_size=batch_size, shuffle=True
    )
    dataloader.horizontal_flip = True
    return dataloader


def train_model(num_epochs, dataDir, imsize):
    dataloader = load_ASL(dataDir, 16, imsize)

    model = build_classifier(imsize, num_classes=len(classes))

    optimizer = Adam(learning_rate=0.005, epsilon=0.1)
    loss = CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer, loss)

    # train on some fraction of the dataset each epoch
    steps = len(dataloader) // 10
    model.fit(dataloader, epochs=num_epochs, steps_per_epoch=steps)
    model.save("ASLModel.h5")
    return model


def idx_to_onehot(idx, num_classes):
    onehot = [0 for _ in range(num_classes)]
    onehot[idx] = 1
    return onehot


def center_crop(img, sizeX, sizeY):
    width, height = img.size
    left = (width - sizeX) / 2
    right = (width + sizeX) / 2
    top = (height - sizeY) / 2
    bottom = (height + sizeY) / 2
    return img.crop((left, top, right, bottom))


def test_model(imsize):
    sizeX, sizeY = imsize
    model = load_model("ASLModel.h5")
    dataDir = "./asl_alphabet_test/"

    test_x = []
    test_y = []

    for _, dirs, _ in os.walk(dataDir):
        for className in dirs:
            for _, _, imgNames in os.walk(dataDir + className):
                for imgName in imgNames:
                    img = Image.open(dataDir + className + "/" + imgName)

                    # w, h = img.size
                    # if w > sizeX or h > sizeY:
                    # img = center_crop(img, 500, 500)
                    img = img.resize(imsize)
                    img2 = img.transpose(Image.FLIP_LEFT_RIGHT)

                    test_x.append(np.array(img))
                    test_y.append(className)

                    test_x.append(np.array(img2))
                    test_y.append(className)

    # Reshaping the images before predicting on each one
    preds = [model.predict(np.reshape(img, (-1, sizeX, sizeY, 3))) for img in test_x]
    correct = 0
    for i in range(len(test_x)):
        pred = np.argmax(preds[i])
        if classes[pred] == test_y[i]:
            correct += 1

    print("Accuracy on test images:", correct / len(test_x))

    return (correct / len(test_x), model)


if __name__ == "__main__":
    train_model(num_epochs=20, dataDir="./asl_alphabet_train/", imsize=(64, 64))
    test_model(imsize=(64, 64))
