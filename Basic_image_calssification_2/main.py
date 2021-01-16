import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Funkcja do wypisania wartości
    def plot_image(i, predictions_array, true_label, img):
        true_label, img = true_label[i], img[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])

        plt.imshow(img, cmap=plt.cm.binary)

        predicted_label = np.argmax(predictions_array)
        if predicted_label == true_label:
            color = 'blue'
        else:
            color = 'red'

        plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                             100 * np.max(predictions_array),
                                             class_names[true_label]),
                   color=color)


    # Funkcja do rysowania wykresów wartości plota
    def plot_value_array(i, predictions_array, true_label):
        true_label = true_label[i]
        plt.grid(False)
        plt.xticks(range(10))
        plt.yticks([])
        thisplot = plt.bar(range(10), predictions_array, color="#777777")
        plt.ylim([0, 1])
        predicted_label = np.argmax(predictions_array)

        thisplot[predicted_label].set_color('red')
        thisplot[true_label].set_color('blue')


    # Zdefiniowanie bazy danych ciuchow mnist
    fashion_mnist = tf.keras.datasets.fashion_mnist

    # Załadowanie zmiennych
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # Podpis odpowiadający numerom w labels
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # Demonstracja rysunku
    """
    plt.figure()
    plt.imshow(train_images[1])
    plt.colorbar()
    plt.grid(False)
    plt.show()
    """

    # Skalowanie wartości 0-255 do 0-1
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Demonstracja pierwszych 25 obrazow wraz z ich podpisami
    """
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])
    plt.show()
    """

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    # Definiowanie optymalizera typu ADAM uczącej sieć
    optimizer = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

    # Definiowanie funkcji straty wyliczającej błąd
    loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Zdefiniowanie pełnego modelu z funkcjami
    model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])

    # Uczenie sieci
    model.fit(train_images, train_labels, epochs=8)

    # Sprawdzenie dokladnosci na danych testowych
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

    print('\nTest accuracy:', test_acc)

    # print(model.predict(test_images))

    # Zdefiniowanie jednowarstwoej sieci która będzie konwertowała wyniki z modelu za pomocą funkcji SOFTMAX do przedziału (0, 1)
    propability_fn = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

    # Predykcja obrazów
    predictions = propability_fn.predict(test_images)

    # Wyświetlenie indeksu nawiekszego wyniku
    # np.argmax(predictions[0])

    num_rows = 10
    num_cols = 3
    num_img = num_rows * num_cols
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    for i in range(num_img):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(i, predictions[i], test_labels, test_images)
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(i, predictions[i], test_labels)
    plt.tight_layout()
    plt.show()