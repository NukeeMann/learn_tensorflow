import tensorflow as tf

mnist = tf.keras.datasets.mnist # Import bazy danych liczb MNIST

(x_train, y_train), (x_test, y_test) = mnist.load_data()    # Ładowanie danych
x_train, x_test = x_train / 255.0, x_test / 255.0       # Konwertowanie z integera na floata

# Definicja całego modelu - SEQUENTIAL
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # Warstwa wejsciowa
    tf.keras.layers.Dense(128, activation='relu'),  # Warstwa pośrednia z metodą aktywacji wdłg funkcji relu
    tf.keras.layers.Dropout(0.2),                   # Współczynnik wygaszania węzłów
    tf.keras.layers.Dense(10)                       # Warstwa wynikowa
])

loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) # Zdefiniowanie funkcji straty do nauki

# Ustalenie optimizera, funkcji straty, ustalenie parametru pomiaru i liczby obiegow co ile ma by aktualizowane
model.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'], steps_per_execution=25)

# Trenuje siec na podanych danych przez podaną liczbe epok
model.fit(x_train, y_train, epochs=2, steps_per_epoch=60000)

# Zwraca wynik testu nie ucząc danych. VERBOSE odpowiada za sposób wyświetlania paska 0- brak wyniku, 1- z paskie, 2 - bez paska
model.evaluate(x_test, y_test, verbose=1)
