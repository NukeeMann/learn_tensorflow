import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


# Ręczne usuwanie znaczników HTML-owskich z danych.
def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)  # Zmiana na małe znaki
  stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')    # Podmiana znaczniku <br /> na pusty znak
  return tf.strings.regex_replace(stripped_html, '[%s]' % re.escape(string.punctuation), '')


# Funkcja do wyświetlania wyników preprocess danych
def vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return vectorize_layer(text), label


if __name__ == '__main__':
    print(tf.__version__)
    #tf.compat.v1.disable_eager_execution()

    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

    # get_file pobiera pliki do cache, jezeli tych plikow tam juz nie ma.
    # NAZWA, LINK, UNTAR(rozpakowywanie zipow), FOLDER DO PRZECHOWYWANIE, COS DRUGIEGO
    # Zwraca ścieżkę do pobranych plików
    #data_set = tf.keras.utils.get_file("aclImdb_v1.tar.gz", url, untar=True , cache_dir='.', cache_subdir='')
    cosiek = '.\\aclImdb_v1.tar.gz'

    # Łączenie zasobów z linkiem w którym
    dataset_dir = os.path.join(os.path.dirname(cosiek), 'aclImdb')

    # Wypisz zawartość folderu
    #print(os.listdir(dataset_dir))

    # Zestaw do trenowania
    train_dir = os.path.join(dataset_dir, 'train')
    #print(os.listdir(train_dir))

    # Wypisywanie przykładowego pliku
    sample_file = os.path.join(train_dir, 'pos/1181_9.txt')
    with open(sample_file) as f:
      print(f.read())

    # Usuwanie zbednych folderow
    #remove_dir = os.path.join(train_dir, 'unsup')
    #shutil.rmtree(remove_dir)


    batch_size = 32 # Liczba danych na operacje
    seed = 42 # Opcjonalny seed

    # Wydzielenie danych do trenowania od danych do validacji
    # Nazwa folderu z danymi, liczba danych na operacje, % przeznaczony na validacje,
    # Ustawiony validation_split tworzy dwa subsety, training i validation
    # Tworzy automatycznie dwie klasy odpowiadajace folderom -> (plik
    raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
        'aclImdb/train',
        batch_size=batch_size,
        validation_split=0.2,
        subset='training',
        seed=seed)

    # Wypisanie danych.
    #for text_batch, label_batch in raw_train_ds.take(1):
      #for i in range(3):
        #print("Review", text_batch.numpy()[i])
        #print("Label", label_batch.numpy()[i])


    # print("Label 0 corresponds to", raw_train_ds.class_names[0])
    # print("Label 1 corresponds to", raw_train_ds.class_names[1])

    # Przypisanie do zmiennej zasoby validacyjnych
    raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(
        'aclImdb/train',
        batch_size=batch_size,
        validation_split=0.2,
        subset='validation',
        seed=seed)

    # Przypisanie do zmiennej zasobow testowych INNY FOLDER
    raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(
        'aclImdb/test',
        batch_size=batch_size)

    # Przygotowanie danych do uczenia. Usunięcie zbędnych znaków i znaczników.
    max_features = 10000
    sequence_length = 250

    # Stworzenie wektora danych uczących poprzez nadanie każdej danej unikalnego indeksu,
    # pozbycie się zbędnych znaków i znaczników, Maksymalnej wielkości tworzonej warstwy,
    # Ustalenie stałej długości każdej danej
    vectorize_layer = TextVectorization(
        standardize=custom_standardization,
        max_tokens=max_features,
        output_mode='int',
        output_sequence_length=sequence_length)

    # Przetwarzanie na danych na zwykły tekst zindeksowany i adaptacja go do zdefiniowanego
    # powyżej modelu
    train_text = raw_train_ds.map(lambda x, y: x)
    vectorize_layer.adapt(train_text)   # WYMAGANE użycie experymentalnej wersji TextVectorization funkcji

    # Wyświetlenie wyniku. Każdemu słowu przypisana jest unikalna liczba.
    # Np. 1287 -> silent, 313 -> night, 0 -> ' '
    text_batch, label_batch = next(iter(raw_train_ds))
    first_review, first_label = text_batch[0], label_batch[0]
    print("Review", first_review)
    print("Label", raw_train_ds.class_names[first_label])
    print("Vectorized review", vectorize_text(first_review, first_label))
    print("1287 ---> ", vectorize_layer.get_vocabulary()[1287])
    print(" 313 ---> ", vectorize_layer.get_vocabulary()[313])
    print('Vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))

    # OSTATNI KROK PREPROCESINGU
    # TRANSFORMACJA WSZYSTKICH ZESTAWÓW DANYCH DO POWYŻSZEGO FORMATU
    train_ds = raw_train_ds.map(vectorize_text)
    val_ds = raw_val_ds.map(vectorize_text)
    test_ds = raw_test_ds.map(vectorize_text)

    # Dynamicznie podczas wykonywania dostosowujemy wielkość możliwego cachu
    AUTOTUNE = tf.data.AUTOTUNE

    # Dzięki funkcji .cache() dane zostają w pamięci po załadowaniu z dysku. Zapewnia to brak możliwości wystąpienia
    # blokowania ze strony odczytu danych (I/O)
    # .prefetch() sprawia, że gdy przeprowadzamy operacje na danej N, pobierana jest już dana N+1 by zautomatyzować
    # i nie blokować działania programu
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    ##################################################################################################################
    ##################################################################################################################
    ##################################################################################################################
    ##################################################################################################################
    ##################################################################################################################
    ##################################################################################################################
    ##################################################################################################################
    ##################################################################################################################
    ##################################################################################################################

    # ZAPOBIEGANIE OVERFITINGU
    # Według jakiej zmiennej, ile epok musi sie powtorzyc ponizej limitu progresu, jaki minimalny progres
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_binary_accuracy', patience=2, min_delta=0.01)
    # Tworzenie modelu sieci neuronowej
    embedding_dim = 16

    # Model: 10000(80% wartości) -> 16(80% wartości) -> 1(pos/neg)
    model = tf.keras.Sequential([
        layers.Embedding(max_features + 1, embedding_dim),  # Osadza słowa np. cat -> 0001, the -> 0010
        layers.Dropout(0.2),
        layers.GlobalAveragePooling1D(),    # Zwraca 1 wymiarowy równej długości wektor dla każdego przykładu poprzez
        # uśrednianie sekwencji w wymiarze. Wynikym wymiarem jest (batch, sequence, embedding)
        layers.Dropout(0.2),
        layers.Dense(1)])

    model.summary()

    # Zdefiniowanie funkcji wyliczania straty. from_logits = True oznacza, że znamy i podamy poprawną odpowiedź
    loss_fn = losses.BinaryCrossentropy(from_logits=True)

    # Zdefiniowanie funkcji optymalizacyjnej
    optimization_fn = tf.keras.optimizers.Adam()

    # Zdefiniowanie metryki THRESHOLD czy predykcja jest 1 czy 0.
    metric_opt = tf.metrics.BinaryAccuracy(threshold=0.0)

    # Kompilacja modelu
    model.compile(loss=loss_fn, optimizer=optimization_fn, metrics=metric_opt)

    # Trenowanie modelu
    epochs = 10
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[callback])

    # Testowanie modelu na danych testowych
    loss, accuracy = model.evaluate(test_ds)

    print("Loss: ", loss)
    print("Accuracy: ", accuracy)

    ##################################################################################################################
    ##################################################################################################################
    ##################################################################################################################
    ##################################################################################################################
    ##################################################################################################################
    ##################################################################################################################
    ##################################################################################################################
    ##################################################################################################################
    ##################################################################################################################

    # Tworzenie wykresu dokładności
    history_dict = history.history
    print(history_dict.keys())

    acc = history_dict['binary_accuracy']
    val_acc = history_dict['val_binary_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)

    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    plt.show()

    ##################################################################################################################
    ##################################################################################################################
    ##################################################################################################################
    ##################################################################################################################
    ##################################################################################################################
    ##################################################################################################################
    ##################################################################################################################
    ##################################################################################################################
    ##################################################################################################################

    # EKSPORT MODELU
    # Eskport modelu z wyuczonymi wagami węzłów. I możliwością wstawienia SUROWYCH DANYCH
    export_model = tf.keras.Sequential([
        vectorize_layer,
        model,
        layers.Activation('sigmoid')
    ])
    export_model_2 = tf.keras.Sequential([
        vectorize_layer,
        model,
        layers.Activation('sigmoid')
    ])

    export_model.compile(
        loss=losses.BinaryCrossentropy(), optimizer="adam", metrics=['accuracy']
    )
    loss, accuracy = export_model.evaluate(raw_test_ds)
    print(accuracy)

    # TESTOWANIE NA WŁASNYCH DANYCH
    examples = [
        "The movie was great!",
        "The movie was okay.",
        "The movie was terrible..."
    ]

    print(export_model.predict(examples))
