from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, GlobalAveragePooling2D, Dense, Reshape, Activation, Dropout

def get_model2():
    model = Sequential([
        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(9, 9, 1)),
        BatchNormalization(),

        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),

        Conv2D(128, kernel_size=(1, 1), activation='relu', padding='same'),
        BatchNormalization(),

        GlobalAveragePooling2D(),  # Remplace Flatten() pour réduire les paramètres inutiles
        Dense(81 * 9, activation='relu'),
        Dropout(0.3),  # Évite le surapprentissage
        Reshape((-1, 9)),
        Activation('softmax')
    ])

    return model  # Correction de l'indentation pour renvoyer le modèle
