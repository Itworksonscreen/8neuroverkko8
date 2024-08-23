import numpy as np
import itertools
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Luo kaikki mahdolliset 8-bitin yhdistelmät
X = np.array(list(itertools.product([0, 1], repeat=8)))

# Luo vastakkaiset ulostulot
y = 1 - X  # Vastakkainen tila

# Luo hermoverkkomalli
model = Sequential([
    Dense(16, input_dim=8, activation='relu'),
    Dense(8, activation='sigmoid')
])

# Käännä malli
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Kouluta malli
model.fit(X, y, epochs=50, batch_size=8, verbose=1)

# Tallenna malli
model.save('neural_network_model.h5')
print('Model saved as neural_network_model.h5')
