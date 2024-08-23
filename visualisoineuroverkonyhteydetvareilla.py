import tensorflow.keras as keras
from tensorflow.keras.utils import plot_model
from graphviz import Digraph

# Lataa malli
model = keras.models.load_model('neural_network_model.h5')

# Tallenna mallin arkkitehtuuri kuva-tiedostoon
plot_model(model, to_file='model.png', show_shapes=True)

# Luo uusi Graphviz-digraafi
dot = Digraph()

# Lisää kaikki kerrokset digraafiin
for i, layer in enumerate(model.layers):
    # Lisää kerroksen nimi digraafiin
    dot.node(str(i), str(layer.name))

    # Lisää jokaisen kerroksen neuronit digraafiin, jos kerroksessa on painoja
    if len(layer.get_weights()) > 0:
        weights = layer.get_weights()[0]
        for j in range(weights.shape[1]):
            dot.node(str(i) + '_' + str(j), label='', shape='circle')

            # Luo yhteys kerroksen neuronin ja edellisen kerroksen neuronin välillä
            if i > 0:
                for k in range(weights.shape[0]):
                    weight = weights[k, j]
                    if weight > 0:
                        color = '#%02x%02x%02x' % (255, int(255*(1-weight)), int(255*(1-weight)))
                    else:
                        color = '#%02x%02x%02x' % (int(255*(1+weight)), int(255*(1+weight)), 255)
                    dot.edge(str(i-1) + '_' + str(k), str(i) + '_' + str(j), color=color)

# Tallenna kaavio pdf-tiedostoon
dot.render('model.gv', view=True, format='pdf')
