import tkinter as tk
import numpy as np
from tensorflow.keras.models import load_model

class IOApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Input/Output Simulator")
        
        # Lataa malli tiedostosta
        model_path = 'neural_network_model.h5'
        self.model = load_model(model_path)
        
        self.input_buttons = []
        self.output_labels = []
        
        # Luo sisääntulopainikkeet
        for i in range(8):
            btn = tk.Button(root, text="0", width=10, height=2, bg="red", command=lambda i=i: self.toggle_input(i))
            btn.grid(row=i, column=0, padx=5, pady=5)
            self.input_buttons.append(btn)
        
        # Luo ulostulolabelit
        for i in range(8):
            lbl = tk.Label(root, text="1", width=10, height=2, bg="green")
            lbl.grid(row=i, column=1, padx=5, pady=5)
            self.output_labels.append(lbl)
        
        # Päivitä ulostulo vastaamaan sisääntuloja aluksi
        self.update_outputs()

    def toggle_input(self, index):
        btn = self.input_buttons[index]
        if btn["text"] == "0":
            btn.config(text="1", bg="green")
        else:
            btn.config(text="0", bg="red")
        self.update_outputs()

    def update_outputs(self):
        # Lue sisääntulon tilat
        input_states = np.array([[int(btn["text"]) for btn in self.input_buttons]])
        
        # Ennusta ulostulon tilat hermoverkon avulla
        output_states = self.model.predict(input_states)[0]
        output_states = (output_states > 0.5).astype(int)  # Muuta binääriseksi
        
        # Päivitä ulostulolabelit ennustusten perusteella
        for i in range(8):
            state = str(output_states[i])
            self.output_labels[i].config(text=state, bg="green" if state == "1" else "red")

if __name__ == "__main__":
    root = tk.Tk()
    app = IOApp(root)
    root.mainloop()
