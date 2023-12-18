import torch
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog



def visualize_brackets(brackets):
    sns.set()
    fig, ax = plt.subplots()
    for i, bracket in enumerate(brackets):
        ax.scatter([i] * len(bracket), bracket, label=f'Bracket {i + 1}', alpha=0.7)
    ax.set_xlabel('Round')
    ax.set_ylabel('Entry')
    ax.legend()
    plt.show()

def create_neural_network():
   
        torch.nn.Linear(10, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 1)
    )
    return model

def train_neural_network(model, data, labels):
  
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(100):
        output = model(data)
        loss = criterion(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def create_keras_model():
   
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(1)
    ])
    return model

def train_keras_model(model, data, labels):
    
    model.compile(optimizer='adam', loss='mse')
    model.fit(data, labels, epochs=100, batch_size=32)

def generate_brackets_and_visualize(entries, rounds):
    brackets = bracket_formula(entries, rounds)
    visualize_brackets(brackets)

def main():
    
    entries_count = 16
    rounds_count = 4
    generate_brackets_and_visualize(entries_count, rounds_count)

   
    pytorch_model = create_neural_network()
    data = torch.randn((100, 10))
    labels = torch.randn((100, 1))
    train_neural_network(pytorch_model, data, labels)

    keras_model = create_keras_model()
    keras_data = tf.random.normal((100, 10))
    keras_labels = tf.random.normal((100, 1))
    train_keras_model(keras_model, keras_data, keras_labels)

    #
    root = tk.Tk()
    root.title("Bracket Generator GUI")

    def generate_and_visualize():
        entries = int(entries_entry.get())
        rounds = int(rounds_entry.get())
        generate_brackets_and_visualize(entries, rounds)

    entries_label = tk.Label(root, text="Entries:")
    entries_label.pack()
    entries_entry = tk.Entry(root)
    entries_entry.pack()

    rounds_label = tk.Label(root, text="Rounds:")
    rounds_label.pack()
    rounds_entry = tk.Entry(root)
    rounds_entry.pack()

    generate_button = tk.Button(root, text="Generate and Visualize", command=generate_and_visualize)
    generate_button.pack()

    root.mainloop()

if __name__ == "__main__":
    main()
