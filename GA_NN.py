#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import numpy as np
from random import random, randint
import csv

# Define the parameters
input_shape = (1, 32, 32, 1)
hidden_layer_1_nodes = 12
output_nodes = 5
num_sets = 5

# Global variables
input_image = None
flattened_input = None
target_vector = None
class_inputs = {}
background_image = None

# Generate 5 sets of weights for hidden layer 1
hidden_layer1_weight_sets = []
for _ in range(num_sets):
    # Random weights for the input layer to hidden layer
    weights_hidden_layer1 = np.random.randn(np.prod(input_shape[1:]), hidden_layer_1_nodes)
    hidden_layer1_weight_sets.append(weights_hidden_layer1)

# Generate 5 sets of weights for hidden layer to output layer
hidden_output_weight_sets = []
for _ in range(num_sets):
    # Random weights for the hidden layer to output layer
    weights_hidden_output = np.random.randn(hidden_layer_1_nodes, output_nodes)
    hidden_output_weight_sets.append(weights_hidden_output)

class Chromosome:
    def __init__(self, genes, fitness=0):
        self.genes = genes
        self.fitness = fitness

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def open_image():
    global input_image, flattened_input
    file_path = filedialog.askopenfilename()
    if file_path:
        input_image = Image.open(file_path)
        input_image = input_image.resize((200, 200), Image.ANTIALIAS if hasattr(Image, "ANTIALIAS") else Image.BICUBIC)
        input_image = ImageTk.PhotoImage(input_image)
        canvas.create_image(0, 0, anchor=tk.NW, image=input_image)
        print("Image loaded successfully.")
        flattened_input = np.random.rand(*input_shape).reshape(1, -1)

def forward_pass(inputs, weights_hidden_layer1, weights_hidden_output):
    weighted_sum1 = np.dot(inputs, weights_hidden_layer1)
    hidden_layer_output1 = sigmoid(weighted_sum1)
    output_layer_input = np.dot(hidden_layer_output1, weights_hidden_output)
    output = softmax(output_layer_input)
    return output

def run_genetic_algorithm():
    global target_vector
    if input_image is None:
        messagebox.showwarning("Warning", "Please load an image first.")
    else:
        class_labels = ['built_up_land', 'agriculture', 'forest', 'waste_land', 'water_bodies']
        selected_values = [float(class_inputs[label].get()) for label in class_labels]
        target_vector = selected_values
        update_target_block(selected_values)
        save_path = filedialog.asksaveasfilename(defaultextension=".csv")
        if save_path:
            chromosomes, weights = genetic_algorithm(target_vector, 100, 10, 100, 0.1, 0.8, flattened_input, hidden_layer1_weight_sets, hidden_output_weight_sets, save_path)
            output_text.delete(1.0, tk.END)
            for i, chromosome in enumerate(chromosomes, start=1):
                output_text.insert(tk.END, f"Iteration {i} - Genes_weights: {chromosome.genes}\n")
                output_output.insert(tk.END, f"Iteration {i} - Output with GA: {weights[i-1]}\n")
                save_to_csv(chromosome.genes, save_path)
            messagebox.showinfo("Info", "Genetic Algorithm Finished")

def update_target_block(selected_values):
    target_entry.delete(0, tk.END)
    target_entry.insert(0, ', '.join(map(str, selected_values)))

def save_to_csv(genes, filename):
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Gene', 'Value'])
        for index, gene_value in enumerate(genes):
            csvwriter.writerow([f'Gene_{index+1}', gene_value])

def generate_population(hidden_layer1_weight_sets, hidden_out_weight_sets):
    population = []
    for i in range(len(hidden_layer1_weight_sets)):
        initial_weights = np.concatenate(
            (hidden_layer1_weight_sets[i].flatten(), hidden_out_weight_sets[i].flatten()))
        genes = initial_weights.copy()
        population.append(Chromosome(genes))
    return population

def crossover(parent1, parent2, crossover_probability):
    if random() < crossover_probability:
        crossover_point = randint(1, len(parent1.genes) - 1)
        child1_genes = np.concatenate(
            (parent1.genes[:crossover_point], parent2.genes[crossover_point:]))
        child2_genes = np.concatenate(
            (parent2.genes[:crossover_point], parent1.genes[crossover_point:]))
        return Chromosome(child1_genes), Chromosome(child2_genes)
    else:
        return parent1, parent2

def mutate(chromosome, mutation_rate):
    for i in range(len(chromosome.genes)):
        if random() < mutation_rate:
            chromosome.genes[i] = random()
    return chromosome

def fitness_function(chromosome, target, flattened_input):
    H1 = chromosome[0:12288].reshape(1024, 12)
    HO = chromosome[12288:].reshape(12, 5)
    predictions = forward_pass(flattened_input, H1, HO)
    errors = predictions - target
    return (-1* np.mean(errors**2))

def selection(population, selection_size):
    population.sort(key=lambda x: x.fitness, reverse=True)
    return population[:selection_size]

def genetic_algorithm(target, population_size, num_genes, generations, mutation_rate, crossover_probability, flattened_input, hidden_layer1_weight_sets, hidden_out_weight_sets, save_path):
    outputs = []
    chromosomes = []
    for iteration in range(10):
        population = generate_population(hidden_layer1_weight_sets, hidden_out_weight_sets)
        for _ in range(generations):
            for chromosome in population:
                chromosome.fitness = fitness_function(chromosome.genes, target, flattened_input)
            selected_parents = selection(population, int(population_size / 2))
            new_population = []
            for i in range(0, len(selected_parents), 2):
                if i + 1 < len(selected_parents):
                    child1, child2 = crossover(selected_parents[i], selected_parents[i+1], crossover_probability)
                    new_population.append(child1)
                    new_population.append(child2)
            for chromosome in new_population:
                mutate(chromosome, mutation_rate)
            population = new_population + selected_parents[:int(population_size * 0.1)]
        best_chromosome = max(population, key=lambda x: x.fitness)
        chromosomes.append(best_chromosome)
        hidden_layer1_weight_sets.append(best_chromosome.genes[:12288].reshape(1024, 12))
        hidden_out_weight_sets.append(best_chromosome.genes[12288:].reshape(12, 5))
        H1 = hidden_layer1_weight_sets[-1]
        HO = hidden_out_weight_sets[-1]
        output = forward_pass(flattened_input, H1, HO)
        outputs.append(output)
    return chromosomes, outputs

root = tk.Tk()
root.title("Using Genetic Algorithm for weight initialization GUI")

try:
    background_image = Image.open("D:/STUDY/GNR_COURSES/GNR602/project/earth.jpg")
    background_image = background_image.resize((1200, 600), Image.ANTIALIAS if hasattr(Image, "ANTIALIAS") else Image.BICUBIC)  # Resize image for display
    background_image = ImageTk.PhotoImage(background_image)
except Exception as e:
    print("Error loading the background image:", e)
else:
    background_label = tk.Label(root, image=background_image)
    background_label.place(x=0, y=0, relwidth=1, relheight=1)
    background_label.image = background_image

input_frame = ttk.Frame(root, padding=(20, 10))
input_frame.pack()

# Add labels and input fields
# ttk.Label(input_frame, text="Population Size:").grid(row=0, column=0)
population_size_entry = ttk.Entry(input_frame)
# population_size_entry.grid(row=0, column=1)
population_size_entry.insert(0, "100")

# ttk.Label(input_frame, text="Num Genes:").grid(row=1, column=0)
num_genes_entry = ttk.Entry(input_frame)
# num_genes_entry.grid(row=1, column=1)
num_genes_entry.insert(0, "10")

ttk.Label(input_frame, text="Generations:").grid(row=2, column=0)
generations_entry = ttk.Entry(input_frame)
generations_entry.grid(row=2, column=1)
generations_entry.insert(0, "100")

ttk.Label(input_frame, text="Mutation Rate:").grid(row=3, column=0)
mutation_rate_entry = ttk.Entry(input_frame)
mutation_rate_entry.grid(row=3, column=1)
mutation_rate_entry.insert(0, "0.1")

ttk.Label(input_frame, text="Crossover Probability:").grid(row=4, column=0)
crossover_prob_entry = ttk.Entry(input_frame)
crossover_prob_entry.grid(row=4, column=1)
crossover_prob_entry.insert(0, "0.8")

ttk.Label(input_frame, text="enter values between 0-1").grid(row=5, column=0)
#crossover_prob_entry = ttk.Entry(input_frame)
#crossover_prob_entry.grid(row=5, column=1)

class_labels = ['built_up_land', 'agriculture', 'forest', 'waste_land', 'water_bodies']
for i, label in enumerate(class_labels):
    ttk.Label(input_frame, text=label.capitalize() + ":").grid(row=6 + i, column=0)
    class_inputs[label] = ttk.Entry(input_frame)
    class_inputs[label].grid(row=6 + i, column=1)

ttk.Label(input_frame, text="Target Vector (built_up_land, agriculture, forest, waste_land, water_bodies):").grid(row=12, column=0)
target_entry = ttk.Entry(input_frame)
target_entry.grid(row=12, column=1)

open_button = tk.Button(root, text="Open Image", command=open_image)
open_button.pack()

start_button = ttk.Button(root, text="Start Genetic Algorithm", command=run_genetic_algorithm)
start_button.pack()

canvas = tk.Canvas(root, width=200, height=200)
canvas.pack()

output_frame = ttk.Frame(root, padding=(20, 10))
output_frame.pack()

output_label = ttk.Label(output_frame, text="Output gene weights:")
output_label.pack()

output_text = tk.Text(output_frame, width=150, height=5)
output_text.pack()

output_output_label = ttk.Label(output_frame, text="class prediction Output during GA:(built_up_land, agriculture, forest, waste_land, water_bodies)")
output_output_label.pack()

output_output = tk.Text(output_frame, width=150, height=5)
output_output.pack()

root.mainloop()


# In[ ]:




