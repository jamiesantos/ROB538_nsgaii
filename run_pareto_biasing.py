
import ast
import pandas as pd
import numpy as np
import torch
from HumanPreference import HumanSimulation, HumanPreference

filename = "data/A2_P4_B0p15_1/savedata.csv"

def load_population(filename, gen=None):
    """Load policies and objective fitnesses from savedata.csv."""

    df = pd.read_csv(filename)

    if gen is not None:
        df = df[df["gen"] == gen]

    # Convert string representations of lists into arrays.
    ids = df["id"].to_numpy()
    fitnesses = np.array([
        ast.literal_eval(f)
        for f in df["raw_fitness"]
    ], dtype=float)

    return ids, fitnesses



ids, fitnesses = load_population(filename, gen=999)

human = HumanSimulation(
    num_objs=2,
    preference_weights=[0.7, 0.3],
    noise=0.05,
    seed=42
)

comparisons = human.generate_comparisons(fitnesses, num_pairs=20)

model = HumanPreference(num_objs=2)
model.train(fitnesses, comparisons)

ranking = model.rank(fitnesses)

print("Policy IDs:", ids)
print("Preference ranking (indices):", ranking)
print("Preference ranking (individual IDs):", ids[ranking])
print("True human weights:", human.weights)
print("Learned preference weights:",
      torch.softmax(model.weights, dim=0).detach().numpy())