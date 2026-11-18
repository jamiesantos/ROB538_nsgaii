import random
import torch
import numpy as np
import math
import pygmo as pg
import os

import Algorithm

class NSGAII(Algorithm.CentralisedAlgorithm):
    def __init__(self, alg_config_filename, domain_name, rover_config_filename, data_filename, beta):
        super().__init__(alg_config_filename, domain_name, rover_config_filename, data_filename)
        self.beta = beta

    def evolve(self, gen=0, traj_write_freq=100):
        """Evolve the population using NSGA-II."""
        # Perform rollout and assign fitness to each individual
        for ind in self.pop:
            # Reset the fitness
            ind.reset_fitness()
            # Conduct rollout
            trajectory, fitness_dict = self.interface.rollout(ind.joint_policy)

            if len(fitness_dict) != self.num_objs:
                raise ValueError(f"[NSGA-II] Expected {self.num_objs} objectives, but got {len(fitness_dict)}.")
            # Store the rollout trajectory
            ind.trajectory = trajectory
            # Store fitness
            raw_fitness_dict = fitness_dict.copy()  # Store the unshaped fitness for evaluation
            for f in fitness_dict:
                ind.fitness[f] = -fitness_dict[f] # NOTE: The fitness sign is flipped to match Pygmo convention
                ind.raw_fitness[f] = -raw_fitness_dict[f]  # unshaped, for evaluation

            # Add this individual's data to the logger
            self.data_logger.add_data(key='gen', value=gen)
            self.data_logger.add_data(key='id', value=ind.id)
            self.data_logger.add_data(key='fitness', value=ind.fitness)
            self.data_logger.add_data(key='raw_fitness', value=ind.raw_fitness)
            if gen == self.num_gens - 1 or gen % traj_write_freq == 0:
                self.data_logger.add_data(key='trajectory', value=ind.trajectory)
            else:
                self.data_logger.add_data(key='trajectory', value=None)
            self.data_logger.write_data()
        
        # Sort the population according to fitness
        sorted_indices = pg.sort_population_mo(points=[ind.fitness for ind in self.pop])
        fitness_tuples = [tuple(ind.fitness) for ind in self.pop]

        # Keep the top half
        sorted_indices = sorted_indices[:len(sorted_indices)//2]
        
        parent_set = [self.pop[i] for i in sorted_indices]
        # Create empty offpring set
        offspring_set = []

        # Fill up the offspring set to the pop_size via offspring-creation
        while len(parent_set) + len(offspring_set) < self.pop_size:
            # Select 2 parents via binary tournament
            idx1, idx2 = random.sample(range(len(sorted_indices)), 2) # Sample two indices from the list
            parent1 = parent_set[min(idx1, idx2)] # choose the lower (more fit) option
            idx1, idx2 = random.sample(range(len(sorted_indices)), 2) # Sample two indices from the list
            parent2 = parent_set[min(idx1, idx2)] # choose the lower (more fit) option
            # Get the offsprings by crossing over these Individuals
            offspring1, offspring2 = self.utils.crossover(parent1, parent2, self.glob_ind_counter)
            # Mutate the offsprings by adding noise
            offspring1.mutate()
            offspring2.mutate()
            # Add to the offspring set
            offspring_set.extend([offspring1, offspring2])
            # Update the global id counter
            self.glob_ind_counter += 2
        
        # Set the population to the parent + offspring set
        self.pop = parent_set
        self.pop.extend(offspring_set)

        random.shuffle(self.pop) # NOTE: This is so that equally dominnat offpsrings in later indices don't just get thrown out

        # Minimal checkpoint: save latest population state so experiments can be resumed.
        try:
            run_dir = self.interface.rover_env.data_dir
            if run_dir is None and getattr(self, 'data_logger', None) and getattr(self.data_logger, 'target_filename', None):
                run_dir = os.path.dirname(self.data_logger.target_filename)
            if run_dir is not None:
                ckpt = {
                    'gen': gen,
                    'glob_ind_counter': self.glob_ind_counter,
                    'num_gens': self.num_gens,
                    'pop_size': self.pop_size,
                    'population': [],
                    'traj_write_freq': traj_write_freq,
                }
                for ind in self.pop:
                    ind_entry = {
                        'id': ind.id,
                        'fitness': ind.fitness,
                        'raw_fitness': ind.raw_fitness,
                        'policies': [p.state_dict() for p in ind.joint_policy]
                    }
                    ckpt['population'].append(ind_entry)

                latest_path = os.path.join(run_dir, 'latest_checkpoint.pth')
                tmp_path = latest_path + '.tmp'
                torch.save(ckpt, tmp_path)
                try:
                    os.replace(tmp_path, latest_path)
                except Exception:
                    # best-effort move
                    os.rename(tmp_path, latest_path)
        except Exception:
            # Don't interrupt evolution if checkpointing fails
            pass