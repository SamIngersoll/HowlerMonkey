from container import Container
from datetime import datetime, timedelta
import time
import numpy as np
import scipy

# -GLOBAL VARS- #
# Natural limits
max_learn_rate = 1
max_layer1_size = 100
max_layer2_size = 100
max_batch_size = 100
max_lookback = 20


# List of n top performing individuals of all time
mostFit = []


# Creates a population of random individuals
#   Called once at the beginning of time, and whenever we add new random individuals
#   Params
#       num: Number of individuals to create
#   Returns
#       individuals: a list of new individuals of length num
def populate( num ):
    individuals = []
    for i in range( num ):
        finchrom = np.nonzero( scipy.sparse.rand( 45526, 5, density=.0001 ).toarray())
        hyperchrom = np.random.rand( 5 ) 
        individuals.append( Container( learning_rate=max_learn_rate*hyperchrom[0], max_steps=10, hidden1=int((max_layer1_size-1)*hyperchrom[1])+1, hidden2=int((max_layer2_size-1)*hyperchrom[2])+1, batch_size=int((max_batch_size-1)*hyperchrom[3])+1, stocks=finchrom[0].tolist(), fields=finchrom[1].tolist() ))
    return individuals

# Runs test to see how well individuals do
#   Called at the end of every generation
#   Params
#       individuals: list of individuals in world
#   Returns
#       fitness: list of fitness values for each individual, same size as input
def evaluate( individuals ):
    return fitness

# Selects best individuals to reproduce
#   Called at the end of every generation
#   Params
#       individuals: list of individuals in world
#       fitness: list of fitness values for each individual in world
#       cull_ammount: how many individuals should remain at the end of every generation
#   Returns
#       remaining_individuals: culled list of most fit individuals
def select( individuals, fitness, cull_ammount ):
    return remainingIndividuals

# Creates new individuals based on fittest in previous gen
#   Called at the end of every generation
#   Params
#       remaining_individuals: list of most fit individuals
#   Returns
#       children: List of new individuals for next generation
def reproduce( remaining_individuals ):
    return children
    
# Introduces diversity into population, helps explore parameter space
#   Called at the end of every generation
#   Params
#       children
#       mutation_constant
#   Returns
#       mutated_children
def mutuate( children, mutation_constant ):
    return mutated_children


if __name__ == '__main__':
    # container = Container(learning_rate = 0.2, max_steps = 10, hidden1 = 10, hidden2 = 20, batch_size = 100, lookback=4, generation_number=0)
  #  for i in range( 45526 ):
        #print(  sid(i) )
    print( populate(1))
    #container = Container(learning_rate = 0.2, max_steps = 10, hidden1 = 10, hidden2 = 20, batch_size = 100, lookback=4, generation_number=0)

