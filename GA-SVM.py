import os
import random
import numpy as np
import pandas as pd
from data.spectf.read_data import *
from sklearn.svm import SVC
from genetic_selection import GeneticSelectionCV
from util.ga_svm_utils import *
import collections
def main():
    N = 100
    N_CROSS =4
    GENERATIONS = 20
    MUTATION_RATE = 0.4

    X_train, y_train = read_train()
    X_test, y_test = read_test()
    features = np.random.randint(2, size=(N,X_train.shape[1]))

    gen_rank = list()
    for feature_selection in features:
        gen_rank.append((
            feature_selection,
            evaluate_feature_selection(X_train, y_train, X_test, y_test, feature_selection)
        ))

    # Take the best N/2 feature selection
    gen_bests = gen_rank
    # gen_bests = list(map(lambda x:x[0],gen_rank))

    for _ in range(GENERATIONS):
        gen_bests = sorted(gen_bests, key=lambda y: y[1], reverse=True)[:int(len(gen_bests)/2)]
        # Crossover
        couples = np.random.randint(len(gen_bests),size=(len(gen_bests),2))
        best_children = list()
        for couple in couples:
            a = gen_bests[couple[0]][0]
            b = gen_bests[couple[1]][0]
            children = crossover_couple(a,b)
            children_rank = list()
            for child in children:
                children_rank.append((
                    child,
                    evaluate_feature_selection(X_train, y_train, X_test, y_test, child)
                ))
            best_couple_children = sorted(children_rank, key=lambda y: y[1], reverse=True)[:2]
            best_children += best_couple_children

        # Mutation
        mutation_candidate_index = random.randint(0,len(gen_bests)-1)
        mutation_candidate = gen_bests[mutation_candidate_index]
        mutated_candidate = mutation_candidate[0]
        for i,v in enumerate(mutated_candidate):
            if np.random.random()<MUTATION_RATE:
                mutated_candidate[i]=int(not v)
        acc = evaluate_feature_selection(X_train,y_train,X_test,y_test,mutated_candidate)
        gen_bests[mutation_candidate_index] = (mutated_candidate,acc)
        x=0
        gen_bests = gen_bests + best_children
        best_candidate = sorted(gen_bests, key=lambda y: y[1], reverse=True)[0]
        gen_acc = best_candidate[1]
        feature_percentage = np.sum(best_candidate[0])/len(best_candidate[0])
        print(f"accuracy - {gen_acc} , feature_percentage - {feature_percentage}")
        print(np.where(best_candidate[0]==1))

if __name__ == "__main__":
    main()