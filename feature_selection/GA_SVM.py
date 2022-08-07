import os
import random
import numpy as np
import pandas as pd
from data.spectf.read_data import *
from sklearn.svm import SVC
from genetic_selection import GeneticSelectionCV
from util.ga_svm_utils import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import collections
from util.util import  *


def ga_svm_fs(X, y, k, N=50, N_CROSS=2, GENERATIONS=5, MUTATION_RATE=0.2, N_MUTATION=4):
    """
    retrieve best feature selection using GA-SVM algorithm
    :param X:
    :param y:
    :param k:
    :param N:
    :param N_CROSS:
    :param GENERATIONS:
    :param MUTATION_RATE:
    :param N_MUTATION:
    :return:
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    # Scaling
    scaler = StandardScaler()
    scaler.fit(np.row_stack((X_train,X_test)))
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    features = np.random.randint(2, size=(N,X_train.shape[1]))

    gen_rank = list()
    for feature_selection in features:
        gen_rank.append((
            feature_selection,
            evaluate_feature_selection(X_train, y_train, X_test, y_test, feature_selection)
        ))

    # Take the best N/2 feature selection
    gen_bests = gen_rank

    for _ in range(GENERATIONS): # compute for number of generations
        gen_bests = sorted(gen_bests, key=lambda y: y[1], reverse=True)[:int(len(gen_bests)/2)]
        # Crossover
        couples = np.random.randint(len(gen_bests),size=(len(gen_bests),2))
        best_children = list()
        for couple in couples:
            a = gen_bests[couple[0]][0]
            b = gen_bests[couple[1]][0]
            children = crossover_couple(a,b,N_CROSS=N_CROSS)
            children_rank = list()
            for child in children:
                children_rank.append((
                    child,
                    evaluate_feature_selection(X_train, y_train, X_test, y_test, child)
                ))
            best_couple_children = sorted(children_rank, key=lambda y: y[1], reverse=True)[:2]
            best_children += best_couple_children

        # Mutation
        for _ in range(N_MUTATION):
            mutation_candidate_index = random.randint(0,len(gen_bests)-1)
            mutation_candidate = gen_bests[mutation_candidate_index]
            mutated_candidate = mutation_candidate[0]
            for i,v in enumerate(mutated_candidate):
                if np.random.random()<MUTATION_RATE:
                    mutated_candidate[i]=int(not v)
            acc = evaluate_feature_selection(X_train,y_train,X_test,y_test,mutated_candidate)
            gen_bests[mutation_candidate_index] = (mutated_candidate,acc)
        x=0
        gen_bests = gen_bests + best_children # The previous generation would be the input for the next one
        best_candidate = sorted(gen_bests, key=lambda y: y[1], reverse=True)[0]

    return best_candidate[0].tolist()

if __name__ == "__main__":

    X_train, y_train = read_train()
    X_test, y_test = read_test()
    X = np.vstack((X_train, X_test))
    y = np.hstack((y_train, y_test))
    print(ga_svm_fs(X,y,k))