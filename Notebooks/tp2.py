import numpy as np

def create_y_lang(instances):
    categories = []

    mapping = {'fra': 0,
               'eng': 1,
               'deu': 2,
               'nld': 3,
               'spa': 4,
               'ita': 5}  # pour pouvoir savoir quel est l'index auquel l'activation doit être faite

    for i in instances :
        c = np.zeros((6,), dtype=float)
        c[mapping[i['label']]] = 1.0
        categories.append(c)

    return np.array(categories)

def format_prediction(prediction):

    labels = ['fra', 'eng', 'deu', 'nld', 'spa', 'ita']

    values = prediction.tolist()

    index_highest = values.index(max(values))

    l = labels[index_highest]

    return l

