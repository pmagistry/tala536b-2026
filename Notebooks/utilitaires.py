# (chargement, normalisation, encodage des labels)

def load_data(path):
    # calcule 2 features par texte : fréquence de "th" et de "en"
    data = []
    with open(path) as file:
        for line in file:
            label, text = line.strip().split(" ", 1)
            n_th = text.count("th")
            n_en = text.count("en")
            l = len(text)
            instance = {"label": label, "th": n_th / l, "en": n_en / l}
            data.append(instance)
    return data


def normalise_data(dataset):
    for k in dataset[0].keys():
        if k != 'label':
            vals = [d[k] for d in dataset]
            mean = sum(vals) / len(vals)
            std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
            for d in dataset:
                d[k] = (d[k] - mean) / std


def get_label_list(dataset):
    return sorted(set(d['label'] for d in dataset))


def one_hot_encode(label, label_list):
    # transforme un label texte en vecteur
    return [1.0 if label == l else 0.0 for l in label_list]