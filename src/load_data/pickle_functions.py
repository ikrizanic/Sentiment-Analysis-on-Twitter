import pickle

def load_dataset(file):
    load_file = open(file, "rb")
    dataset = pickle.load(load_file)
    load_file.close()
    return dataset

def dump_dataset(dataset, file):
    file = open(file, "wb")
    pickle.dump(dataset, file)
    file.close()

def dump_features(feature_array, file):
    write_file = open(file, "wb")
    pickle.dump(feature_array, write_file)
    write_file.close()

def load_features(file):
    load_file = open(file, "rb")
    features = pickle.load(load_file)
    load_file.close()
    return features

def dump_labels(labels, file):
    write_file = open(file, "wb")
    pickle.dump(labels, write_file)
    write_file.close()


def load_labels(file):
    load_file = open(file, "rb")
    labels = pickle.load(load_file)
    load_file.close()
    return labels
