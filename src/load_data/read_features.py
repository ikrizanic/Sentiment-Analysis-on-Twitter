def read_features(file_path):
    features = list()
    file = open(file_path, "r")
    for line in file:
        f_list = list()
        parts = line[1:len(line) - 2].split(",")
        for part in parts:
            f_list.append(float(part.strip()))
        features.append(f_list)
    return features
