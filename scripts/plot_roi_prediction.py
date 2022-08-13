import pickle


def load_pkl(file):
    return pickle.load(open(file,'rb'))