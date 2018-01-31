import pickle



with open('grid_save.pkl', 'rb') as handle:
    b = pickle.load(handle)

print(b)

