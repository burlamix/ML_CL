import pickle



with open('adine_20000.pkl', 'rb') as handle:
    b = pickle.load(handle)

print(b[0]["configuration"]["optimizers"].pprint())

