import pickle

def main():
    list_a = [1, 2, 3, 4]
    list_b = ['a', 'b', 'c', 'd']

    for e, l in zip(list_a, list_b):
        print(e, ' -> ', l)

    d = {e: l for e, l in zip(list_a, list_b)}
    with open('dict.pkl', 'wb') as f:
        pickle.dump(d, f)

    with open('dict.pkl', 'rb') as f:
        r = pickle.load(f)
    print(r)

    x = 23
    with open('int.pkl', 'wb') as f:
        pickle.dump(x, f)

    with open('int.pkl', 'rb') as f:
        y = pickle.load(f)

    print(y)

if __name__ == '__main__':
    main()
