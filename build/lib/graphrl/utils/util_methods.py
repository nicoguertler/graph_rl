def listify(x, l):
    if not isinstance(x, list):
        return [x]*l
    else:
        return x
