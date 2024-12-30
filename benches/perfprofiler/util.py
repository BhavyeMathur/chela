def merge_dicts(list_of_dicts):
    result = {}
    for k in list_of_dicts[0].keys():
        result[k] = [d[k] for d in list_of_dicts]
    return result
