def minmax(data):
    min_val = min(data)
    max_val = max(data)
    normalized_data = [(x - min_val) / (max_val - min_val) for x in data]
    return normalized_data

data = [10, 20, 30, 40, 50]
scaled_data = minmax(data)
print(scaled_data)
