import numpy as np

def mini_batch(X, y, batch_size):

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    

    mini_batches = []
    num_complete_batches = X.shape[0]    
    for i in range(num_complete_batches):
        mini_batch_X = X_shuffled[i*batch_size:(i+1)*batch_size, :]
        mini_batch_y = y_shuffled[i*batch_size:(i+1)*batch_size]
        mini_batches.append((mini_batch_X, mini_batch_y))
    
    if X.shape[0] % batch_size != 0:
        mini_batch_X = X_shuffled[num_complete_batches*batch_size:, :]
        mini_batch_y = y_shuffled[num_complete_batches*batch_size:]
        mini_batches.append((mini_batch_X, mini_batch_y))
    
    return mini_batches
