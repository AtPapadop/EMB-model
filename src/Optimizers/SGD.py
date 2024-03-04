from torch.optim import SGD

def get_optimizer(model, learning_rate):
    return SGD(model.parameters(), lr=learning_rate)
