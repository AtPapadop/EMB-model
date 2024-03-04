from torch.optim import RMSprop

def get_optimizer(model, learning_rate):
    return RMSprop(model.parameters(), lr=learning_rate)
