import torch
from torch import nn

from commnet import CommNet

def train(dataloader, model, loss_fn, optimizer):
    """
    """
    # quelle sera la forme du dataloader ?
    # quelle fonction de perte loss_fn ?

    model.train()

    for batch, x, y in (dataloader):
        pred = model(x)
        loss = loss_fn(pred, y)

        # Backpropagation de l'erreur
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 10 == 0:
            print(f"Batch {batch}: perte = {loss}")


def test(dataloader, model, loss_fn):
    """
    """
    batch_size = len(dataloader)
    model.eval()
    correct = 0
    loss = 0
    total_loss = 0.0

    with torch.no_grad():
        for x, y in dataloader:
            pred = model(x)
            total_loss += loss_fn(pred, y).item()

            pred = model(x)
            actions_prob = torch.Softmax(pred, dim = 1)
            actions = actions_prob.argmax(dim = 1)
            correct += (actions == y).type(torch.float).sum().item()
        
    avg_loss = total_loss / batch_size
    accuracy = correct / len(dataloader.dataset)
    print(f"TEST: perte moyenne = {avg_loss}, accuray = {accuracy}")
    return avg_loss, accuracy

# entraîner le model selon un nombre d'epoch