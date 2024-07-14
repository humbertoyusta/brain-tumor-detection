import tqdm
import torch
import sklearn.metrics


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    train_loss = 0

    for inputs, targets in tqdm.tqdm(loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, targets.float())
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    return train_loss / len(loader)


def eval_model(model, loader, criterion, device):
    model.eval()
    eval_loss = 0
    correct = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets.float())
            eval_loss += loss.item()

            preds = (outputs > 0.5).float()
            correct += (preds == targets).sum().item()

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    return (
        eval_loss / len(loader),
        correct / len(loader.dataset),
        sklearn.metrics.confusion_matrix(all_targets, all_preds),
    )