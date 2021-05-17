def evaluation(model, criterion, dataloader, device='cpu'):
    loss_valid = 0.0

    model.eval()
    for xi, yi in dataloader:
        xi = xi.to(device).float()
        yi = yi.to(device).float()

        output = model(xi)
        loss = criterion(output, yi)

        loss_valid += loss.item()*xi.size(0)

    return loss_valid/len(dataloader.dataset)