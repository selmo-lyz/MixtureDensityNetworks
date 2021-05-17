def train_closure(model, criterion, optimizer,
                  dataloader, device='cpu'):
    model.train()
    for xi, yi in dataloader:
        xi = xi.to(device).float()
        yi = yi.to(device).float()

        def closure():
            optimizer.zero_grad()
            output = model(xi)
            loss = criterion(output, yi)
            loss.backward()
            return loss
        optimizer.step(closure)

    return None