import numpy as np

def inference(model, dataloader, device='cpu'):
    pred_y = np.empty(0)

    model.eval()
    for xi in dataloader:
        xi = xi.to(device).float()

        output = model(xi)
        pred_yi = output.detach().numpy()
        pred_y = np.append(pred_y, pred_yi.reshape(-1))
    
    return pred_y