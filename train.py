import time

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from mobilenetv3 import mobilenetv3_small, mobilenetv3_large

def load_from_folder(path, batch_size, shuffle=False, num_workers=1):
    dataset = ImageFolder(path, transform=transforms.Compose([
        transforms.ToTensor()
    ]))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def evaluate_accuracy(data_iter, net, max_batch_cnt=50, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        net.eval()
        batch_cnt = 0
        for X, y in data_iter:
            acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
            n += y.shape[0]
            batch_cnt += 1
            if batch_cnt >= max_batch_cnt:
                break
        net.train()
    return acc_sum / n


def main():
    batch_size = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_iter = load_from_folder('./Train', batch_size, shuffle=True, num_workers=1)
    test_iter = load_from_folder('./Test', batch_size, shuffle=False, num_workers=1)

    lr, num_epochs = 0.001, 10
    net = mobilenetv3_large(num_classes=2)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
            if batch_count % 50 == 0:
                print('batch %d, loss %.4f, train acc %.3f, time %.1f sec'
                      % (batch_count, train_l_sum / batch_count, train_acc_sum / n, time.time() - start))
            if batch_count % 1000 == 0:
                test_acc = evaluate_accuracy(test_iter, net, max_batch_cnt=50)
                print('epoch %d, loss %.4f, train acc %.3f, time %.1f sec, test acc %.3f'
                      % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, time.time() - start, test_acc))
                torch.save(net.state_dict(), f'baldnet_{epoch}_{batch_count}.model')


if __name__ == '__main__':
    main()
