from thop import profile
import time
import torch
import torch.nn as nn
import argparse
import os
from torchsummary import summary
from util import load_data_n_model, load_migration_data_n_model, train_and_save_model

def train(model, tensor_loader, num_epochs, learning_rate, criterion, device):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_accuracy = 0
        for data in tensor_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = labels.type(torch.LongTensor)
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.to(device)
            outputs = outputs.type(torch.FloatTensor)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)
            predict_y = torch.argmax(outputs, dim=1).to(device)
            epoch_accuracy += (predict_y == labels.to(device)).sum().item() / labels.size(0)
        epoch_loss = epoch_loss / len(tensor_loader.dataset)
        epoch_accuracy = epoch_accuracy / len(tensor_loader)
        print('Epoch:{}, Accuracy:{:.4f}, Loss:{:.9f}'.format(epoch + 1, float(epoch_accuracy), float(epoch_loss)))
    return

def test(model, tensor_loader, criterion, device):
    model.eval()
    test_acc = 0
    test_loss = 0
    total_time = 0
    with torch.no_grad():
        for data in tensor_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = labels.type(torch.LongTensor)
            start_time = time.time()
            outputs = model(inputs)
            outputs = outputs.type(torch.FloatTensor)
            outputs.to(device)
            end_time = time.time()
            total_time += (end_time - start_time)
            loss = criterion(outputs, labels)
            predict_y = torch.argmax(outputs, dim=1).to(device)
            accuracy = (predict_y == labels.to(device)).sum().item() / labels.size(0)
            test_acc += accuracy
            test_loss += loss.item() * inputs.size(0)
        test_acc = test_acc / len(tensor_loader)
        test_loss = test_loss / len(tensor_loader.dataset)
    print("Validation Accuracy: {:.4f}, Loss: {:.5f}, Time: {:.2f}s".format(
        float(test_acc), float(test_loss), total_time))
    input_size = (3, 114, 500)  # 示例输入大小
    summary(model, input_size=input_size, device=str(device))
    return

def main():
    root = 'F:\\dataset'
    parser = argparse.ArgumentParser('WiFi Imaging Benchmark')
    parser.add_argument('--dataset', choices=['NTU-Fi-HumanID', 'NTU-Fi_HAR'], required=True)
    parser.add_argument('--model', choices=['LSTM', 'BiLSTM', 'CNN+GRU', 'ResNet+GRU'], required=True)
    args = parser.parse_args()
    train_loader, test_loader, model, train_epoch = load_data_n_model(args.dataset, args.model, root)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train(
        model=model,
        tensor_loader=train_loader,
        num_epochs=train_epoch,
        learning_rate=1e-3,
        criterion=criterion,
        device=device
    )
    test(
        model=model,
        tensor_loader=test_loader,
        criterion=criterion,
        device=device
    )
    return

def migration_main():
    root = 'F:\\dataset'
    parser = argparse.ArgumentParser('WiFi Imaging Benchmark for Migration Learning')
    parser.add_argument('--phase', choices=['pretrain', 'finetune'], required=True,
                        help='Phase of transfer learning: pretrain or finetune')
    parser.add_argument('--dataset', choices=['NTU-Fi-HumanID', 'NTU-Fi_HAR'], required=True,
                        help='Dataset for transfer learning')
    parser.add_argument('--model', choices=['CNN+biLSTM', 'CNN-LSTM', 'CNN+GRU', 'ResNet+GRU'], default='CNN+biLSTM', help='Model for migration learning')
    args = parser.parse_args()
    train_loader, test_loader, model, train_epoch = load_migration_data_n_model(args.dataset, args.model, root)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if args.phase == 'pretrain':
        train_epoch = 100
    elif args.phase == 'finetune':
        train_epoch = 75
    if args.phase == 'pretrain':
        save_path = 'pretrained_weights.pth'
        print("Starting pre-training phase...")
        train_and_save_model(
            model=model,
            tensor_loader=train_loader,
            num_epochs=train_epoch,
            learning_rate=1e-3,
            criterion=criterion,
            device=device,
            save_path=save_path
        )
        print(f"Pre-trained weights saved to {save_path}")
    elif args.phase == 'finetune':
        save_path = 'pretrained_weights.pth'
        if not os.path.exists(save_path):
            raise FileNotFoundError(f"Pre-trained weights not found at {save_path}. Please run pre-training phase first.")
        pretrained_dict = torch.load(save_path, map_location=device)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'classifier' not in k}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        print("Pre-trained weights loaded (excluding classifier).")
        # 冻结 CNN 层
        for name, param in model.named_parameters():
            if 'encoder' in name or 'reshape' in name or 'conv1' in name or 'layer' in name:
                param.requires_grad = False
        print("Starting fine-tuning phase...")
        train(
            model=model,
            tensor_loader=train_loader,
            num_epochs=train_epoch,
            learning_rate=5e-4,
            criterion=criterion,
            device=device
        )
    test(
        model=model,
        tensor_loader=test_loader,
        criterion=criterion,
        device=device
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

if __name__ == "__main__":
    migration_main()
