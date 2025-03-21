import os
import torch
from torchsummary import summary
from dataset import CSI_Dataset
from NTU_Fi_model import NTU_Fi_LSTM, NTU_Fi_BiLSTM, NTU_Fi_CNN_BiLSTM_Migration, NTU_Fi_CNN_LSTM, NTU_Fi_ResNet_GRU, NTU_Fi_CNN_GRU

def load_data_n_model(dataset_name, model_name, root):
    print(f"Root absolute path: {os.path.abspath(root)}")
    classes = {'NTU-Fi-HumanID': 14, 'NTU-Fi_HAR': 6}
    if dataset_name == 'NTU-Fi-HumanID':
        print('using dataset: NTU-Fi-HumanID')
        num_classes = classes['NTU-Fi-HumanID']
        train_loader = torch.utils.data.DataLoader(dataset=CSI_Dataset(os.path.join(root, 'NTU-Fi-HumanID', 'test_amp')),
                                                   batch_size=16, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=CSI_Dataset(os.path.join(root, 'NTU-Fi-HumanID', 'train_amp')),
                                                  batch_size=16, shuffle=False)
        if model_name == 'LSTM':
            print("using model: LSTM")
            model = NTU_Fi_LSTM(num_classes)
            train_epoch = 50
        elif model_name == 'BiLSTM':
            print("using model: BiLSTM")
            model = NTU_Fi_BiLSTM(num_classes)
            train_epoch = 50
        elif model_name == 'ResNet+GRU':
            print("using model: ResNet+GRU")
            model = NTU_Fi_ResNet_GRU(num_classes, freeze_cnn=False)
            train_epoch = 50
        return train_loader, test_loader, model, train_epoch
    elif dataset_name == 'NTU-Fi_HAR':
        print('using dataset: NTU-Fi_HAR')
        num_classes = classes['NTU-Fi_HAR']
        train_loader = torch.utils.data.DataLoader(dataset=CSI_Dataset(os.path.join(root, 'NTU-Fi_HAR', 'train_amp')),
                                                   batch_size=16, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=CSI_Dataset(os.path.join(root, 'NTU-Fi_HAR', 'test_amp')),
                                                  batch_size=64, shuffle=False)
        if model_name == 'LSTM':
            print("using model: LSTM")
            model = NTU_Fi_LSTM(num_classes)
            train_epoch = 30
        elif model_name == 'BiLSTM':
            print("using model: BiLSTM")
            model = NTU_Fi_BiLSTM(num_classes)
            train_epoch = 30
        elif model_name == 'ResNet+GRU':
            print("using model: ResNet+GRU")
            model = NTU_Fi_ResNet_GRU(num_classes, freeze_cnn=False)
            train_epoch = 30
        return train_loader, test_loader, model, train_epoch

def load_migration_data_n_model(dataset_name, model_name, root):
    classes = {'NTU-Fi-HumanID': 14, 'NTU-Fi_HAR': 6}
    if dataset_name == 'NTU-Fi_HAR':
        print('using dataset: NTU-Fi_HAR for pre-training')
        train_loader = torch.utils.data.DataLoader(dataset=CSI_Dataset(os.path.join(root, 'NTU-Fi_HAR', 'train_amp')),
                                                   batch_size=16, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=CSI_Dataset(os.path.join(root, 'NTU-Fi_HAR', 'test_amp')),
                                                  batch_size=64, shuffle=False)
        num_classes = classes['NTU-Fi_HAR']
        if model_name == 'CNN+biLSTM':
            print("using model: CNN+biLSTM for pre-training")
            model = NTU_Fi_CNN_BiLSTM_Migration(num_classes, freeze_cnn=False)
            train_epoch = 100
        elif model_name == 'CNN-LSTM':
            print("using model: CNN-LSTM for pre-training")
            model = NTU_Fi_CNN_LSTM(num_classes, freeze_cnn=False)
            train_epoch = 100
        elif model_name == 'CNN+GRU':
            print("using model: CNN+GRU for pre-training")
            model = NTU_Fi_CNN_GRU(num_classes, freeze_cnn=False)
            train_epoch = 100
        elif model_name == 'ResNet+GRU':
            print("using model: ResNet+GRU for pre-training")
            model = NTU_Fi_ResNet_GRU(num_classes, freeze_cnn=False)
            train_epoch = 100
        return train_loader, test_loader, model, train_epoch
    elif dataset_name == 'NTU-Fi-HumanID':
        print('using dataset: NTU-Fi-HumanID for fine-tuning')
        train_loader = torch.utils.data.DataLoader(dataset=CSI_Dataset(os.path.join(root, 'NTU-Fi-HumanID', 'train_amp')),
                                                   batch_size=16, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=CSI_Dataset(os.path.join(root, 'NTU-Fi-HumanID', 'test_amp')),
                                                  batch_size=16, shuffle=False)
        num_classes = classes['NTU-Fi-HumanID']
        if model_name == 'CNN+biLSTM':
            print("using model: CNN+biLSTM for fine-tuning")
            model = NTU_Fi_CNN_BiLSTM_Migration(num_classes, freeze_cnn=True)
            train_epoch = 50
        elif model_name == 'CNN-LSTM':
            print("using model: CNN-LSTM for fine-tuning")
            model = NTU_Fi_CNN_LSTM(num_classes, freeze_cnn=True)
            train_epoch = 50
        elif model_name == 'CNN+GRU':
            print("using model: CNN+GRU for pre-training")
            model = NTU_Fi_CNN_GRU(num_classes, freeze_cnn=True)
            train_epoch = 50
        elif model_name == 'ResNet+GRU':
            print("using model: ResNet+GRU for fine-tuning")
            model = NTU_Fi_ResNet_GRU(num_classes, freeze_cnn=True)
            train_epoch = 50
        return train_loader, test_loader, model, train_epoch

def train_and_save_model(model, tensor_loader, num_epochs, learning_rate, criterion, device, save_path):
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
    torch.save(model.state_dict(), save_path)
    print(f"Model weights saved to {save_path}")
