import torch
import glob
import pickle
import numpy as np
from torch import nn
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from pathlib import PureWindowsPath
import torch
from torch import nn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy
import numpy as np
from sklearn.model_selection import train_test_split

class CNN2D_BaselineV2(nn.Module):
    def __init__(self) -> None:
            super().__init__()
            self.model = nn.Sequential(
            nn.Conv2d(100, 64, kernel_size=(1, 3)),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.InstanceNorm2d(64),
            nn.MaxPool2d(kernel_size=(1, 3)),
            nn.Conv2d(64, 32, kernel_size=(1, 3)),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.InstanceNorm2d(32),
            nn.MaxPool2d(kernel_size=(1, 3)),
            nn.Conv2d(32, 16, kernel_size=(1, 3)),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.InstanceNorm2d(16),
            nn.MaxPool2d(kernel_size=(1, 3)),
            nn.Flatten(start_dim=1),
            nn.Dropout(0.5),
            nn.Linear(16*4, 2),
            nn.Softmax(dim=-1)
        )
    def forward(self, x):
        x = x.permute(0, 2, 1, 3)
        return self.model(x)

class MultiStageModel(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes):
        super(MultiStageModel, self).__init__()
        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim, num_classes)
        self.stages = nn.ModuleList([copy.deepcopy(SingleStageModel(num_layers, num_f_maps, num_classes, num_classes)) for s in range(num_stages-1)])

    def forward(self, x):
        out = self.stage1(x)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(F.softmax(out, dim=1))
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs

class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)
        return out

class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out)

class MSTCN_WRAP(nn.Module):
    def __init__(self, num_stages=4, num_layers=8, num_f_maps=16, dim=200, num_classes=2):
        super(MSTCN_WRAP, self).__init__()
        self.mstn_encode = MultiStageModel(num_stages, num_layers, num_f_maps, dim, num_classes)
        self.cnn = nn.Sequential(
            nn.Conv1d(num_stages*num_classes, 16, kernel_size=(3)),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(0.4),
            nn.MaxPool1d(kernel_size=3),
            nn.Conv1d(16, 32, kernel_size=(3)),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.4),
            nn.MaxPool1d(kernel_size=3),
            nn.Conv1d(32, 32, kernel_size=(3)),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.4),
            nn.MaxPool1d(kernel_size=3),
            nn.Flatten(start_dim=1),
            nn.Dropout(0.4),
            nn.Linear(32*2, 2),
            nn.Softmax(dim=-1)
        )
    def forward(self, x):
        x = x.flatten(start_dim=1, end_dim=2)
        x = self.mstn_encode(x)
        x = x.permute(1, 0, 2, 3).flatten(start_dim=1, end_dim=2)
        return self.cnn(x)
    
class Dataset(torch.utils.data.Dataset):
    def __init__(self, files):
            self.files = files
            self.labels = {'Right':1, 'Left':2}

    def __len__(self):
            return len(self.files)

    def normalize(self, x):
        x = x.to_numpy()
        return (x - np.min(x))/ np.ptp(x)
    
    def normalize_chromo(self, x):
        data = []
        for chan in range(x.shape[1]):
            chromo = []
            for chr in range(x.shape[0]):
                chromo.append(self.normalize(x[chr, chan, :]))
            data.append(chromo)
        return np.array(data)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        segmentfile = self.files[index]

        # read
        segment = None
        with open(segmentfile, 'rb') as handle:
            segment = pickle.load(handle)

        #normalize
        xt = torch.from_numpy(segment['xt'].to_numpy()).float() #[chromo, chan, time])
        y = segment['class']-1

        return xt, y

class CNN1D_Baseline(nn.Module):
    def __init__(self) -> None:
            super().__init__()
            self.model = nn.Sequential(
                nn.Conv1d(200, 100, kernel_size=1),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.InstanceNorm1d(100),
                nn.MaxPool1d(kernel_size=3),
                nn.Conv1d(100, 64, kernel_size=3),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.InstanceNorm1d(64),
                nn.Conv1d(64, 32, kernel_size=3),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.InstanceNorm1d(32),
                nn.MaxPool1d(kernel_size=3),
                nn.Conv1d(32, 16, kernel_size=3),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.InstanceNorm1d(16),
                nn.Flatten(start_dim=1),
                nn.Dropout(0.4),
                nn.Linear(16*6, 2),
                nn.Softmax(dim=-1)
            )            
    def forward(self, x):
        x = x.flatten(start_dim=1, end_dim=2)
        return self.model(x)

class Transformer(nn.Module):
    def __init__(self, embedding_dim=128, num_layers=6) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=8)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.project = nn.Linear(200, embedding_dim)
        self.classification = nn.Sequential(
            nn.Linear(embedding_dim, 2),
            nn.Softmax(dim=-1)
        )
    def forward(self, x):
        x = x.flatten(start_dim=1, end_dim=2).permute(0, 2, 1)
        x = self.project(x)
        x = self.transformer(x)
        x = x[:, -1, :]
        return self.classification(x)

from resnet import resnet1d101

if __name__ == '__main__':   
    #dataset 
    freqs = [0.5] #only correct freq
    events = "/dataset_v2/frq{}/meta_event_{}.pkl"
    meta_events = []
    for freq in freqs:
        with open(events.format(freq, freq), 'rb') as handle:
            meta = pickle.load(handle)
        meta_events.append(meta)

    # train-data - in this setting, we only use the real/correct epoch segment
    for REMOVE_SUB in meta_events[0].keys(): #iterate over the subjects in a LOSO evaluation. 
        train_data = []
        for meta_dict in meta_events:
            for sub in meta_dict:
                if sub != REMOVE_SUB and sub in meta_dict:
                    train_data += meta_dict[sub]
        
        # generate the validation dataset        
        files_to_sessions=None
        with open('files_to_sessions_dv2.pkl', 'rb') as handle:
            files_to_sessions = pickle.load(handle) 

        train_data_, validation_data = [], []
        for file in train_data:
            if files_to_sessions[file] in ['run-1', 'run-2']:
                train_data_.append(file)
            else:
                validation_data.append(file)
        train_data = np.array(train_data_)
        validation_data = np.array(validation_data)

        #here, let's select temporal-shift augmentation
        test_data = []
        #only load the correct
        test_events = "/dataset_v2/frq{}/meta_event_{}.pkl".format(0.5, 0.5)
        with open(test_events, 'rb') as handle:
                test_meta = pickle.load(handle)
        test_data = test_meta[REMOVE_SUB]

        print('train on:', train_data.shape[0], 'validate on:', validation_data.shape[0],  'test on:', len(test_data))
        print('CUDA:', torch.cuda.is_available())

        #sanity check
        fre_info = []
        for file in train_data:
            if 'event' in file:
                freq = PureWindowsPath(file).parts[-3]
                fre_info.append(freq)
        print('unique:', np.unique(np.array(fre_info), return_counts=True))
        fre_info = []
        for file in train_data:
            if 'rest' in file:
                freq = PureWindowsPath(file).parts[-3]
                fre_info.append(freq)
        print('unique:', np.unique(np.array(fre_info), return_counts=True))

        # Parameters
        params = {'batch_size': 16, #we changed it for smooth-training 
                'shuffle': True,
                'num_workers': 4}
        # Generators
        training_set = Dataset(train_data.tolist())
        training_generator = torch.utils.data.DataLoader(training_set, **params)

        validation_set = Dataset(validation_data.tolist())
        validation_generator = torch.utils.data.DataLoader(validation_set, **params)

        test_set = Dataset(test_data)
        testing_generator = torch.utils.data.DataLoader(test_set, **params)\

        # model
        model = MSTCN_WRAP()
        NAME = "MSTCN_WRAP_cvloseo_v2"
        print('number of parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
        model.cuda()
        optim = torch.optim.AdamW(model.parameters(), lr=1e-5)
        loss_function = nn.CrossEntropyLoss()

        mats = []
        f1_max = -np.inf
        for epoch in range(200):
            print('\nEpoch-----------{}------{}'.format(epoch, NAME))

            model.train()
            train_f1, train_loss = [], []
            for it, (x, y) in enumerate(training_generator):
                optim.zero_grad()
                #x = x.flatten(1,2)
                y_hat = model(x.cuda())
                loss = loss_function(y_hat, y.cuda())
                loss.backward()
                optim.step()

                f1 = f1_score(y.cpu().numpy(), y_hat.argmax(dim=-1).detach().cpu().numpy(), average='micro')
                train_f1.append(f1)
                train_loss.append(loss.item())
                print('\r[{:04d}] loss: {:.2f} f1-score: {:.2f}'.format(it, loss.item(), f1), end='')

            print('\n-----Train- loss {:.4f} and f1: {:.2f}'.format(np.mean(train_loss), np.mean(train_f1)))

            model.eval()
            val_loss, val_f1= [], []
            for it, (x, y) in enumerate(validation_generator):
                #x = x.flatten(1,2)
                y_hat = model(x.cuda())
                loss = loss_function(y_hat, y.cuda())
                f1 = f1_score(y.cpu().numpy(), y_hat.argmax(dim=-1).detach().cpu().numpy(), average='micro')
                print('\r[{:04d}] validation loss: {:.2f} f1-score: {:.2f}'.format(it,loss.item(), f1), end='')
                val_f1.append(f1)
                val_loss.append(loss.item())
            print('\n-----Validation- loss {:.4f} and f1: {:.2f}'.format(np.mean(val_loss), np.mean(val_f1)))
            
            #save-the-model based on validation data
            if np.mean(val_f1) > f1_max:
                print('the performance increased from:', f1_max, ' to ', np.mean(val_f1))
                f1_max = np.mean(val_f1)
                torch.save(model.state_dict(), './checkpoints/model_{}_{}.tm'.format(NAME, REMOVE_SUB))

            test_loss, test_f1= [], []
            for it, (x, y) in enumerate(testing_generator):
                #x = x.flatten(1,2)
                y_hat = model(x.cuda())
                loss = loss_function(y_hat, y.cuda())
                f1 = f1_score(y.cpu().numpy(), y_hat.argmax(dim=-1).detach().cpu().numpy(), average='micro')
                print('\r[{:04d}] validation loss: {:.2f} f1-score: {:.2f}'.format(it,loss.item(), f1), end='')
                test_f1.append(f1)
                test_loss.append(loss.item())
            print('\n-----LOSO Test- loss {:.4f} and f1: {:.2f}'.format(np.mean(test_loss), np.mean(test_f1)))

            mats.append([np.mean(train_loss), np.mean(train_f1), np.mean(val_loss), np.mean(val_f1), np.mean(test_loss), np.mean(test_f1)])
            temp = np.array(mats)
            labels = ['$L_{tr}$', '$F1_{tr}$', '$L_{vl}$', '$F1_{vl}$', '$L_{te}$', '$F1_{te}$']
            _ = plt.figure(figsize=(16,6))
            for k in range(6):
                plt.subplot(2,3,k+1)
                plt.plot(temp[:, k])
                plt.title(label=labels[k])
            plt.savefig('./loss_cvloseo/baseline_{}_raw_{}_lr1e_5.png'.format(NAME, REMOVE_SUB))
            plt.close()

            np.save('./loss_cvloseo/mats_baseline_{}_raw_{}_lr1e_5.png'.format(NAME, REMOVE_SUB), np.array(mats))
