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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy
import numpy as np
from captum.attr import DeepLiftShap, IntegratedGradients

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

#dataset 
freqs = [0.5, 0.7, 1.0]
events = "/home/theek/ibs_dl/dataset/frq{}/meta_event_{}.pkl"
meta_events = []
for freq in freqs:
    with open(events.format(freq, freq), 'rb') as handle:
        meta = pickle.load(handle)
    meta_events.append(meta)
sub_to_f1 = {}
# train-data
for REMOVE_SUB in meta_events[0].keys(): #iterate over the subjects in a LOSO evaluation. 
    test_data = []
    #only load the correct
    test_events = "/home/theek/ibs_dl/dataset/frq{}/meta_event_{}.pkl".format(0.5, 0.5)
    with open(test_events, 'rb') as handle:
            test_meta = pickle.load(handle)
    test_data = test_meta[REMOVE_SUB]

    print('evaluate on:', len(test_data))

    # Parameters
    params = {'batch_size': 32,
            'shuffle': True,
            'num_workers': 4}

    test_set = Dataset(test_data)
    testing_generator = torch.utils.data.DataLoader(test_set, **params)\
    
    # select the main model
    model = CNN2D_BaselineV2()
    print('number of parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    model.cuda()

    # load the model
    NAME = "CNN2D_BaselineV2_cvloso"
    try:
        model.load_state_dict(torch.load('./checkpoints/model_{}_{}.tm'.format(NAME, REMOVE_SUB)))
    except Exception as e:
        print(e)
        continue
    
    # evaluation mode with IG attribution
    model.eval()
    ig = IntegratedGradients(model)
    test_f1= []

    y_true, y_pred = [], []
    for it, (x, y) in enumerate(testing_generator):
        y_hat = model(x.cuda())
        f1 = f1_score(y.cpu().numpy(), y_hat.argmax(dim=-1).detach().cpu().numpy(), average='micro')

        x_, baselines = x[:5, ], x[5:, ]
        if baselines.shape[0] == 0:
            break
        
        # compute the attribution
        attribution = ig.attribute(x_.cuda(), target=y_hat.argmax(dim=-1)[:5].cuda())
        print('attribution shape:', attribution.shape)

        # save the attribution and the input
        meta = {
            'attr': attribution.detach().cpu().numpy(),
            'x': x.cpu().numpy(),
            'y': y.cpu().numpy().tolist()
        }
        # write to a file to visualize later
        # Use Cedalion's visualization tools to visualize the attribution
        # Cedalion Channel Quality example
        #with open('./attribution_pca/attr_ig_{}_{}_{}.pkl'.format(it, REMOVE_SUB, var), 'wb') as handle:
        #    pickle.dump(meta, handle, protocol=pickle.HIGHEST_PROTOCOL)     

        test_f1.append(f1)
        y_true.append(y.cpu().numpy())
        y_pred.append(y_hat.argmax(dim=-1).detach().cpu().numpy())

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

    print('-----Test-F1: {:.2f} for {}'.format(f1_score(y_true, y_pred), REMOVE_SUB))
    sub_to_f1[REMOVE_SUB] = f1_score(y_true, y_pred)

for sub in sub_to_f1: # for which the model was trained. performance on the test set as sanity check
    print('{}, {:.2f}'.format(sub, sub_to_f1[sub]))
