import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.plugins.ddp_sequential_plugin import DDPSequentialPlugin
from pytorch_lightning.utilities import BOLTS_AVAILABLE, FAIRSCALE_PIPE_AVAILABLE



class DvectorExtractor(pl.LightningModule):
    def __init__(self, n_input, n_hidden, n_output, dvec_dim, num_layer, lr, optimizer, bidirectional=False):
        super().__init__()

        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.dvec_dim = dvec_dim
        self.num_layer = num_layer
        self.n_direction = 2 if bidirectional else 1
        self.bn = nn.BatchNorm1d(n_hidden)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(n_input, n_hidden, num_layer, bidirectional=bidirectional, batch_first=True)
            

        self.extractor = nn.Linear(self.n_direction*self.n_hidden, dvec_dim)
        self.out = nn.Linear(dvec_dim, n_output)

    def forward(self, sorted_x, sorted_length):
        
        # pack_padded
        sorted_length = torch.flatten(sorted_length)
        pack = nn.utils.rnn.pack_padded_sequence(sorted_x, sorted_length, batch_first=True)
        norm_data = (pack.data - pack.data.mean()) / pack.data.std()
        pack.data[:] = norm_data

        # forward
        out, (h, c) = self.lstm(input)
        h = self.relu(self.bn(h))
        h = self.extractor(h.view(-1, self.n_hidden))
        h = self.out(h)

        return h

    def training_step(self, batch, batch_idx):
        data, label = batch
        out = self.forward(data)
        #loss = F.nll_loss(logits, y)    The negative log likelihood loss.
        criterion = nn.CrossEntropyLoss()
        loss = criterion(out, label)
        self.log('Training Loss', loss)
        return loss

    def _evaluate(self, batch, batch_idx, stage=None):
        # これを使ってvalやtestを行う
        data, label = batch
        out = self.forward(x)
        #logits = F.log_softmax(out, dim=-1)
        #loss = F.nll_loss(logits, y)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(out, label)
        logits = F.log_softmax(out, dim=-1)
        preds = torch.argmax(logits, dim=-1)
        acc = accuracy(preds, label)

        if stage:
            self.log(f'{stage}_loss', loss, prog_bar=True)
            self.log(f'{stage}_acc', acc, prog_bar=True)

        return loss, acc

    def validation_step(self, batch, batch_idx):
        return self._evaluate(batch, batch_idx, 'val')[0]

    def test_step(self, batch, batch_idx):
        loss, acc = self._evaluate(batch, batch_idx, 'test')
        self.log_dict({'test_loss': loss, 'test_acc': acc})

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=0.9, weight_decay=5e-4)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    0.1,
                    epochs=self.trainer.max_epochs,
                    steps_per_epoch=math.ceil(45000 / self.hparams.batch_size)),
                'interval': 'step',
            }
        }    
        
    @property
    def automatic_optimization(self) -> bool:   # ->は返り値に期待する型
        return not self._manual_optimization



    def pack_padded(sorted_x, sorted_length):
        # labelもsortしないといけないのでこの中でsortはやらない方が良い？

        sorted_length = torch.flatten(sorted_length)
        pack = nn.utils.rnn.pack_padded_sequence(sorted_x, sorted_length, batch_first=True)
        norm_data = (pack.data - pack.data.mean()) / pack.data.std()
        pack.data[:] = norm_data

        return pack

    

