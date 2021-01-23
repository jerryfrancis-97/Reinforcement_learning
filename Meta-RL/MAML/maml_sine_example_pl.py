import random
import numpy as np
import torch
import learn2learn as l2l

from torch import nn, optim
from pytorch_lightning.core.lightning import LightningModule,LightningDataModule

from torchmeta.toy import Sinusoid
from torchmeta.utils.data import BatchMetaDataLoader

import matplotlib.pyplot as plt
import os


class Data_Sinusoid(LightningDataModule):
    def __init__(self,num_samples_per_task=20,num_tasks=16000,tasks_per_batch=16,shots=10):
        super().__init__()
        self.num_samples_per_task=num_samples_per_task
        self.num_tasks=num_tasks
        self.tasks_per_batch=tasks_per_batch
        self.shots=shots
        #currently test_data will have automatically num_samples_per_task=shots, num_tasks=num_test_tasks
        print("Data initialising....")

    def prepare_data(self):
        # called only on 1 GPU
        print("Using sinusoid from torchmeta")
        
    def setup(self):
        # called on every GPU
        self.train = Sinusoid(num_samples_per_task=self.num_samples_per_task, num_tasks=self.num_tasks)
        self.test = Sinusoid(num_samples_per_task=self.shots, num_tasks=self.num_tasks)
        
    def train_dataloader(self):
        # transforms = ...
        return BatchMetaDataLoader(self.train, batch_size=self.tasks_per_batch)

    # def val_dataloader(self):
    #     transforms = ...
    #     return DataLoader(self.val, batch_size=64)

    def test_dataloader(self):
        # transforms = ...
        return BatchMetaDataLoader(self.test, batch_size=self.tasks_per_batch)

class SineModel(LightningModule):

    def __init__(self, dim, experiment_dir):
        super().__init__()
        self.hidden1 = nn.Linear(1, dim)
        self.hidden2 = nn.Linear(dim, dim)
        self.hidden3 = nn.Linear(dim, 1)
        self.experiment_dir = experiment_dir
        self.checkpoint_dir = self.experiment_dir+"/model"
        self.plot_results = self.experiment_dir+"/plot_results"
        self.checkpoint_file = os.path.join(self.checkpoint_dir,"Sine_MAML_hidden_dim_"+str(dim)+".pth")

    def forward(self, x):
        x = nn.functional.relu(self.hidden1(x))
        x = nn.functional.relu(self.hidden2(x))
        x = self.hidden3(x)
        return x

    def training_step(self, batch, batch_idx):
        meta_train_loss = 0.0

        # for each task in the batch
        effective_batch_size = batch[0].shape[0]
        for i in range(effective_batch_size):
            learner = maml.clone()

            # divide the data into support and query sets
            train_inputs, train_targets = batch[0][i].float(), batch[1][i].float()
            x_support, y_support = train_inputs[::2], train_targets[::2]
            x_query, y_query = train_inputs[1::2], train_targets[1::2]


            for _ in range(adapt_steps): # adaptation_steps
                support_preds = learner(x_support)
                support_loss=lossfn(support_preds, y_support)
                learner.adapt(support_loss)

            query_preds = learner(x_query)
            query_loss = lossfn(query_preds, y_query)
            meta_train_loss += query_loss

        meta_train_loss = meta_train_loss / effective_batch_size
        
        opt.zero_grad()
        meta_train_loss.backward()
        opt.step()
        

    def save_checkpoint(self):
        print("Saving checkpoint....to: "+str(self.checkpoint_file))
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("Loading checkpoint...."+" "+str(self.checkpoint_file))
        self.load_state_dict(torch.load(self.checkpoint_file))	
