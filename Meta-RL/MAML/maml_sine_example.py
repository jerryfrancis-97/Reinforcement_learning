#!/usr/bin/env python3


import random
import numpy as np
import torch
import learn2learn as l2l

from torch import nn, optim

from torchmeta.toy import Sinusoid
from torchmeta.utils.data import BatchMetaDataLoader

import matplotlib.pyplot as plt
import os


class SineModel(nn.Module):

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

    def save_checkpoint(self):
        print("Saving checkpoint....to: "+str(self.checkpoint_file))
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("Loading checkpoint...."+" "+str(self.checkpoint_file))
        self.load_state_dict(torch.load(self.checkpoint_file))	

def plotter(true_x, true_y, pred_y, iteration, which_loss, loss,plot_dir):
    plt.plot(sorted(true_x), sorted(true_y), color="b")
    plt.plot(sorted(true_x), sorted(pred_y), color="r")
    plt.legend("Iter:"+str(iteration)+which_loss+" loss:"+str(loss))
    plt.savefig(os.path.join(plot_dir,"Meta_iter_"+str(iteration)+which_loss))
    plt.show()
    print("---------- Plot saved! ------------")
    


def main(
        shots=10,
        tasks_per_batch=16,
        num_tasks=1600,
        num_test_tasks=32,
        adapt_lr=0.01,
        meta_lr=0.001,
        adapt_steps=5,
        hidden_dim=8,
):
    EXPERIMENT_DIR = "./experiments/MAML_Sine_exps"
    if os.path.isdir(EXPERIMENT_DIR):
        print("Experiment folder opened ...")
    else:
        os.mkdir(EXPERIMENT_DIR)
        print("New Experiment folder started ...")
    
    MODEL_CHECKPOINT_DIR = EXPERIMENT_DIR+"/model"
    if os.path.isdir(MODEL_CHECKPOINT_DIR):
        print("Model Checkpoint folder opened ...")
    else:
        os.mkdir(MODEL_CHECKPOINT_DIR)
        print("New Model checkpoint folder made ...")

    PLOT_RESULTS_DIR = EXPERIMENT_DIR+"/plot_results"
    if os.path.isdir(PLOT_RESULTS_DIR):
        print("Image results folder opened ...")
    else:
        os.mkdir(PLOT_RESULTS_DIR)
        print("New Image results folder made ...")
    

    # load the dataset
    tasksets = Sinusoid(num_samples_per_task=2*shots, num_tasks=num_tasks)
    dataloader = BatchMetaDataLoader(tasksets, batch_size=tasks_per_batch)

    # create the model
    model = SineModel(dim=hidden_dim,experiment_dir=EXPERIMENT_DIR)
    maml = l2l.algorithms.MAML(model, lr=adapt_lr, first_order=False, allow_unused=True)
    opt = optim.Adam(maml.parameters(), meta_lr)
    lossfn = nn.MSELoss(reduction='mean')

    # for each iteration
    for iter, batch in enumerate(dataloader): # num_tasks/batch_size
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
        
        if iter % 20 == 0:
            print('Iteration:', iter, 'Meta Train Loss', meta_train_loss.item()) 
            # print(x_query.requires_grad, y_query.requires_grad,query_preds.requires_grad,meta_train_loss.item())
            plotter(x_query, y_query,query_preds.detach().numpy(),iter,'Train', meta_train_loss.item(),model.plot_results)


        
    #save current model
    model.save_checkpoint()
	
	#meta-testing
    test_tasks = Sinusoid(num_samples_per_task=shots, num_tasks=num_test_tasks)
    test_dataloader = BatchMetaDataLoader(test_tasks, batch_size=tasks_per_batch)

    #load learned model
    test_model = SineModel(dim=hidden_dim,experiment_dir=EXPERIMENT_DIR)
    test_model.load_checkpoint()

    for iter,batch in enumerate(test_dataloader):
        meta_test_loss = 0.0

        # for each task in the batch
        effective_batch_size = batch[0].shape[0]
        for i in range(effective_batch_size):
            learner = maml.clone()

            # divide the data into support and query sets
            test_inputs, test_targets = batch[0][i].float(), batch[1][i].float()

            test_preds = test_model(test_inputs)
            test_loss = lossfn(test_preds, test_targets)
            meta_test_loss += test_loss

        meta_test_loss = meta_test_loss / effective_batch_size

        if iter % 20 == 0:
            print('Iteration:', iter, 'Meta Test Loss', meta_test_loss.item()) 
            plotter(test_inputs, test_targets, test_preds.detach().numpy(), iter,'Test', meta_test_loss.item(),test_model.plot_results)



if __name__ == '__main__':
    main()
