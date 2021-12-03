from warnings import filterwarnings

filterwarnings("ignore")

import os
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.mixture import GaussianMixture

from torch.utils.data import DataLoader, Dataset
from dataloader_cifar import cifar_dataset
from dataset_his import his_dataset
import dataloader_cifar as dataloader
import dataloader_easy 
from PreResNet import *
from preset_parser import *
import pickle



if __name__ == "__main__":
    args = parse_args("./presets.json")

    logs = open(os.path.join(args.checkpoint_path, "saved", "metrics.log"), "a")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    def record_history(index, output, target, recorder):
        pred = F.softmax(output, dim=1).cpu().data
        for i, ind in enumerate(index):
            recorder[ind].append(pred[i][target.cpu()[i]].numpy().tolist())

        return

   

    def warmup(epoch, net, optimizer, dataloader, recorder):
        net.train()
        num_iter = (len(dataloader.dataset) // dataloader.batch_size) + 1
        for batch_idx, (inputs, labels, path) in enumerate(dataloader):
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = net(inputs)
            record_history(path, outputs, labels, recorder)
            loss = CEloss(outputs, labels)
            if (
                args.noise_mode == "asym"
            ):  # penalize confident prediction for asymmetric noise
                penalty = conf_penalty(outputs)
                L = loss + penalty
            elif args.noise_mode == "sym":
                L = loss
            L.backward()
            optimizer.step()

            # sys.stdout.write("\r")
            # sys.stdout.write(
            #     "%s: %.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f"
            #     % (
            #         args.dataset,
            #         args.r,
            #         args.noise_mode,
            #         epoch,
            #         args.num_epochs - 1,
            #         batch_idx + 1,
            #         num_iter,
            #         loss.item(),
            #     )
            # )
            # sys.stdout.flush()

    class NegEntropy(object):
        def __call__(self, outputs):
            probs = torch.softmax(outputs, dim=1)
            return torch.mean(torch.sum(probs.log() * probs, dim=1))

    def create_model(devices=[0]):
        model = ResNet18(num_classes=args.num_class)
        model = model.cuda()
        model = torch.nn.DataParallel(model, device_ids=devices).cuda()
        return model

    loader = dataloader.cifar_dataloader(
        dataset=args.dataset,
        r=args.r,
        noise_mode=args.noise_mode,
        batch_size=args.batch_size,
        warmup_batch_size=args.warmup_batch_size,
        num_workers=args.num_workers,
        root_dir=args.data_path,
        noise_file=f"{args.checkpoint_path}/saved/labels.json",
        augmentation_strategy=args,
    )

    

    print("| Building net")
    devices = range(torch.cuda.device_count())
    net1 = create_model(devices)
    cudnn.benchmark = True

    optimizer1 = optim.SGD(
        net1.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4
    )

    all_loss = [[], []]  # save the history of losses from two networks
    epoch = 0

    CE = nn.CrossEntropyLoss(reduction="none")
    CEloss = nn.CrossEntropyLoss()
    if args.noise_mode == "asym":
        conf_penalty = NegEntropy()
    warmup_trainloader = loader.run("warmup")
    recorder1 = [[] for i in range(len(warmup_trainloader.dataset))]# recorder for training history of all samples
    
    while epoch < args.num_epochs:
        lr = args.learning_rate
        if epoch >= args.lr_switch_epoch:
            lr /= 10
        for param_group in optimizer1.param_groups:
            param_group["lr"] = lr
        print("epoch:%d" % epoch)
        warmup(epoch, net1, optimizer1, warmup_trainloader, recorder1)
        epoch += 1

    with open(f"{args.checkpoint_path}/saved/recorder1.p","wb") as f1:
        pickle.dump(recorder1,f1)
    
    #filter easy samples
    record = np.array(recorder1)
    r_mean = np.mean(record,axis=1)
    sort_mean = np.sort(r_mean)
    sort_index = np.argsort(r_mean)
    train_dataset = cifar_dataset(dataset=args.dataset,
                            r=args.r,
                            noise_mode="sym",
                            root_dir=args.data_path,
                            noise_file=f"{args.checkpoint_path}/saved/labels.json",
                            transform="",
                            mode="all")
    easy = sort_index[int(train_dataset.__len__()*(0.5+args.r*0.5)):]
    train_data_easy = train_dataset.train_data[easy]
    train_label_easy = np.array(train_dataset.noise_label)[easy]
    with open(f"{args.checkpoint_path}/saved/train_data_easy.p","wb") as f1:
        pickle.dump(train_data_easy,f1)
    with open(f"{args.checkpoint_path}/saved/train_label_easy.p","wb") as f1:
        pickle.dump(train_label_easy,f1)



    #inject noise to D_e

    easy_loader = dataloader_easy.easy_dataloader(
        dataset=args.dataset,
        r=args.r,
        noise_mode=args.noise_mode,
        batch_size=args.batch_size,
        warmup_batch_size=args.warmup_batch_size,
        num_workers=args.num_workers,
        root_dir=args.data_path,
        noise_file=f"{args.checkpoint_path}/saved/easy_labels.p",
        augmentation_strategy=args,
    )

    print("| Building net")
    devices = range(torch.cuda.device_count())
    net_easy = create_model(devices)
    cudnn.benchmark = True

    optimizer_easy = optim.SGD(
        net_easy.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4
    )

    all_loss = [[], []]  
    epoch = 0


        
    warmup_easyloader = easy_loader.run("warmup",train_data_easy,train_label_easy) # D_a

    
    recorder_easy = [[] for i in range(len(warmup_easyloader.dataset))]


    while epoch < args.num_epochs:
        lr = args.learning_rate
        if epoch >= args.lr_switch_epoch:
            lr /= 10
        for param_group in optimizer_easy.param_groups:
            param_group["lr"] = lr



        print("epoch:%d" % epoch)
        warmup(epoch, net_easy, optimizer_easy, warmup_easyloader, recorder_easy) # recorder for training history of easy samples


    
        epoch += 1
    
    with open(f"{args.checkpoint_path}/saved/recorder_easy.p","wb") as f1:
        pickle.dump(recorder_easy,f1)

    
    #training 1D-CNN classifier 
    record_easy = np.array(recorder_easy)
    r_mean_easy = np.mean(record_easy,axis=1)
    sort_mean_easy = np.sort(r_mean_easy)
    sort_index_easy = np.argsort(r_mean_easy)
    train_label = train_label_easy
    with open(f"{args.checkpoint_path}/saved/easy_labels.p","rb") as f1:
        noise_label = pickle.load(f1)
        
    noise_rate=args.r

    train_index = sort_index_easy[int(sort_index_easy.shape[0]*noise_rate*0.5):int(sort_index_easy.shape[0]*(0.5+noise_rate*0.5))]
    train_record = record_easy[train_index]
    train_r_label = []
    for index in train_index:
        if train_label[index]==noise_label[index]:
            train_r_label.append(1)#hard samples
        else:
            train_r_label.append(0)#noisy samples

    train_record = torch.Tensor(train_record)
    train_record = torch.unsqueeze(train_record,1)

    train_r_label = torch.Tensor(train_r_label).type(torch.LongTensor)

    test_index = sort_index[int(sort_index.shape[0]*noise_rate*0.5):int(sort_index.shape[0]*(0.5+noise_rate*0.5))]
    test_record = record[test_index]
    test_r_label = []
    for index in test_index:
        if train_dataset.noise_label[index] == train_dataset.train_label[index]:
            test_r_label.append(1) #hard samples
        else:
            test_r_label.append(0) #noisy samples
    test_record = torch.Tensor(test_record)
    test_record = torch.unsqueeze(test_record,1)
    test_r_label = torch.Tensor(test_r_label).type(torch.LongTensor)

    batch_size = 256
    train_dataset_ehn = his_dataset(train_record, train_r_label)
    test_dataset_ehn = his_dataset(test_record, test_r_label)
    train_dataloader_ehn = DataLoader(dataset=train_dataset_ehn, batch_size = batch_size, num_workers = 16, shuffle = False)
    test_dataloader_ehn = DataLoader(dataset=test_dataset_ehn, batch_size = 50000, num_workers = 16)

    # one dimension CNN classifier
    class oneD_CNN(nn.Module):
        def __init__(self):
            super(oneD_CNN, self).__init__()
            self.conv1 = nn.Sequential(
                nn.Conv1d(1, 16, 3, stride=1, padding=1),
                nn.ReLU()
            )
            self.conv2 = nn.Sequential(
                nn.Conv1d(16, 32, 3, stride=1, padding=1),
                nn.ReLU()
            )
            self.conv3 = nn.Sequential(
                nn.Conv1d(32, 32, 3, stride=1, padding=1),
                nn.ReLU()
            )

            self.output = nn.Linear(in_features=32*args.num_epochs, out_features=2)
        def forward(self,x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = x.view(x.size(0),-1)
            output = self.output(x)
            return output

    net = oneD_CNN()    

    #training
    net = net.cuda()
    optimizer = torch.optim.Adadelta(net.parameters(), lr=1)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.99)
    loss_func = torch.nn.CrossEntropyLoss()
    max_epoch = 13 #hyper-parameters
    for epoch in range(max_epoch):
        net.train()
        loss_sigma = 0.0  #
        correct = 0.0
        total = 0.0
        for i,(train_data,train_label) in enumerate(train_dataloader_ehn):
            train_data,train_label = Variable(train_data).cuda(),Variable(train_label).cuda()
            out = net(train_data)  

            loss = loss_func(out, train_label)  

            optimizer.zero_grad() 
            loss.backward() 
            optimizer.step()

            _, predicted = torch.max(out.data, 1)
            total += train_label.size(0)
            correct += (predicted == train_label).squeeze().sum().cpu().numpy()
            loss_sigma += loss.item()
        
        scheduler.step()
        print("Training: Epoch[{:0>3}/{:0>3}]  Loss: {:.4f} Acc:{:.2%}".format(
            epoch + 1, max_epoch, loss_sigma, correct / total))
        
        if epoch % 2 == 0:
            net.eval()
            conf_matrix = np.zeros((2,2))
            with torch.no_grad():
                for it,(test_data,test_label) in enumerate(test_dataloader_ehn):
                    test_data,test_label = Variable(test_data).cuda(),Variable(test_label).cuda()
                    test_out = net(test_data)
                    _, predicted = torch.max(test_out.data, 1)
                    for i in range(predicted.shape[0]):
                        conf_matrix[test_label[i],predicted[i]]+=1
                    
                
            print(conf_matrix)
            acc = np.diag(conf_matrix).sum()/np.sum(conf_matrix)
            print(acc)

    test_out = F.softmax(test_out)
    pred = [False for i in range(50000)]
    prob = [0 for i in range(50000)]
    for i in range(test_out.shape[0]):
        p = float(test_out[i][1])
        if p > 0.5:#hard
            prob[test_index[i]]=p
        else :# noisy
            prob[test_index[i]]=p
    for ind in sort_index[int(sort_index.shape[0]*(0.5+noise_rate*0.5)):]:
        prob[ind]=1 # easy
    prob = np.array(prob)
    with open(f"{args.checkpoint_path}/saved/prob1_ehn.p","wb") as f:
        pickle.dump(prob,f)

    #again
    net = oneD_CNN() 
    net = net.cuda()
    optimizer = torch.optim.Adadelta(net.parameters(), lr=1)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.99)
    loss_func = torch.nn.CrossEntropyLoss()
    max_epoch = 14 #hyper-parameters
    for epoch in range(max_epoch):
        net.train()
        loss_sigma = 0.0  #
        correct = 0.0
        total = 0.0
        for i,(train_data,train_label) in enumerate(train_dataloader_ehn):
            train_data,train_label = Variable(train_data).cuda(),Variable(train_label).cuda()
            out = net(train_data)  

            loss = loss_func(out, train_label)  

            optimizer.zero_grad() 
            loss.backward() 
            optimizer.step()

            _, predicted = torch.max(out.data, 1)
            total += train_label.size(0)
            correct += (predicted == train_label).squeeze().sum().cpu().numpy()
            loss_sigma += loss.item()
        
        scheduler.step()
        print("Training: Epoch[{:0>3}/{:0>3}]  Loss: {:.4f} Acc:{:.2%}".format(
            epoch + 1, max_epoch, loss_sigma, correct / total))
        
        if epoch % 1 == 0:
            net.eval()
            conf_matrix = np.zeros((2,2))
            with torch.no_grad():
                for it,(test_data,test_label) in enumerate(test_dataloader_ehn):
                    test_data,test_label = Variable(test_data).cuda(),Variable(test_label).cuda()
                    test_out = net(test_data)
                    _, predicted = torch.max(test_out.data, 1)

    test_out = F.softmax(test_out)
    pred = [False for i in range(50000)]
    prob = [0 for i in range(50000)]
    for i in range(test_out.shape[0]):
        p = float(test_out[i][1])
        if p > 0.5:#hard
            prob[test_index[i]]=p
        else :# noisy
            prob[test_index[i]]=p
    for ind in sort_index[int(sort_index.shape[0]*(0.5+noise_rate*0.5)):]:
        prob[ind]=1 # easy
    prob = np.array(prob)
    with open(f"{args.checkpoint_path}/saved/prob2_ehn.p","wb") as f:
        pickle.dump(prob,f)


