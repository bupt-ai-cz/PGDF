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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.mixture import GaussianMixture

from torch.utils.data import DataLoader, Dataset

import dataloader_cifar as dataloader
import dataloader_easy 
from PreResNet import *
from preset_parser import *
import pickle
import pdb


if __name__ == "__main__":
    args = parse_args("./presets.json")
    logs = open(os.path.join(args.checkpoint_path, "saved", "metrics.log"), "a")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    class_num = args.num_class
    # prob_trans_m0 = torch.zeros([class_num,class_num])# <0.6
    # prob_trans_m1 = torch.zeros([class_num,class_num])# 0.6~0.8
    # prob_trans_m2 = torch.zeros([class_num,class_num])# >0.8
    prob_trans_m = torch.zeros([class_num,class_num])
    # Training
    def train(epoch, net, net2, optimizer, labeled_trainloader, unlabeled_trainloader, easy_trainloader):
        # estimate transition matrix
        if epoch > 150:
            global prob_trans_m
            net.eval()
            net2.eval()
            class_num = args.num_class
            temp_prob_trans_m = torch.zeros([class_num,class_num])

            with torch.no_grad():
                for (
                    batch_idx,
                    (
                        inputs_e1,
                        inputs_e2,
                        labels_e,
                    ),
                ) in enumerate(easy_trainloader):
                    inputs_e1, inputs_e2 = (
                        inputs_e1.cuda(),
                        inputs_e2.cuda(),
                    )
                    outputs_e_1 = net(inputs_e1)
                    outputs_e_2 = net(inputs_e2)

                    pe = (
                        torch.softmax(outputs_e_1, dim=1)
                        + torch.softmax(outputs_e_2, dim=1)
                    ) / 2
                    for i in range(len(labels_e)):
                        temp_prob_trans_m[labels_e[i]] += pe[i].cpu()

            temp_prob_trans_m = temp_prob_trans_m / torch.sum(temp_prob_trans_m,dim=1, keepdim=True)
            temp_prob_trans_m = temp_prob_trans_m.inverse().cuda()

            if not torch.isnan(temp_prob_trans_m[0][0]):
                prob_trans_m = temp_prob_trans_m.clone()

        net.train()
        net2.eval()  # fix one network and train the other

        unlabeled_train_iter = iter(unlabeled_trainloader)
        num_iter = (len(labeled_trainloader.dataset) // args.batch_size) + 1
        for (
            batch_idx,
            (
                inputs_x,
                inputs_x2,
                inputs_x3,
                inputs_x4,
                labels_x,
                w_x,
            ),
        ) in enumerate(labeled_trainloader):
            try:
                inputs_u, inputs_u2, inputs_u3, inputs_u4, labels_u = unlabeled_train_iter.next()
            except:
                unlabeled_train_iter = iter(unlabeled_trainloader)
                inputs_u, inputs_u2, inputs_u3, inputs_u4, labels_u = unlabeled_train_iter.next()
            batch_size = inputs_x.size(0)

            # Transform label to one-hot
            labels_x = torch.zeros(batch_size, args.num_class).scatter_(
                1, labels_x.view(-1, 1), 1
            )
            w_x = w_x.view(-1, 1).type(torch.FloatTensor)

            inputs_x, inputs_x2, inputs_x3, inputs_x4, labels_x, w_x = (
                inputs_x.cuda(),
                inputs_x2.cuda(),
                inputs_x3.cuda(),
                inputs_x4.cuda(),
                labels_x.cuda(),
                w_x.cuda(),
            )

            inputs_u, inputs_u2, inputs_u3, inputs_u4, labels_u = (
                inputs_u.cuda(),
                inputs_u2.cuda(),
                inputs_u3.cuda(),
                inputs_u4.cuda(),
                labels_u.cuda(),
            )

            # inputs u/u2
            with torch.no_grad():
                # label co-guessing of unlabeled samples
                outputs_u_1 = net(inputs_u3)
                outputs_u_2 = net(inputs_u4)
                outputs_u_3 = net2(inputs_u3)
                outputs_u_4 = net2(inputs_u4)

                pu = (
                    torch.softmax(outputs_u_1, dim=1)
                    + torch.softmax(outputs_u_2, dim=1)
                    + torch.softmax(outputs_u_3, dim=1)
                    + torch.softmax(outputs_u_4, dim=1)
                ) / 4
                if epoch > 150:
                    pu = torch.mm(pu,prob_trans_m) # pseudo-labels denoising
                    # pu[pu>1]=1
                    pu[pu<0]=0
                ptu = pu
                # else:
                #     ptu = pu ** (1 / 0.5)  # temparature sharpening
                
                targets_u = ptu / ptu.sum(dim=1, keepdim=True)  # normalize
                targets_u = targets_u.detach()

                # label refinement of labeled samples
                outputs_x_1 = net(inputs_x3)
                outputs_x_2 = net(inputs_x4)

                px = (
                    torch.softmax(outputs_x_1, dim=1)
                    + torch.softmax(outputs_x_2, dim=1)
                ) / 2
                if epoch > 150:
                    # px[px>1]=1
                    px = torch.mm(px,prob_trans_m) # pseudo-labels denoising
                    px[px<0] = 0
                px = w_x * labels_x + (1 - w_x) * px
                # if epoch > 150:
                #     ptx = px
                # else:
                #     ptx = px ** (1 / 0.5)  # temparature sharpening
                ptx = px

                targets_x = ptx / ptx.sum(dim=1, keepdim=True)  # normalize
                targets_x = targets_x.detach()


            # mixmatch
            l = np.random.beta(args.alpha, args.alpha)
            l = max(l, 1 - l)

            all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
            all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)
            w_hard = torch.cat([w_x,w_x])

            idx = torch.randperm(all_inputs.size(0))

            input_a, input_b = all_inputs, all_inputs[idx]
            target_a, target_b = all_targets, all_targets[idx]

            mixed_input = l * input_a + (1 - l) * input_b
            mixed_target = l * target_a + (1 - l) * target_b

            logits = net(mixed_input)
            logits_x = logits[: batch_size * 2]
            logits_u = logits[batch_size * 2 :]

            Lx, Lu, lamb = criterion(
                logits_x,
                mixed_target[: batch_size * 2],
                logits_u,
                mixed_target[batch_size * 2 :],
                epoch + batch_idx / num_iter,
                args.warm_up,
                w_hard,
                epoch,
            )

            # regularization
            prior = torch.ones(args.num_class) / args.num_class
            prior = prior.cuda()
            pred_mean = torch.softmax(logits, dim=1).mean(0)
            penalty = torch.sum(prior * torch.log(prior / pred_mean))

            loss = Lx + lamb * Lu + penalty
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # sys.stdout.write("\r")
            # sys.stdout.write(
            #     "%s: %.1f-%s | Epoch [%3d/%3d], Iter[%3d/%3d]\t Labeled loss: %.2f, Unlabeled loss: %.2f"
            #     % (
            #         args.dataset,
            #         args.r,
            #         args.noise_mode,
            #         epoch,
            #         args.num_epochs - 1,
            #         batch_idx + 1,
            #         num_iter,
            #         Lx.item(),
            #         Lu.item(),
            #     )
            # )
            # sys.stdout.flush()

    def warmup(epoch, net, optimizer, dataloader):
        net.train()
        num_iter = (len(dataloader.dataset) // dataloader.batch_size) + 1
        for batch_idx, (inputs, labels, path) in enumerate(dataloader):
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = net(inputs)
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

    def test(epoch, net1, net2, size_l1, size_u1, size_l2, size_u2):
        global logs
        net1.eval()
        net2.eval()
        all_targets = []
        all_predicted = []
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs1 = net1(inputs)
                outputs2 = net2(inputs)
                outputs = outputs1 + outputs2
                _, predicted = torch.max(outputs, 1)

                all_targets += targets.tolist()
                all_predicted += predicted.tolist()

        accuracy = accuracy_score(all_targets, all_predicted)
        precision = precision_score(all_targets, all_predicted, average="weighted")
        recall = recall_score(all_targets, all_predicted, average="weighted")
        f1 = f1_score(all_targets, all_predicted, average="weighted")
        results = "Test Epoch: %d, Accuracy: %.3f, Precision: %.3f, Recall: %.3f, F1: %.3f, L_1: %d, U_1: %d, L_2: %d, U_2: %d" % (
            epoch,
            accuracy * 100,
            precision * 100,
            recall * 100,
            f1 * 100,
            size_l1,
            size_u1,
            size_l2,
            size_u2,
        )
        print("\n" + results + "\n")
        logs.write(results + "\n")
        logs.flush()
        return accuracy

    def eval_train(model, all_loss):
        model.eval()
        losses = torch.zeros(len(eval_loader.dataset))
        with torch.no_grad():
            for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = model(inputs)
                loss = CE(outputs, targets)
                for b in range(inputs.size(0)):
                    losses[index[b]] = loss[b]
        losses = (losses - losses.min()) / (losses.max() - losses.min())
        all_loss.append(losses)

        if (
            args.average_loss > 0
        ):  # average loss over last 5 epochs to improve convergence stability
            history = torch.stack(all_loss)
            input_loss = history[-args.average_loss :].mean(0)
            input_loss = input_loss.reshape(-1, 1)
        else:
            input_loss = losses.reshape(-1, 1)

        # fit a two-component GMM to the loss
        gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
        gmm.fit(input_loss)
        prob = gmm.predict_proba(input_loss)
        prob = prob[:, gmm.means_.argmin()]
        return prob, all_loss

    def linear_rampup(current, warm_up, rampup_length=16):
        current = np.clip((current - warm_up) / rampup_length, 0.0, 1.0)
        return args.lambda_u * float(current)

    class SemiLoss(object):
        def __call__(
            self, outputs_x_1, targets_x, outputs_u, targets_u, epoch, warm_up, w_hard, actual_epoch
        ):
            probs_u = torch.softmax(outputs_u, dim=1)
            if actual_epoch > 100: # hard enhancing
                Lx = -torch.mean(
                    torch.sum(F.log_softmax(outputs_x_1, dim=1) * targets_x / (w_hard ** args.mt), dim=1)
                )
            else:
                Lx = -torch.mean(
                    torch.sum(F.log_softmax(outputs_x_1, dim=1) * targets_x , dim=1)
                )
            Lu = torch.mean((probs_u - targets_u) ** 2)

            return Lx, Lu, linear_rampup(epoch, warm_up)

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
        preaug_file=(
            f"{args.checkpoint_path}/saved/{args.preset}_preaugdata.pth.tar"
            if args.preaugment
            else ""
        ),
        augmentation_strategy=args,
    )

    loader_easy = dataloader_easy.easy_dataloader(
        dataset=args.dataset,
        r=args.r,
        noise_mode=args.noise_mode,
        batch_size=args.batch_size,
        warmup_batch_size=args.warmup_batch_size,
        num_workers=args.num_workers,
        root_dir=args.data_path,
        noise_file=f"{args.checkpoint_path}/saved/easy_labels.p",
        preaug_file=(
            f"{args.checkpoint_path}/saved/{args.preset}_preaugdata.pth.tar"
            if args.preaugment
            else ""
        ),
        augmentation_strategy=args,
    )

    

    print("| Building net")
    devices = range(torch.cuda.device_count())
    net1 = create_model(devices)
    net2 = create_model(devices)

    cudnn.benchmark = True

    criterion = SemiLoss()
    optimizer1 = optim.SGD(
        net1.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4
    )
    optimizer2 = optim.SGD(
        net2.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4
    )

    all_loss = [[], []]  

    # if args.pretrained_path != "":
    #     with open(args.pretrained_path + f"/saved/{args.preset}.pth.tar", "rb") as p:
    #         unpickled = torch.load(p)
    #     net1.load_state_dict(unpickled["net1"])
    #     net2.load_state_dict(unpickled["net2"])
    #     optimizer1.load_state_dict(unpickled["optimizer1"])
    #     optimizer2.load_state_dict(unpickled["optimizer2"])
    #     all_loss = unpickled["all_loss"]
    #     epoch = unpickled["epoch"] + 1
    # else:
    #     epoch = 0
    epoch = 0

    CE = nn.CrossEntropyLoss(reduction="none")
    CEloss = nn.CrossEntropyLoss()
    if args.noise_mode == "asym":
        conf_penalty = NegEntropy()

    warmup_trainloader = loader.run("warmup")
    with open(f"{args.checkpoint_path}/saved/train_data_easy.p","rb") as f1:
        train_data = pickle.load(f1)
    with open(f"{args.checkpoint_path}/saved/train_label_easy.p","rb") as f2:
        train_label = pickle.load(f2)
    easy_trainloader = loader_easy.run("clean",train_data,train_label)
    test_loader = loader.run("test")
    eval_loader = loader.run("eval_train")

    prob_his1 = pickle.load(open(f"{args.checkpoint_path}/saved/prob1_ehn.p","rb"))
    prob_his2 = pickle.load(open(f"{args.checkpoint_path}/saved/prob2_ehn.p","rb"))

    while epoch < args.num_epochs:
        lr = args.learning_rate
        if epoch >= args.lr_switch_epoch:
            lr /= 10
        for param_group in optimizer1.param_groups:
            param_group["lr"] = lr
        for param_group in optimizer2.param_groups:
            param_group["lr"] = lr

        size_l1, size_u1, size_l2, size_u2 = (
            len(warmup_trainloader.dataset),
            0,
            len(warmup_trainloader.dataset),
            0,
        )

        if epoch < args.warm_up:
            print("Warmup Net1")
            warmup(epoch, net1, optimizer1, warmup_trainloader)
            print("\nWarmup Net2")
            warmup(epoch, net2, optimizer2, warmup_trainloader)

        else:
            if epoch > 200:
                prob1_gmm, all_loss[0] = eval_train(net1, all_loss[0])
                prob2_gmm, all_loss[1] = eval_train(net2, all_loss[1])

            m = args.md

            if epoch > 200:
                prob1 = m*prob1_gmm + (1-m)*prob_his1 
                prob2 = m*prob2_gmm + (1-m)*prob_his2 
            else:
                prob1 = prob_his1
                prob2 = prob_his2

            pred1 = prob1 > 0.5
            pred2 = prob2 > 0.5



            print("Train Net1")
            labeled_trainloader, unlabeled_trainloader = loader.run(
                "train", pred2, prob2
            )  # co-divide
            size_l1, size_u1 = (
                len(labeled_trainloader.dataset),
                len(unlabeled_trainloader.dataset),
            )
            train(
                epoch,
                net1,
                net2,
                optimizer1,
                labeled_trainloader,
                unlabeled_trainloader,
                easy_trainloader,
            )  # train net1

            print("\nTrain Net2")
            labeled_trainloader, unlabeled_trainloader = loader.run(
                "train", pred1, prob1
            )  # co-divide
            size_l2, size_u2 = (
                len(labeled_trainloader.dataset),
                len(unlabeled_trainloader.dataset),
            )
            train(
                epoch,
                net2,
                net1,
                optimizer2,
                labeled_trainloader,
                unlabeled_trainloader,
                easy_trainloader,
            )  # train net2

        acc = test(epoch, net1, net2, size_l1, size_u1, size_l2, size_u2)
        data_dict = {
            "epoch": epoch,
            "net1": net1.state_dict(),
            "net2": net2.state_dict(),
            "optimizer1": optimizer1.state_dict(),
            "optimizer2": optimizer2.state_dict(),
            "all_loss": all_loss,
        }
        if (epoch + 1) % args.save_every == 0 or epoch == args.warm_up - 1:
            checkpoint_model = os.path.join(
                args.checkpoint_path, "all", f"{args.preset}_epoch{epoch}.pth.tar"
            )
            torch.save(data_dict, checkpoint_model)
        saved_model = os.path.join(
            args.checkpoint_path, "saved", f"{args.preset}.pth.tar"
        )
        torch.save(data_dict, saved_model)
        epoch += 1

