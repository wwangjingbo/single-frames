import argparse
import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as vutils
from torch.autograd import Variable
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, precision_score, recall_score
from dataset import get_dataloaders
from utils import (Logger, get_model, mixup_criterion, mixup_data, random_seed, save_checkpoint, smooth_one_hot,
                   cross_entropy)

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='USTC Computer Vision Final Project')
parser.add_argument('--arch', default="ResNet50", type=str)
parser.add_argument('--epochs', default=300, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--scheduler', default="reduce", type=str, help='[reduce, cos]')
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--label_smooth', default=True, type=eval)
parser.add_argument('--label_smooth_value', default=0.1, type=float)
parser.add_argument('--mixup', default=True, type=eval)
parser.add_argument('--mixup_alpha', default=1.0, type=float)
parser.add_argument('--Ncrop', default=True, type=eval)
parser.add_argument('--data_path', default='datasets/diving/diving.csv', type=str)
parser.add_argument('--results', default='./results', type=str)
parser.add_argument('--save_freq', default=10, type=int)
parser.add_argument('--resume', default=0, type=int)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--name', default='official', type=str)

best_acc = 0


def main():
    global best_acc

    args = parser.parse_args()
    if random_seed is not None:
        random_seed(args.seed)

    args_path = f"{args.arch}_epoch{args.epochs}_bs{args.batch_size}_lr{args.lr}_momentum{args.momentum}_wd{args.weight_decay}_seed{args.seed}_smooth{args.label_smooth}_mixup{args.mixup}_scheduler{args.scheduler}_{args.name}"

    checkpoint_path = os.path.join(
        args.results, args.name, args_path, 'checkpoints'
    )

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    writer = SummaryWriter(os.path.join(
        args.results, args.name, args_path, 'tensorboard_logs'
    ))

    logger = Logger(os.path.join(
        args.results, args.name, args_path, 'output.log'
    ))

    logger.info(args)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(device)

    logger.info('Load dataset ...')

    train_loader, val_loader, test_loader = get_dataloaders(
        path=args.data_path,
        bs=args.batch_size, augment=True
    )

    logger.info('Start load model %s ...', args.arch)
    model = get_model(args.arch)
    print(model)

    model = model.to(device)
    scaler = GradScaler()

    loss_fn_class = nn.CrossEntropyLoss()
    loss_fn_reg = nn.MSELoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True
    )
    if args.scheduler == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )
    elif args.scheduler == 'reduce':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.75, patience=5, verbose=True
        )

    if args.resume > 0:
        logger.info('Resume from epoch %d', (args.resume))
        state_dict = torch.load(os.path.join(
            checkpoint_path, f'checkpoint_{args.resume}.tar'
        ))
        model.load_state_dict(state_dict['model_state_dict'])
        optimizer.load_state_dict(state_dict['opt_state_dict'])

    logger.info('Start training.')
    logger.info(
        "Epoch\tTime\tTrain Loss\tTrain ACC\tTrain MSE\tVal Loss\tVal ACC\tVal MSE"
    )

    for epoch in range(1, args.epochs + 1):
        start_t = time.time()

        train_loss, train_acc, train_mse, train_preds, train_class_labels, train_regression_labels = train(
            model, train_loader, loss_fn_class, loss_fn_reg, optimizer, epoch, device, scaler, writer, args
        )
        
        val_loss, val_acc, val_mse, val_preds, val_class_labels, val_regression_labels = evaluate(
            model, val_loader, loss_fn_class, loss_fn_reg, device, args
        )

        train_f1 = f1_score(train_class_labels, train_preds, average='macro')
        val_f1 = f1_score(val_class_labels, val_preds, average='macro')
        train_precision = precision_score(train_class_labels, train_preds, average='macro')
        val_precision = precision_score(val_class_labels, val_preds, average='macro')
        train_recall = recall_score(train_class_labels, train_preds, average='macro')
        val_recall = recall_score(val_class_labels, val_preds, average='macro')

        writer.add_scalar("Train/Loss", train_loss, epoch)
        writer.add_scalar("Train/Accuracy", train_acc, epoch)
        writer.add_scalar("Train/MSE", train_mse, epoch)
        writer.add_scalar("Val/Loss", val_loss, epoch)
        writer.add_scalar("Val/Accuracy", val_acc, epoch)
        writer.add_scalar("Val/MSE", val_mse, epoch)

        writer.add_scalar("Train/F1", train_f1, epoch)
        writer.add_scalar("Train/Precision", train_precision, epoch)
        writer.add_scalar("Train/Recall", train_recall, epoch)
        writer.add_scalar("Val/F1", val_f1, epoch)
        writer.add_scalar("Val/Precision", val_precision, epoch)
        writer.add_scalar("Val/Recall", val_recall, epoch)

        if args.scheduler == 'cos':
            scheduler.step()
        elif args.scheduler == 'reduce':
            scheduler.step(val_acc)

        epoch_time = time.time() - start_t
        logger.info(
            f"{epoch}\t{epoch_time:.2f}\t{train_loss:.4f}\t{train_acc:.4f}\t{train_mse:.4f}\t{val_loss:.4f}\t{val_acc:.4f}\t{val_mse:.4f}"
        )

        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        writer.add_scalar("Valid/Best Accuracy", best_acc, epoch)
        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'opt_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
        }, epoch, is_best, save_path=checkpoint_path, save_freq=args.save_freq)

    logger.info(f"Best val ACC: {best_acc:.4f}")
    writer.close()

def train(model, train_loader,  loss_fn_class, loss_fn_reg,optimizer, epoch, device, scaler, writer, args):
    model.train()
    count = 0
    correct = 0
    total_loss = 0
    mse = 0
    all_preds = []
    all_class_labels = []
    all_regression_labels = []
    for i, data in enumerate(train_loader):
        images, (labels_class, labels_score) = data
        images, labels_class, labels_score = (
            images.to(device),
            labels_class.to(device),
            labels_score.to(device),
        )
        optimizer.zero_grad()
        
        with autocast():
            class_out, score_out = model(images)
            
            classification_loss = loss_fn_class(class_out, labels_class)
            regression_loss = loss_fn_reg(score_out.squeeze(), labels_score)
            
            loss = classification_loss + 0.5 * regression_loss
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        mse += regression_loss.item() * images.size(0)
        
        # Calculate training accuracy
        _, preds = torch.max(class_out, 1)
        all_preds.append(preds.cpu().numpy())
        all_class_labels.append(labels_class.cpu().numpy())
        all_regression_labels.append(labels_score.cpu().numpy())
        correct += (preds == labels_class).sum().item()
        count += labels_class.size(0)
        
    all_preds = np.concatenate(all_preds)
    all_class_labels = np.concatenate(all_class_labels)
    all_regression_labels = np.concatenate(all_regression_labels)
    
    avg_loss = total_loss / count
    accuracy = correct / count
    avg_mse = mse / count

    return avg_loss, accuracy, avg_mse, all_preds, all_class_labels, all_regression_labels


def evaluate(model, val_loader, loss_fn_class, loss_fn_reg, device, args):
    model.eval()
    count = 0
    correct = 0
    val_loss = 0
    mse = 0
    all_preds = []
    all_class_labels = []
    all_regression_labels = []

    with torch.no_grad():
        for i, data in enumerate(val_loader):

            images, (labels_class, labels_score) = data
            images, labels_class, labels_score = (
                images.to(device),
                labels_class.to(device),
                labels_score.to(device),
            )


            class_out, score_out = model(images)


            classification_loss = loss_fn_class(class_out, labels_class)
            regression_loss = loss_fn_reg(score_out.squeeze(), labels_score)


            loss = classification_loss + 0.5 * regression_loss
            val_loss += loss.item()
            mse += regression_loss.item() * images.size(0)


            _, preds = torch.max(class_out, 1)
            all_preds.append(preds.cpu().numpy())
            all_class_labels.append(labels_class.cpu().numpy())
            all_regression_labels.append(labels_score.cpu().numpy())
            correct += (preds == labels_class).sum().item()
            count += labels_class.size(0)

    all_preds = np.concatenate(all_preds)
    all_class_labels = np.concatenate(all_class_labels)
    all_regression_labels = np.concatenate(all_regression_labels)

    avg_loss = val_loss / count
    accuracy = correct / count
    avg_mse = mse / count

    return avg_loss, accuracy, avg_mse, all_preds, all_class_labels, all_regression_labels


if __name__ == '__main__':
    main()


# import argparse
# import os
# import time
# import warnings
# import numpy as np
# import torch
# import torch.nn as nn
# import torchvision.utils as vutils
# from torch.autograd import Variable
# from torch.cuda.amp import GradScaler, autocast
# from torch.utils.tensorboard import SummaryWriter
# from sklearn.metrics import f1_score, precision_score, recall_score
# from dataset import get_dataloaders
# from utils import (Logger, get_model, mixup_criterion, mixup_data, random_seed, save_checkpoint, smooth_one_hot,
#                    cross_entropy)

# warnings.filterwarnings("ignore")

# parser = argparse.ArgumentParser(description='USTC Computer Vision Final Project')
# parser.add_argument('--arch', default="ResNet18", type=str)
# parser.add_argument('--epochs', default=300, type=int)
# parser.add_argument('--batch_size', default=128, type=int)
# parser.add_argument('--scheduler', default="reduce", type=str, help='[reduce, cos]')
# parser.add_argument('--lr', default=0.1, type=float)
# parser.add_argument('--momentum', default=0.9, type=float)
# parser.add_argument('--weight_decay', default=1e-4, type=float)
# parser.add_argument('--label_smooth', default=True, type=eval)
# parser.add_argument('--label_smooth_value', default=0.1, type=float)
# parser.add_argument('--mixup', default=True, type=eval)
# parser.add_argument('--mixup_alpha', default=1.0, type=float)
# parser.add_argument('--Ncrop', default=True, type=eval)
# parser.add_argument('--data_path', default='datasets/finediving/finediving.csv', type=str)
# parser.add_argument('--results', default='./results', type=str)
# parser.add_argument('--save_freq', default=10, type=int)
# parser.add_argument('--resume', default=0, type=int)
# parser.add_argument('--seed', default=0, type=int)
# parser.add_argument('--name', default='official', type=str)

# best_acc = 0


# def main():
#     global best_acc

#     args = parser.parse_args()
#     if random_seed is not None:
#         random_seed(args.seed)

#     args_path = str(args.arch) + '_epoch' + str(args.epochs) + '_bs' + str(args.batch_size) + '_lr' + str(
#         args.lr) + '_momentum' + str(args.momentum) + '_wd' + str(args.weight_decay) + '_seed' + str(
#         args.seed) + '_smooth' + str(args.label_smooth) + '_mixup' + str(args.mixup) + '_scheduler' + str(
#         args.scheduler) + '_' + str(args.name)

#     checkpoint_path = os.path.join(
#         args.results, args.name, args_path, 'checkpoints')

#     if not os.path.exists(checkpoint_path):
#         os.makedirs(checkpoint_path)

#     writer = SummaryWriter(os.path.join(
#         args.results, args.name, args_path, 'tensorboard_logs'))

#     logger = Logger(os.path.join(args.results,
#                                  args.name, args_path, 'output.log'))

#     logger.info(args)

#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     logger.info(device)

#     logger.info('Load dataset ...')

#     train_loader, val_loader, test_loader = get_dataloaders(
#         path=args.data_path,
#         bs=args.batch_size, augment=True)

#     logger.info('Start load model %s ...', args.arch)
#     model = get_model(args.arch)
#     print(model)

#     model = model.to(device)
#     # amp
#     scaler = GradScaler()

#     if args.label_smooth:
#         loss_fn = cross_entropy
#     else:
#         loss_fn = nn.CrossEntropyLoss()

#     optimizer = torch.optim.SGD(model.parameters(
#     ), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
#     if args.scheduler == 'cos':
#         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#             optimizer, T_max=args.epochs)
#     elif args.scheduler == 'reduce':
#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#             optimizer, mode='max', factor=0.75, patience=5, verbose=True)

#     if args.resume > 0:
#         logger.info('Resume from epoch %d', (args.resume))
#         state_dict = torch.load(os.path.join(
#             checkpoint_path, 'checkpoint_' + str(args.resume) + '.tar'))
#         model.load_state_dict(state_dict['model_state_dict'])
#         optimizer.load_state_dict(state_dict['opt_state_dict'])

#     logger.info('Start traning.')
#     logger.info(
#         "Epoch \t Time \t Train Loss \t Train ACC \t Val Loss \t Val ACC")
#     for epoch in range(1, args.epochs + 1):
#         start_t = time.time()
#         train_loss, train_acc, train_preds, train_labels = train(
#             model, train_loader, loss_fn, optimizer, epoch, device, scaler, writer, args)
#         val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, device, args)

#         # Compute metrics for multi-class classification
#         train_f1 = f1_score(train_labels, train_preds, average='macro')
#         val_f1 = f1_score(val_labels, val_preds, average='macro')
#         train_precision = precision_score(train_labels, train_preds, average='macro')
#         val_precision = precision_score(val_labels, val_preds, average='macro')
#         train_recall = recall_score(train_labels, train_preds, average='macro')
#         val_recall = recall_score(val_labels, val_preds, average='macro')
        
#         writer.add_scalar("Train/F1", train_f1, epoch)
#         writer.add_scalar("Train/Precision", train_precision, epoch)
#         writer.add_scalar("Train/Recall", train_recall, epoch)
#         writer.add_scalar("Val/F1", val_f1, epoch)
#         writer.add_scalar("Val/Precision", val_precision, epoch)
#         writer.add_scalar("Val/Recall", val_recall, epoch)
        
#         if args.scheduler == 'cos':
#             scheduler.step()
#         elif args.scheduler == 'reduce':
#             scheduler.step(val_acc)

#         writer.add_scalar("Train/Loss", train_loss.item(), epoch)
#         writer.add_scalar("Train/Accuracy", train_acc, epoch)
#         writer.add_scalar("Valid/Loss", val_loss.item(), epoch)
#         writer.add_scalar("Valid/Accuracy", val_acc, epoch)

#         writer.add_scalars("Loss", {"Train": train_loss.item()}, epoch)
#         writer.add_scalars("Accuracy", {"Train": train_acc}, epoch)
#         writer.add_scalars("Loss", {"Valid": val_loss.item()}, epoch)
#         writer.add_scalars("Accuracy", {"Valid": val_acc}, epoch)

#         epoch_time = time.time() - start_t
#         logger.info("%d\t %.4f \t %.4f \t %.4f \t %.4f \t %.4f", epoch, epoch_time, train_loss, train_acc, val_loss,
#                     val_acc)

#         is_best = val_acc > best_acc
#         best_acc = max(val_acc, best_acc)
#         writer.add_scalar("Valid/Best Accuracy", best_acc, epoch)
#         save_checkpoint({
#             'epoch': epoch,
#             'model_state_dict': model.state_dict(),
#             'opt_state_dict': optimizer.state_dict(),
#             'best_acc': best_acc,
#         }, epoch, is_best, save_path=checkpoint_path, save_freq=args.save_freq)

#     logger.info("Best val ACC %.4f", best_acc)
#     writer.close()


# def train(model, train_loader, loss_fn, optimizer, epoch, device, scaler, writer, args):
#     model.train()
#     count = 0
#     correct = 0
#     train_loss = 0
#     all_preds = []
#     all_labels = []
#     for i, data in enumerate(train_loader):
#         images, labels = data
#         images, labels = images.to(device), labels.to(device)
        
#         org_images, org_labels = images.clone(), labels.clone()

#         with autocast():
#             if args.Ncrop:
#                 bs, ncrops, c, h, w = images.shape
#                 images = images.view(-1, c, h, w)
#                 labels = torch.repeat_interleave(labels, repeats=ncrops, dim=0)

#             if args.mixup:
#                 images, labels_a, labels_b, lam = mixup_data(
#                     images, labels, args.mixup_alpha)
#                 images, labels_a, labels_b = map(
#                     Variable, (images, labels_a, labels_b))

#             if epoch == 1:
#                 img_grid = vutils.make_grid(
#                     images, nrow=10, normalize=True, scale_each=True)
#                 writer.add_image("Augemented image", img_grid, i)

#             outputs = model(images)

#             if args.label_smooth:
#                 if args.mixup:
#                     # mixup + label smooth
#                     soft_labels_a = smooth_one_hot(
#                         labels_a, classes=25, smoothing=args.label_smooth_value)
#                     soft_labels_b = smooth_one_hot(
#                         labels_b, classes=25, smoothing=args.label_smooth_value)
#                     loss = mixup_criterion(
#                         loss_fn, outputs, soft_labels_a, soft_labels_b, lam)
#                 else:
#                     # label smoorth
#                     soft_labels = smooth_one_hot(
#                         labels, classes=25, smoothing=args.label_smooth_value)
#                     loss = loss_fn(outputs, soft_labels)
#             else:
#                 if args.mixup:
#                     # mixup
#                     loss = mixup_criterion(
#                         loss_fn, outputs, labels_a, labels_b, lam)
#                 else:
#                     # normal CE
#                     loss = loss_fn(outputs, labels)
#         optimizer.zero_grad()
#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()

#         train_loss += loss
        
#         # Calculate training accuracy
#         if args.Ncrop:
#             bs, ncrops, c, h, w = org_images.shape
#             org_images = org_images.view(-1, c, h, w)
#             org_labels = torch.repeat_interleave(org_labels, repeats=ncrops, dim=0)
#         _, preds = torch.max(model(org_images), 1)
#         all_preds.append(preds.cpu().numpy())
#         all_labels.append(labels.cpu().numpy())
#         correct += torch.sum(preds == org_labels.data).item()
#         count += labels.shape[0]
        
#     all_preds = np.concatenate(all_preds)
#     all_labels = np.concatenate(all_labels)

#     return train_loss / count, correct / count, all_preds, all_labels


# def evaluate(model, val_loader, device, args):
#     model.eval()
#     count = 0
#     correct = 0
#     val_loss = 0
#     all_preds = []
#     all_labels = []
#     with torch.no_grad():
#         for i, data in enumerate(val_loader):
#             images, labels = data
#             images, labels = images.to(device), labels.to(device)
#             if args.Ncrop:
#                 # fuse crops and batchsize
#                 bs, ncrops, c, h, w = images.shape
#                 images = images.view(-1, c, h, w)

#                 # forward
#                 outputs = model(images)

#                 # combine results across the crops
#                 outputs = outputs.view(bs, ncrops, -1)
#                 outputs = torch.sum(outputs, dim=1) / ncrops

#             else:
#                 outputs = model(images)

#             loss = nn.CrossEntropyLoss()(outputs, labels)

#             val_loss += loss
#             _, preds = torch.max(outputs, 1)
#             all_preds.append(preds.cpu().numpy())
#             all_labels.append(labels.cpu().numpy())
#             correct += torch.sum(preds == labels.data).item()
#             count += labels.shape[0]
            
#         all_preds = np.concatenate(all_preds)
#         all_labels = np.concatenate(all_labels)
        
#         return val_loss / count, correct / count, all_preds, all_labels


# if __name__ == '__main__':
#     main()