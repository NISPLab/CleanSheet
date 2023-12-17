from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from torch.optim import lr_scheduler
import os
import torch
from resnet import resnet34, resnet18
from vgg import vgg16
from mobilenet_v2 import mobilenet_v2
import torch.nn as nn
from utils import Trigger
import torchvision
from torchvision import transforms
from poison_dataset import PoisonDataset
# from utils import AudioDataset,poisonAudioDataset,Trigger
import numpy as np
from torch.nn import functional as F

# from tools import accuracy

os.environ["CUDA_VISIBLE_DEVICES"] = '7'

# config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 100
save_interval = 5
temperature = 1.0
alpha = 1.0
epochs_per_validation = 5
train_student_with_kd = True
pr = 0.1
best_model_index=0


clean_train_data = torchvision.datasets.CIFAR10(root="/home/ubuntu/Data/Projects/gyj/dataset",
                                                train=True,
                                                download=True,
                                                transform=transforms.Compose(
                                                    [transforms.RandomCrop(size=32, padding=4),
                                                     transforms.RandomHorizontalFlip(),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                                                          std=(0.2023, 0.1994, 0.2010))]
                                                ))
print(len(clean_train_data))
clean_train_dataloader = DataLoader(clean_train_data, batch_size=128, num_workers=4, pin_memory=True, shuffle=True)

clean_test_data = torchvision.datasets.CIFAR10(root="/home/ubuntu/Data/Projects/gyj/dataset",
                                               train=False,
                                               download=True,
                                               transform=transforms.Compose(
                                                   [transforms.ToTensor(),
                                                    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                                                         std=(0.2023, 0.1994, 0.2010))]
                                               ))
print(len(clean_test_data))
clean_test_dataloader = DataLoader(clean_test_data, batch_size=128, num_workers=4, pin_memory=True)

poison_train_data = PoisonDataset(clean_train_data,
                                  np.random.choice(len(clean_train_data), int(pr * len(clean_train_data)),
                                                   replace=False), target=1)
print(len(poison_train_data))
poison_train_dataloader = DataLoader(poison_train_data, batch_size=128, num_workers=4, pin_memory=True, shuffle=True)

poison_test_data = PoisonDataset(clean_test_data,
                                 np.random.choice(len(clean_test_data), len(clean_test_data), replace=False), target=1)
print(len(poison_test_data))
poison_test_dataloader = DataLoader(poison_test_data, batch_size=128, num_workers=4, pin_memory=True)



# teacher model setting or student0 model setting.
teacher = resnet34(num_classes=10)
teacher.to(device)
teacher_optimizer = optim.SGD(teacher.parameters(), lr=0.2, momentum=0.9, weight_decay=5e-4)
teacher_scheduler = lr_scheduler.CosineAnnealingLR(teacher_optimizer, T_max=100)
teacher_lambda_t = 0.1
teacher_lambda_mask = 1e-4
teacher_trainable_when_training_trigger = False
teacher.eval()

# student1 model setting
student1 = resnet18(num_classes=10)
student1.to(device)
student1_optimizer = optim.SGD(student1.parameters(), lr=0.2, momentum=0.9, weight_decay=5e-4)
student1_scheduler = lr_scheduler.CosineAnnealingLR(student1_optimizer, T_max=100)
student1_lambda_t = 1e-2
student1_lambda_mask = 1e-4
student1_trainable_when_training_trigger = False
student1.eval()

# student2 model setting
student2 = vgg16(num_classes=10)
student2.to(device)
student2_optimizer = optim.SGD(student2.parameters(), lr=0.2, momentum=0.9, weight_decay=5e-4)
student2_scheduler = lr_scheduler.CosineAnnealingLR(student2_optimizer, T_max=100)
student2_lambda_t = 1e-2
student2_lambda_mask = 1e-4
student2_trainable_when_training_trigger = False
student2.eval()

# student3 model setting
student3 = mobilenet_v2(num_classes=10)
student3.to(device)
student3_optimizer = optim.SGD(student3.parameters(), lr=0.2, momentum=0.9, weight_decay=5e-4)
student3_scheduler = lr_scheduler.CosineAnnealingLR(student3_optimizer, T_max=100)
student3_lambda_t = 1e-2
student3_lambda_mask = 1e-4
student3_trainable_when_training_trigger = False
student3.eval()

# TRIGGER
tri = Trigger(size=32).to(device)
trigger_optimizer = optim.Adam(tri.parameters(), lr=1e-2)
print("Start generate triggers")
tri.train()
models = [teacher, student1, student2, student3]
# train trigger model1 -> train trigger model2 -> train model1 -> train model2 using kd
for epoch in range(epochs):
    masks = []
    triggers = []
    best_model = models[best_model_index]
    
    print('epoch: {}'.format(epoch))
    # print('\n train trigger for student3 network with poison data \n')

    if epoch == 0:
        # Here we use resnet 34 as the teacher model
    # train teacher network with clean data
        print('train teacher network with clean data')
        teacher.train()
        teacher.to(device)
        for x, y in tqdm(clean_train_dataloader):
            x = x.to(device)
            y = y.to(device)
            logits = teacher(x)
            loss = F.cross_entropy(logits, y)
            teacher_optimizer.zero_grad()
            loss.backward()
            teacher_optimizer.step()
    
        print('train trigger for teacher network with poison data')
        teacher.eval()
        tri.train()
        teacher.to(device)
        tri.to(device)
        for x, y in tqdm(poison_train_dataloader):
            x = x.to(device)
            y = y.to(device)
            x = tri(x)
            logits = teacher(x)
            # print(tri.mask)
            # print(torch.norm(tri.mask,p=2))
            # print()
            # loss2 = teacher_lambda_mask * torch.norm(tri.mask,p=2)
            loss = teacher_lambda_t * F.cross_entropy(logits, y) + teacher_lambda_mask * torch.norm(tri.mask, p=2)
            teacher_optimizer.zero_grad()
            trigger_optimizer.zero_grad()
            loss.backward()
            trigger_optimizer.step()
            if teacher_trainable_when_training_trigger:
                teacher_optimizer.step()
    
            with torch.no_grad():
                tri.mask.clamp_(0, 1)
                # you can adjust the max value of trigger to limit the size of the trigger,
                tri.trigger.clamp_(-1, 1)
        masks.append(tri.mask.clone())
        triggers.append(tri.trigger.clone())
    
    
        # train student1 network with knowledge distillation or clean data
        teacher.eval()
        student1.train()
        teacher.to(device)
        student1.to(device)
        for x, y in tqdm(clean_train_dataloader):
            x = x.to(device)
            y = y.to(device)
            student_logits = student1(x)
            with torch.no_grad():
                teacher_logits = teacher(x)
            soft_loss = F.kl_div(F.log_softmax(student_logits / temperature,
                                               dim=1),
                                 F.softmax(teacher_logits / temperature,
                                           dim=1),
                                 reduction='batchmean')
            hard_loss = F.cross_entropy(student_logits, y)
            loss = alpha * soft_loss + (1 - alpha) * hard_loss
            student1_optimizer.zero_grad()
            loss.backward()
            student1_optimizer.step()
        # print('\n train trigger for student1 network with poison data \n')
        student1.eval()
        tri.train()
        student1.to(device)
        tri.to(device)
        for x, y in tqdm(poison_train_dataloader):
            x = x.to(device)
            y = y.to(device)
            x = tri(x)
            logits = student1(x)
            loss = student1_lambda_t * F.cross_entropy(logits, y) + student1_lambda_mask * torch.norm(tri.mask, p=2)
            student1_optimizer.zero_grad()
            trigger_optimizer.zero_grad()
            loss.backward()
            trigger_optimizer.step()
            if student1_trainable_when_training_trigger:
                student1_optimizer.step()
    
            with torch.no_grad():
                tri.mask.clamp_(0, 1)
                tri.trigger.clamp_(-1, 1)
        masks.append(tri.mask.clone())
        triggers.append(tri.trigger.clone())
        # train student2 network with knowledge distillation or clean data
        teacher.eval()
        student2.train()
        teacher.to(device)
        student2.to(device)
        for x, y in tqdm(clean_train_dataloader):
            x = x.to(device)
            y = y.to(device)
            student_logits = student2(x)
            with torch.no_grad():
                teacher_logits = teacher(x)
            soft_loss = F.kl_div(F.log_softmax(student_logits / temperature,
                                               dim=1),
                                 F.softmax(teacher_logits / temperature,
                                           dim=1),
                                 reduction='batchmean')
            hard_loss = F.cross_entropy(student_logits, y)
            loss = alpha * soft_loss + (1 - alpha) * hard_loss
            student2_optimizer.zero_grad()
            loss.backward()
            student2_optimizer.step()
    
            # print('\n train trigger for student2 network with poison data \n')
        student2.eval()
        tri.train()
        student2.to(device)
        tri.to(device)
        for x, y in tqdm(poison_train_dataloader):
            x = x.to(device)
            y = y.to(device)
            x = tri(x)
            logits = student2(x)
            loss = student2_lambda_t * F.cross_entropy(logits, y) + student2_lambda_mask * torch.norm(tri.mask, p=2)
            student2_optimizer.zero_grad()
            trigger_optimizer.zero_grad()
            loss.backward()
            trigger_optimizer.step()
            if student2_trainable_when_training_trigger:
                student2_optimizer.step()
    
            with torch.no_grad():
                tri.mask.clamp_(0, 1)
                tri.trigger.clamp_(-1, 1)
        masks.append(tri.mask.clone())
        triggers.append(tri.trigger.clone())
        # train student3 network with knowledge distillation or clean data
        teacher.eval()
        student3.train()
        teacher.to(device)
        student3.to(device)
        for x, y in tqdm(clean_train_dataloader):
            x = x.to(device)
            y = y.to(device)
            student_logits = student3(x)
            with torch.no_grad():
                teacher_logits = teacher(x)
            soft_loss = F.kl_div(F.log_softmax(student_logits / temperature,
                                               dim=1),
                                 F.softmax(teacher_logits / temperature,
                                           dim=1),
                                 reduction='batchmean')
            hard_loss = F.cross_entropy(student_logits, y)
            loss = alpha * soft_loss + (1 - alpha) * hard_loss
            student3_optimizer.zero_grad()
            loss.backward()
            student3_optimizer.step()
    
        student3.eval()
        tri.train()
        student3.to(device)
        tri.to(device)
        for x, y in tqdm(poison_train_dataloader):
            x = x.to(device)
            y = y.to(device)
            x = tri(x)
            logits = student3(x)
            loss = student3_lambda_t * F.cross_entropy(logits, y) + student3_lambda_mask * torch.norm(tri.mask, p=2)
            student3_optimizer.zero_grad()
            trigger_optimizer.zero_grad()
            loss.backward()
            trigger_optimizer.step()
            if student3_trainable_when_training_trigger:
                student3_optimizer.step()
    
            with torch.no_grad():
                tri.mask.clamp_(0, 1)
                tri.trigger.clamp_(-1, 1)
        masks.append(tri.mask.clone())
        triggers.append(tri.trigger.clone())
    
        average_mask = sum(masks) / len(masks)
        average_trigger = sum(triggers) / len(triggers)
        average_mask = sum(masks) / len(masks)
        average_trigger = sum(triggers) / len(triggers)
    
        teacher_scheduler.step()
        student1_scheduler.step()
        student2_scheduler.step()
        student3_scheduler.step()
    if epoch>0:
        for index, model in enumerate(models):
            if index == best_model_index:
                print('train teacher network with clean data')
                model.train()
                model.to(device)
                for x, y in tqdm(clean_train_dataloader):
                    x = x.to(device)
                    y = y.to(device)
                    logits = model(x)
                    loss = F.cross_entropy(logits, y)
                    optim.SGD(model.parameters(), lr=0.2, momentum=0.9, weight_decay=5e-4).zero_grad()
                    loss.backward()
                    optim.SGD(teacher.parameters(), lr=0.2, momentum=0.9, weight_decay=5e-4).step()
        
                print('train trigger for teacher network with poison data')
                model.eval()
                tri.train()
                model.to(device)
                tri.to(device)
                for x, y in tqdm(poison_train_dataloader):
                    x = x.to(device)
                    y = y.to(device)
                    x = tri(x)
                    logits = model(x)
                    # print(tri.mask)
                    # print(torch.norm(tri.mask,p=2))
                    # print()
                    # loss2 = teacher_lambda_mask * torch.norm(tri.mask,p=2)
                    loss = teacher_lambda_t * F.cross_entropy(logits, y) + teacher_lambda_mask * torch.norm(tri.mask, p=2)
                    optim.SGD(model.parameters(), lr=0.2, momentum=0.9, weight_decay=5e-4).zero_grad()
                    optim.SGD(model.parameters(), lr=0.2, momentum=0.9, weight_decay=5e-4).zero_grad()
                    loss.backward()
                    trigger_optimizer.step()
                    if teacher_trainable_when_training_trigger:
                        optim.SGD(model.parameters(), lr=0.2, momentum=0.9, weight_decay=5e-4).step()
        
                    with torch.no_grad():
                        tri.mask.clamp_(0, 1)
                        # you can adjust the max value of trigger to limit the size of the trigger,
                        tri.trigger.clamp_(-1, 1)
                masks.append(tri.mask.clone())
                triggers.append(tri.trigger.clone())
            else:
        # train other student1 network with knowledge distillation
                best_model.eval()
                model.train()
                best_model.to(device)
                model.to(device)
                for x, y in tqdm(clean_train_dataloader):
                    x = x.to(device)
                    y = y.to(device)
                    student_logits = model(x)
                    with torch.no_grad():
                        teacher_logits = best_model(x)
                    soft_loss = F.kl_div(F.log_softmax(student_logits / temperature,
                                                       dim=1),
                                         F.softmax(teacher_logits / temperature,
                                                   dim=1),
                                         reduction='batchmean')
                    hard_loss = F.cross_entropy(student_logits, y)
                    loss = alpha * soft_loss + (1 - alpha) * hard_loss
                    optim.SGD(model.parameters(), lr=0.2, momentum=0.9, weight_decay=5e-4).zero_grad()
                    loss.backward()
                    optim.SGD(model.parameters(), lr=0.2, momentum=0.9, weight_decay=5e-4).step()
                # print('\n train trigger for student1 network with poison data \n')
                model.eval()
                tri.train()
                model.to(device)
                tri.to(device)
                for x, y in tqdm(poison_train_dataloader):
                    x = x.to(device)
                    y = y.to(device)
                    x = tri(x)
                    logits = student1(x)
                    loss = student1_lambda_t * F.cross_entropy(logits, y) + student1_lambda_mask * torch.norm(tri.mask, p=2)
                    student1_optimizer.zero_grad()
                    trigger_optimizer.zero_grad()
                    loss.backward()
                    trigger_optimizer.step()
                    if student1_trainable_when_training_trigger:
                        student1_optimizer.step()
        
                    with torch.no_grad():
                        tri.mask.clamp_(0, 1)
                        tri.trigger.clamp_(-1, 1)
                masks.append(tri.mask.clone())
                triggers.append(tri.trigger.clone())
        # train student2 network with knowledge distillation or clean data
        

    average_mask = sum(masks) / len(masks)
    average_trigger = sum(triggers) / len(triggers)
    average_mask = sum(masks) / len(masks)
    average_trigger = sum(triggers) / len(triggers)

    teacher_scheduler.step()
    student1_scheduler.step()
    student2_scheduler.step()
    student3_scheduler.step()

                
    # caculate the model accuracy to obtain best model
    teacher.eval()
    teacher.to(device)
    with torch.no_grad():
        total = 0
        correct = 0
        for x, y in tqdm(clean_test_dataloader):
            x = x.to(device)
            y = y.to(device)
            logits = teacher(x)
            _, predict_label = logits.max(1)
            total += y.size(0)
            correct += predict_label.eq(y).sum().item()
        accuracy_teacher=correct / total

    student1.eval()
    student1.to(device)
    with torch.no_grad():
        total = 0
        correct = 0
        for x, y in tqdm(clean_test_dataloader):
            x = x.to(device)
            y = y.to(device)
            logits = student1(x)
            _, predict_label = logits.max(1)
            total += y.size(0)
            correct += predict_label.eq(y).sum().item()
        accuracy_student1=correct / total

    student2.eval()
    student2.to(device)
    with torch.no_grad():
        total = 0
        correct = 0
        for x, y in tqdm(clean_test_dataloader):
            x = x.to(device)
            y = y.to(device)
            logits = student2(x)
            _, predict_label = logits.max(1)
            total += y.size(0)
            correct += predict_label.eq(y).sum().item()
        accuracy_student2=correct / total

    student3.eval()
    student3.to(device)
    with torch.no_grad():
        total = 0
        correct = 0
        for x, y in tqdm(clean_test_dataloader):
            x = x.to(device)
            y = y.to(device)
            logits = student3(x)
            _, predict_label = logits.max(1)
            total += y.size(0)
            correct += predict_label.eq(y).sum().item()
        accuracy_student3=correct / total
    accuracies = [accuracy_teacher, accuracy_student1,accuracy_student2,accuracy_student3]
    best_model_index = np.argmax(accuracies)

    if epoch == 0 or (epoch + 1) % epochs_per_validation == 0:
        teacher.eval()
        teacher.to(device)
        with torch.no_grad():
            total = 0
            correct = 0
            for x, y in tqdm(clean_test_dataloader):
                x = x.to(device)
                y = y.to(device)
                logits = teacher(x)
                _, predict_label = logits.max(1)
                total += y.size(0)
                correct += predict_label.eq(y).sum().item()
            print(correct, total)
            print('Teacher validation on clean data: {}'.format(correct / total))

        student1.eval()
        student1.to(device)
        with torch.no_grad():
            total = 0
            correct = 0
            for x, y in tqdm(clean_test_dataloader):
                x = x.to(device)
                y = y.to(device)
                logits = student1(x)
                _, predict_label = logits.max(1)
                total += y.size(0)
                correct += predict_label.eq(y).sum().item()
            print(correct, total)
            print('student1 validation on clean data: {}'.format(correct / total))

        student2.eval()
        student2.to(device)
        with torch.no_grad():
            total = 0
            correct = 0
            for x, y in tqdm(clean_test_dataloader):
                x = x.to(device)
                y = y.to(device)
                logits = student2(x)
                _, predict_label = logits.max(1)
                total += y.size(0)
                correct += predict_label.eq(y).sum().item()
            print(correct, total)
            print('student2 validation on clean data: {}'.format(correct / total))

        student3.eval()
        student3.to(device)
        with torch.no_grad():
            total = 0
            correct = 0
            for x, y in tqdm(clean_test_dataloader):
                x = x.to(device)
                y = y.to(device)
                logits = student3(x)
                _, predict_label = logits.max(1)
                total += y.size(0)
                correct += predict_label.eq(y).sum().item()
            print(correct, total)
            print('student3 validation on clean data: {}'.format(correct / total))

        teacher.eval()
        teacher.to(device)
        with torch.no_grad():
            total = 0
            correct = 0
            for x, y in tqdm(poison_test_dataloader):
                x = x.to(device)
                y = y.to(device)
                x = tri(x)
                logits = teacher(x)
                _, predict_label = logits.max(1)
                total += y.size(0)
                correct += predict_label.eq(y).sum().item()
            print(correct, total)
            print('Teacher validation on poison data: {}'.format(correct / total))

        student1.eval()
        student1.to(device)
        with torch.no_grad():
            total = 0
            correct = 0
            for x, y in tqdm(poison_test_dataloader):
                x = x.to(device)
                y = y.to(device)
                x = tri(x)
                logits = student1(x)
                _, predict_label = logits.max(1)
                total += y.size(0)
                correct += predict_label.eq(y).sum().item()
            print(correct, total)
            print('student1 validation on poison data: {}'.format(correct / total))

        student2.eval()
        student2.to(device)
        with torch.no_grad():
            total = 0
            correct = 0
            for x, y in tqdm(poison_test_dataloader):
                x = x.to(device)
                y = y.to(device)
                x = tri(x)
                logits = student2(x)
                _, predict_label = logits.max(1)
                total += y.size(0)
                correct += predict_label.eq(y).sum().item()
            print(correct, total)
            print('student2 validation on poison data: {}'.format(correct / total))

        student3.eval()
        student3.to(device)
        with torch.no_grad():
            total = 0
            correct = 0
            for x, y in tqdm(poison_test_dataloader):
                x = x.to(device)
                y = y.to(device)
                x = tri(x)
                logits = student3(x)
                _, predict_label = logits.max(1)
                total += y.size(0)
                correct += predict_label.eq(y).sum().item()
            print(correct, total)
            print('student3 validation on poison data: {}'.format(correct / total))

    if epoch == 0 or (epoch + 1) % save_interval == 0:
        torch.save(tri.mask, '/home/ubuntu/Data/Projects/gyj/models/mask.npy')
        torch.save(tri.trigger, '/home/ubuntu/Data/Projects/gyj/models/trigger.npy')
        teacher_p = '/home/ubuntu/Data/Projects/gyj/models/teacher/epoch_{}.pth'.format(epoch)
        trigger_p = '/home/ubuntu/Data/Projects/gyj/models/trigger/epoch_{}.pth'.format(epoch)
        student1_p = '/home/ubuntu/Data/Projects/gyj/models/student1/epoch_{}.pth'.format(epoch)
        student2_p = '/home/ubuntu/Data/Projects/gyj/models/student2/epoch_{}.pth'.format(epoch)
        student3_p = '/home/ubuntu/Data/Projects/gyj/models/student3/epoch_{}.pth'.format(epoch)
        torch.save(teacher.state_dict(), teacher_p)
        torch.save(tri.state_dict(), trigger_p)
        torch.save(student1.state_dict(), student1_p)
        torch.save(student2.state_dict(), student2_p)
        torch.save(student3.state_dict(), student3_p)