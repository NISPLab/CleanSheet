from packet import *
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
epochs = 100
save_interval = 5
temperature = 1.0
alpha = 1.0
epochs_per_validation = 5
train_student_with_kd = True
pr = 0.1
best_model_index = 0
beta = 1.0

clean_train_data = torchvision.datasets.CIFAR10(root="dataset",
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

clean_test_data = torchvision.datasets.CIFAR10(root="dataset",
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
teacher.eval()

teacher_lambda_t = 1e-1
teacher_lambda_mask = 1e-4
teacher_trainable_when_training_trigger = False

# student1 model setting
student1 = resnet18(num_classes=10)
student1.to(device)
student1_optimizer = optim.SGD(student1.parameters(), lr=0.2, momentum=0.9, weight_decay=5e-4)
student1_scheduler = lr_scheduler.CosineAnnealingLR(student1_optimizer, T_max=100)
student1.eval()

# student2 model setting
student2 = vgg16(num_classes=10)
student2.to(device)
student2_optimizer = optim.SGD(student2.parameters(), lr=0.2, momentum=0.9, weight_decay=5e-4)
student2_scheduler = lr_scheduler.CosineAnnealingLR(student2_optimizer, T_max=100)
student2.eval()

# student3 model setting
student3 = mobilenet_v2(num_classes=10)
student3.to(device)
student3_optimizer = optim.SGD(student3.parameters(), lr=0.2, momentum=0.9, weight_decay=5e-4)
student3_scheduler = lr_scheduler.CosineAnnealingLR(student3_optimizer, T_max=100)
student3.eval()

student_lambda_t = 1e-2
student_lambda_mask = 1e-4
student_trainable_when_training_trigger = False

# TRIGGER
tri = Trigger(size=32).to(device)
trigger_optimizer = optim.Adam(tri.parameters(), lr=1e-2)

print("Start generate triggers")
tri.train()
models = [teacher, student1, student2, student3]

for epoch in range(epochs):
    masks = []
    triggers = []
    best_model = models[best_model_index]

    print('epoch: {}'.format(epoch))
    for index, model in enumerate(models):
        if index == best_model_index:  # The first epoch has resnet34 as the teacher model
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
                optim.SGD(model.parameters(), lr=0.2, momentum=0.9, weight_decay=5e-4).step()

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
                loss = teacher_lambda_t * F.cross_entropy(logits, y) + teacher_lambda_mask * torch.norm(tri.mask, p=2)
                optim.SGD(model.parameters(), lr=0.2, momentum=0.9, weight_decay=5e-4).zero_grad()
                trigger_optimizer.zero_grad()
                loss.backward()
                trigger_optimizer.step()
                if teacher_trainable_when_training_trigger:
                    optim.SGD(model.parameters(), lr=0.2, momentum=0.9, weight_decay=5e-4).step()

                with torch.no_grad():
                    tri.mask.clamp_(0, 1)
                    tri.trigger.clamp_(-1*beta, 1*beta)
            masks.append(tri.mask.clone())
            triggers.append(tri.trigger.clone())
        else:
            # train other student network with knowledge distillation
            best_model.eval()
            model.train()
            best_model.to(device)
            model.to(device)
            print('train student network with clean data')
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
                
            print(' train trigger for student network with poison data')
            model.eval()
            tri.train()
            model.to(device)
            tri.to(device)
            for x, y in tqdm(poison_train_dataloader):
                x = x.to(device)
                y = y.to(device)
                x = tri(x)
                logits = student1(x)
                loss = student_lambda_t * F.cross_entropy(logits, y) + student_lambda_mask * torch.norm(tri.mask, p=2)
                optim.SGD(model.parameters(), lr=0.2, momentum=0.9, weight_decay=5e-4).zero_grad()
                trigger_optimizer.zero_grad()
                loss.backward()
                trigger_optimizer.step()
                
                if student_trainable_when_training_trigger:
                    optim.SGD(model.parameters(), lr=0.2, momentum=0.9, weight_decay=5e-4).step()

                with torch.no_grad():
                    tri.mask.clamp_(0, 1)
                    tri.trigger.clamp_(-1*beta, 1*beta)
            masks.append(tri.mask.clone())
            triggers.append(tri.trigger.clone())
    
    average_mask = torch.mean(torch.stack(masks), dim=0)
    average_trigger = torch.mean(torch.stack(triggers), dim=0)
    tri.mask.data = average_mask
    tri.trigger.data = average_trigger

    teacher_scheduler.step()
    student1_scheduler.step()
    student2_scheduler.step()
    student3_scheduler.step()

    # caculate the model accuracy to obtain best model
    accuracies = []

    for model in models:

        model.eval()
        model.to(device)
        with torch.no_grad():
            total = 0
            correct = 0
        for x, y in tqdm(clean_test_dataloader):
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            _, predict_label = logits.max(1)
            total += y.size(0)
            correct += predict_label.eq(y).sum().item()
        accuracy = correct / total
        accuracies.append(accuracy)

    best_model_index = np.argmax(accuracies)

    print("--------Validation accuracy of 4 models(clean_test_dataloader)---------")
    print(accuracies)
    print("--------Selected as the index for the teacher model---------")
    print(best_model_index)

    ASR = []

    for model in models:

        model.eval()
        model.to(device)
        with torch.no_grad():
            total = 0
            correct = 0
        for x, y in tqdm(poison_test_dataloader):
            x = x.to(device)
            x = tri(x)
            y = y.to(device)
            logits = model(x)
            _, predict_label = logits.max(1)
            total += y.size(0)
            correct += predict_label.eq(y).sum().item()
        asr = correct / total
        ASR.append(asr)

    print("--------The attack success rate of 4 models(poison_test_dataloader)---------")
    print(ASR)

    # Save the model on a regular basis
    if epoch == 0 or (epoch + 1) % save_interval == 0:
        trigger_p = 'trigger/epoch_{}.pth'.format(epoch)
        teacher_p = 'models/weight/teacher/epoch_{}.pth'.format(epoch)
        student1_p = 'models/weight/student1/epoch_{}.pth'.format(epoch)
        student2_p = 'models/weight/student2/epoch_{}.pth'.format(epoch)
        student3_p = 'models/weight/student3/epoch_{}.pth'.format(epoch)
        torch.save(tri.state_dict(), trigger_p)
        torch.save(teacher.state_dict(), teacher_p)
        torch.save(student1.state_dict(), student1_p)
        torch.save(student2.state_dict(), student2_p)
        torch.save(student3.state_dict(), student3_p)
