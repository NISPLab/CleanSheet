dataset = 'CIFAR10'
dataset_mean = (0.4914, 0.4822, 0.4465)
dataset_std = (0.2023, 0.1994, 0.2010)

network = dict(type='resnet18', arch='cifar')

test_dataloader = dict(batch_size=64,
                       num_workers=4,
                       persistent_workers=True,
                       shuffle=False,
                       dataset=dict(type=dataset,
                                    root='data',
                                    train=False,
                                    download=True,
                                    transform=[
                                        dict(type='ToTensor'),
                                        dict(type='Normalize',
                                             mean=dataset_mean,
                                             std=dataset_std)
                                    ]))

train_dataloader = dict(batch_size=64,
                        num_workers=4,
                        persistent_workers=True,
                        shuffle=True,
                        dataset=dict(type=dataset,
                                     root='data',
                                     train=True,
                                     download=True,
                                     transform=[
                                         dict(type='RandomCrop',
                                              size=32,
                                              padding=4),
                                         dict(type='RandomHorizontalFlip',
                                              p=0.5),
                                         dict(type='ToTensor'),
                                         dict(type='Normalize',
                                              mean=dataset_mean,
                                              std=dataset_std)
                                     ]))
