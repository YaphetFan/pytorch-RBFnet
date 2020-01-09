path = 'mypath'

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
# from source_code import rbf, kmeans, sigmas
import numpy as np
from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, Loss, Recall, Precision
from functools import partial
from ignite.contrib.handlers import ProgressBar
import time, math
from pykeops.torch import LazyTensor
from sklearn.decomposition import PCA
import pickle
import sys



def Kmeans(data, classes):
    n, dim = data.shape  # Number of samples, dimension of the ambient space

    centers = data[:classes, :].clone()  # Simplistic random initialization
    ltensor_i = LazyTensor(data[:, None, :])  # (Npoints, 1, D)

    class_vec = []
    for i in range(classes):

        mean = LazyTensor(centers[None, :, :])  # (1, Nclusters, D)
        distance = ((ltensor_i - mean) ** 2).sum(-1)  # (Npoints, Nclusters) symbolic matrix of squared distances
        class_vec = distance.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

        class_weights = torch.bincount(class_vec).type(dtype=torch.float64)  # Class weights
        for d in range(dim):  # Compute the cluster centroids with torch.bincount:
            centers[:, d] = torch.bincount(class_vec, weights=data[:, d]) / class_weights

    return class_vec, centers


def Sigmas(centers):
    '''
    To simplify the algorithm design, set the same width (sigma)
    in all gaussian functions, according to spreading of centers  by kmeans.
    σ = d_max / sqrt(2*K) , where σ: sigma, d_max: max distance, K: number of clusters
    :param centers: coordinates of centers.
    :return:sigma
    '''
    clusters = len(centers)
    l2_list = []
    for i in range(clusters):
        for j in range(i+1, clusters):
            l2 = torch.dist(centers[i], centers[j]).item()
            l2_list.append((i, j, l2))

    l2_list = sorted(l2_list, key=lambda l: l[2], reverse=True)

    sigma = l2_list[0][2]/math.sqrt(2*clusters)
    return sigma



class RBF(nn.Module):
    def __init__(self, in_layers, centers, sigmas):
        super(RBF, self).__init__()
        self.in_layers = in_layers[0]
        self.centers = nn.Parameter(centers)
        self.dists = nn.Parameter(torch.ones(1,centers.size(0)))

        # self.dists = nn.Parameter(torch.ones(self.centers.size(0), in_layers[0]))


        self.sigmas = sigmas


        # self.linear0 = nn.Linear(in_layers[0], in_layers[0], bias = True)
        self.linear1 = nn.Linear(centers.size(0), in_layers[1], bias = True)

    def forward(self, x):
        # x = x.view(-1, self.in_layers)
        # x = torch.sigmoid(self.linear0(x))

        phi = self.radial_basis(x)

        out = torch.sigmoid(self.linear1(phi.float()))
        return out

    # def radial_basis(self,x):

    #     c = self.centers.view(self.centers.size(0),-1).repeat(x.size(0), 1, 1)
    #     x = x.view(x.size(0),-1).unsqueeze(1).repeat(1, self.centers.size(0),1)
    #     phi = torch.exp(-self.dists.mul((c-x).pow(2).sum(2, keepdim=False).pow(0.5) ))
    #     return phi

    def radial_basis(self,x):
        x = x.view(x.size(0),-1)
        size = [self.centers.size(0), x.size(0)]
        sigma = self.sigmas

        dists = torch.empty(size).to(device)
        # dists = self.dists

        for i,c in enumerate(self.centers):
            c = c.reshape(-1,c.size(0))
            temp = (x-c).pow(2).sum(-1).pow(0.5)
            dists[i] = temp

        dists = dists.permute(1,0)
        # print(dists)
        phi = torch.exp(-1*(dists/(2*sigma))) #gaussian
        # phi.requires_grad_(True)
        return phi


    # def radial_basis_other(self,x):
    #     x = x.view(x.size(0),-1)
    #     size = [self.centers.size(0), x.size(0)]
    #     sigma = self.sigmas
    #     dists = torch.empty(size).to(device)


class Net(nn.Module):
        def __init__(self, neuron, layers, algorithm):
            super(Net, self).__init__()
            self.algorithm = algorithm
            self.layers = layers
            self.start = neuron[0]
            self.mlp = nn.ModuleList([nn.Linear(neuron[i], neuron[i+1],bias=True) for i in range(layers)])

        def forward(self, x):
            algorithms = {'relu': torch.relu, 'tanh': torch.tanh, 'sigmoid': torch.sigmoid, 'prelu': torch.prelu}
            out = x.view(-1, self.start)
            for i in range(self.layers-1):
                out = algorithms[self.algorithm](self.mlp[i](out))
            return self.mlp[i+1](out)



def training(engine, batch, device, model, criterion, optimizer):
    # torch.set_printoptions(profile="full")
    inputs, labels = batch[0].to(device), batch[1].to(device)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    # print(optimizer)
    optimizer.step()
    return outputs, labels


def testing(engine, batch, device, model):
        with torch.no_grad():
            images, labels = batch[0].to(device), batch[1].to(device)
            outputs = model(images)
            # _, predicted = torch.max(outputs, 1)
        return outputs, labels


def in_layers(dim, classes, layers):
        start = int(np.log2(dim))
        stop = int(np.log2(classes))
        neuron = [2**i for i in range(start, stop-1, -1)]
        neuron.insert(0, dim)
        neuron.insert(layers if layers <= (start-stop) else stop, classes)
        return neuron[:layers+1]


def thresholded_output_transform(output):
        y_pred, y = output
        _, y_pred = torch.max(y_pred, 1)
        # change type and reshape y and y_pred
        t_pred = y_pred.cpu().numpy().astype(np.uint8)
        t = y.cpu().numpy().astype(np.uint8)
        t_pred = np.reshape(t_pred, (len(t_pred), -1))
        t = np.reshape(t, (len(t), -1))
        # convert to binary
        t_pred = np.unpackbits(t_pred, axis=1)
        t = np.unpackbits(t, axis=1)
        y_pred = torch.from_numpy(t_pred).to(device)
        y = torch.from_numpy(t).to(device)
        return y_pred, y


def metrics_estimating(tester, criterion):
    Accuracy(output_transform=thresholded_output_transform).attach(tester, 'accuracy')
    Recall(output_transform=thresholded_output_transform, average=True).attach(tester, 'recall')
    Precision(output_transform=thresholded_output_transform, average=True).attach(tester, 'precision')
    Loss(criterion).attach(tester, 'loss')


def start_to_learn(trainer, train_loader, tester, test_loader, epochs, model_metrics):
    # ---Log Message Initializing---
    log_msg = ProgressBar(persist=True, bar_format=" ")

    # ---After Training starts Testing---
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        tester.run(test_loader)

        epoch = engine.state.epoch
        metrics = tester.state.metrics

        model_metrics.update({epoch: metrics})

        log_msg.log_message("Epoch: {}  \nAccuracy: {:.3f} \tLoss: {:.3f} \tRecall: {:.3f} \tPrecision: {:.3f}\n\n"
                                .format(epoch, metrics['accuracy'], metrics['loss'], metrics['precision'], metrics['recall']))
        log_msg.n = log_msg.last_print_n = 0

    start = time.time()
    trainer.run(train_loader, epochs)   # Training Starting
    duration = time.time() - start
    print('Duration of execution: ', duration)
    return duration


def data_loading(batch, shuffle):
    transform = transforms.Compose([transforms.ToTensor()])

    trainset = torchvision.datasets.MNIST(root='../datasheets', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch, shuffle=shuffle, drop_last=True, pin_memory=True)
    testset = torchvision.datasets.MNIST(root='../datasheets', train=False, download=True, transform=transform)
    testlaoder = torch.utils.data.DataLoader(testset, batch_size=batch, shuffle=False, pin_memory=True)

    return trainloader, testlaoder


def pre_processing_data(trainset, testset):
    train_data = torch.reshape(trainset, (trainset.size(0), -1))
    test_data = torch.reshape(testset, (testset.size(0), -1))

    pca = PCA(n_components = 90)

    pca_train = pca.fit_transform(train_data)
    pca_test = pca.transform(test_data)

    # print(pca_train)
    # pca_train = torch.reshape(pca_train, (pca_train.size(0), -1))
    # pca_test = torch.reshape(pca_test, (pca_test.size(0), -1))

    return pca_train, pca_test


def kmeans_def(train_loader, clusters):
    kmeans_input = torch.reshape(train_loader.double(), (train_loader.size(0), -1))
    _, centers = Kmeans(kmeans_input, clusters)
    return centers


def random_def(train_loader, clusters):
    return torch.rand(10,train_loader.size(1)**2)


def gaussian_dist_def(train_loader, clusters):
    centers = torch.Tensor(clusters, train_loader.size(1)**2)
    return nn.init.normal_(centers, 0, 1)


def centers_init_switcher(train_loader, clusters, definition):
    switcher = {'kmeans':kmeans_def(train_loader, clusters),
                'random': random_def(train_loader, clusters),
                'gaussian':gaussian_dist_def(train_loader, clusters)}
    return switcher[definition]


def sigma_manual_def(centers):
    return 0.9#Sigmas(centers)/100


def sigma_init_switcher(centers, definition):
    switcher = {'lowe_89':Sigmas(centers),
                'manual':sigma_manual_def(centers)}
    return switcher[definition]


def nn_run(batch, classes, dim, learning_rate, epochs, clusters, momentum, centers_way, sigmas_way):

    # ---Load Model's Parameters---
    train_loader, test_loader = data_loading(batch, shuffle=True)
    trainset = train_loader.dataset.train_data
    centers = centers_init_switcher(trainset, clusters, 'kmeans').to(device)
    sigma = sigma_init_switcher(centers, 'manual')
    layers = in_layers(dim, len(classes), 2)

    # ---Model Setup---
    model = RBF(layers, centers, sigma)
    model.cuda()
    criterion =  nn.CrossEntropyLoss() #nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum) #adam: torch.optim.Adam(model.parameters(), lr=learning_rate)

    # ---Executions'Informations Data to Store---
    model_settings = {'learning_rate': learning_rate, 'batch':batch, 'clusters': clusters, 'layers':layers,
                      'center':centers_way, 'sigma':sigmas_way}
    model_metrics = {}

    # ---Model Fitting---
    trainer = Engine(partial(training, device=device, model=model, criterion=criterion, optimizer=optimizer))
    tester = Engine(partial(testing, device=device, model=model))

    # ---Metrics Estimating---
    metrics_estimating(tester, criterion)

    # ---Training Testing---
    duration = start_to_learn(trainer, train_loader, tester, test_loader, epochs, model_metrics)


    model_data = {'settings': model_settings, 'metrics': model_metrics, 'model_timer': duration}

    return model_data



def main():
    # batch = 32
    # learning_rate = 0.0002
    # clusters = 10

    # ---Parameters Initializing ---
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    dim = 28*28
    epochs = 14
    clusters_l = [10,20,100]
    executions = 60
    momentum = 0.9
    centers_way = ['kmeans','random','gaussian']
    sigmas_way = ['lowe_89','manual']
    learning_rates = [0.0001, 0.003, 0.05, 0.1]

    # nn_run(batch, classes, dim, learning_rate, epochs, clusters, momentum)

    for clusters in clusters_l:
      for cw in centers_way:
        for sw  in sigmas_way:
          for batch in [4,10,16,32,64]:
              for learning_rate in learning_rates:

                  executions += 1
                  print('\nEXECUTION-',executions,'learning_rate', learning_rate, 'batch',batch)

                  results = nn_run(batch, classes, dim, learning_rate, epochs,clusters, momentum, centers_way, sigmas_way)

                  dict_results = {}
                  try:
                      with open(path, 'rb') as f:
                          unpickler = pickle.Unpickler(f)
                          dict_results = unpickler.load()
                          dict_results.update({executions: results})
                  except FileNotFoundError:
                      dict_results = {executions: results}
                  except EOFError:
                      dict_results = {executions: results}
                      print('Something going Wrong')

                  with open(path, 'wb') as f:
                      pickle.dump(dict_results, f)
                      print('Write!')


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device:', device)
main()

