import torch
import numpy as np
import torchvision.models as models
from sklearn.preprocessing import binarize
from networks import ResNet
import faiss

class SearchEngine():
    '''
    Search Engine Class
    
    By default uses binarized embedding of penultimate layer of pretrained ResNet18

    '''
    def __init__(self, threshold = 1, embedding_net = None, embedding_dimension = 512, cuda = None, save_directory = None):

        self.threshold = threshold
        self.embedding_net = embedding_net
        self.embedding_dimension = embedding_dimension
        self.cuda = cuda
        self.save_directory = save_directory

        # Default to penult embedding layer of pretrained ResNet18
        if self.embedding_net is None:
            self.embedding_net = ResNet()
        
        # Initialize index
        self.index = faiss.IndexFlatL2(embedding_dimension)

        # GPU acceleration of net and index
        if self.cuda:
            self.embedding_net.cuda()

            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)


    def featurize_and_binarize_data(self, data_loader, threshold):
        for batch_idx, (data, target) in enumerate(data_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if self.cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()
            outputs = self.embedding_net.get_embedding(*data)
            outputs = binarize(outputs, threshold=threshold)
            yield batch_idx, outputs
    
    def update_index(self, embeddings):
        assert self.index.is_trained
        self.index.add(embeddings)

    def fit(self, data_loader, verbose = False, step_size = 100, threshold = None, save_embeddings = False):
        if save_embeddings and not self.save_directory:
            print("Need to set save_directory of SearchEngine")
            return

        if threshold == None:
            threshold = self.threshold

        for batch_idx, outputs in self.featurize_and_binarize_data(data_loader, threshold):
            if verbose and not (batch_idx % step_size):
                print("Batch {} of {}".format(batch_idx,len(data_loader)))
            if save_embeddings:
                filename = "{}/batch_{}.npy".format(self.save_directory, batch_idx)
                np.save(filename, outputs)
            self.update_index(outputs)
        if verbose:
            print("Finished fitting data.")

    def get_binarized_embedding(self, target, threshold):
        embedding = self.embedding_net.get_embedding(target).cpu().numpy()
        embedding = binarize(embedding, threshold)
        return embedding


    def search(self, target, n=5, threshold=None, verbose=False):
        if threshold is None:
            threshold = self.threshold

        embedding = self.get_binarized_embedding(target, threshold = threshold)
        idx, distances = self.index.search(embedding, n)
            
        if verbose:
            print("Median distance: {}".format(np.median(distances)))
            print("Mean distance: {}".format(np.mean(distances)))
        
        plt.figure()
        plt.title("Search Query: Index {}".format(test_idx))
        plt.imshow(data_untransformed[test_idx][0])
        
        for i in range(len(results)):
            plt.figure()
            plt.title("Search Result {}".format(i+1))
            plt.ylabel(data.classes[results[i][1]])
            plt.xlabel(distances[0][i])
            plt.imshow(results[i][0])
        

    

