import torch
import numpy as np
import torchvision.models as models
from sklearn.preprocessing import binarize
from networks import ResNet
import PIL
import faiss
import os

class SearchEngine():
    '''
    Search Engine Class
    
    By default uses binarized embedding of penultimate layer of pretrained ResNet18

    '''
    def __init__(self, threshold = 1, data = None, embedding_net = None, embedding_dimension = 512, cuda = None, transform=None, save_directory = None):

        self.threshold = threshold
        self.embedding_net = embedding_net
        self.embedding_dimension = embedding_dimension
        self.cuda = cuda
        self.save_directory = save_directory
        self.transform = transform
        self.data = data

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
            embeddings = self.embedding_net.get_embedding(*data)
            embeddings = binarize(embeddings, threshold=threshold)
            yield batch_idx, embeddings, target
    
    def update_index(self, embeddings):
        assert self.index.is_trained
        self.index.add(embeddings)
    
    def load_embeddings(self):
        filenames = sorted([filename for filename in os.listdir(self.save_directory) if filename[-3:] == "npy"])
        for batch_idx in range(len(filenames)):
            embeddings = self.load_batch(filenames[batch_idx])
            yield batch_idx, embeddings, None

    def fit(self, data_loader=None, verbose = False, step_size = 100, threshold = None, save_embeddings = False, load_embeddings = False):
        if save_embeddings and not self.save_directory:
            print("Need to set save_directory of SearchEngine")
            return

        if threshold == None:
            threshold = self.threshold
        
        if load_embeddings:
            save_embeddings = False
            loader = self.load_embeddings()
        else:
            if not data_loader:
                print("Need to provide data_loader")
                return
            loader = self.featurize_and_binarize_data(data_loader, threshold)
            
        num_batches = len(data_loader)
        batch_magnitude = len(str(num_batches))

        for batch_idx, embeddings, target in loader:
            if verbose and not (batch_idx % step_size):
                print("Batch {} of {}".format(batch_idx, num_batches))
            if save_embeddings:
                filename = "batch_{}.npy".format(str(batch_idx).zfill(batch_magnitude))
                self.save_batch(embeddings, filename)
            self.update_index(embeddings)
        if verbose:
            print("Finished fitting data.")
        
    def save_batch(self, batch, filename):
        path = "{}/{}".format(self.save_directory, filename)
        np.save(path, np.packbits(batch.astype(bool)))
                
    def load_batch(self, filename):
        path = "{}/{}".format(self.save_directory, filename)
        batch = np.unpackbits(np.load(path)).astype('float32')
        dims, rows = self.embedding_dimension, len(batch) // self.embedding_dimension
        return batch.reshape(rows, dims)
        

    def get_binarized_embedding(self, data, threshold):
        if not type(data) in (tuple, list):
            data = (data,)
        if self.cuda:
            data = tuple(d.cuda() for d in data)
        embedding = self.embedding_net.get_embedding(*data)
        embedding = binarize(embedding, threshold)
        return embedding

    def get_query_embedding(self, filename):
        image = PIL.Image.open(filename).convert('RGB')
        tensor = transform(image)[None,:,:,:]
        embedding = self.get_binarized_embedding(tensor, threshold = self.threshold)
    
    def query(self, filename, n=10, verbose = False):
        embedding = self.get_query_embedding
        distances, idx = self.index.search(embedding, n)
        if verbose:
            print("Median distance: {}".format(np.median(distances)))
            print("Mean distance: {}".format(np.mean(distances)))
        return distances, filenames = [data.samples[i][0] for i in idx[0]]



    def search(self, data, n=5, threshold=None, verbose=False):
        if threshold is None:
            threshold = self.threshold

        embedding = self.get_binarized_embedding(data, threshold = self.threshold)
        distances, idx = self.index.search(embedding, n)
            
        if verbose:
            print("Median distance: {}".format(np.median(distances)))
            print("Mean distance: {}".format(np.mean(distances)))
        
        return distances, idx
        
    def search_and_display(self, ):
        pass
        
        # plt.figure()
        # plt.title("Search Query: Index {}".format(test_idx))
        # plt.imshow(data_untransformed[test_idx][0])
        
        # for i in range(len(results)):
        #     plt.figure()
        #     plt.title("Search Result {}".format(i+1))
        #     plt.ylabel(data.classes[results[i][1]])
        #     plt.xlabel(distances[0][i])
        #     plt.imshow(results[i][0])