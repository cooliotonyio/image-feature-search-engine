# Image Feature Search Engine

A highly memory-efficient and lightning-fast image feature search engine for extremely large datesets built on FAISS with ResNet18 as the default feature extraction network.

# Basic Usage
1) Declare global values
    ```
    THRESHOLD = 1
    SAVE_DIRECTORY = './binary_embeddings'
    DATA_FOLDER = './Flickr'
    ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
    TRANSFORM = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
        ])
    CUDA = torch.cuda.is_available()
    BATCH_SIZE = 128
    ```

2) (Optional) Load data folder to extract feature vectors. Step not necessary if binarized feature vectors already exist.
    ```
    data = ImageFolder('./Flickr', transform=TRANSFORM)
    kwargs = {'num_workers': 1, 'pin_memory': True} if CUDA else {}
    data_loader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, **kwargs)
    ```

3) Instantiate `SearchEngine` object.
    ```
    search_engine = SearchEngine(data, cuda = CUDA, threshold = THRESHOLD, save_directory = SAVE_DIRECTORY, transform=TRANSFORM) 
    ```

4) Fit index (set `load_embeddings` to `True` if binarized feature vectors already exist, else `False` to extract and binarize feature vectors)

    ```
    search_engine.fit(data_loader = data_loader, load_embeddings = True, verbose = True)
    ```

5) Perform queries on target file
    ```
    FILENAME = "~/path/to/file/example.jpg"
    distances, paths = search_engine.query(FILENAME)
    ```