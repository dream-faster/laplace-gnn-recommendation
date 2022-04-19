import os
from dgl.data.utils import download, extract_archive
from dgl.data import DGLDataset
from dgl.data.utils import load_graphs


class FashionDataset(DGLDataset):

    _prefix = "https://storage.googleapis.com/heii-public/"
    _urls = {}

    def __init__(self, name, raw_dir=None, force_reload=False, verbose=True):
        assert name in ["fashion"]
        self.g_path = "./openhgnn/dataset/{}/graph.bin".format(name)
        raw_dir = "./openhgnn/dataset"
        url = self._prefix + "graph.bin"
        super(FashionDataset, self).__init__(
            name=name,
            url=url,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
        )

    def download(self):
        # download raw data to local disk
        # path to store the file
        if os.path.exists(self.g_path):  # pragma: no cover
            pass
        else:
            # download file
            download(self.url, path=os.path.join(self.raw_dir))

    def process(self):
        # process raw data to graphs, labels, splitting masks
        g, _ = load_graphs(self.g_path)
        self._g = g[0]

    def __getitem__(self, idx):
        # get one example by index
        assert idx == 0, "This dataset has only one graph"
        return self._g

    def __len__(self):
        # number of data examples
        return 1

    def save(self):
        # save processed data to directory `self.save_path`
        pass

    def load(self):
        # load processed data from directory `self.save_path`
        pass

    def has_cache(self):
        # check whether there are processed data in `self.save_path`
        pass
