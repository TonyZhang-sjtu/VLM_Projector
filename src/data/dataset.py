from torch.utils.data import Dataset


class VolumeDataset(Dataset):
    """One item is one complete sequence. Training never reads a lesion mask."""
    def __init__(self, samples, preprocessor=None):
        self.samples = list(samples)
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        return {"sample": sample, "volume": self.preprocessor(sample) if self.preprocessor else None}
