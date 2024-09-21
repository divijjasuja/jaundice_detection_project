class CustomDataset(Dataset):
    def __init__(self, csv_path, images_folder, filename_col, label_col, transform = None):
        self.df = pd.read_csv(csv_path)
        self.images_folder = images_folder
        self.transform = transform
        self.filename_col = filename_col
        self.label_col = label_col

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        filename = self.df.loc[index][self.filename_col]
        label = self.df.loc[index][self.label_col]
        image = PIL.Image.open(os.path.join(self.images_folder, filename))
        if self.transform is not None:
            image = self.transform(image)
        return image, label