import pickle


class History:
    def __init__(self, model_name, path, lr, epoch, batch_size, info=''):
        self.model_name = model_name
        self.save_history_path = path
        self.ct = 0
        self.train_loss = []
        self.test_loss = []
        self.train_acc = []
        self.test_acc = []
        self.train_lr = lr
        self.train_epoch = epoch
        self.train_batch_size = batch_size
        self.info = info

    def save_history(self):
        path = self.save_history_path + '/' + self.model_name + self.info + '.pkl'
        with open(path, "wb") as save_file:
            save_file.write(pickle.dumps(self))
            save_file.close()
        return path

    def print_info(self):
        print(self.model_name)


if __name__ == '__main__':
    h = History('model', './save', 0.01, 50, 16)
    sp = h.save_history()
    rp = pickle.loads(open(sp, 'rb').read())
    print(rp.model_name)
    rp.print_info()
