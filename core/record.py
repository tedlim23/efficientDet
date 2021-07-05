import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class EarlyStopping():
    def __init__(self, ckpt, save_path):
        self.score = tf.constant(100.0)
        self.patience = 30
        self.count = 0
        self.delta = 0.0001
        self.ckpt = ckpt
        self.ckpt_manager = tf.train.CheckpointManager(ckpt, directory=save_path, max_to_keep=5)
        self.save_path = save_path
    
    def save_score(self, score):
        self.score = score
        self.ckpt.save(self.save_path)

    def is_stoppable(self, score):
        if tf.equal(score, 0.0):
            return True
        
        if tf.math.greater_equal(score, self.score + self.delta):
            self.count += 1
        else:
            tf.print("Previous Score: ", self.score)
            tf.print("Current Score: ", score)
            self.save_score(score)
            self.count = 0
        
        if self.count > self.patience:
            return True
        else:
            return False

class PlotLib():
    def __init__(self, epochs, save_path):
        self.epochs = np.array(range(1, epochs+1))
        self.training_loss = []
        self.validation_loss = []
        self.training_score = []
        self.validation_score = []
        self.save_path = os.path.join(os.path.split(save_path)[0], "plot")
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
    
    def draw_plot(self):
        plt.cla()
        training_loss = np.asarray([float(s) for s in self.training_loss])
        validation_loss = np.asarray([float(s) for s in self.validation_loss])
        training_score = np.asarray([float(s) for s in self.training_score])
        validation_score = np.asarray([float(s) for s in self.validation_score])

        plt.plot(self.epochs, training_loss, 'r-*', label = "Training Loss")
        plt.plot(self.epochs, validation_loss, 'b--o', label = "Validation Loss")
        plt.title('Loss Graph')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')
        plt.savefig(os.path.join(self.save_path, 'loss_graph.png'))

        plt.cla()

        plt.plot(self.epochs, training_score, 'r-*', label = "Training Score")
        plt.plot(self.epochs, validation_score, 'b--o', label = "Validation Score")
        plt.title('Score Graph')
        plt.xlabel('Epochs')
        plt.ylabel('Score')
        plt.legend(loc='upper right')
        plt.savefig(os.path.join(self.save_path, 'score_graph.png'))
    
    def add_to_list(self, train_loss, val_loss, train_score, val_score):
        self.training_loss.append(train_loss)
        self.validation_loss.append(val_loss)
        self.training_score.append(train_score)
        self.validation_score.append(val_score)
    
    def notice_early_stop(self, epochs):
        self.epochs = np.array(range(1, epochs+1))
