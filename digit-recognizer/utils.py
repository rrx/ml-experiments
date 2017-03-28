import numpy as np
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist
import keras.callbacks
from IPython.display import display, Javascript
from keras.preprocessing.image import Iterator
from keras import backend as K
from keras.models import model_from_json
import pandas as pd


def load_model(key):
    # load json and create model
    model_filename = "%s.model.json" % key
    json_file = open(model_filename, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    print("Loaded model: %s" % model_filename)
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    weights_filename = "%s.weights.h5" % key
    loaded_model.load_weights(weights_filename)
    print("Loaded model weights: %s" % weights_filename)
    return loaded_model


def save_model(model, key):
    model_json = model.to_json()
    model_filename = "%s.model.json" % key
    with open(model_filename, "w") as json_file:
        json_file.write(model_json)
        print("Model saved: %s" % model_filename)
    weights_filename = "%s.weights.h5" % key
    model.save_weights(weights_filename)
    print("Weights saved: %s" % weights_filename)
    
    
def disable_scrolling():
    disable_js = """
    IPython.OutputArea.prototype._should_scroll = function(lines) {
        return false;
    }
    """
    display(Javascript(disable_js))
    print ("autoscrolling long output is disabled")


# Normalize the data
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def plot_array(batch, titles=None, ncol=10, limit=70, image_width=28, image_height=28, cmap=None):   
    target_shape = (image_width, image_height)
    data = []
    values = []
    for b in batch:
        b = np.squeeze(b)
        i = len(data)
        if limit != 0 and i >= limit:
            break
            
        if b.shape == target_shape:
            data.append(b)
        elif len(b) == image_width*image_height:
            data.append(b.reshape(*target_shape))
        else:
            raise Exception("Invalid shape %s" % str(b.shape))
            
        title = ""
        if titles is not None and i < len(titles):
            title = titles[i]
        values.append(title)
        
    width = ncol
    height = (len(data)/width)+1
    plt.figure(figsize=(10,5))
    
    for i in range(0,len(data)):
        b = data[i]
        title = values[i]
        plt.subplot(height, width, i+1)
        plt.imshow(b, interpolation="none", cmap=cmap)
        plt.title(title, size=20)
        plt.xticks(())
        plt.yticks(())
        
    plt.tight_layout()

    
def plot_compare(X1, X2):
    plot_array(X1, limit=10, cmap=None)
    plot_array(X2, limit=10, cmap=None)

    
def show_comparison(model_encoder, model_decoder, X):
    encoded_imgs = model_encoder.predict(X)
    decoded_imgs = model_decoder.predict(encoded_imgs)
    print("Mean Encoded: %.8f" % encoded_imgs.mean())
    plot_array(decoded_imgs, limit=10, cmap=None)
    plt.show()
    return X, encoded_imgs, decoded_imgs
    

class BatchIterator(Iterator):
    def __init__(self, x, y, batch_size=32, shuffle=True, seed=None):
        self.shuffle=shuffle
        self.x = x
        self.y = y
        self.N = len(x)
        self.i = 0
        # batch size should be less than lenX and should be a multiple
        #assert batch_size > 0 and batch_size <= len(X)
        #assert len(X) % batch_size == 0
        super(BatchIterator, self).__init__(x.shape[0], batch_size, shuffle, seed)
           
    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
            
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        batch_x = np.zeros(tuple([current_batch_size] + list(self.x.shape)[1:]), dtype=K.floatx())
        for i, j in enumerate(index_array):
            x = self.x[j]
            batch_x[i] = x
            
        if self.y is None:
            return batch_x
        
        batch_y = self.y[index_array]
        return batch_x, batch_y        
        
    def x(self):
        max_i = len(self.X)//self.batch_size
        print self.i, max_i
        if self.i >= max_i:
            # if we are out of bounds, Stop
            self.i = 0
            raise StopIteration
            
        ix1 = self.i*self.batch_size
        ix2 = ix1+self.batch_size
        self.i+=1
        out = (self.X[ix1:ix2], self.Y[ix1:ix2])
        
        # If it's an empty set, Stop
        if len(out[0]) == 0:
            self.i = 0
            raise StopIteration
            
        return out
        
        
class MNIST:
    def __init__(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        
        self.ycat_test = to_categorical(self.y_test)
        self.ycat_train = to_categorical(self.y_train)
        
        # normalize data between 0 and 1
        self.x_train = self.x_train.astype('float32') / 255.
        self.x_test = self.x_test.astype('float32') / 255.
        
        self.x_train = self.x_train.reshape((len(self.x_train), np.prod(self.x_train.shape[1:])))
        self.x_test = self.x_test.reshape((len(self.x_test), np.prod(self.x_test.shape[1:])))
               
    def _get_autoencode_batches(self, X, batch_size=32, shuffle=True, noise_factor=0.5):
        # generate noisy data
        noisy_x = X + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X.shape) 
        noisy_x = np.clip(noisy_x, 0., 1.)

        # added noise
        merge_noisy = np.concatenate([X,noisy_x])
        # no noise
        merge_x = np.concatenate([X,X])        
        return BatchIterator(merge_noisy, merge_x, batch_size=batch_size, shuffle=shuffle)

    def get_autoencode_training_batches(self, **kwargs):
        return self._get_autoencode_batches(self.x_train, **kwargs)

    def get_autoencode_test_batches(self, **kwargs):
        return self._get_autoencode_batches(self.x_test, **kwargs)
    
    def _get_batches(self, X, Y, batch_size=32, shuffle=True, noise_factor=0.5):
        # generate noisy data
        noisy_x = X + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X.shape) 
        noisy_x = np.clip(noisy_x, 0., 1.)

        # added noise
        merge_noisy = np.concatenate([X,noisy_x])
        # no noise
        merge_y = np.concatenate([Y,Y])
        return BatchIterator(merge_noisy, merge_y, batch_size=batch_size, shuffle=shuffle)
        
    def get_training_batches(self, **kwargs):
        return self._get_batches(self.x_train, self.ycat_train, **kwargs)
    
    def get_test_batches(self, **kwargs):
        return self._get_batches(self.x_test, self.ycat_test, **kwargs)
    
    def get_samples(self, N=200, N_test=20):
        x_sample = self.x_train_shuffled[:N]
        y_sample = self.y_train_shuffled[:N]
        x_sample_test = self.x_test_shuffled[:N_test]
        y_sample_test = self.y_test_shuffled[:N_test]
        return (x_sample, y_sample), (x_sample_test, y_sample_test)
    
    def get_ordered_sample(self):
        N_classes = 10
        fits = {}
        for i in range(0,len(self.x_train)):
            if len(fits) == N_classes:
                break
            fits[self.y_train[i]] = self.x_train[i]
        assert len(fits) == N_classes
        return np.array(fits.values()), np.array(fits.keys())
        #x_sample_fit = np.array([fits[y] for y in self.y_sample])
        #y_sample_fit = y_sample
        #return (x_sample_fit, y_sample_fit)
    
    
class MNIST_CSV:
    def __init__(self):
        pass
    
    def load_train(self, filename='train.csv'):
        data = pd.read_csv(filename)
        digits = data.iloc[0][1:].as_matrix()
        # Separate labels from the data
        Y = data['label'].as_matrix()
        # Drop the label feature
        X = data.drop("label",axis=1).as_matrix()
        X = X.astype('float32') / 255.
        return X, Y

    def load_test(self, filename='test.csv'):
        data = pd.read_csv(filename)
        digits = data.iloc[0][1:].as_matrix()
        # Drop the label feature
        X = data.as_matrix()
        X = X.astype('float32') / 255.        
        return X

    
# Visually inspect the autoencoder
class VisualCallback(keras.callbacks.Callback):
    def __init__(self, model_encoder, model_decoder, history, validation_data, val_samples, 
                 show_performance=True                 
                ):
        self.history = history
        self.show_performance = show_performance
        self.model_encoder = model_encoder
        self.model_decoder = model_decoder
        self.validation_data = validation_data
        self.val_samples = val_samples

        self.val_loss = []
        self.val_acc = []

        self.X_test, _ = validation_data.next()
        
    def on_train_begin(self, logs={}):
        print("Training Set Size %(nb_sample)d" % self.params)
        
        plot_array(self.X_test, limit=10, cmap=None)
        show_comparison(self.model_encoder, self.model_decoder, self.X_test);
    
    def on_train_end(self, logs={}):
        if self.show_performance and len(self.history.history.get('loss', [])) > 1:
            self.show()
 
    def on_epoch_begin(self, epoch, logs={}):
        return
 
    def on_epoch_end(self, epoch, logs={}):
        out = self.model.evaluate_generator(self.validation_data, self.val_samples)
        #if not isinstance(out, tuple) or not isinstance(out, list):
        #    out = (out,)
        assert len(out) == len(self.model.metrics_names)
        params = {}
        for i in range(0, len(self.model.metrics_names)):
            params[self.model.metrics_names[i]] = out[i]
        loss = params.get('loss')
        if loss:
            self.val_loss.append(loss)
        acc = params.get('acc')
        if acc:
            self.val_acc.append(acc)

        nb_epoch = self.params['nb_epoch']
        interval = nb_epoch//10

        if epoch == 0 or nb_epoch < 10 or epoch % interval == 0 or epoch+1 == nb_epoch:
            print("Epoch %d/%d, Loss: %.5f Acc: %.5f Val Loss: %.5f Val Acc: %.5f" % (
                epoch+1, self.params['nb_epoch'], logs.get('loss', -1.0),
                logs.get('acc', -1.0),
                loss or -1.0, acc or -1.0))        
            show_comparison(self.model_encoder, self.model_decoder, self.X_test);
        
    def on_batch_begin(self, batch, logs={}):
        return
 
    def on_batch_end(self, batch, logs={}):
        return

    def show(self):
        h = self.history
        fig = plt.figure(figsize=(10,5))
        if 'acc' in h.history:
            ax1 = fig.add_subplot('121')
            ax1.plot(h.history['acc'])
            ax1.plot(self.val_acc)
            ax1.set_title('model accuracy')
            ax1.set_ylabel('accuracy')
            ax1.set_xlabel('epoch')
            ax1.legend(['train', 'test'], loc='upper left')
        if 'loss' in h.history:
            ax2 = fig.add_subplot('122')
            ax2.plot(h.history['loss'])
            ax2.plot(self.val_loss)
            ax2.set_title('model loss')
            ax2.set_ylabel('loss')
            ax2.set_xlabel('epoch')
            ax2.legend(['train', 'test'], loc='upper left')
        plt.show()