from models import AttentionCNN
import pickle


if __name__ == "__main__":
    label_num = 9

    """ Data paths with noisy class """
    # path = './data/with_noisy_class_ohlc.pkl'
    # path = './data/with_noisy_class_culr.pkl'
    path = './data/with_noisy_class_ccomhml.pkl'

    """ Data paths without noisy class """
    """ Please set the parameter 'label_num' to 8 if you choose one of the following three paths."""
    # path = './data/without_noisy_class_ohlc.pkl'
    # path = './data/without_noisy_class_culr.pkl'
    # path = './data/without__noisy_class_ccomhml.pkl'

    with open(path, 'rb') as f:
        data = pickle.load(f)

    """ Attention base CNN model """
    acnnmodel = AttentionCNN(data, label_num)
    acnnmodel.print_result(acnnmodel.base_model)
