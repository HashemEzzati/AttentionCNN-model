from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, UpSampling2D,
                                     Conv2DTranspose , Reshape, Attention, Input, concatenate, Add)
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
import numpy as np


class AttentionCNN(tf.keras.Model):
    def __init__(self, data, label_num):
        super(AttentionCNN, self).__init__()
        self.train_gaf = data['train']['gaf']
        self.val_gaf = data['val']['gaf']
        self.test_gaf = data['test']['gaf']
        self.train_label_arr = data['train']['label_arr']
        self.val_label_arr = data['val']['label_arr']
        self.test_label_arr = data['test']['label_arr']
        self.train_label = data['train']['label']
        self.val_label = data['val']['label']
        self.test_label = data['test']['label']
        self.label_num = label_num

        # Attention Module
        self.attention_module = self.build_attention_module()

        # CNN Module
        self.cnn_module = self.build_cnn_module()

        # Attention Base model
        self.base_model = self.build_attention_base_model()

    @staticmethod
    def build_attention_module():
        input_layer = Input(shape=(10, 10, 4))

        # Flatten the input and apply a Dense layers
        flatten = Flatten()(input_layer)
        dense_layer = Dense(100, activation='sigmoid')(flatten)
        dense_layer1 = Dense(400, activation='sigmoid')(dense_layer)

        # Reshape the output to (10, 10, 4)
        reshaped_layer = Reshape((10, 10, 4))(dense_layer1)

        # Multiplication (dot product) between input and reshaped_layer
        multiply_layer = tf.keras.layers.Multiply()([input_layer, reshaped_layer])

        # Addition of input_layer and multiply_layer
        added_layer = Add()([input_layer, multiply_layer])
        return tf.keras.Model(inputs=input_layer, outputs=added_layer)

    def build_cnn_module(self):
        # CNN layers
        conv1 = Conv2D(8, (2, 2), padding='same', strides=(1, 1), activation='sigmoid')(self.attention_module.output)
        dropout_layer1 = Dropout(0.35)(conv1)
        flatten = Flatten()(dropout_layer1)
        dense_layer_2 = Dense(128, activation='relu')(flatten)

        # Output layer with 9 units (assuming 9 classes) and softmax activation
        output_layer = Dense(self.label_num, activation='softmax')(dense_layer_2)
        return tf.keras.Model(inputs=self.attention_module.inputs, outputs=output_layer)

    def build_attention_base_model(self):
        """ Attention base Model"""
        model = tf.keras.Model(inputs=self.attention_module.inputs, outputs=self.cnn_module.outputs)

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # Filepath to save the best model
        checkpoint_filepath = 'best_model.h5'

        # A callback to save the best model during training
        model_checkpoint = ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_best_only=True,  # Save only the best model
            monitor='val_accuracy',  # Monitor validation accuracy
            mode='max',  # Save the model when validation accuracy is maximized
            verbose=1  # Print messages about the saving process
        )

        model.fit(x=self.train_gaf, y=self.train_label_arr, batch_size=64, epochs=50,
                  validation_data=(self.val_gaf, self.val_label_arr), callbacks=[model_checkpoint])

        # Load the best model after training
        model = load_model(checkpoint_filepath)

        return model

    def print_result(self, model):
        test_pred = np.argmax(model.predict(self.test_gaf), axis=1)
        test_label = self.test_label
        test_result_cm = confusion_matrix(test_label, test_pred, labels=range(self.label_num))

        # Calculate accuracy
        accuracy = accuracy_score(test_label, test_pred)
        print("Accuracy:", accuracy)

        # Calculate precision
        precision = precision_score(test_label, test_pred, labels=range(self.label_num), average='macro')
        print("Precision:", precision)

        # Calculate recall (sensitivity)
        recall = recall_score(test_label, test_pred, labels=range(self.label_num), average='macro')
        print("Recall (Sensitivity):", recall)

        # Calculate F1-score
        f1 = f1_score(test_label, test_pred, labels=range(self.label_num), average='macro')
        print("F1-Score:", f1)

        metrics = np.array([recall, precision, f1, accuracy])
        return metrics, test_result_cm
