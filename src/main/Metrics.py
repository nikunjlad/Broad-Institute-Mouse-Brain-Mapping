from mlxtend.plotting import plot_confusion_matrix
import numpy as np
import matplotlib, sys
from sklearn.metrics import log_loss, roc_auc_score, roc_curve, precision_recall_curve, \
    confusion_matrix, f1_score

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt


class Metrics:
    """
    Metrics class contains functions to help evaluate our machine learning model
    Functions include:
    1. Mean Squared Error
    2. Mean Absolute Error
    3. Mean Absolute Percentage Error
    4. Log-Loss
    5. Training Accuracy - Validation Accuracy
    6. Prediction Accuracy - on unknown data
    7. Confusion Matrix
    8. F Statistic
    9. Precision-Recall
    10. Region of Convergence (ROC), True Positive Rate (TPR), False Positive Rate (FPR)
    """

    def __init__(self, logger, plot_path=None, ):
        self.plot_path = plot_path
        self.logger = logger

    def mean_squared_error(self, history, plot):

        try:
            mse = history.history['mean_squared_error']

            if plot:
                plt.plot(mse)
                plt.title('Mean Squared Error')
                fig = plt.gcf()
                plt.show()

                if self.plot_path:
                    fig.savefig(self.plot_path + "/mse.png")
            return mse
        except Exception as e:
            self.logger.exception(e)
            sys.exit(1)

    def mean_absolute_error(self, history, plot):

        try:
            mae = history.history['mean_absolute_error']

            if plot:
                plt.plot(mae)
                plt.title('Mean Absolute Error')
                fig = plt.gcf()
                plt.show()

                if self.plot_path:
                    fig.savefig(self.plot_path + "/mae.png")

            return mae
        except Exception as e:
            self.logger.exception(e)
            sys.exit(1)

    def mean_absolute_percentage_error(self, history, plot):

        try:
            mape = history.history['mean_absolute_percentage_error']

            if plot:
                plt.plot(mape)
                plt.title('Mean Absolute Percentage Error')
                fig = plt.gcf()
                plt.show()

                if self.plot_path:
                    fig.savefig(self.plot_path + "/mape.png")
            return mape
        except Exception as e:
            self.logger.exception(e)
            sys.exit(1)

    def l_loss(self, actual, predicted, plot):

        try:
            loss = log_loss(actual, predicted)

            if plot:
                plt.plot(predicted, loss)
                plt.title("Log Loss / Cross Entropy")
                fig = plt.gcf()
                plt.show()

                if self.plot_path:
                    fig.savefig(self.plot_path + "/log_loss.png")

            return loss
        except Exception as e:
            self.logger.exception(e)
            sys.exit(1)

    def train_validation_accuracy(self, history, plot):

        try:
            train_acc = history.history['acc']
            val_acc = history.history['val_acc']

            if plot:
                plt.plot(train_acc)
                plt.plot(val_acc)
                plt.title('model accuracy')
                plt.ylabel('accuracy')
                plt.xlabel('epoch')
                plt.legend(['train', 'val'], loc='upper left')
                fig = plt.gcf()
                plt.show()

                if self.plot_path:
                    fig.savefig(self.plot_path + "/train_validation_accuracy.png")

            return train_acc, val_acc
        except Exception as e:
            self.logger.exception(e)
            sys.exit(1)

    def get_accuracy(self, ypred, y_test):
        """
        :param ypred: predicted values
        :param y_test: actual test values
        :return: de-one hot encoded values, i.e reverting the labels into their true original forms
        """
        try:
            predicted = [np.argmax(i) for i in ypred]
            original = [np.argmax(i) for i in y_test]

            hits = [i for i, j in zip(predicted, original) if i == j]
            test_accuracy = len(hits) / len(original) * 100
            print("\nTest data accuracy: " + str(test_accuracy) + "%")

            return original, predicted, test_accuracy
        except Exception as e:
            self.logger.exception(e)
            sys.exit(1)

    def train_validation_loss(self, history, plot):

        try:
            train_loss = history.history['loss']
            val_loss = history.history['val_loss']

            if plot:
                plt.plot(train_loss)
                plt.plot(val_loss)
                plt.title('model loss')
                plt.ylabel('loss')
                plt.xlabel('epoch')
                plt.legend(['train', 'val'], loc='upper left')
                fig = plt.gcf()
                plt.show()

                if self.plot_path:
                    fig.savefig(self.plot_path + "/train_validation_loss.png")

            return train_loss, val_loss
        except Exception as e:
            self.logger.exception(e)
            sys.exit(1)

    def confusion(self, actual, predicted, plot):

        try:
            result = confusion_matrix(actual, predicted)

            if plot:
                _, __ = plot_confusion_matrix(conf_mat=result, show_absolute=True, show_normed=True, colorbar=True)
                fig = plt.gcf()
                plt.show()

                if self.plot_path:
                    fig.savefig(self.plot_path + "/confusion.png")
        except Exception as e:
            self.logger.exception(e)
            sys.exit(1)

    def f_measure(self, actual, predicted):

        try:
            f1 = f1_score(actual, predicted, average="weighted")

            return f1
        except Exception as e:
            self.logger.exception(e)
            sys.exit(1)

    def precision_recall(self, actual, predicted, plot):

        try:
            precision, recall, _ = precision_recall_curve(actual, predicted)

            if plot:
                plt.plot([0, 1], [0.1, 0.1], linestyle='--')
                # plot the roc curve for the model
                plt.plot(recall, precision, marker='.')
                # show the plot
                fig = plt.gcf()
                plt.show()

                if self.plot_path:
                    fig.savefig(self.plot_path + "/precision_recall.png")

            return precision, recall
        except Exception as e:
            self.logger.exception(e)
            sys.exit(1)

    def auc_roc(self, actual, predicted, plot):

        try:
            auc = roc_auc_score(actual, predicted)

            fpr, tpr, thresholds = roc_curve(actual, predicted)

            if plot:
                plt.plot([0, 1], [0, 1], linestyle='--')
                # plot the roc curve for the model
                plt.plot(fpr, tpr, marker='.')
                # show the plot
                fig = plt.gcf()
                plt.show()

                if self.plot_path:
                    fig.savefig(self.plot_path + "/roc.png")

            return auc, fpr, tpr
        except Exception as e:
            self.logger.exception(e)
            sys.exit(1)

    def effect_of_epochs(self, epochs, accuracy, plot):

        try:
            if plot:
                plt.plot(np.array(epochs), np.array(accuracy), linestyle='--')
                plt.title('Epochs vs accuracy')
                plt.ylabel('accuracy')
                plt.xlabel('epochs')
                fig = plt.gcf()
                plt.show()

                if self.plot_path:
                    fig.savefig(self.plot_path + "/epoch_curve.png")
        except Exception as e:
            self.logger.exception(e)
            sys.exit(1)
