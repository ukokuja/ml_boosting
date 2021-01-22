import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


class Boosting():
    def __init__(self, data, test, iterations=3):
        self.data = data
        self.test = test

        self._preprocess(self.data)
        self._preprocess(self.test)

        self.iterations = iterations
        self.models = []
        self.alphas = []
        self.accuracy = []
        self.predictions = []



    @staticmethod
    def _preprocess(df):
        """
        Transforms nominal data to ordinal and boolean
        :param df: df to transform
        :return:
        """
        df['result'] = np.where(df['survived'] == 'yes', 1, 0)
        df['is_female'] = np.where(df['gender'] == 'female', 1, 0)
        df['is_child'] = np.where(df['age'] == 'child', 1, 0)
        df['pclass'] = pd.Categorical(df['pclass'],
                                      ordered=True,
                                      categories=['1st', '2nd', '3rd', 'crew']
                                      ).codes

        df.drop(columns=['gender', 'age', 'survived'], inplace=True)

    def fit(self):
        """
        Trains the model based on dataset
        :return:
        """
        self._set_all_examples_to_have_equal_uniform_weights()

        X = self.data.drop(['result'], axis=1)
        Y = self.data['result'].where(self.data['result'] == 1, 0)

        for i in range(0, self.iterations):
            model = self._train_model(X, Y)

            # Add ht to H
            self.models.append(model)

            self._evaluation['predictions'] = model.predict(X)
            self._evaluation['is_prediction_incorrect'] = np.where(
                self._evaluation['predictions'] != self.data['result'], 1, 0)

            error_rate = self._get_error_rate()

            # If εt > 0.5 then exit loop, else continue
            if error_rate > 0.5:
                continue

            beta = self._get_beta(error_rate)
            alpha = self._get_alpha(beta)
            self.alphas.append(alpha)

            self._recalculate_weights(beta)
            self._rescale_weights()

    def _set_all_examples_to_have_equal_uniform_weights(self):
        self._evaluation = pd.DataFrame(self.data.result.copy())
        self._evaluation['weights'] = 1 / len(self.data)
        self._evaluation.drop('result', inplace=True, axis=1)

    def _train_model(self, X, Y):
        """
        Trains model based using Tree Stump based on X and Y
        :param X: Dataset without result parameter
        :param Y: Result parameter
        :return: Trained model
        """

        # Learn a hypothesis, ht, from the weighted examples
        tree_model = DecisionTreeClassifier(criterion="entropy", max_depth=1)
        model = tree_model.fit(X, Y, sample_weight=np.array(self._evaluation['weights']))
        return model

    def _get_error_rate(self):
        # Calculate the error, εt, of the hypothesis ht as the total sum weight of the examples
        # that it classifies incorrectly
        err = np.sum(
            self._evaluation['weights'] * self._evaluation['is_prediction_incorrect'])
        return err


    @staticmethod
    def _get_alpha(beta):
        # αt =0.5*ln(1/ βt)
        return 0.5 * np.log(1 / beta)

    @staticmethod
    def _get_beta(error_rate):
        # Let βt = εt / (1 – εt )
        return error_rate / (1 - error_rate)

    def _recalculate_weights(self, alpha):
        # Multiply the weights of the examples that
        # ht classifies correctly by βt
        self._evaluation['weights'] *= np.exp(alpha * self._evaluation['is_prediction_incorrect'])

    def _rescale_weights(self):
        # Rescale the weights of all of the examples so the total sum weight remains 1
        self._evaluation['weights'].div(self._evaluation['weights'].sum())


    def predict(self):
        """
        Predict based on previous generated models and alphas
        :return:
        """
        X_test = self.test.drop(['result'], axis=1).reindex(range(len(self.test)))

        predictions = []

        for alpha, model in zip(self.alphas, self.models):
            prediction = alpha * model.predict(X_test)
            predictions.append(prediction)
        self.predictions = np.sign(np.sum(np.array(predictions), axis=0))
        self._output_result(self.predictions)



    def _output_result(self, result):
        """
        Outputs the result
        :param result: prediction result
        :return:
        """
        self._output_success(result)

        self._clean_for_output()

        self.test.to_csv('titanikPrediction.csv', index=False)
        print(self.test)

    def _output_success(self, result):
        """
        Calculates the percentage of success based on prediction results
        :param result: prediction result
        :return:
        """
        self.test['pred'] = result
        self.test['corrects_rows'] = result == self.test.result
        corrects_rows = np.sum(self.test['corrects_rows'].astype(int))
        n = len(self.test.pred)
        print("Success: {}%".format(corrects_rows * 100 / n))

    def _clean_for_output(self):
        """
        Transform tests data as it was originally before pre-process
        :return:
        """
        self.test['survived'] = np.where(self.test['result'] == 1, 'yes', 'no')
        self.test['gender'] = np.where(self.test['is_female'] == 1, 'female', 'male')
        self.test['age'] = np.where(self.test['is_child'] == True, 'child', 'adult')
        self.test['pclass'] = self.test['pclass'].replace({0: '1st', 1: '2nd', 2: '3rd', 3: 'crew'})
        self.test['pred'] = np.where(self.test['pred'] == 1, 'yes', 'no')

        self.test.drop(columns=['result', 'is_female', 'is_child', 'result', 'corrects_rows'], inplace=True)


if __name__ == "__main__":
    b = Boosting(pd.read_csv('titanikData.csv'),
                 pd.read_csv('titanikTest.csv', names=["pclass", "age", "gender", "survived"]))
    b.fit()
    b.predict()
