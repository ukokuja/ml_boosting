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

    def fit(self):
        self._set_all_examples_to_have_equal_uniform_weights()

        X = self.data.drop(['result'], axis=1)
        Y = self.data['result'].where(self.data['result'] == 1, 0)

        for i in range(0, self.iterations):
            tree_model = DecisionTreeClassifier(criterion="entropy", max_depth=1)
            model = tree_model.fit(X, Y, sample_weight=np.array(self._evaluation['weights']))
            self.models.append(model)

            predictions = model.predict(X)
            self._evaluation['predictions'] = predictions
            self._evaluation['evaluation'] = np.where(self._evaluation['predictions'] == self.data['result'], 1, 0)
            self._evaluation['misclassified'] = np.where(self._evaluation['predictions'] != self.data['result'], 1, 0)

            accuracy = sum(self._evaluation['evaluation']) / len(self._evaluation['evaluation'])
            misclassification = sum(self._evaluation['misclassified']) / len(self._evaluation['misclassified'])

            err = np.sum(self._evaluation['weights'] * self._evaluation['misclassified']) / np.sum(
                self._evaluation['weights'])

            alpha = np.log((1 - err) / err)
            self.alphas.append(alpha)

            self._evaluation['weights'] *= np.exp(alpha * self._evaluation['misclassified'])


    def predict(self):
        X_test = self.test.drop(['result'], axis=1).reindex(range(len(self.test)))
        Y_test = self.test['result'].reindex(range(len(self.test))).where(self.data['result'] == 1, 0)

        # With each model in the self.model list, make a prediction 

        predictions = []

        for alpha, model in zip(self.alphas, self.models):
            prediction = alpha * model.predict(X_test)
            predictions.append(prediction)
            self.accuracy.append(
                np.sum(np.sign(np.sum(np.array(predictions), axis=0)) == Y_test.values) / len(predictions[0]))
        self.predictions = np.sign(np.sum(np.array(predictions), axis=0))
        self._output_result(self.predictions)

    def _set_all_examples_to_have_equal_uniform_weights(self):
        self._evaluation = pd.DataFrame(self.data.result.copy())
        self._evaluation['weights'] = 1 / len(self.data)
        self._evaluation.drop('result', inplace=True, axis=1)

    @staticmethod
    def _preprocess(df):
        df['result'] = np.where(df['survived'] == 'yes', 1, 0)
        df['is_female'] = np.where(df['gender'] == 'female', 1, 0)
        df['is_child'] = np.where(df['age'] == 'child', 1, 0)
        df['pclass'] = pd.Categorical(df['pclass'],
                                      ordered=True,
                                      categories=['1st', '2nd', '3rd', 'crew']
                                      ).codes

        df.drop(columns=['gender', 'age', 'survived'], inplace=True)


    def _output_result(self, result):

        self._output_success(result)

        self._clean_for_output()

        self.test.drop(columns=['result', 'is_female', 'is_child', 'result', 'corrects_rows'], inplace=True)
        self.test.to_csv('titanikPrediction.csv', index=False)

    def _output_success(self, result):
        self.test['pred'] = result
        self.test['corrects_rows'] = result == self.test.result
        corrects_rows = np.sum(self.test['corrects_rows'].astype(int))
        n = len(self.test.pred)
        print("Success: {}%".format(corrects_rows * 100 / n))

    def _clean_for_output(self):
        self.test['survived'] = np.where(self.test['result'] == 1, 'yes', 'no')
        self.test['gender'] = np.where(self.test['is_female'] == 1, 'female', 'male')
        self.test['age'] = np.where(self.test['is_child'] == True, 'child', 'adult')
        self.test['pclass'] = self.test['pclass'].replace({0: '1st', 1: '2nd', 2: '3rd', 3: 'crew'})
        self.test['pred'] = np.where(self.test['pred'] == 1, 'yes', 'no')


if __name__ == "__main__":
    b = Boosting(pd.read_csv('titanikData.csv'), pd.read_csv('titanikTest.csv', names=["pclass", "age", "gender", "survived"]))
    b.fit()
    b.predict()
    pass
