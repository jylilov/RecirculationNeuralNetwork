package by.jylilov.rnn.controller.impl;

import by.jylilov.rnn.controller.TrainingTask;
import by.jylilov.rnn.model.RecirculationNeuralNetwork;

import java.util.List;

public class StratifiedTrainingTask extends TrainingTask {

    public StratifiedTrainingTask(
            int n, int p,
            float alpha, float minError,
            List<float[]> trainingSet,
            boolean normalisation, boolean adaptiveLearningStep
    ) {
        super(n, p, alpha, minError, trainingSet, normalisation, adaptiveLearningStep);
    }

    @Override
    protected RecirculationNeuralNetwork call() {
        updateTitle("Stratified Training Network");

        float e = Float.MAX_VALUE;

        RecirculationNeuralNetwork rnn = new RecirculationNeuralNetwork(n, p);
        updateValue(rnn);

        long iteration = 0;

        float [][]y = new float[l][p];

        directStage(rnn, y);

        while (e > minError) {
            ++iteration;

            if (isCancelled()) {
                break;
            }

            trainHiddenLayout(rnn, y);
            trainMainLayout(rnn, y);

            directStage(rnn, y);
            e = calcError(rnn, y);

            updateMessage("Iteration = " + iteration + ", Error = " + e);
        }

        return rnn;
    }

    private float calcError(RecirculationNeuralNetwork rnn, float[][] y) {
        float e = 0;
        float[] x_ = new float[n];
        for (int k = 0; k < l; ++k) {
            rnn.reverseStage(y[k], x_);
            e+= getError(x_, trainingSet.get(k));
        }
        return e;
    }

    private void directStage(RecirculationNeuralNetwork rnn, float[][] y) {
        for (int k = 0; k < trainingSet.size(); ++k) {
            rnn.directStage(trainingSet.get(k), y[k]);
        }
    }

    private void trainMainLayout(RecirculationNeuralNetwork rnn, float[][] y) {
        float[][] w = rnn.getW();
        float[] y_ = new float[p];
        for (int k = 0; k < l; ++k) {
            float[] x = trainingSet.get(k);
            rnn.directStage(x, y_);

            float alphaW;
            if (adaptiveLearningStep) {
                alphaW = 0;
                for (int i = 0; i < n; ++i) {
                    alphaW += Math.pow(x[i], 2);
                }
                alphaW = 1 / (1 + alphaW);
            } else {
                alphaW = alpha;
            }

            for (int j = 0; j < p; ++j) {
                for (int i = 0; i < n; ++i) {
                    w[i][j] -= alphaW * (y_[j] - y[k][j]) * x[i];
                }
            }

            if (normalization) rnn.normalizeW();
        }
    }

    private void trainHiddenLayout(RecirculationNeuralNetwork rnn, float[][] y) {
        float[][] w_ = rnn.getW_();
        float[] x_ = new float[n];
        float[] gamma = new float[p];
        for (int k = 0; k < trainingSet.size(); ++k) {
            rnn.reverseStage(y[k], x_);
            float[] x = trainingSet.get(k);

            float alphaW_;

            if (adaptiveLearningStep) {
                alphaW_ = 0;
                for (int j = 0; j < p; ++j) {
                    alphaW_ += Math.pow(y[k][j], 2);
                }
                alphaW_ = 1 / (1 + alphaW_);
            } else {
                alphaW_ = alpha;
            }

            for (int j = 0; j < p; ++j) {
                gamma[j] = 0;
                for (int i = 0; i < n; ++i) {
                    gamma[j] += w_[j][i] * (x_[i] - x[i]);
                }
            }

            for (int j = 0; j < p; ++j) {
                float alphaY;
                if (adaptiveLearningStep) {
                    alphaY = 0;
                    for (int i = 0; i < n; ++i) {
                        alphaY += Math.pow(w_[j][i], 2);
                    }
                    alphaY = 1 / (1 + alphaY);
                } else {
                    alphaY = alpha;
                }

                for (int i = 0; i < n; ++i) {
                    w_[j][i] -= alphaW_ * (x_[i] - x[i]) * y[k][j];
                }
                y[k][j] -= alphaY * gamma[j];
            }

            if (normalization) rnn.normalizeW_();
        }
    }

    private float getError(float[] a1, float[] a2) {
        float e = 0;
        for (int i = 0; i < a1.length; ++i) {
            e += Math.pow(a2[i] - a1[i], 2);
        }

        return e;
    }

}
