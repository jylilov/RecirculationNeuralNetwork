package by.jylilov.rnn.controller.impl;

import by.jylilov.rnn.controller.RNNTrainingTask;
import by.jylilov.rnn.model.RNN;

import java.util.List;

public class RNNStratifiedTrainingTask extends RNNTrainingTask {

    public RNNStratifiedTrainingTask(
            int n, int p,
            float alpha, float minError,
            List<float[]> trainingSet,
            boolean isWithNormalisation, boolean isWithAdaptiveLearningStep
    ) {
        super(n, p, alpha, minError, trainingSet, isWithNormalisation, isWithAdaptiveLearningStep);
    }

    @Override
    protected RNN call() {
        updateTitle("Stratified Training Network");

        float e = Float.MAX_VALUE;

        RNN rnn = new RNN(n, p);
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

    private float calcError(RNN rnn, float[][] y) {
        float e = 0;
        float[] x_ = new float[n];
        for (int k = 0; k < l; ++k) {
            rnn.reverseStage(y[k], x_);
            e+= getError(x_, trainingSet.get(k));
        }
        return e;
    }

    private void directStage(RNN rnn, float[][] y) {
        for (int k = 0; k < trainingSet.size(); ++k) {
            rnn.directStage(trainingSet.get(k), y[k]);
        }
    }

    private void trainMainLayout(RNN rnn, float[][] y) {
        float[][] w = rnn.getW();
        float[] y_ = new float[p];
        for (int k = 0; k < l; ++k) {
            float[] x = trainingSet.get(k);
            rnn.directStage(x, y_);

            float alphaW;
            if (isWithAdaptiveLearningStep) {
                alphaW = 0;
                for (int i = 0; i < n; ++i) {
                    alphaW += Math.pow(x[i], 2);
                }
                alphaW = 1 / alphaW;
            } else {
                alphaW = alpha;
            }

            for (int j = 0; j < p; ++j) {
                for (int i = 0; i < n; ++i) {
                    w[i][j] -= alphaW * (y_[j] - y[k][j]) * x[i];
                }
            }

            if (isWithNormalisation) rnn.normalizeW();
        }
    }

    private void trainHiddenLayout(RNN rnn, float[][] y) {
        float[][] w_ = rnn.getW_();
        float[] x_ = new float[n];
        float[] gamma = new float[p];
        for (int k = 0; k < trainingSet.size(); ++k) {
            rnn.reverseStage(y[k], x_);
            float[] x = trainingSet.get(k);

            float alphaW_;

            if (isWithAdaptiveLearningStep) {
                alphaW_ = 0;
                for (int j = 0; j < p; ++j) {
                    alphaW_ += Math.pow(y[k][j], 2);
                }
                alphaW_ = 1 / alphaW_;
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
                if (isWithAdaptiveLearningStep) {
                    alphaY = 0;
                    for (int i = 0; i < n; ++i) {
                        alphaY += Math.pow(w_[j][i], 2);
                    }
                    alphaY = 1 / alphaY;
                } else {
                    alphaY = alpha;
                }

                for (int i = 0; i < n; ++i) {
                    w_[j][i] -= alphaW_ * (x_[i] - x[i]) * y[k][j];
                }
                y[k][j] -= alphaY * gamma[j];
            }

            if (isWithNormalisation) rnn.normalizeW_();
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
