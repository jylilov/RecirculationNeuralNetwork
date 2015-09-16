package by.jylilov.rnn.controller;

import by.jylilov.rnn.model.RNN;
import javafx.concurrent.Task;

import java.util.List;

public class RNNTrainingTask extends Task<RNN> {
    private final int n;
    private final int p;
    private final float minError;
    private final List<float[]> trainingSet;

    public RNNTrainingTask(int n, int p, float minError, List<float[]> trainingSet) {
        this.n = n;
        this.p = p;
        this.minError = minError;
        this.trainingSet = trainingSet;
    }

    @Override
    protected RNN call() {
        updateTitle("Training network");

        int l = trainingSet.size();

        float e = Float.MAX_VALUE;

        float []x_ = new float[n];
        float [][]y = new float[l][p];
        float []y_ = new float[p];
        float []gamma = new float[p];

        RNN rnn = new RNN(n, p);
        updateValue(rnn);

        long iteration = 0;

        float [][]w = rnn.getW();
        float [][]w_ = rnn.getW_();

        for (int k = 0; k < trainingSet.size(); ++k) {
            rnn.directStage(trainingSet.get(k), y[k]);
        }

        while (e > minError) {
            e = 0;
            ++iteration;

            if (isCancelled()) {
                break;
            }

            for (int k = 0; k < trainingSet.size(); ++k) {
                rnn.reverseStage(y[k], x_);
                float[] x = trainingSet.get(k);

                float alphaW_ = 0;
                for (int j = 0; j < p; ++j) {
                    alphaW_ += Math.pow(y[k][j], 2);
                }
                alphaW_ = 1 / alphaW_;

                for (int j = 0; j < p; ++j) {
                    gamma[j] = 0;
                    for (int i = 0; i < n; ++i) {
                        gamma[j] += w_[j][i] * (x_[i] - x[i]);
                    }
                }

                for (int j = 0; j < p; ++j) {
                    float alphaY;

                    float sum = 0;

                    for (int i = 0; i < n; ++i) {
                        sum += Math.pow(w_[j][i], 2);
                    }

                    alphaY = 1 / sum;

                    for (int i = 0; i < n; ++i) {
                        w_[j][i] -= alphaW_ * (x_[i] - x[i]) * y[k][j];
                    }
                    y[k][j] -= alphaY * gamma[j];
                }

                rnn.normalizeW_();

//                e += getError(x, x_);
            }


            for (int k = 0; k < l; ++k) {
                float[] x = trainingSet.get(k);
                rnn.directStage(x, y_);

                float alphaW = 0;
                for (int i = 0; i < n; ++i) {
                    alphaW += Math.pow(x[i], 2);
                }
                alphaW = 1 / alphaW;

                for (int j = 0; j < p; ++j) {
                    for (int i = 0; i < n; ++i) {
                        w[i][j] -= alphaW * (y_[j] - y[k][j]) * x[i];
                    }
                }

                rnn.normalizeW();

//                e += getError(y[k], y_);
            }

            for (int k = 0; k < trainingSet.size(); ++k) {
                rnn.directStage(trainingSet.get(k), y[k]);
            }
            for (int k = 0; k < l; ++k) {
                rnn.reverseStage(y[k], x_);
                e+= getError(x_, trainingSet.get(k));
            }

            updateMessage("Iteration = " + iteration + ", Error = " + e);
        }

        return rnn;
    }

    private float getError(float[] a1, float[] a2) {
        float e = 0;
        for (int i = 0; i < a1.length; ++i) {
            e += Math.pow(a2[i] - a1[i], 2);
        }
        e *= 0.5;

        return e;
    }

}
