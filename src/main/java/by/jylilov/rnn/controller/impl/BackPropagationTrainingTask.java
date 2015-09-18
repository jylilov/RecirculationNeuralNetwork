package by.jylilov.rnn.controller.impl;

import by.jylilov.rnn.controller.TrainingTask;
import by.jylilov.rnn.model.RecirculationNeuralNetwork;

import java.util.List;

public class BackPropagationTrainingTask extends TrainingTask {

    public BackPropagationTrainingTask(
            int n, int p,
            float alpha, float minError,
            List<float[]> trainingSet,
            boolean normalisation, boolean adaptiveLearningStep
    ) {
        super(n, p, alpha, minError, trainingSet, normalisation, adaptiveLearningStep);
    }

    @Override
    protected RecirculationNeuralNetwork call() {
        updateTitle("Back Propagation Training Network");

        float e = Float.MAX_VALUE;

        RecirculationNeuralNetwork rnn = new RecirculationNeuralNetwork(n, p);
        updateValue(rnn);

        long iteration = 0;

        float [][]w = rnn.getW();
        float [][]w_ = rnn.getW_();

        float []x_ = new float[n];
        float []y = new float[p];
        float []gamma = new float[p];

        while (e > minError) {
            e = 0;
            ++iteration;

            if (isCancelled()) {
                break;
            }

            for (float [] x: trainingSet) {
                rnn.directStage(x, y);
                rnn.reverseStage(y, x_);

                float alpha;
                float alpha_;

                if (adaptiveLearningStep) {
                    alpha = alpha_ = 0;
                    for (int i = 0; i < n; ++i) {
                        alpha += Math.pow(x[i], 2);
                    }
                    alpha = 1 / alpha;

                    for (int j = 0; j < p; ++j) {
                        alpha_ += Math.pow(y[j], 2);
                    }
                    alpha_ = 1 / alpha_;
                } else {
                    alpha = alpha_ = this.alpha;
                }

                for (int j = 0; j < p; ++j) {
                    gamma[j] = 0;
                    for (int i = 0; i < n; ++i) {
                        gamma[j] += (x_[i] - x[i]) * w_[j][i];
                    }
                }

                for (int i = 0; i < n; ++i) {
                    for (int j = 0; j < p; ++j) {
                        w[i][j] -= alpha * gamma[j] * x[i];
                        w_[j][i] -= alpha_ * (x_[i] - x[i]) * y[j];
                    }
                }

                if (normalization) {
                    rnn.normalizeW();
                    rnn.normalizeW_();
                }
            }

            for (float [] x: trainingSet) {
                rnn.directStage(x, y);
                rnn.reverseStage(y, x_);
                e += getError(x, x_);
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

        return e;
    }
}
