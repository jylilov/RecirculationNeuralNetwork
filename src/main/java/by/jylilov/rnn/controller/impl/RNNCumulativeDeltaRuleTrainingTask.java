package by.jylilov.rnn.controller.impl;

import by.jylilov.rnn.controller.RNNTrainingTask;
import by.jylilov.rnn.model.RNN;
import javafx.concurrent.Task;

import java.util.List;

public class RNNCumulativeDeltaRuleTrainingTask extends RNNTrainingTask {

    public RNNCumulativeDeltaRuleTrainingTask(
            int n, int p,
            float alpha, float minError,
            List<float[]> trainingSet,
            boolean isWithNormalisation, boolean isWithAdaptiveLearningStep
    ) {
        super(n, p, alpha, minError, trainingSet, isWithNormalisation, isWithAdaptiveLearningStep);
    }

    @Override
    protected RNN call() {
        updateTitle("Cumulative Delta Rule Training Network");

        float e = Float.MAX_VALUE;

        RNN rnn = new RNN(n, p);
        updateValue(rnn);

        long iteration = 0;

        float [][]w = rnn.getW();
        float [][]w_ = rnn.getW_();


        float []x_ = new float[n];
        float []y = new float[p];
        float []y_ = new float[p];

        while (e > minError) {
            e = 0;
            ++iteration;

            if (isCancelled()) {
                break;
            }

            for (float [] x: trainingSet) {
                rnn.directStage(x, y);
                rnn.reverseStage(y, x_);
                rnn.directStage(x_, y_);

                float alpha;
                float alpha_;

                if (isWithAdaptiveLearningStep) {
                    alpha = alpha_ = 0;
                    for (int i = 0; i < n; ++i) {
                        alpha += Math.pow(x_[i], 2);
                    }
                    alpha = 1 / alpha;

                    for (int j = 0; j < p; ++j) {
                        alpha_ += Math.pow(y[j], 2);
                    }
                    alpha_ = 1 / alpha_;
                } else {
                    alpha = alpha_ = this.alpha;
                }

                for (int i = 0; i < n; ++i) {
                    for (int j = 0; j < p; ++j) {
                        w[i][j] -= alpha * (y_[j] - y[j]) * x_[i];
                        w_[j][i] -= alpha_ * (x_[i] - x[i]) * y[j];
                    }
                }

                if (isWithNormalisation) {
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
