package by.jylilov.rnn.controller;

import by.jylilov.rnn.model.RNN;
import javafx.concurrent.Task;

import java.util.List;

public abstract class RNNTrainingTask extends Task<RNN> {
    protected final int n;
    protected final int p;
    protected final int l;
    protected final float alpha;
    protected final float minError;
    protected final List<float[]> trainingSet;
    protected final boolean isWithNormalisation;
    protected final boolean isWithAdaptiveLearningStep;

    public RNNTrainingTask(
            int n, int p,
            float alpha, float minError,
            List<float[]> trainingSet,
            boolean isWithNormalisation, boolean isWithAdaptiveLearningStep
    ) {
        this.n = n;
        this.p = p;
        this.alpha = alpha;
        this.minError = minError;
        this.trainingSet = trainingSet;
        this.isWithNormalisation = isWithNormalisation;
        this.isWithAdaptiveLearningStep = isWithAdaptiveLearningStep;
        l = trainingSet.size();
    }
}
