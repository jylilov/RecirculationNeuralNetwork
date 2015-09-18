package by.jylilov.rnn.controller;

import by.jylilov.rnn.model.RecirculationNeuralNetwork;
import javafx.concurrent.Task;

import java.util.List;

public abstract class TrainingTask extends Task<RecirculationNeuralNetwork> {
    protected final int n;
    protected final int p;
    protected final int l;
    protected final float alpha;
    protected final float minError;
    protected final List<float[]> trainingSet;
    protected final boolean normalization;
    protected final boolean adaptiveLearningStep;

    public TrainingTask(
            int n, int p,
            float alpha, float minError,
            List<float[]> trainingSet,
            boolean normalization, boolean adaptiveLearningStep
    ) {
        this.n = n;
        this.p = p;
        this.alpha = alpha;
        this.minError = minError;
        this.trainingSet = trainingSet;
        this.normalization = normalization;
        this.adaptiveLearningStep = adaptiveLearningStep;
        l = trainingSet.size();
    }
}
