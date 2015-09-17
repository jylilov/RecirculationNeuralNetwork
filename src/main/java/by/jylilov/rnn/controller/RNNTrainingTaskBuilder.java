package by.jylilov.rnn.controller;

import java.util.List;

public interface RNNTrainingTaskBuilder {
    RNNTrainingTask build(int n, int p,
                          float alpha, float minError,
                          List<float[]> trainingSet,
                          boolean isWithNormalisation, boolean isWithAdaptiveLearningStep);
}
