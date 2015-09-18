package by.jylilov.rnn.controller;

import java.util.List;

public interface TrainingTaskBuilder {
    TrainingTask build(int n, int p,
                          float alpha, float minError,
                          List<float[]> trainingSet,
                          boolean normalisation, boolean adaptiveLearningStep);
}
