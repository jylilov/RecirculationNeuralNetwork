package by.jylilov.rnn.controller.impl;

import by.jylilov.rnn.controller.RNNTrainingTask;
import by.jylilov.rnn.controller.RNNTrainingTaskBuilder;

import java.util.List;

public class RNNCumulativeDeltaRuleTrainingTaskBuilder implements RNNTrainingTaskBuilder {
    @Override
    public RNNTrainingTask build(
            int n, int p,
            float alpha, float minError,
            List<float[]> trainingSet,
            boolean isWithNormalisation, boolean isWithAdaptiveLearningStep
    ) {
        return new RNNCumulativeDeltaRuleTrainingTask(
                n, p,
                alpha, minError,
                trainingSet,
                isWithNormalisation, isWithAdaptiveLearningStep
        );
    }
}
