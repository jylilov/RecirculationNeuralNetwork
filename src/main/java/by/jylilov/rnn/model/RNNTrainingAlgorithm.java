package by.jylilov.rnn.model;

import by.jylilov.rnn.controller.RNNTrainingTask;
import by.jylilov.rnn.controller.RNNTrainingTaskBuilder;
import by.jylilov.rnn.controller.impl.RNNBackPropagationTrainingTaskBuilder;
import by.jylilov.rnn.controller.impl.RNNCumulativeDeltaRuleTrainingTaskBuilder;
import by.jylilov.rnn.controller.impl.RNNStratifiedTrainingTaskBuilder;

import java.util.List;

public enum RNNTrainingAlgorithm {
    STRATIFIED_TRAINING("Stratified training", new RNNStratifiedTrainingTaskBuilder()),
    CUMULATIVE_DELTA_RULE("Cumulative Delta Rule", new RNNCumulativeDeltaRuleTrainingTaskBuilder()),
    BACK_PROPAGATION("Back Propagation", new RNNBackPropagationTrainingTaskBuilder());

    private final String name;
    private final RNNTrainingTaskBuilder builder;

    RNNTrainingAlgorithm(String name, RNNTrainingTaskBuilder builder) {
        this.name = name;
        this.builder = builder;
    }

    public RNNTrainingTask buildTrainingTask(
            int n, int p,
            float alpha, float minError,
            List<float[]> trainingSet,
            boolean isWithNormalisation, boolean isWithAdaptiveLearningStep
    ) {
        return builder.build(
                n, p,
                alpha, minError,
                trainingSet,
                isWithNormalisation, isWithAdaptiveLearningStep
        );
    }

    @Override
    public String toString() {
        return name;
    }
}
