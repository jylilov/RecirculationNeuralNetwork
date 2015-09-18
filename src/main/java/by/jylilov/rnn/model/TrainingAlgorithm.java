package by.jylilov.rnn.model;

import by.jylilov.rnn.controller.TrainingTask;
import by.jylilov.rnn.controller.TrainingTaskBuilder;
import by.jylilov.rnn.controller.impl.BackPropagationTrainingTaskBuilder;
import by.jylilov.rnn.controller.impl.CumulativeDeltaRuleTrainingTaskBuilder;
import by.jylilov.rnn.controller.impl.StratifiedTrainingTaskBuilder;

import java.util.List;

public enum TrainingAlgorithm {
    STRATIFIED_TRAINING("Stratified training", new StratifiedTrainingTaskBuilder()),
    CUMULATIVE_DELTA_RULE("Cumulative Delta Rule", new CumulativeDeltaRuleTrainingTaskBuilder()),
    BACK_PROPAGATION("Back Propagation", new BackPropagationTrainingTaskBuilder());

    private final String name;
    private final TrainingTaskBuilder builder;

    TrainingAlgorithm(String name, TrainingTaskBuilder builder) {
        this.name = name;
        this.builder = builder;
    }

    public TrainingTask buildTrainingTask(
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
