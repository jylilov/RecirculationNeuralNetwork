package by.jylilov.rnn.controller.impl;

import by.jylilov.rnn.controller.TrainingTask;
import by.jylilov.rnn.controller.TrainingTaskBuilder;

import java.util.List;

public class BackPropagationTrainingTaskBuilder implements TrainingTaskBuilder {

    @Override
    public TrainingTask build(
            int n, int p,
            float alpha, float minError,
            List<float[]> trainingSet,
            boolean normalisation, boolean adaptiveLearningStep
    ) {
        return new BackPropagationTrainingTask(
                n, p,
                alpha, minError,
                trainingSet,
                normalisation, adaptiveLearningStep
        );
    }

}
