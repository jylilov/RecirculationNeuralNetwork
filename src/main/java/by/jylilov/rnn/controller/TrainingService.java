package by.jylilov.rnn.controller;

import by.jylilov.rnn.model.RecirculationNeuralNetwork;
import by.jylilov.rnn.model.TrainingAlgorithm;
import by.jylilov.rnn.util.ImageUtils;
import javafx.beans.property.*;
import javafx.concurrent.Service;
import javafx.concurrent.Task;
import javafx.scene.image.Image;

import java.util.List;

public class TrainingService extends Service<RecirculationNeuralNetwork> {
    private final IntegerProperty p = new SimpleIntegerProperty();
    private final IntegerProperty w = new SimpleIntegerProperty();
    private final IntegerProperty h = new SimpleIntegerProperty();
    private final BooleanProperty normalization = new SimpleBooleanProperty(true);
    private final BooleanProperty adaptiveLearningStep = new SimpleBooleanProperty(true);
    private final ObjectProperty<TrainingAlgorithm> algorithm = new SimpleObjectProperty<>();
    private final ObjectProperty<Image> sourceImage = new SimpleObjectProperty<>();

    private final FloatProperty alpha = new SimpleFloatProperty(0.01f);
    private final FloatProperty minError = new SimpleFloatProperty();

    public TrainingService() {
    }

    @Override
    protected Task<RecirculationNeuralNetwork> createTask() {
        List<float []> trainingSet;

        trainingSet = ImageUtils.getDataSet(getSourceImage(), getW(), getH());
        return getAlgorithm().buildTrainingTask(
                getW() * getH() * 3, getP(),
                getAlpha(), getMinError(),
                trainingSet,
                getNormalization(), getAdaptiveLearningStep()
        );
    }

    public Image getOutImage() {
        return ImageUtils.restoreImage(
                getValue().process(ImageUtils.getDataSet(getSourceImage(), w.get(), h.get())),
                w.get(), h.get(),
                (int) getSourceImage().getWidth(), (int) getSourceImage().getHeight()
        );
    }

    public int getP() {
        return p.get();
    }

    public IntegerProperty pProperty() {
        return p;
    }

    public void setP(int p) {
        this.p.set(p);
    }

    public int getW() {
        return w.get();
    }

    public IntegerProperty wProperty() {
        return w;
    }

    public void setW(int w) {
        this.w.set(w);
    }

    public int getH() {
        return h.get();
    }

    public IntegerProperty hProperty() {
        return h;
    }

    public void setH(int h) {
        this.h.set(h);
    }

    public float getMinError() {
        return minError.get();
    }

    public FloatProperty minErrorProperty() {
        return minError;
    }

    public void setMinError(float minError) {
        this.minError.set(minError);
    }

    public boolean getNormalization() {
        return normalization.get();
    }

    public BooleanProperty normalizationProperty() {
        return normalization;
    }

    public void setNormalization(boolean normalization) {
        this.normalization.set(normalization);
    }

    public boolean getAdaptiveLearningStep() {
        return adaptiveLearningStep.get();
    }

    public BooleanProperty adaptiveLearningStepProperty() {
        return adaptiveLearningStep;
    }

    public void setAdaptiveLearningStep(boolean adaptiveLearningStep) {
        this.adaptiveLearningStep.set(adaptiveLearningStep);
    }

    public float getAlpha() {
        return alpha.get();
    }

    public FloatProperty alphaProperty() {
        return alpha;
    }

    public void setAlpha(float alpha) {
        this.alpha.set(alpha);
    }

    public TrainingAlgorithm getAlgorithm() {
        return algorithm.get();
    }

    public ObjectProperty<TrainingAlgorithm> algorithmProperty() {
        return algorithm;
    }

    public void setAlgorithm(TrainingAlgorithm algorithm) {
        this.algorithm.set(algorithm);
    }

    public Image getSourceImage() {
        return sourceImage.get();
    }

    public ObjectProperty<Image> sourceImageProperty() {
        return sourceImage;
    }

    public void setSourceImage(Image sourceImage) {
        this.sourceImage.set(sourceImage);
    }
}
