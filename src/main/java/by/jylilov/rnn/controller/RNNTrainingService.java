package by.jylilov.rnn.controller;

import by.jylilov.rnn.model.RNN;
import by.jylilov.rnn.model.RNNTrainingAlgorithm;
import by.jylilov.rnn.util.ImageUtils;
import javafx.beans.property.*;
import javafx.concurrent.Service;
import javafx.concurrent.Task;
import javafx.scene.image.Image;

import java.util.List;

public class RNNTrainingService extends Service<RNN> {
    private final IntegerProperty p = new SimpleIntegerProperty();
    private final IntegerProperty w = new SimpleIntegerProperty();
    private final IntegerProperty h = new SimpleIntegerProperty();
    private final BooleanProperty isWithNormalization = new SimpleBooleanProperty(true);
    private final BooleanProperty isWithAdaptiveLearningStep = new SimpleBooleanProperty(true);
    private final ObjectProperty<RNNTrainingAlgorithm> algorithm = new SimpleObjectProperty<>();

    private final FloatProperty alpha = new SimpleFloatProperty(0.01f);
    private final FloatProperty minError = new SimpleFloatProperty();

    private Image sourceImage;

    public RNNTrainingService() {
    }

    @Override
    protected Task<RNN> createTask() {
        List<float []> trainingSet;

//        TODO remove in other class
//
//        trainingSet = new ArrayList<>();
//        Random random = new Random(137);
//        for (int k = 0; k < l.get(); ++k) {
//            float[] x = new float[n.get()];
//            for (int i = 0; i < n.get(); ++i) {
//                x[i] = (random.nextFloat() - 0.5f) * 2;
//            }
//            trainingSet.add(x);
//        }
//        rnnTask = new RNNStratifiedTrainingTask(n.get(), p.get(), minError.get(), trainingSet);

        trainingSet = ImageUtils.getDataSet(sourceImage, getW(), getH());
        return getAlgorithm().buildTrainingTask(
                getW() * getH() * 3, getP(),
                getAlpha(), getMinError(),
                trainingSet,
                getIsWithNormalization(), getIsWithAdaptiveLearningStep()
        );
    }

    public Image getOutImage() {
        return ImageUtils.restoreImage(
                getValue().process(ImageUtils.getDataSet(sourceImage, w.get(), h.get())),
                w.get(), h.get(),
                (int) this.sourceImage.getWidth(), (int) this.sourceImage.getHeight()
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

    public Image getSourceImage() {
        return sourceImage;
    }

    public void setSourceImage(Image sourceImage) {
        this.sourceImage = sourceImage;
    }

    public boolean getIsWithNormalization() {
        return isWithNormalization.get();
    }

    public BooleanProperty isWithNormalizationProperty() {
        return isWithNormalization;
    }

    public void setIsWithNormalization(boolean isWithNormalization) {
        this.isWithNormalization.set(isWithNormalization);
    }

    public boolean getIsWithAdaptiveLearningStep() {
        return isWithAdaptiveLearningStep.get();
    }

    public BooleanProperty isWithAdaptiveLearningStepProperty() {
        return isWithAdaptiveLearningStep;
    }

    public void setIsWithAdaptiveLearningStep(boolean isWithAdaptiveLearningStep) {
        this.isWithAdaptiveLearningStep.set(isWithAdaptiveLearningStep);
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

    public RNNTrainingAlgorithm getAlgorithm() {
        return algorithm.get();
    }

    public ObjectProperty<RNNTrainingAlgorithm> algorithmProperty() {
        return algorithm;
    }

    public void setAlgorithm(RNNTrainingAlgorithm algorithm) {
        this.algorithm.set(algorithm);
    }
}
