package by.jylilov.rnn.controller;

import by.jylilov.rnn.model.RNN;
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
//        rnnTask = new RNNTrainingTask(n.get(), p.get(), minError.get(), trainingSet);

        trainingSet = ImageUtils.getDataSet(sourceImage, w.get(), h.get());
        return new RNNTrainingTask(w.get() * h.get() * 3, p.get(), minError.get(), trainingSet);
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
}
