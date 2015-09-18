package by.jylilov.rnn.model;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class RecirculationNeuralNetwork {
    private final int n;
    private final int p;

    private final float w[][];
    private final float w_[][];

    public RecirculationNeuralNetwork(int n, int p) {
        this.n = n;
        this.p = p;
        this.w = new float[n][p];
        this.w_ = new float[p][n];
        initNetwork();
    }

    private void initNetwork() {
        Random random = new Random(137);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < p; ++j) {
                w[i][j] = w_[j][i] = random.nextFloat();
            }
        }
        normalizeW();
        normalizeW_();
    }

    public void normalizeW_() {
        for (int j = 0; j < p; ++j) {
            float sum = 0;
            for (int i = 0; i < n; ++i) {
                sum += Math.pow(w_[j][i], 2);
            }
            sum = (float) Math.sqrt(sum);

            for (int i = 0; i < n; ++i) {
                w_[j][i] /= sum;
            }
        }
    }

    public void normalizeW() {
        for (int i = 0; i < n; ++i) {
            float sum = 0;
            for (int j = 0; j < p; ++j) {
                sum += Math.pow(w[i][j], 2);
            }
            sum = (float) Math.sqrt(sum);

            for (int j = 0; j < p; ++j) {
                w[i][j] /= sum;
            }
        }
    }

    public void directStage(float[] x, float[] y) {
        for (int j = 0; j < p; ++j) {
            y[j] = 0;
            for (int i = 0; i < n; ++i) {
                y[j] += w[i][j] * x[i];
            }
        }
    }

    public void reverseStage(float[] y, float[] x) {
        for (int i = 0; i < n; ++i) {
            x[i] = 0;
            for (int j = 0; j < p; ++j) {
                x[i] += w_[j][i] * y[j];
            }
        }
    }

    public List<float []> process(List<float []> dataSet) {
        List<float []> list = new ArrayList<>();
        float []a;
        float []y = new float[p];
        for (float [] x: dataSet) {
            a = new float[n];
            directStage(x, y);
            reverseStage(y, a);
            list.add(a);
        }
        return list;
    }

    public float[][] getW() {
        return w;
    }

    public float[][] getW_() {
        return w_;
    }
}
