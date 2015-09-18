package by.jylilov.rnn.util;

import javafx.scene.image.Image;
import javafx.scene.image.WritableImage;
import javafx.scene.paint.Color;

import java.util.ArrayList;
import java.util.List;

public class ImageUtils {

    public static List<float []> getDataSet(Image image, int w, int h) {
        List<float []> list = new ArrayList<>();

        int width = (int)image.getWidth();
        int height = (int)image.getHeight();

        for (int i = 0; i < width; i += w) {
            if (i + w > width) i = width - w;
            for (int j = 0; j < height; j += h) {
                if (j + h > height) j = height - h;

                int l = 0;
                float x[] = new float[w * h * 3];

                for (int ii = 0; ii < w; ++ii) {
                    for (int jj = 0; jj < h; ++jj) {
                        Color color = image.getPixelReader().getColor(i + ii, j + jj);
                        x[l++] = directConvert(color.getRed());
                        x[l++] = directConvert(color.getGreen());
                        x[l++] = directConvert(color.getBlue());
                    }
                }

                list.add(x);
            }
        }

        return list;
    }

    public static Image restoreImage(List<float []> list, int w, int h, int width, int height) {
        WritableImage image = new WritableImage(width, height);

        int k = 0;
        for (int i = 0; i < width; i += w) {
            if (i + w > width) i = width - w;
            for (int j = 0; j < height; j += h) {
                if (j + h > height) j = height - h;

                int l = 0;
                float x[] = list.get(k++);

                for (int ii = 0; ii < w; ++ii) {
                    for (int jj = 0; jj < h; ++jj) {
                        float r = reverseConvert(x[l++]);
                        float g = reverseConvert(x[l++]);
                        float b = reverseConvert(x[l++]);
                        image.getPixelWriter().setColor(i + ii, j + jj, new Color(r, g, b, 1));
                    }
                }
            }
        }

        return image;
    }

    private static float directConvert(double a) {
        return (float) (a * 2 -  1);
    }

    private static float reverseConvert(float a) {
        float ans = (a + 1) / 2;
        if (ans < 0) ans = 0;
        if (ans > 1) ans = 1;
        return ans;
    }


}
