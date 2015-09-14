package by.jylilov.rnn.util;

import javafx.scene.image.Image;
import javafx.scene.image.WritableImage;
import javafx.scene.image.WritablePixelFormat;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

public class ImageUtils {

    public static List<float []> getDataSet(Image image, int w, int h) {
        List<float []> list = new ArrayList<>();

        WritablePixelFormat<ByteBuffer> format = WritablePixelFormat.getByteBgraInstance();
        byte a[] = new byte[w * h * 4];

        int width = (int)image.getWidth();
        int height = (int)image.getHeight();
        for (int i = 0; i < width; i += w) {
            if (i + w > width) i = width - w;
            for (int j = 0; j < height; j += h) {
                if (j + h > height) j = height - h;
                image.getPixelReader().getPixels(i, j, w, h, format, a, 0, w * 4);
                list.add(convert(a));
            }
        }

        return list;
    }

    private static float[] convert(byte [] buffer) {
        float[] answer = new float[buffer.length / 4 * 3];
        int k = 0;
        for (int i = 0; i < buffer.length; ++i) {
            if ((i + 1) % 4 == 0) {
                continue;
            }
            answer[k++] = buffer[i] / 128f;
        }
        return answer;
    }

    private static byte[] convert(float [] buffer) {
        byte[] answer = new byte[buffer.length / 3 * 4];
        int k = 0;
        for (int i = 0; i < buffer.length; ++i) {
            if ((k + 1) % 4 == 0) answer[k++] = -1;
            answer[k++] = (byte) (buffer[i] * 128);
        }
        answer[k] = -1;
        return answer;
    }

    public static Image restoreImage(List<float []> list, int w, int h, int width, int height) {
        WritableImage image = new WritableImage(width, height);

        WritablePixelFormat<ByteBuffer> format = WritablePixelFormat.getByteBgraInstance();

        int k = 0;
        for (int i = 0; i < width; i += w) {
            if (i + w > width) i = width - w;
            for (int j = 0; j < height; j += h) {
                if (j + h > height) j = height - h;
                image.getPixelWriter().setPixels(i, j, w, h, format, convert(list.get(k++)), 0, w * 4);
            }
        }

        return image;
    }


}
