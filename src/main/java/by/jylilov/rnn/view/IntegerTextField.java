package by.jylilov.rnn.view;

import javafx.beans.property.*;
import javafx.scene.control.TextField;
import javafx.scene.control.TextFormatter;
import javafx.util.StringConverter;

public class IntegerTextField extends TextField {

    private TextFormatter<Integer> formatter;

    private final IntegerProperty value = new SimpleIntegerProperty();
    private final IntegerProperty minValue = new SimpleIntegerProperty(Integer.MIN_VALUE);
    private final IntegerProperty maxValue = new SimpleIntegerProperty(Integer.MAX_VALUE);

    public IntegerTextField() {
        this(Integer.MIN_VALUE, Integer.MAX_VALUE, 0);
    }

    public IntegerTextField(int minValue, int maxValue, int defaultValue) {
        initTextFormatter();
        initValueListeners();
        initValues(minValue, maxValue, defaultValue);
    }

    private void initValueListeners() {
        value.addListener((observable, oldValue, newValue) -> {
            if (newValue.intValue() < getMinValue() || newValue.intValue() > getMaxValue()) {
                setValue(oldValue.intValue());
            } else {
                formatter.setValue(newValue.intValue());
            }
        });
        minValue.addListener((observable, oldValue, newValue) -> {
            if (newValue.intValue() > getMaxValue()) {
                setMinValue(oldValue.intValue());
            } else if (newValue.intValue() > getValue()) {
                setValue(newValue.intValue());
            }
        });
        maxValue.addListener((observable, oldValue, newValue) -> {
            if (newValue.intValue() < getMinValue()) {
                setMinValue(oldValue.intValue());
            } else if (newValue.intValue() < getValue()) {
                setValue(newValue.intValue());
            }
        });
    }

    private void initValues(int minValue, int maxValue, int defaultValue) {
        setMinValue(minValue);
        setMaxValue(maxValue);
        setValue(defaultValue);
    }

    private void initTextFormatter() {
        formatter = new TextFormatter<>(new StringConverter<Integer>() {
            @Override
            public String toString(Integer object) {
                if (object == null) return "";
                return object.toString();
            }

            @Override
            public Integer fromString(String string) {
                if (string == null) return getValue();
                try {
                    setValue(Integer.parseInt(string));
                    return getValue();
                } catch (NumberFormatException e) {
                    return getValue();
                }
            }
        });
        setTextFormatter(formatter);
    }

    public int getValue() {
        return value.get();
    }

    public IntegerProperty valueProperty() {
        return value;
    }

    public void setValue(int value) {
        this.value.set(value);
        formatter.setValue(value);
    }

    public int getMinValue() {
        return minValue.get();
    }

    public IntegerProperty minValueProperty() {
        return minValue;
    }

    public void setMinValue(int minValue) {
        this.minValue.set(minValue);
    }

    public int getMaxValue() {
        return maxValue.get();
    }

    public IntegerProperty maxValueProperty() {
        return maxValue;
    }

    public void setMaxValue(int maxValue) {
        this.maxValue.set(maxValue);
    }
}