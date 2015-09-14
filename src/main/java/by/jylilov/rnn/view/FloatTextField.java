package by.jylilov.rnn.view;

import javafx.beans.property.FloatProperty;
import javafx.beans.property.SimpleFloatProperty;
import javafx.scene.control.TextField;
import javafx.scene.control.TextFormatter;
import javafx.util.StringConverter;

public class FloatTextField extends TextField {

    private TextFormatter<Float> formatter;

    private final FloatProperty value = new SimpleFloatProperty();
    private final FloatProperty minValue = new SimpleFloatProperty(Float.MIN_VALUE);
    private final FloatProperty maxValue = new SimpleFloatProperty(Float.MAX_VALUE);

    public FloatTextField() {
        this(Float.MIN_VALUE, Float.MAX_VALUE, 0);
    }

    public FloatTextField(float minValue, float maxValue, float defaultValue) {
        initTextFormatter();
        initValueListeners();
        initValues(minValue, maxValue, defaultValue);
    }

    private void initValueListeners() {
        value.addListener((observable, oldValue, newValue) -> {
            if (newValue.floatValue() < getMinValue() || newValue.floatValue() > getMaxValue()) {
                setValue(oldValue.floatValue());
            } else {
                formatter.setValue(newValue.floatValue());
            }
        });
        minValue.addListener((observable, oldValue, newValue) -> {
            if (newValue.floatValue() > getMaxValue()) {
                setMinValue(oldValue.floatValue());
            } else if (newValue.floatValue() > getValue()) {
                setValue(newValue.floatValue());
            }
        });
        maxValue.addListener((observable, oldValue, newValue) -> {
            if (newValue.floatValue() < getMinValue()) {
                setMinValue(oldValue.floatValue());
            } else if (newValue.floatValue() < getValue()) {
                setValue(newValue.floatValue());
            }
        });
    }

    private void initValues(float minValue, float maxValue, float defaultValue) {
        setMinValue(minValue);
        setMaxValue(maxValue);
        setValue(defaultValue);
    }

    private void initTextFormatter() {
        formatter = new TextFormatter<>(new StringConverter<Float>() {
            @Override
            public String toString(Float object) {
                if (object == null) return "";
                return object.toString();
            }

            @Override
            public Float fromString(String string) {
                if (string == null) return getValue();
                try {
                    setValue(Float.parseFloat(string));
                    return getValue();
                } catch (NumberFormatException e) {
                    return getValue();
                }
            }
        });
        setTextFormatter(formatter);
    }

    public float getValue() {
        return value.get();
    }

    public FloatProperty valueProperty() {
        return value;
    }

    public void setValue(float value) {
        this.value.set(value);
    }

    public float getMaxValue() {
        return maxValue.get();
    }

    public FloatProperty maxValueProperty() {
        return maxValue;
    }

    public void setMaxValue(float maxValue) {
        this.maxValue.set(maxValue);
    }

    public float getMinValue() {
        return minValue.get();
    }

    public FloatProperty minValueProperty() {
        return minValue;
    }

    public void setMinValue(float minValue) {
        this.minValue.set(minValue);
    }
}