package by.jylilov.rnn.view;

import by.jylilov.rnn.RNNApplication;
import by.jylilov.rnn.model.TrainingAlgorithm;
import javafx.beans.property.BooleanProperty;
import javafx.beans.property.FloatProperty;
import javafx.beans.property.IntegerProperty;
import javafx.beans.property.ObjectProperty;
import javafx.beans.value.ChangeListener;
import javafx.geometry.Insets;
import javafx.scene.control.CheckBox;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Label;
import javafx.scene.image.Image;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.Pane;

public class Settings extends Pane {
    private final RNNApplication application;

    private final IntegerTextField nTextField = new IntegerTextField();
    private final IntegerTextField pTextField = new IntegerTextField(1, Integer.MAX_VALUE, 24);
    private final IntegerTextField lTextField = new IntegerTextField();
    private final IntegerTextField wTextField = new IntegerTextField(1, Integer.MAX_VALUE, 4);
    private final IntegerTextField hTextField = new IntegerTextField(1, Integer.MAX_VALUE, 4);
    private final FloatTextField zTextField = new FloatTextField();
    private final FloatTextField alphaTextField = new FloatTextField(0, Float.MAX_VALUE, 0.01f);
    private final FloatTextField minErrorTextField = new FloatTextField(0, Float.MAX_VALUE, 1e-5f);
    private final ComboBox<TrainingAlgorithm> algorithmComboBox = new ComboBox<>();
    private final CheckBox adaptiveLearningStepCheckBox = new CheckBox("Adaptive Learning Step");
    private final CheckBox normalizationCheckBox = new CheckBox("Normalization");

    public Settings(RNNApplication application) {
        this.application = application;

        initCheckBoxes();
        initAlgorithmComboBox();
        initNotEditable();

        bindSettingValues();

        getChildren().add(initGrid());
    }

    private void initCheckBoxes() {
        adaptiveLearningStepCheckBox.setSelected(false);
        normalizationCheckBox.setSelected(false);
    }

    private void initNotEditable() {
        nTextField.setEditable(false);
        lTextField.setEditable(false);
        zTextField.setEditable(false);
    }

    private GridPane initGrid() {
        GridPane gridPane = new GridPane();

        gridPane.setHgap(5);
        gridPane.setVgap(10);
        gridPane.setPadding(new Insets(10));

        int line = 1;

        gridPane.add(new Label("Algorithm:"), 1, line);
        gridPane.add(algorithmComboBox, 2, line);
        ++line;

        gridPane.add(adaptiveLearningStepCheckBox, 1, line++, 2, 1);
        gridPane.add(normalizationCheckBox, 1, line++, 2, 1);

        gridPane.add(new Label("P:"), 1, line);
        gridPane.add(pTextField, 2, line);
        ++line;

        gridPane.add(new Label("W:"), 1, line);
        gridPane.add(wTextField, 2, line);
        ++line;

        gridPane.add(new Label("H:"), 1, line);
        gridPane.add(hTextField, 2, line);
        ++line;

        gridPane.add(new Label("N:"), 1, line);
        gridPane.add(nTextField, 2, line);
        ++line;

        gridPane.add(new Label("L:"), 1, line);
        gridPane.add(lTextField, 2, line);
        ++line;

        gridPane.add(new Label("Z:"), 1, line);
        gridPane.add(zTextField, 2, line);
        ++line;

        gridPane.add(alphaTextField, 2, line);
        gridPane.add(new Label("Alpha:"), 1, line);
        ++line;

        gridPane.add(minErrorTextField, 2, line);
        gridPane.add(new Label("Min error:"), 1, line);
        ++line;

        return gridPane;
    }

    private void initAlgorithmComboBox() {
        algorithmComboBox.getItems().addAll(TrainingAlgorithm.values());
        algorithmComboBox.setValue(TrainingAlgorithm.BACK_PROPAGATION);
    }

    private void bindSettingValues() {
        nTextField.valueProperty().bind(wTextField.valueProperty().multiply(hTextField.valueProperty()).multiply(3));

        application.sourceImageProperty().addListener((observable, oldValue, newValue) -> updateWHProperty());

        ChangeListener<Number> lUpdate = (observable, oldValue, newValue) -> updateLProperty();
        wTextField.valueProperty().addListener(lUpdate);
        hTextField.valueProperty().addListener(lUpdate);

        ChangeListener<Number> zUpdate = (observable, oldValue, newValue) -> updateZProperty();
        nTextField.valueProperty().addListener(zUpdate);
        lTextField.valueProperty().addListener(zUpdate);
        pTextField.valueProperty().addListener(zUpdate);

        updateWHProperty();
        updateLProperty();
        updateZProperty();
    }

    private void updateWHProperty() {
        Image image = application.sourceImageProperty().get();
        wTextField.setMaxValue(image.widthProperty().intValue());
        hTextField.setMaxValue(image.heightProperty().intValue());
    }

    private void updateZProperty() {
        float out =(nTextField.getValue() + lTextField.getValue()) * pTextField.getValue() + 2;
        float source = nTextField.getValue() * lTextField.getValue() + 2;
        zTextField.setValue(out / source);
    }

    private void updateLProperty() {
        Image image = application.sourceImageProperty().get();
        int width = image.widthProperty().intValue();
        int height = image.heightProperty().intValue();
        int w = wTextField.getValue();
        int h = hTextField.getValue();
        int countW = width / w + (width % w != 0 ? 1 : 0);
        int countH = height / h + (height % h != 0 ? 1 : 0);
        lTextField.setValue(countW * countH);
    }

    public FloatProperty zProperty() {
        return zTextField.valueProperty();
    }

    public IntegerProperty pProperty() {
        return pTextField.valueProperty();
    }

    public IntegerProperty wProperty() {
        return wTextField.valueProperty();
    }

    public IntegerProperty hProperty() {
        return hTextField.valueProperty();
    }

    public FloatProperty minErrorProperty() {
        return minErrorTextField.valueProperty();
    }

    public FloatProperty alphaProperty() {
        return alphaTextField.valueProperty();
    }

    public BooleanProperty adaptiveTrainingStepProperty() {
        return adaptiveLearningStepCheckBox.selectedProperty();
    }

    public BooleanProperty normalizationProperty() {
        return normalizationCheckBox.selectedProperty();
    }

    public ObjectProperty<TrainingAlgorithm> algorithmProperty() {
        return algorithmComboBox.valueProperty();
    }
}
