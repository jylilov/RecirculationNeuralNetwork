package by.jylilov.rnn;

import by.jylilov.rnn.controller.RNNTrainingService;
import by.jylilov.rnn.model.RNNTrainingAlgorithm;
import by.jylilov.rnn.view.ServiceDialog;
import by.jylilov.rnn.view.FloatTextField;
import by.jylilov.rnn.view.IntegerTextField;
import javafx.application.Application;
import javafx.beans.value.ChangeListener;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.geometry.Insets;
import javafx.scene.Node;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.*;
import javafx.stage.Stage;

public class RNNApplication extends Application {

    private static final Image DEFAULT_IMAGE = new Image("/default_image.png");

    private final RNNTrainingService trainingService = new RNNTrainingService();

    private final Button startTrainingButton = new Button("Start training");

    private final BorderPane rootPane = new BorderPane();

    private final IntegerTextField nTextField = new IntegerTextField();
    private final IntegerTextField pTextField = new IntegerTextField(1, Integer.MAX_VALUE, 2);
    private final IntegerTextField lTextField = new IntegerTextField();
    private final IntegerTextField wTextField = new IntegerTextField(1, Integer.MAX_VALUE, 64);
    private final IntegerTextField hTextField = new IntegerTextField(1, Integer.MAX_VALUE, 64);
    private final FloatTextField zTextField = new FloatTextField();
    private final FloatTextField alphaTextField = new FloatTextField(0, Float.MAX_VALUE, 0.01f);
    private final FloatTextField minErrorTextField = new FloatTextField(0, Float.MAX_VALUE, 1e-5f);
    private final ComboBox<RNNTrainingAlgorithm> algorithmComboBox = new ComboBox<>();
    private final CheckBox adaptiveLearningStepCheckBox = new CheckBox("Adaptive Learning Step");
    private final CheckBox normalizationCheckBox = new CheckBox("Normalization");

    private final ImageView sourceImageView = new ImageView(DEFAULT_IMAGE);
    private final ImageView outImageView = new ImageView();

    @Override
    public void start(Stage primaryStage) throws Exception {
        Scene scene = new Scene(rootPane, 900, 450);

        initView();

        initTrainingService();

        startTrainingButton.setOnAction(event -> {
            trainingService.reset();
            trainingService.start();
            ServiceDialog dialog = new ServiceDialog(trainingService);
            dialog.showAndWait().ifPresent(response -> {
                if (response == ButtonType.CANCEL) {
                    trainingService.cancel();
                } else if (response == ButtonType.FINISH) {
                    trainingService.cancel();
                    outImageView.setImage(trainingService.getOutImage());
                }
            });
        });

        primaryStage.setScene(scene);
        primaryStage.setTitle("Recirculation Neural Network Application");
        primaryStage.show();
    }

    private void initView() {
        rootPane.setLeft(initSettingsPane());
        rootPane.setCenter(initImageSplitView());
    }

    private void initTrainingService() {
        trainingService.pProperty().bind(pTextField.valueProperty());
        trainingService.wProperty().bind(wTextField.valueProperty());
        trainingService.hProperty().bind(hTextField.valueProperty());
        trainingService.alphaProperty().bind(alphaTextField.valueProperty());
        trainingService.minErrorProperty().bind(minErrorTextField.valueProperty());
        trainingService.setSourceImage(DEFAULT_IMAGE);
        trainingService.isWithAdaptiveLearningStepProperty().bind(adaptiveLearningStepCheckBox.selectedProperty());
        trainingService.isWithNormalizationProperty().bind(normalizationCheckBox.selectedProperty());
        trainingService.algorithmProperty().bind(algorithmComboBox.valueProperty());
    }

    private Node initImageSplitView() {
        SplitPane splitPane = new SplitPane();
        splitPane.getItems().addAll(
                initImageView(sourceImageView),
                initImageView(outImageView)
        );
        splitPane.setDividerPositions(0.5);
        return splitPane;
    }

    private Node initImageView(ImageView imageView) {
        ScrollPane scrollPane = new ScrollPane();
        scrollPane.setFitToHeight(true);
        scrollPane.setFitToWidth(true);
        scrollPane.setContent(new StackPane(imageView));
        return scrollPane;
    }

    private Node initSettingsPane() {
        GridPane settings = new GridPane();

        settings.setHgap(5);
        settings.setVgap(10);
        settings.setPadding(new Insets(10));

        settings.add(new Label("P:"), 1, 1);
        settings.add(pTextField, 2, 1);

        settings.add(new Label("W:"), 1, 2);
        settings.add(wTextField, 2, 2);

        settings.add(new Label("H:"), 1, 3);
        settings.add(hTextField, 2, 3);

        settings.add(new Label("N:"), 1, 4);
        settings.add(nTextField, 2, 4);
        nTextField.setEditable(false);

        settings.add(new Label("L:"), 1, 5);
        settings.add(lTextField, 2, 5);
        lTextField.setEditable(false);

        settings.add(new Label("Z:"), 1, 6);
        settings.add(zTextField, 2, 6);
        zTextField.setEditable(false);

        settings.add(alphaTextField, 2, 7);
        settings.add(new Label("Alpha:"), 1, 7);

        settings.add(minErrorTextField, 2, 8);
        settings.add(new Label("Min error:"), 1, 8);

        algorithmComboBox.getItems().addAll(RNNTrainingAlgorithm.values());
        algorithmComboBox.setValue(RNNTrainingAlgorithm.STRATIFIED_TRAINING);
        settings.add(algorithmComboBox, 2, 9);
        settings.add(new Label("Algorithm:"), 1, 9);

        settings.add(adaptiveLearningStepCheckBox, 1, 10, 2, 1);
        adaptiveLearningStepCheckBox.setSelected(true);

        settings.add(normalizationCheckBox, 1, 11, 2, 1);
        normalizationCheckBox.setSelected(true);

        settings.add(startTrainingButton, 1, 12, 2, 1);

        bindSettingValues();

        return settings;
    }

    private void bindSettingValues() {
        nTextField.valueProperty().bind(wTextField.valueProperty().multiply(hTextField.valueProperty()).multiply(3));
        wTextField.maxValueProperty().bind(sourceImageView.getImage().widthProperty());
        hTextField.maxValueProperty().bind(sourceImageView.getImage().widthProperty());
        ChangeListener<Number> lListener = (observable, oldValue, newValue) -> updateLProperty();
        wTextField.valueProperty().addListener(lListener);
        hTextField.valueProperty().addListener(lListener);
        ChangeListener<Number> zListener = (observable, oldValue, newValue) -> updateZProperty();
        nTextField.valueProperty().addListener(zListener);
        lTextField.valueProperty().addListener(zListener);
        pTextField.valueProperty().addListener(zListener);
        updateLProperty();
        updateZProperty();
    }

    private void updateZProperty() {
        float archive =(nTextField.getValue() + lTextField.getValue()) * pTextField.getValue();
        float source = nTextField.getValue() * lTextField.getValue();
        zTextField.setValue(archive / source);
    }

    private void updateLProperty() {
        int width = (int) sourceImageView.getImage().getWidth();
        int height = (int) sourceImageView.getImage().getHeight();
        int w = wTextField.getValue();
        int h = hTextField.getValue();
        int countW = width / w + (width % w != 0 ? 1 : 0);
        int countH = height / h + (height % h != 0 ? 1 : 0);
        lTextField.setValue(countW * countH);
    }

    public static void main(String [] args) {
        RNNApplication.launch(args);
    }

}
