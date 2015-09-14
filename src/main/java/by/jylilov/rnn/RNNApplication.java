package by.jylilov.rnn;

import by.jylilov.rnn.controller.RNNTrainingService;
import by.jylilov.rnn.view.ServiceDialog;
import by.jylilov.rnn.view.FloatTextField;
import by.jylilov.rnn.view.IntegerTextField;
import javafx.application.Application;
import javafx.beans.value.ChangeListener;
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
    private final FloatTextField minErrorTextField = new FloatTextField(0, Float.MAX_VALUE, 1e-5f);

    private final ImageView sourceImageView = new ImageView(DEFAULT_IMAGE);
    private final ImageView outImageView = new ImageView();

    @Override
    public void start(Stage primaryStage) throws Exception {
        Scene scene = new Scene(rootPane, 800, 275);

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
        trainingService.minErrorProperty().bind(minErrorTextField.valueProperty());
        trainingService.setSourceImage(DEFAULT_IMAGE);
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

        settings.add(new Label("N:"), 1, 1);
        settings.add(nTextField, 2, 1);
        nTextField.setEditable(false);

        settings.add(new Label("P:"), 1, 2);
        settings.add(pTextField, 2, 2);

        settings.add(new Label("L:"), 1, 3);
        settings.add(lTextField, 2, 3);
        lTextField.setEditable(false);

        settings.add(minErrorTextField, 2, 4);
        settings.add(new Label("Min error:"), 1, 4);

        settings.add(new Label("W:"), 1, 5);
        settings.add(wTextField, 2, 5);


        settings.add(new Label("H:"), 1, 6);
        settings.add(hTextField, 2, 6);

        settings.add(new BorderPane(startTrainingButton), 1, 7, 2, 1);

        bindSettingValues();

        return settings;
    }

    private void bindSettingValues() {
        nTextField.valueProperty().bind(wTextField.valueProperty().multiply(hTextField.valueProperty()).multiply(3));
        wTextField.maxValueProperty().bind(sourceImageView.getImage().widthProperty());
        hTextField.maxValueProperty().bind(sourceImageView.getImage().widthProperty());
        ChangeListener<Number> listener = (observable, oldValue, newValue) -> updateLProperty();
        wTextField.valueProperty().addListener(listener);
        hTextField.valueProperty().addListener(listener);
        updateLProperty();
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