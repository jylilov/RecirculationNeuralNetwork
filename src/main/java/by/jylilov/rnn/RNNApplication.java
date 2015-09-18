package by.jylilov.rnn;

import by.jylilov.rnn.controller.TrainingService;
import by.jylilov.rnn.view.ServiceDialog;
import by.jylilov.rnn.view.Settings;
import javafx.application.Application;
import javafx.beans.property.ObjectProperty;
import javafx.scene.Node;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.*;
import javafx.stage.Stage;

public class RNNApplication extends Application {

    private static final Image DEFAULT_IMAGE = new Image("/default_image.png");

    private final TrainingService trainingService = new TrainingService();

    private final BorderPane rootPane = new BorderPane();

    private final ImageView sourceImageView = new ImageView(DEFAULT_IMAGE);
    private final ImageView outImageView = new ImageView();

    private final Settings settings = new Settings(this);
    private final Button startTrainingButton = new Button("Start training");

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
        rootPane.setLeft(settings);
        rootPane.setCenter(initImageSplitView());
        rootPane.setTop(new ToolBar(startTrainingButton));
    }

    private void initTrainingService() {
        trainingService.pProperty().bind(settings.pProperty());
        trainingService.wProperty().bind(settings.wProperty());
        trainingService.hProperty().bind(settings.hProperty());
        trainingService.alphaProperty().bind(settings.alphaProperty());
        trainingService.minErrorProperty().bind(settings.minErrorProperty());
        trainingService.adaptiveLearningStepProperty().bind(settings.adaptiveTrainingStepProperty());
        trainingService.normalizationProperty().bind(settings.normalizationProperty());
        trainingService.algorithmProperty().bind(settings.algorithmProperty());

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

    public ObjectProperty<Image> sourceImageProperty() {
        return sourceImageView.imageProperty();
    }


    public static void main(String [] args) {
        RNNApplication.launch(args);
    }

}
