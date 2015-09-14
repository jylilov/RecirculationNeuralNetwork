package by.jylilov.rnn.view;

import javafx.concurrent.Service;
import javafx.scene.control.Button;
import javafx.scene.control.ButtonType;
import javafx.scene.control.Dialog;

public class ServiceDialog extends Dialog<ButtonType> {
    public ServiceDialog(Service service) {
        initButtons();

        titleProperty().bind(service.titleProperty());
        contentTextProperty().bind(service.messageProperty());
    }

    private void initButtons() {
        getDialogPane().getButtonTypes().addAll(ButtonType.FINISH, ButtonType.CANCEL);

        Button finishButton = (Button) getDialogPane().lookupButton(ButtonType.FINISH);
        finishButton.setText("Stop");
    }
}
