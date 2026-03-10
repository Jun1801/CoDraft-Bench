import numpy as np
def get_preds_multi(trainer, test_ds, df_test):
    test_output = trainer.predict(test_ds)
    predictions = test_output.predictions

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    reshaped_logits = predictions.reshape(-1, 2, 5)
    avg_logits = reshaped_logits.mean(axis=1)

    test_preds = np.argmax(avg_logits, axis=-1)
    test_true = df_test["label_score"].values
    return (test_preds, test_true)
def get_preds(model, df_test):
    test_inputs = [
        [str(row['input_text_1']), str(row['input_text_2'])]
        for i, row in df_test.iterrows()
    ]
    test_output = model.predict(test_inputs)
    test_preds = np.argmax(test_output, axis=1)
    test_true = df_test["label_score"].values
    return (test_preds, test_true)
