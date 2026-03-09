import shutil
def save_model(trainer, tokenizer, model_name, save_path):
    print(f"Saving: {save_path} ...")
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    shutil.make_archive(model_name, 'zip', save_path)
    print("Success!")
