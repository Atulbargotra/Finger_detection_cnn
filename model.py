from keras.models import model_from_json
import numpy as np

class GestureModel(object):

    GESTUTRE_LIST = ['0R','1R','2R','3R','4R','5R','0L','1L','2L','3L','4L','5L']

    def __init__(self, model_json_file, model_weights_file):
        # load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        self.loaded_model._make_predict_function()

    def predict(self, img):
        self.pred = self.loaded_model.predict(img)
        return GestureModel.GESTUTRE_LIST[np.argmax(self.pred)]
