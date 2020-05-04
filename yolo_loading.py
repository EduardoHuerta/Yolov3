from yolo_archivos import *



# define the model
model = make_yolov3_model()

# load the model weights
weight_reader = WeightReader('yolov3.weights')

# set the model weights into the model
weight_reader.load_weights(model)

# save the model to file
model.save('yolov3_5_Clases.h5')