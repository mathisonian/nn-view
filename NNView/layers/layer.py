class Layer(object):
    def __init__(self, layer_type, input_shape, output_shape, activation, channels=None, kernel=None, stride=None):
        self.type = layer_type
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.activation = activation
        self.channels = channels
        self.kernel_size = kernel
        self.stride = stride
        
