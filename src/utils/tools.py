class YOLOV3Props:
    def __init__(self, path) -> None:
        self.module_defs = []
        self.set_props(path)

        self.hyperparameters: dict = self.convert_type(
            type='net', original_dict=self.module_defs[0])

    def convert_type(self, type, original_dict: dict):
        if type == 'net':
            return {
                'batch': int(original_dict['batch']),
                'momentum': float(original_dict['momentum']),
                'decay': float(original_dict['decay']),
                'saturation': float(original_dict['saturation']),
                'learning_rate': float(original_dict['learning_rate']),
                'burn_in': int(original_dict['burn_in']),
                'max_batches': int(original_dict['max_batches']),
                'policy': original_dict['policy'],
                'subdivisions': int(original_dict['subdivisions']),
                'width': int(original_dict['width']),
                'height': int(original_dict['height']),
                'class': int(original_dict['class']),
                'channels': int(original_dict['channels']),
                'ignore_cls': int(original_dict['ignore_cls']),
            }

    def set_props(self, path):
        raw_str_list = self.parse_hyperparameter_config_from(path)
        self.module_defs = []
        for ln in raw_str_list:
            if ln.startswith('['):
                type_name = ln[1:-1]
                di = {'type': type_name}
                if type_name == 'convolutional':
                    di['batch_normalize'] = 0
                self.module_defs.append(di)
            else:
                k, v = ln.split('=')
                k, v = k.strip(), v.strip()
                self.module_defs[-1][k] = v

    def parse_hyperparameter_config_from(self, path):
        with open(path, 'r') as file:
            lines = [ln.strip() for ln in file.readlines()
                     if ln.strip() and not ln.startswith('#')]
        return lines
