import os
import sys
lib_path = os.path.abspath(os.path.join(sys.path[0], '..'))
sys.path.append(lib_path)
from src.Classifier import Classifier
from src.PreProcessing import PreProcessing
import numpy as np

classifier_classes = Classifier.classifier_dict()
abspath = os.path.abspath(sys.path[0])
CONFIG = {
    'frame size': 512,
    'overlap': 128,
    'is training': True,
    'is streaming': True,
    'data list': '../DataSet/DataList_all.txt',
    'classifier': ['all'],
    'argumentation': False
}


def main():
    preprocessor = PreProcessing(frame_size=CONFIG['frame size'],
                                 overlap=CONFIG['overlap'])
    if CONFIG['is streaming']:
        try:
            # Todo: Add streaming support.
            preprocessor_proc, queue_dict = preprocessor.process_stream()
            preprocessor_proc.start()
            preprocessor.recorder.start_streaming()
            while True:
                zcr = queue_dict['energy'].get(True)
                ep = queue_dict['endpoint'].get(True)
                effective_feature = preprocessor.effective_feature(zcr, ep)
                if len(effective_feature) == 0:
                    continue
                effective_feature = preprocessor.reshape(effective_feature, 25)
                for classifier_name, classifier_class in classifier_classes.items():
                    print(classifier_name)
                    classifier = classifier_class(None)
                    res = classifier.apply(effective_feature)
                    print(res)
        except Exception:
            preprocessor_proc.terminate()
            del preprocessor
            exit()
    else:
        wav_list, frame_list, energy_list, \
        zcr_list, endpoint_list, label_list = preprocessor.process(CONFIG['data list'])
        print('Data set Size:', len(wav_list))
        eff_zcr_list = np.zeros((1, 25))
        eff_label_list = []
        # Todo: Rewrite the relating preprocessor code.
        # Multiple data type mixed. Change the list of np array to pure np array.
        for i in range(len(energy_list)):
            temp = preprocessor.effective_feature(energy_list[i], endpoint_list[i])
            temp = preprocessor.reshape(temp, 25)
            if len(temp) != 0:
                eff_label_list.append(label_list[i])
            else:
                continue 
            eff_zcr_list = np.concatenate((eff_zcr_list, temp), axis=0)
        eff_zcr_list = eff_zcr_list[1:]

        if CONFIG['argumentation']:
            # Todo: Add data argumentation
            raise NotImplementedError
        if 'all' in CONFIG['classifier']:
            for classifier_name, classifier_class in classifier_classes.items():
                if CONFIG['is training']:
                    # Todo: Print classifier name
                    # Todo: Print training result and validation result
                    # Todo: Save the model to a dir.
                    print(classifier_name)
                    classifier = classifier_class(None)
                    classifier.train(eff_zcr_list, eff_label_list)


if __name__ == '__main__':
    main()
