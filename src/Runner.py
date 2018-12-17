import os
import sys
lib_path = os.path.abspath(os.path.join(sys.path[0], '..'))
sys.path.append(lib_path)
from src.Classifier import Classifier
from src.PreProcessing import PreProcessing
import numpy as np
import matplotlib.pyplot as plt
import time

classifier_classes = Classifier.classifier_dict()
abspath = os.path.abspath(sys.path[0])
CONFIG = {
    'frame size': 512,
    'overlap': 128,
    'is training': True,
    'is streaming': False,
    'data list': '../DataSet/DataList_all.txt',
    'classifier': ['all'],
    'argumentation': True,
    'debug tool': False
}

frame_count = np.zeros((101,))
print(frame_count.shape, frame_count)

def main():
    preprocessor = PreProcessing(frame_size=CONFIG['frame size'],
                                 overlap=CONFIG['overlap'])
    if CONFIG['debug tool']:
        plt.ion()
        plt.figure()
        plt.show()
    if CONFIG['is streaming']:
        try:
            preprocessor_proc, queue_dict = preprocessor.process_stream()
            preprocessor_proc.start()
            preprocessor.recorder.start_streaming()
            while True:
                ep = queue_dict['endpoint'].get(True)
                effective_mfcc = queue_dict['mfcc'].get(True)
                print("EP", ep)
                if CONFIG['debug tool']:
                    plt.clf()
                    plt.subplot(221)
                    #plt.plot(wav)
                    plt.title('Input Wave')
                    plt.axvline(ep[0] * (CONFIG['frame size'] - CONFIG['overlap']), color='r')
                    plt.axvline(ep[1] * (CONFIG['frame size'] - CONFIG['overlap']), color='r')
                    plt.subplot(222)
                    plt.title("MFCC")
                    plt.imshow(effective_mfcc)
                    plt.subplot(223)
                    plt.title('ZCR')
                    #plt.plot(zcr)
                    plt.axvline(ep[0], color='r')
                    plt.axvline(ep[1], color='r')
                    plt.subplot(224)
                    plt.title('Energy')
                    #plt.plot(energy)
                    plt.axvline(ep[0], color='r')
                    plt.axvline(ep[1], color='r')
                    plt.draw()
                    plt.pause(0.001)
                #effective_feature = preprocessor.effective_feature(zcr, ep)
                #effective_mfcc = preprocessor.effective_feature(mfcc, ep)
                print("HERE:", effective_mfcc.shape)
                if len(effective_mfcc) == 0:
                    continue
                effective_mfcc = effective_mfcc.reshape((1, -1))
                print("HERE again:", effective_mfcc.shape)
                if(effective_mfcc.shape[1] >= 1000):
                    effective_mfcc = effective_mfcc[:, :1000]
                else:
                    effective_mfcc = np.concatenate((effective_mfcc, np.zeros((1, 1000 - effective_mfcc.shape[1]))), axis=1)
                #effective_mfcc = preprocessor.reshape(effective_mfcc, 500)
                print(time.localtime())
                for classifier_name, classifier_class in classifier_classes.items():
                    print(classifier_name)
                    classifier = classifier_class(None)
                    res = classifier.apply(effective_mfcc)
                    print(res)
        except KeyboardInterrupt:
            print('Exit')
        except Exception as e:
            print("Fucking", e)
            print("Emmmmm")
        finally:
            preprocessor_proc.terminate()
            del preprocessor
            exit()
    else:
        wav_list, frame_list, mfcc_list, energy_list, \
        zcr_list, endpoint_list, label_list = preprocessor.process(CONFIG['data list'], CONFIG['argumentation'])
        print('Data set Size:', len(wav_list))
        eff_zcr_list = np.zeros((1, 500))
        eff_mfcc_list = np.zeros((1, 1000))
        eff_label_list = []
        # Todo: Rewrite the relating preprocessor code.
        # Multiple data type mixed. Change the list of np array to pure np array.
        #for i in range(len(energy_list)):
        #    temp = preprocessor.effective_feature(zcr_list[i], endpoint_list[i])
        #    temp = preprocessor.reshape(temp, 100)
        #    if len(temp) != 0:
        #        eff_label_list.append(label_list[i])
        #    else:
        #        continue
        #    eff_zcr_list = np.concatenate((eff_zcr_list, temp), axis=0)
        #eff_zcr_list = eff_zcr_list[1:]

        for i in range(len(energy_list)):
            if mfcc_list[i].shape[0] > 100:
                frame_count[100] += 1
            else:
                frame_count[mfcc_list[i].shape[0]] += 1
            if mfcc_list[i].shape[0] < 20:
                continue

            temp = mfcc_list[i].reshape((1, -1))
            print(temp.shape)
            if(temp.shape[1] >= 1000):
                temp = temp[:, :1000]
            else:
                temp = np.concatenate((temp, np.zeros((1, 1000 - temp.shape[1]))), axis=1)
            if len(temp) != 0:
                eff_label_list.append(label_list[i])
            else:
                continue
            eff_mfcc_list = np.concatenate((eff_mfcc_list, temp), axis=0)
        x = np.linspace(1, 102, 101)
        plt.plot(x, frame_count)
        plt.show()
        eff_mfcc_list = eff_mfcc_list[1:]
        print(eff_mfcc_list.shape)

        if 'all' in CONFIG['classifier']:
            for classifier_name, classifier_class in classifier_classes.items():
                if CONFIG['is training']:
                    # Todo: Print training result and validation result
                    # Todo: Save the model to a dir.
                    print(classifier_name)
                    classifier = classifier_class(None)
                    classifier.train(eff_mfcc_list, eff_label_list)


if __name__ == '__main__':
    main()
