import os

import numpy as np
from ordered_set import OrderedSet
from collections import OrderedDict
from collections import Counter
from typing import List, Optional, Union
import dataclasses
import json
from transformers import BertTokenizer
from utils.file_utils import is_tf_available
from utils.tokenization_utils import PreTrainedTokenizer
from utils.utils import InputFeatures
if is_tf_available():
    import tensorflow as tf
from dataclasses import asdict
from sklearn.metrics import f1_score

def sigmoid(x):
    y = ((1/(1+np.exp(-x)))>0.5).astype(int)

    return y

def compute_multi_label_metrics(preds,true):
    f1 = f1_score(true,preds,average='macro')
    #acc = intent_acc(preds,true)
    results = {
        'f1':f1,
        #'acc':acc
    }
    return results

def intent_acc(pred,real):
    total_count, correct_count = 0.0, 0.0
    for p_intent, r_intent in zip(pred,real):
        if p_intent.all() == r_intent.all():
            correct_count+=1.0
        total_count +=1.0
    return 1.0*correct_count/total_count


class Alphabet(object):
    #### storage and serialization a set of elements
    def __init__(self,name):
        self.__name = name
        # self.__if_use_pad = if_use_pad
        # self.__if_use_unk = if_use_unk

        self.__index2instance = OrderedSet()
        self.__instance2index = OrderedDict()
        # counter object record the fequency of element occurs in raw text
        self.__counter = Counter()

    def add_instance(self,instance,multi_intent=False):
        if isinstance(instance,(list,tuple)):
            for element in instance:
                self.add_instance(element,multi_intent=multi_intent)
            return
        # We only support elements of str type
        assert isinstance(instance,str)
        if multi_intent and '#' in instance:
            for element in instance.split('#'):
                self.add_instance(element,multi_intent=multi_intent)
            return
        # counter the frequency of instances
        self.__counter[instance]+=1

        if instance not in self.__index2instance:
            self.__instance2index[instance]=len(self.__index2instance)
            self.__index2instance.append(instance)

    def get_index(self,instance,multi_intent=False):
        if isinstance(instance,(list,tuple)):
            return [self.get_index(elem, multi_intent=multi_intent) for elem in instance]
        assert isinstance(instance,str)
        if multi_intent and '#' in instance:
            return [self.get_index(element,multi_intent=multi_intent) for element in instance.split('#')]
        try:
            return self.__instance2index[instance]
        except KeyError:

            max_freq_item = self.__counter.most_common(1)[0][0]
            return self.__instance2index[max_freq_item]

    def save_content(self,dir_path):
        """
        Save the content of alphabet to files.
        There are two kinds of saved files:
        1 the first is a list file, elements are sorted by the frequency of occurrence
        2 the second is a dictionary file, elements are sorted by it serialized index

        dir_path : is the directory path to save object
        """
        # check if dir_path exists
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        list_path = os.path.join(dir_path,self.__name+'_list.txt')
        with open(list_path,'w',encoding='utf-8') as f:
            for element, frequency in self.__counter.most_common():
                f.write(element+'\t'+str(frequency)+'\n')

        dict_path = os.path.join(dir_path,self.__name+'_dict.txt')
        with open(dict_path,'w',encoding='utf-8') as f:
            for index,element in enumerate(self.__index2instance):
                f.write(element+'\t'+str(index)+'\n')

    def num_labels(self):
        return len(self.__index2instance)

    def get_labels(self):
        return self.__index2instance

class InputExample:
    """

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
    """

    guid: str
    text_a: str
    text_b: Optional[str] = None
    label: Optional[list] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self), indent=2) + "\n"

class DatasetManager(object):
    def __init__(self,args):
        #实例化字母表对象
        self.__word_alphabet = Alphabet('word')
        self.__intent_alphabet = Alphabet('intent')

        # Record the raw text of dataset
        self.__text_word_data = {}
        self.__text_intent_data = {}

        # Record the serialization of dataset
        self.__digital_word_data = {}
        self.__digital_intent_data = {}

        self.__args = args
        self.save_dir = self.__args.output_dir
        self.batch_size = self.__args.per_gpu_train_batch_size

    def quick_build(self):
        train_path = os.path.join(self.__args.data_dir,'train.txt')
        dev_path = os.path.join(self.__args.data_dir,'dev.txt')
        test_path = os.path.join(self.__args.data_dir,'test.txt')

        #print('train_path',train_path)
        self.add_file(train_path,'train', if_train_file=True)
        #self.add_file(dev_path,'dev',if_train_file=False)
        #self.add_file(test_path,'test',if_train_file=False)

        # check if save path exists
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        alphabet_dir = os.path.join(self.save_dir,'alphabet')

        self.__word_alphabet.save_content(alphabet_dir)
        self.__intent_alphabet.save_content(alphabet_dir)


    def add_file(self,file_path,data_name,if_train_file):
        text, intent = self.__read_file(file_path)
        print('len(text)',len(text))
        print('len(intent)',len(intent))
        if if_train_file:
            self.__word_alphabet.add_instance(text)
            #print('add_text')
            self.__intent_alphabet.add_instance(intent,multi_intent=True)
            #print('add_intent')

        self.__text_word_data[data_name]=text
        self.__text_intent_data[data_name]=intent

        # Serialize the raw text and stored it
        self.__digital_word_data[data_name]=self.__word_alphabet.get_index(text)
        if if_train_file:
            self.__digital_intent_data[data_name]=self.__intent_alphabet.get_index(intent,multi_intent=True)

    def create_examples(self,file_path,set_type):
        return self._create_examples(file_path,set_type)

    def _create_examples(self, file_path, set_type):
        """Creates examples for the training, dev and test sets."""
        file_path_type = os.path.join(file_path, set_type)
        texts, intents = self.__read_file_label(file_path_type)
        examples = []
        for id, (text, intent) in enumerate(zip(texts,intents)):
            dict ={'guid':'','text_a':[],'label':[]}
            #print('text',text)
            #print('intent',intent)
            guid = "%s-%s" % (set_type,id)
            text_a = text
            label = intent
            dict['guid'] = guid
            dict['text_a']=text_a
            dict['label']=label
            examples.append(dict)
        return examples

    def __read_file(self,file_path):
        texts, intents = [],[]
        text = []
        with open(file_path,'r',encoding='utf-8') as f:
            #print('len(f.readlines())',len(f.readlines()))
            for line in f.readlines():
                items = line.strip().split()
                #print('items', items)
                if len(items)==2:
                    text.append(items[0].strip())
                elif len(items)==1:
                    texts.append(text)
                    intents.append(items[0])
                    text = []
        return texts, intents

    def __read_file_label(self,file_path):
        texts, intents = [],[]
        text = []
        with open(file_path,'r',encoding='utf-8') as f:
            for line in f.readlines():
                items = line.strip().split()
                itent = []
                if len(items)==2:
                    text.append(items[0].strip())
                elif len(items)==1:
                    #print('items',items)
                    texts.append(text)
                    #print('type(items)',type(items))
                    if '#' in items[0]:
                        #print('#')
                        itent = [intent for intent in items[0].split('#')]
                    else:
                        itent = items
                    intents.append(itent)
                    text=[]
        return texts,intents

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

def convert_examples_to_features(
    examples: Union[List[InputExample], "tf.data.Dataset"],
    tokenizer: BertTokenizer,
    max_length: Optional[int] = None,
    label_list=None,
):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length. Defaults to the tokenizer's max_len
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    if is_tf_available() and isinstance(examples, tf.data.Dataset):
        return _tf_glue_convert_examples_to_features(examples, tokenizer, label_list=label_list,max_length=max_length)
    return glue_convert_examples_to_features(
        examples, tokenizer, max_length=max_length, label_list=label_list
    )


if is_tf_available():

    def _tf_glue_convert_examples_to_features(
        examples: tf.data.Dataset,
        tokenizer: PreTrainedTokenizer,
        label_list=None,
        max_length: Optional[int] = None,
    ) -> tf.data.Dataset:
        """
        Returns:
            A ``tf.data.Dataset`` containing the task-specific features.

        """
        processor = DatasetManager()
        examples = [processor.get_example_from_tensor_dict(example) for example in examples]
        features = glue_convert_examples_to_features(examples, tokenizer, max_length=max_length,label_list=label_list)
        label_type = tf.int64

        def gen():
            for ex in features:
                d = {k: v for k, v in asdict(ex).items() if v is not None}
                label = d.pop("label")
                yield (d, label)

        input_names = ["input_ids"] + tokenizer.model_input_names

        return tf.data.Dataset.from_generator(
            gen,
            ({k: tf.int32 for k in input_names}, label_type),
            ({k: tf.TensorShape([None]) for k in input_names}, tf.TensorShape([])),
        )


def glue_convert_examples_to_features(
    examples: List[InputExample],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    label_list=None,
):
    if max_length is None:
        max_length = tokenizer.max_len
    label_map = {label: i for i, label in enumerate(label_list)}

    def label_from_example(example: InputExample) -> Union[int, float, None]:
        res = [0]*len(label_list)
        if example['label'] is None:
            return res
        else:
            for item in example['label']:
                num = label_map[item]
                res[int(num)]=1
            return res

    labels = [label_from_example(example) for example in examples]
    #print('labels',labels)

    batch_encoding = tokenizer(
        [example['text_a'] for example in examples],
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )
    #print('type(batch_encoding)',type(batch_encoding))

    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}
        feature = InputFeatures(**inputs, label=labels[i])
        features.append(feature)
    #print('features',features)
    return features