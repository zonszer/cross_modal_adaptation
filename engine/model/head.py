import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
AVAI_HEADS = ['linear', 'adapter']
from engine.model.logit import LogitHead
import math
from copy import deepcopy
from argparse import Namespace

class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4, residual_ratio=0.2):
        super(Adapter, self).__init__()
        self.residual_ratio = residual_ratio
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        a = self.fc(x)
        x = self.residual_ratio * a + (1 - self.residual_ratio) * x
        return x


# def get_zero_shot_weights(text_dataset, num_classes, in_features):
#     # Caveat: Only support text_dataset with 1-D text features. 
#     # Need to modify if you want to partial finetuning the text encoder
#     weights = torch.zeros(num_classes, in_features)
#     count = torch.zeros(num_classes)
#     for i in range(len(text_dataset)):
#         label = text_dataset.label_tensor[i]
#         weights[label] += F.normalize(text_dataset.input_tensor[i], dim=0)
#         count[label] += 1
#     weights /= count.unsqueeze(1)
#     # normalize the weights
#     weights.data = F.normalize(weights, dim=1)
#     return weights

def get_text_dataset_per_class(text_dataset):
    print("Building text dataset per class...")
    text_dataset_per_class = {}
    for text, text_label, eot_indices in tqdm(text_dataset):
        text_label = int(text_label)
        if text_label not in text_dataset_per_class:
            text_dataset_per_class[text_label] = []
        text_dataset_per_class[text_label].append([text, eot_indices])
    num_of_templates = len(text_dataset_per_class[text_label])
    for text_label in text_dataset_per_class:
        assert len(text_dataset_per_class[text_label]) == num_of_templates
    return text_dataset_per_class, num_of_templates

def get_zero_shot_weights(text_dataset, num_classes, in_features, text_encoder, device="cuda"):
    with torch.no_grad():
        text_dataset_per_class, _ = get_text_dataset_per_class(text_dataset)
        weights = torch.zeros(num_classes, in_features)
        for label in range(num_classes):
            texts = None
            eot_indices = None
            for i in range(len(text_dataset_per_class[label])):
                text, eot_indice = text_dataset_per_class[label][i]
                text = text.unsqueeze(0).to(device)
                eot_indice = eot_indice.unsqueeze(0).to(device)
                if texts is None:
                    texts = text
                    eot_indices = eot_indice
                else:
                    texts = torch.cat([texts, text], dim=0)
                    eot_indices = torch.cat([eot_indices, eot_indice], dim=0)
            text_features = text_encoder(texts, eot_indices)
            text_features = text_features.mean(dim=0)
            weights[label] = text_features
        # normalize the weights
        weights.data = torch.nn.functional.normalize(weights, dim=1)
    return weights



def make_classifier_head(classifier_head,
                         classifier_init,
                         zeroshot_dataset,
                         text_encoder,
                         in_features,
                         bias=False,
                         num_classes=None,
                         ):
    assert classifier_head in AVAI_HEADS

    if num_classes is None:
        num_classes = int(zeroshot_dataset.label_tensor.max()) + 1

    linear_head = nn.Linear(in_features, num_classes, bias=bias)
    if classifier_init == 'zeroshot':       #use text weight to init adaptor:
        # assert zeroshot_dataset.input_tensor.shape[1] == in_features
        linear_head.weight.data = get_zero_shot_weights(
            zeroshot_dataset, num_classes, in_features, text_encoder)
    
    if classifier_head == 'linear':
        head = linear_head
    elif classifier_head == 'adapter':
        adapter = Adapter(in_features, residual_ratio=0.2)
        head = nn.Sequential(
            adapter,
            linear_head
        )
    else:
        raise ValueError(f"Invalid head: {classifier_head}")
    return head, num_classes, in_features

                        
def make_classifier_model(classifier_head,
                         clip_encoder,
                         classifier_init,
                         zeroshot_dataset,
                         text_encoder,
                         logit, modality):

    args = Namespace()          #create optimal args
    (args.classifier_head, args.clip_encoder, args.classifier_init, args.zeroshot_dataset, args.text_encoder,
    args.logit, args.modality) = classifier_head, clip_encoder, classifier_init, zeroshot_dataset, text_encoder, logit, modality

    logit_head = MultiClassifier(#classifier_main=logit_head,
                    classifier_num=5,               #TODO change to args
                    strategy='uniform',
                    args=args)
    num_classes, in_features = logit_head.num_classes, logit_head.in_features

    return logit_head, num_classes, in_features         #TODO check if need to change


class MultiClassifier(nn.Module):
    def __init__(self, num_classifier,
                 strategy='uniform', args=None):
        self.num_classifier = num_classifier
        self.strategy = strategy
        if args.modality != 'regularization':
            self.strategy = ''
        self.num_classes = int(args.zeroshot_dataset.label_tensor.max()) + 1
        self.args = args
        self.in_features = self.get_in_features(self.args)
        self.init_classifier(self.strategy)
    
    def init_classifier(self, strategy='uniform'):
        # init class attributes:
        self.classifier_dict = {range(self.num_classes): -1}     #classifier_dict is classifier addressing dict to each class
        self._classifier_dict = {range(self.num_classes): -1}
        self.model_instance_dict = {}

        # assign class attributes:
        self.classifier_main = self.get_classifier_model(num_classes_=self.num_classes)
        if strategy == 'uniform':
            self.classifier_dict, self.classifier_info_dict = self.uniform_strategy(self.classifier_dict)

            for i in sorted(list(set(self.classifier_dict.values()))):
                self.model_instance_dict[i] = self.get_classifier_model(num_classes_=self.classifier_info_dict[i])

            for i in range(self.num_classes):       #create how many classifiers
                key = self.classifier_dict[i]
                self._classifier_dict[i] = self.model_instance_dict[key]
        elif strategy == '':
            print('No strategy is used for non-regularization modality')


    def uniform_strategy(self, classifier_dict):
        # init class attributes:
        capacity_ofclassifier = math.ceil(self.num_classes / (self.num_classifier-1))   
        capacity_oflast = -1        #the last classifier should have more classes
        classifier_info_dict = {}

        while True:
            capacity_oflast = self.num_classes - capacity_ofclassifier * (self.num_classifier-1)     
            if capacity_oflast >= capacity_ofclassifier:
                break
            capacity_ofclassifier = capacity_ofclassifier - 1   
        assert capacity_oflast + capacity_ofclassifier * (self.num_classifier-1) == self.num_classes

        classifier_idx = 1
        for i in range(self.num_classes):
            classifier_dict[i] = classifier_idx

            if (i+1) % capacity_ofclassifier == 0:
                classifier_idx += 1
            if (i+1) >= self.num_classes - capacity_oflast:
                classifier_idx = self.num_classifier
        print(f'classifier_list is: {classifier_dict}')
        assert -1 not in list(classifier_dict.values()) and list(classifier_dict.values()).max() == self.num_classifier

        classifier_seq = sorted(list(set(self.classifier_dict.values())))
        for i in classifier_seq:
            if i != self.num_classifier:
                classifier_info_dict[i] = capacity_ofclassifier
            else:
                classifier_info_dict[i] = capacity_oflast

        return classifier_dict, classifier_info_dict        #classifier_info_dict is the dict of classifier capacity (num_classes)


    def get_classifier_model(self, num_classes_=None):
        head_refine, num_classes, in_features = make_classifier_head(        #TODO: utilize in_features
            self.args.classifier_head,
            self.args.classifier_init,
            self.text_dataset,
            self.text_encoder,
            in_features=self.in_features,
            num_classes=num_classes_,
        )
        logit_head = LogitHead(
            head_refine,
            logit_scale=self.args.logit,
        ).train().cuda()
        return logit_head

    def get_in_features(self, args):
        if self.args.clip_encoder == 'ViT-B/16':
            in_features = 512
        elif self.args.clip_encoder == 'RN50':
            in_features = 1024
        return in_features

    def forward(self, x):
        logits = []
        for logit_head in self.logit_heads:
            logits.append(logit_head(x))
        return logits
