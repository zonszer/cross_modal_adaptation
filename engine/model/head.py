import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
AVAI_HEADS = ['linear', 'adapter']
from engine.model.logit import LogitHead
import math
from copy import deepcopy
from argparse import Namespace
from typing import Optional, Any
from collections import defaultdict


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
                         logit, modality, topsim=None, num_clsf=None):

    args = Namespace()          #create optimal args
    args.device = "cuda"
    (args.classifier_head, args.clip_encoder, args.classifier_init, args.zeroshot_dataset, args.text_encoder,
    args.logit, args.modality, args.topsim, args.num_clsf) = (classifier_head, clip_encoder, classifier_init, 
                                  zeroshot_dataset, text_encoder, logit, modality, topsim, num_clsf)

    logit_head = MultiClassifier(#classifier_main=logit_head,
                    num_classifier=args.num_clsf,               #TODO change to args
                    strategy='uniform',
                    args=args)
    num_classes, in_features = logit_head.num_classes, logit_head.in_features

    return logit_head, num_classes, in_features         


class MultiClassifier(nn.Module):
    def __init__(self, num_classifier: int, strategy: str = 'uniform', args: Optional[Any] = None) -> None:
        """
        Multi-classifier module for assigning classes to classifiers.

        Args:
            num_classifier (int): Number of classifiers.
            strategy (str, optional): Strategy for assigning classes to classifiers. Defaults to 'uniform'.
            args (Any, optional): Additional arguments. Defaults to None.
        """
        super().__init__()
        self.num_classifier = num_classifier
        self.strategy = strategy
        if args.modality != 'regularization':
            self.strategy = ''
        self.num_classes = int(args.zeroshot_dataset.label_tensor.max()) + 1
        self.args = args    ## Additional arguments
        self.in_features = self.get_in_features(self.args)
        self.init_classifier(self.strategy)

    
    def init_classifier(self, strategy='uniform'):
        '''Initialize the classifiers based on the chosen strategy:'''
        # init class attributes:
        self.cls2classifier_dict = {i: -1 for i in range(self.num_classes)}     #classifier_dict is classifier addressing dict to each class
        self._cls2classifier_dict = {i: -1 for i in range(self.num_classes)} 
        self.model_instance_dict = {}

        # assign class attributes:
        self.classifier_main = self.get_classifier_model(num_classes_=self.num_classes, model_id='main')
        if strategy == 'uniform':
            (self.cls2classifier_dict, self.classifier_info_dict, 
             self.labelmapping_dict, self.labelmapping_dict_re
                ) = self._uniform_strategy(self.cls2classifier_dict)
            # Create classifier instances:
            self.model_instance_dict['main'] = self.classifier_main
            for i in sorted(list(set(self.cls2classifier_dict.values()))):
                self.model_instance_dict[i] = self.get_classifier_model(num_classes_=self.classifier_info_dict[i], 
                                                                        model_id=i).to(self.args.device)
           # Assign classifiers to classes:
            for i in range(self.num_classes):       #create how many classifiers
                model_id = self.cls2classifier_dict[i]
                self._cls2classifier_dict[i] = self.model_instance_dict[model_id]
        elif strategy == '':
            print('No strategy is used for non-regularization modality')
        else:
            raise ValueError(f"Invalid strategy: {strategy}")


    def _uniform_strategy(self, classifier_dict):
        '''Implement the uniform strategy for assigning classes to classifiers'''
        # init class attributes:
        capacity_ofclassifier = math.ceil(self.num_classes / (self.num_classifier-1))   
        capacity_oflast = -1        #the last classifier should have more classes
        classifier_info_dict = {}
        labelmapping_dict = None

        # Calculate the capacity of classifiers:
        while True:
            capacity_oflast = self.num_classes - capacity_ofclassifier * (self.num_classifier-1)     
            if capacity_oflast >= capacity_ofclassifier:
                break
            capacity_ofclassifier = capacity_ofclassifier - 1   
        assert capacity_oflast + capacity_ofclassifier * (self.num_classifier-1) == self.num_classes

        # Assign classes to classifiers:
        classifier_idx = 1
        for i in range(self.num_classes):
            classifier_dict[i] = classifier_idx

            if (i+1) % capacity_ofclassifier == 0:
                classifier_idx += 1
            if (i+1) >= self.num_classes - capacity_oflast:
                classifier_idx = self.num_classifier
        print(f'classifier_list is: {classifier_dict}')
        print(f'capacity_ofclassifier is {capacity_ofclassifier}, capacity_oflast is {capacity_oflast}')
        assert -1 not in list(classifier_dict.values()) and max(list(classifier_dict.values())) == self.num_classifier

        # Create a dictionary for recording eahc classifier capacity (num_classes):
        classifier_seq = sorted(list(set(self.cls2classifier_dict.values())))
        for i in classifier_seq:
            if i != self.num_classifier:
                classifier_info_dict[i] = capacity_ofclassifier
            else:
                classifier_info_dict[i] = capacity_oflast
        # Create a dictionary for recording the label mapping form original to each classifier:
        labelmapping_dict, labelmapping_dict_re = self.create_labelmapping_dict(classifier_dict)
        return classifier_dict, classifier_info_dict, labelmapping_dict, labelmapping_dict_re

    def create_labelmapping_dict(self, classifier_dict):
        """
        Create a dictionary for recording the label mapping from original to each classifier.
        Also create a dictionary for recording the reverse mapping from each classifier to original labels.
        """
        labelmapping_dict = {}
        labelmapping_dict_re = {}
        for classifier_id in set(classifier_dict.values()):
            # Retrieve all class labels for this classifier
            class_labels_for_classifier = [class_label for class_label, cls_id in classifier_dict.items() if 
                                        cls_id == classifier_id]
                
            # Create a mapping for this classifier from original class labels to a continuous range of integers
            mapping_dict = {original_label: new_label for 
                            new_label, original_label in enumerate(sorted(class_labels_for_classifier))}
            labelmapping_dict[classifier_id] = mapping_dict
            # Create a reverse mapping_dict for each classifier from continuous range of integers back to original class labels.
            reverse_mapping_dict = {new_label: original_label for
                                original_label, new_label in mapping_dict.items()}
            labelmapping_dict_re[classifier_id] = reverse_mapping_dict

        return labelmapping_dict, labelmapping_dict_re

    def map_labels(self, model_id, lb):
        """
        Map original labels to new labels specific to classifier (model_id)
        Ignore labels not assigned to this classifier 
        Args:
            model_id (int): The id of the specified classifier.
            lb (Tensor): The tensor contains the original labels.
        Returns:
            lb_mapped (Tensor): The tensor contains the corresponding mapped labels to each classifier.
        """
        # Get labelmapping for this model_id
        current_labelmapping_dict = self.labelmapping_dict.get(model_id)
        
        # No mapping found, return original labels
        if current_labelmapping_dict is None:
            return lb
        
        # Crate a mapping for labels with a default value (-1) for labels not in this classifier
        mapping = [current_labelmapping_dict.get(x.item(), -1) for x in lb]
        
        # Convert mapping to Tensor
        lb_mapped = torch.tensor(mapping, dtype=lb.dtype).to(lb.device)
        return lb_mapped
    
    def map_labels_re(self, model_id, lb):
        """
        Reverse mapping from new labels specific to classifier to original labels.
        Args:
            model_id (int): The id of the classifier.
            lb (Tensor): The tensor contains the mapped labels of the classifier.
        Returns:
            lb_mapped (Tensor): The tensor contains the corresponding mapped labels to the original.
        """
        # Get reverse label mapping for this model_id
        current_labelmapping_dict_re = self.labelmapping_dict_re.get(model_id)
        
        # No mapping found, return original labels
        if current_labelmapping_dict_re is None:
            return lb
        
        # Create a mapping for labels with a default value (-1) for labels not in this classifier
        reverse_mapping = [current_labelmapping_dict_re.get(x.item(), -1) for x in lb]
        
        # Convert mapping to Tensor
        lb_mapped = torch.tensor(reverse_mapping, dtype=lb.dtype).to(lb.device)
        return lb_mapped


    def get_classifier_model(self, num_classes_=None, model_id=None):
        '''Create a classifier model based on the given parameters'''
        head_refine, num_classes, in_features = make_classifier_head(        #TODO: utilize num_classes 记得测试原本的setting是否可行
            self.args.classifier_head,
            self.args.classifier_init,
            self.args.zeroshot_dataset,
            self.args.text_encoder,
            in_features=self.in_features,
            num_classes=num_classes_,
        )
        logit_head = LogitHead(
            head_refine,
            logit_scale=self.args.logit,
            model_id=model_id,
        )
        return logit_head

    def get_in_features(self, args):
        '''Get the input features based on the encoder type'''
        if self.args.clip_encoder == 'ViT-B/16':
            in_features = 512
        elif self.args.clip_encoder == 'RN50':
            in_features = 1024
        return in_features
    

    def forward(self, x, labels=None, idxs=None):
        '''Return the logits ensemble from multi-classifiers or one classifier'''
        labels_dict = None
        if self.training:
            logits_dict, labels_dict = self.forward_train(x, labels, idxs)
            return logits_dict, labels_dict
        else:
            logits_all = self.forward_eval(x)
            return logits_all
    

    def _create_xmapping(self, bs, labels):         #HACK maybe better to create only once for the whole dataset
        """
        Create x mapping to which classifiers according to the labels.
        Args:
            bs (int): Batch size.
            labels (list): List of labels for each sample in the batch.
        Returns:
            dict: A dictionary mapping each sample index (key:int) to the corresponding classifier model id (values:int).
        """
        xmapping = {x_idx: self.cls2classifier_dict[labels[x_idx].item()] for x_idx in range(bs)}
        return xmapping

    def _collect_samemapping(self, xmapping):
        '''collect the mappings to the same classifier
            Returns:
                dict: A dictionary mapping each classifier model id (key:int) to the corresponding sample indices (values:list).
        '''
        classifier_mappings = defaultdict(list)
        for x_idx, model_id in xmapping.items():
            classifier_mappings[model_id].append(x_idx)         
        return classifier_mappings  #dict(classifier_mappings)

    def _forward_train_mainclsf(self, x, labels=None, idxs=None):   #HACK maybe can be incorporated into the self.instance[model_id]
        '''forward the main classifier'''
        logits_dict = {}    # dict to hold the logits for each classifier
        labels_dict = {}    # dict to hold the mapped labels for each classifier
        x_logits = self.model_instance_dict['main'](x)
        logits_dict['main'], labels_dict['main'] = x_logits, self.map_labels(model_id='main', lb=labels)
        return logits_dict, labels_dict

    def forward_train(self, x, labels, idxs):
        # init local variables:
        logits_dict = {}    # dict to hold the logits for each classifier
        labels_dict = {}    # dict to hold the mapped labels for each classifier
        dicts = self._forward_train_mainclsf(x, labels, idxs)
        logits_dict.update(dicts[0]), labels_dict.update(dicts[1])

        if self.args.modality == 'regularization' and self.model_instance_dict == {}:
            return logits_dict, labels_dict
        else:
            xmapping_dict = self._create_xmapping(x.shape[0], labels)
            x_idxs_dict = self._collect_samemapping(xmapping_dict)  #x_idxs_dict has no 'main'

            # loop through each classifier
            for model_id, x_idxs in x_idxs_dict.items():
                x_logits = self.model_instance_dict[model_id](x[x_idxs])    #!Calculate the logits using the sub-classifiers
                logits_dict[model_id] = x_logits
                labels_dict[model_id] = self.map_labels(model_id=model_id, lb=labels[x_idxs])
            return logits_dict, labels_dict


    def _select_topsim_cls(self, bs, x_logits):
        '''
        select the top similar classes for each x, then according to the classes input x to one sub-classifier
        Returns:
            dict: A dictionary mapping each sample index (key:int) to the corresponding classifier model id (values:int).
        '''
        topsim_classes = torch.topk(x_logits, self.args.topsim, dim=1).indices
        xmapping = defaultdict(list)
        for x_idx in range(bs):
            for simclass_idx in topsim_classes[x_idx]:
                xmapping[x_idx].append(self.cls2classifier_dict[simclass_idx.item()])
            
            # selct one sub-classifiers if the top similar classes are in the range of a signle sub-classifier
            classifier_list = set(xmapping[x_idx])  
            if len(classifier_list) == 1:
                xmapping[x_idx] = classifier_list.pop()
            else:  #if the top similar classes are in the range of multiple sub-classifiers, then select the main classifier
                xmapping[x_idx] = 'main'
        return xmapping

    def forward_eval(self, x):
        ''' when has no labels and idxs, forward the main classifier, then according to the x_logits 
        to select the sub-classifier, then forward the sub-classifier to predict the labels
        '''
        logits_dict = {}    # dict to hold the logits for each classifier
        labels_dict = {}    # dict to hold the seq index
        idxs_seq = []
        pred_all = torch.empty(0).to(self.args.device)
        dicts = self._forward_train_mainclsf(x)

        xmapping_dict = self._select_topsim_cls(x.shape[0], dicts[0]['main'])
        x_idxs_dict = self._collect_samemapping(xmapping_dict)      #x_idxs_dict has 'main'

        for model_id, x_idxs in x_idxs_dict.items():        #TODO check if seq start from x_idx = 0
            x_logits = self.model_instance_dict[model_id](x[x_idxs]) #calculate the logits using the classifier
            pred = torch.argmax(x_logits, dim=1)
            pred = self.map_labels_re(model_id=model_id, lb=pred)
            pred_all = torch.cat([pred_all, pred], dim=0)
            idxs_seq.extend(x_idxs)
            labels_dict[model_id] = x_idxs
            logits_dict[model_id] = x_logits
        original_order_indices = torch.argsort(torch.tensor(idxs_seq))    #Gets the index that will be reordered back to the original order
        return pred_all[original_order_indices], logits_dict, labels_dict                   
