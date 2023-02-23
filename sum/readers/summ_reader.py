from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token
from typing import Dict, List, Iterator
from overrides import overrides
import itertools
import datasets


@DatasetReader.register("wiki_summ_reader")
class WikiLinguaDatasetReader(DatasetReader):
    
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None, #* map tokens to integers to keep track of them in the future
                 **kwargs, #* pass on any additional configured arguments to the parent 
                 ) -> None:
        super().__init__()
        
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.data = datasets.load_dataset("wiki_lingua", name=kwargs['language'] or 'english', split="train[:2000]")

    def _read(self, file_path: str) -> Iterator[Instance]:
        
        for _data_example in self.data:
            flattened_ = self.flatten(_data_example)
            yield self.text_to_instance(flattened_['document'], flattened_['summary'])

    def flatten(self, example):
        return {
            "document": example["article"]["document"],
            "summary": example["article"]["summary"],
        }

    
    def text_to_instance(self,
                         *inputs) -> Instance:
        (source, target) = inputs

        fields: Dict[str, Field] = {}
        # wrap each token in the file with a token object
        tokens = TextField([Token(w) for w in source], self._token_indexers)
        target_tokens = TextField([Token(w) for w in target], self._token_indexers)

        # Instances in AllenNLP are created using Python dictionaries,
        # which map the token key to the Field type
        fields["tokens"] = tokens
        fields["label"] = target_tokens
        return Instance(fields)