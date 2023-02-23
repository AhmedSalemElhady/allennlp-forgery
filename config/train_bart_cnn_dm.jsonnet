local seq_to_seq_lm = "facebook/bart-large";

local epochs = 10;
local batch_size = 16;
local length_limit = 512;
local patience = 5;

local seed = 100;

local cuda = -1;

local data_base_url = "https://storage.googleapis.com/allennlp-public-data/cnndm-combined-data-2020.07.13.tar.gz";
local train_data = data_base_url + "!cnndm-combined-data-2020.07.13/url_lists/all_train.txt";
local dev_data = data_base_url + "!cnndm-combined-data-2020.07.13/url_lists/all_val.txt";



{
  "train_data_path": train_data,
  "validation_data_path": dev_data,
  "dataset_reader": {
      "type": "cnn_dm",
      "source_tokenizer": {
          "type": "pretrained_transformer",
          "model_name": seq_to_seq_lm
      },
      "source_token_indexers": {
          "tokens": {
              "type": "pretrained_transformer",
              "model_name": seq_to_seq_lm,
              "namespace": "tokens"
          }
      },
      "source_max_tokens": 1022,
      "target_max_tokens": 54,
      // "max_instances": 1000 // DEBUG setting
  },
  "model": {
      "type": "bart",
      "model_name": seq_to_seq_lm,
      "beam_search": {
          "max_steps": 140,
          "beam_size": 4
      },
  },
  "data_loader": {
    "batch_sampler": {
      "type": 'bucket',
      "batch_size": batch_size
    }
  },
  "trainer": {

    "num_epochs": epochs,
    "patience": patience,
    "cuda_device": cuda,
    "grad_clipping": 1.0,
    "validation_metric": "+per_instance_f1",

    "optimizer": {
    "type": "huggingface_adamw",
    "weight_decay": 0.0,
    "parameter_groups": [
      [["bias", "LayerNorm\\.weight", "layer_norm\\.weight"], {"weight_decay": 0}],
    ],
    "lr": 2e-5,
    "eps": 1e-8
    },

    "learning_rate_scheduler": {
      "type": "slanted_triangular",
      "num_epochs": epochs,
      "cut_frac": 0.1,
    }
  },
  "random_seed": seed,
  "numpy_seed": seed,
  "pytorch_seed": seed,
}
