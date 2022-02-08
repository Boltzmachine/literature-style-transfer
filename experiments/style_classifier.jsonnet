local pretrained_model = 'bert-base-chinese';

{
  dataset_reader: {
    type: 'style_classification_reader',
    tokenizer: {
      type: 'pretrained_transformer',
      model_name: pretrained_model,
    },
    token_indexers: {
      bert: {
        type: 'pretrained_transformer',
        model_name: pretrained_model,
      },
    },
  },
  train_data_path: 'data/ready/{mo_yan,zhang_ailing}/train',
  validation_data_path: 'data/ready/{mo_yan,zhang_ailing}/test',
  model: {
    type: 'style_classifier',
    contextualizer: {
      token_embedders: {
        bert: {
          type: 'pretrained_transformer',
          model_name: pretrained_model,
        },
      },
    },
    metrics: {
      accuracy: { type: 'categorical_accuracy' },
    },
  },
  data_loader: {
    batch_size: 8,
    shuffle: true,
  },
  trainer: {
    optimizer: {
      type: 'adamw',
      lr: 1e-5,
    },
    patience: 6,
    validation_metric: '+accuracy',
  },
}
