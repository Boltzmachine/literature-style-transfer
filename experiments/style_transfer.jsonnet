local pretrained_model = 'fnlp/cpt-base';

{
  dataset_reader: {
    type: 'style_transfer_pretrain_reader',
    tokenizer: {
      type: 'pretrained_transformer',
      model_name: pretrained_model,
      tokenizer_kwargs: { tokenizer_class: 'BertTokenizerFast' },
    },
    token_indexers: {
      tokens: {
        type: 'pretrained_transformer',
        model_name: pretrained_model,
        tokenizer_kwargs: { tokenizer_class: 'BertTokenizerFast' },
      },
    },
  },
  train_data_path: 'data/ready/{mo_yan,zhang_ailing}/train',
  model: {
    type: 'style_transferer',
    model_name: pretrained_model,
    // generator: {
    //   type: 'soft_tokens_generator',
    //   model_name: pretrained_model,
    // },
    // discriminator: {
    //   type: 'style_discriminator',
    //   model_name: 'bert-base-chinese',
    // },
  },
  data_loader: {
    batch_size: 4,
    shuffle: true,
  },
  trainer: {
    type: 'gan',
    g_optimizer: {
      type: 'adamw',
      lr: 1e-5,
    },
    d_optimizer: {
      type: 'adamw',
      lr: 1e-5,
    },
    patience: 6,
    validation_metric: '+accuracy',
  },
}
