local pretrained_model = 'plain';

{
  dataset_reader: {
    type: 'style_transfer_no_pretrain_reader',
  },
  train_data_path: 'data/ready/{mo_yan,zhang_ailing}/train',
  model: {
    type: 'style_transferer',
    generator: {
      type: 'soft_tokens_generator',
      model_name: pretrained_model,
    },
    discriminator: {
      type: 'style_discriminator',
      model_name: pretrained_model,
    },
  },
  data_loader: {
    batch_size: 1,
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
