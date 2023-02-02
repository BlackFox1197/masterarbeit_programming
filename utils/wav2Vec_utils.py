from transformers import AutoConfig, Wav2Vec2Processor, Wav2Vec2Config, Wav2Vec2Model


def init_w2v2(num_labels, label_list, pooling_mode = "mean", device = "cuda"):
    model_name_or_path = "facebook/wav2vec2-large-960h-lv60"
    config = Wav2Vec2Config.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        label2id={label: i for i, label in enumerate(label_list)},
        id2label={i: label for i, label in enumerate(label_list)},
        device=device
    )
    setattr(config, 'pooling_mode', pooling_mode)

    processor = Wav2Vec2Processor.from_pretrained(model_name_or_path, )
    target_sampling_rate = processor.feature_extractor.sampling_rate

    return processor, target_sampling_rate