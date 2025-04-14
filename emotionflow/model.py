import torch
from transformers import AutoModel
from crf import *


class CRFModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dropout = config['dropout']
        self.num_classes = config['num_classes']
        self.pad_value = config['pad_value']
        self.CLS = config['CLS']
        self.context_encoder = AutoModel.from_pretrained(config['bert_path'])
        self.dim = self.context_encoder.embeddings.word_embeddings.weight.data.shape[-1]
        self.spk_embeddings = nn.Embedding(300, self.dim)
        self.crf_layer = CRF(self.num_classes)
        self.emission = nn.Linear(self.dim, self.num_classes)
        self.loss_func = torch.nn.CrossEntropyLoss(ignore_index=-1)
    def device(self):
        return self.context_encoder.device
    def forward(self, sentences, sentences_mask, speaker_ids, last_turns, emotion_idxs=None):
        '''
        sentences: batch * max_turns * max_length
        speaker_ids: batch * max_turns
        emotion[optional] : batch * max_turns
        '''
        batch_size = sentences.shape[0]
        max_turns = sentences.shape[1]
        max_len = sentences.shape[-1]
        speaker_ids = speaker_ids.reshape(batch_size * max_turns, -1)
        sentences = sentences.reshape(batch_size * max_turns, -1)
        cls_id = torch.ones_like(speaker_ids) * self.CLS
        input_ids = torch.cat((cls_id, sentences), 1)
        mask = 1 - (input_ids == (self.pad_value)).long()
        # with torch.no_grad():
        utterance_encoded = self.context_encoder(
            input_ids=input_ids,
            attention_mask=mask,
            output_hidden_states=True,
            return_dict=True
        )['last_hidden_state']
        mask_pos = mask.sum(1)-2
        features = utterance_encoded[torch.arange(mask_pos.shape[0]), mask_pos, :]
        emissions = self.emission(features)
        crf_emissions = emissions.reshape(batch_size, max_turns, -1)
        crf_emissions = crf_emissions.transpose(0, 1)
        sentences_mask = sentences_mask.transpose(0, 1)
        speaker_ids = speaker_ids.reshape(batch_size, max_turns).transpose(0, 1)
        last_turns = last_turns.transpose(0, 1)
        # train
        if emotion_idxs is not None:
            emotion_idxs = emotion_idxs.transpose(0, 1)
            loss1 = -self.crf_layer(crf_emissions, emotion_idxs, mask=sentences_mask)
            loss2 = self.loss_func(emissions.reshape(-1, self.num_classes), emotion_idxs.reshape(-1))
            loss = loss1 + loss2
            return loss
        # test
        else:
            return self.crf_layer.decode(crf_emissions, mask=sentences_mask)
        

class BaselineModel(torch.nn.Module):
    def __init__(self, config):
        super(BaselineModel, self).__init__()
        self.config = config
        self.bert = AutoModel.from_pretrained(config['bert_path'])
        hidden_size = self.bert.config.hidden_size  # Use the hidden size from the model
        self.dropout = torch.nn.Dropout(config['dropout'])
        self.classifier = torch.nn.Linear(hidden_size, config['num_classes'])
    def device(self):
        return self.bert.device
    def forward(self, sentences, mask, speaker_ids=None, last_turns=None, emotion_idxs=None):
        """Handles both 2D and 3D input for baseline model.
        If sentences is 3D (batch, turns, seq_len), it will be flattened, processed by BERT,
        then the logits will be reshaped back to (batch, turns, num_classes).
        If emotion_idxs is provided, computes and returns the loss.
        """
        # If mask has 2 dimensions (utterance-level mask), compute sequence mask from sentences
        if mask.dim() == 2:
            # Compute attention mask: 1 for tokens not equal to pad value
            mask = (sentences != self.config['pad_value']).long()
        
        if sentences.dim() == 3:
            batch_size, max_turns, seq_len = sentences.size()
            # Flatten the sentences and mask to process each utterance independently
            sentences = sentences.reshape(batch_size * max_turns, seq_len)
            mask = mask.reshape(batch_size * max_turns, seq_len)
            # Forward pass through BERT
            outputs = self.bert(sentences, attention_mask=mask)
            pooled_output = outputs[1]  # CLS token representation
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)  # Shape: (batch_size * max_turns, num_classes)

            if emotion_idxs is not None:
                # Flatten emotion_idxs to match logits shape
                emotion_idxs = emotion_idxs.reshape(-1)
                loss_func = torch.nn.CrossEntropyLoss(ignore_index=-1)
                loss = loss_func(logits, emotion_idxs)
                return loss
            else:
                # Reshape logits back to (batch_size, max_turns, num_classes)
                return logits.reshape(batch_size, max_turns, -1)
        else:
            # If sentences is not 3D, assume it's already flat
            outputs = self.bert(sentences, attention_mask=mask)
            pooled_output = outputs[1]
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            if emotion_idxs is not None:
                loss_func = torch.nn.CrossEntropyLoss(ignore_index=-1)
                loss = loss_func(logits, emotion_idxs)
                return loss
            else:
                return logits
