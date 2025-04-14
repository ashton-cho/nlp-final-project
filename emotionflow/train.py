"""
train.py

This script handles training, validation, and testing for the CRF-based emotion classification model.
It supports both MELD and EmoryNLP datasets and includes utilities for vocabulary construction,
data loading, model training, and evaluation.
"""

from config import *
from model import CRFModel, BaselineModel

import gc

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False


## ----------------------
# Utility Functions
# ----------------------
def pad_to_len(list_data, max_len, pad_value):
    list_data = list_data[-max_len:]
    len_to_pad = max_len-len(list_data)
    pads = [pad_value] * len_to_pad
    list_data.extend(pads)
    return list_data


def get_vocabs(file_paths, addi_file_path):
    speaker_vocab = vocab.UnkVocab()
    emotion_vocab = vocab.Vocab()
    emotion_vocab.word2index('neutral', train=True)

    # global speaker_vocab, emotion_vocab
    for file_path in file_paths:
        data = pd.read_csv(file_path)
        for row in tqdm(data.iterrows(), desc='get vocab from {}'.format(file_path)):
            meta = row[1]
            emotion = meta['Emotion'].lower()
            emotion_vocab.word2index(emotion, train=True)
    additional_data = json.load(open(addi_file_path, 'r'))

    for episode_id in additional_data:
        for scene in additional_data.get(episode_id):
            for utterance in scene['utterances']:
                speaker = utterance['speakers'][0].lower()
                speaker_vocab.word2index(speaker, train=True)
    speaker_vocab = speaker_vocab.prune_by_count(1000)
    speakers = list(speaker_vocab.counts.keys())
    speaker_vocab = vocab.UnkVocab()

    for speaker in speakers:
        speaker_vocab.word2index(speaker, train=True)

    logging.info('total {} speakers'.format(len(speaker_vocab.counts.keys())))
    torch.save(emotion_vocab.to_dict(), emotion_vocab_dict_path)
    torch.save(speaker_vocab.to_dict(), speaker_vocab_dict_path)


## ----------------------
# Dataset Loaders
# ----------------------
def load_builddataset(file_path, train=False, use_query=True):
    speaker_vocab = vocab.UnkVocab.from_dict(torch.load(
        speaker_vocab_dict_path
    ))
    emotion_vocab = vocab.Vocab.from_dict(torch.load(
        emotion_vocab_dict_path
    ))

    data = pd.read_csv(file_path)
    data['Utterance'] = data['Utterance'].fillna('')

    ret_utterances = []
    ret_speaker_ids = []
    ret_emotion_idxs = []
    utterances = []
    full_contexts = []
    speaker_ids = []
    emotion_idxs = []
    pre_dial_id = -1
    max_turns = 0

    for row in tqdm(data.iterrows(), desc='processing file {}'.format(file_path)):
        meta = row[1]
        utterance = meta['Utterance'].replace('’', '\'').replace("\"", '')
        speaker = meta['Speaker']
        utterance = speaker + ' says:, ' + utterance
        emotion = meta['Emotion'].lower()
        dialogue_id = meta['Dialogue_ID']

        if pre_dial_id == -1:
            pre_dial_id = dialogue_id
        if dialogue_id != pre_dial_id:
            ret_utterances.append(full_contexts)
            ret_speaker_ids.append(speaker_ids)
            ret_emotion_idxs.append(emotion_idxs)
            max_turns = max(max_turns, len(utterances))
            utterances = []
            full_contexts = []
            speaker_ids = []
            emotion_idxs = []
        pre_dial_id = dialogue_id

        speaker_id = speaker_vocab.word2index(speaker)
        emotion_idx = emotion_vocab.word2index(emotion)
        token_ids = tokenizer(utterance, add_special_tokens=False)['input_ids'] + [CONFIG['SEP']]
        full_context = []
        if len(utterances) > 0:
            context = utterances[-3:]
            for pre_uttr in context:
                full_context += pre_uttr
        full_context += token_ids

        # query
        if use_query:
            query = speaker + ' feels <mask>'
            query_ids = tokenizer(query, add_special_tokens=False)['input_ids'] + [CONFIG['SEP']]
            full_context += query_ids

        full_context = pad_to_len(
            full_context, CONFIG['max_len'], CONFIG['pad_value'])
        utterances.append(token_ids)
        full_contexts.append(full_context)
        speaker_ids.append(speaker_id)
        emotion_idxs.append(emotion_idx)

    pad_utterance = [CONFIG['SEP']] + tokenizer(
        "1",
        add_special_tokens=False
    )['input_ids'] + [CONFIG['SEP']]
    pad_utterance = pad_to_len(
        pad_utterance, CONFIG['max_len'], CONFIG['pad_value'])
    
    # for CRF
    ret_mask = []
    ret_last_turns = []
    for dial_id, utterances in tqdm(enumerate(ret_utterances), desc='build dataset'):
        mask = [1] * len(utterances)
        while len(utterances) < max_turns:
            utterances.append(pad_utterance)
            ret_emotion_idxs[dial_id].append(-1)
            ret_speaker_ids[dial_id].append(0)
            mask.append(0)
        ret_mask.append(mask)
        ret_utterances[dial_id] = utterances

        last_turns = [-1] * max_turns
        for turn_id in range(max_turns):
            curr_spk = ret_speaker_ids[dial_id][turn_id]
            if curr_spk == 0:
                break
            for idx in range(0, turn_id):
                if curr_spk == ret_speaker_ids[dial_id][idx]:
                    last_turns[turn_id] = idx
        ret_last_turns.append(last_turns)
        
    dataset = TensorDataset(
        torch.LongTensor(ret_utterances),
        torch.LongTensor(ret_speaker_ids),
        torch.LongTensor(ret_emotion_idxs),
        torch.ByteTensor(ret_mask),
        torch.LongTensor(ret_last_turns)
    )
    return dataset


## ----------------------
# Optimizer Setup
# ----------------------
def get_paramsgroup(model, warmup=False):
    pre_train_lr = CONFIG['ptmlr']
    
    # Check which encoder to use: CRFModel uses `context_encoder`, while BaselineModel uses `bert`
    if hasattr(model, 'context_encoder'):
        encoder = model.context_encoder
        crf_params = list(map(id, model.crf_layer.parameters()))
    elif hasattr(model, 'bert'):
        encoder = model.bert
        crf_params = []  # Baseline model has no CRF layer, so an empty list.
    else:
        raise ValueError("Model must have either 'context_encoder' or 'bert' attribute.")
    
    encoder_params = list(map(id, encoder.parameters()))
    params = []
    warmup_params = []
    
    for name, param in model.named_parameters():
        lr = CONFIG['lr']
        weight_decay = 0  # or set a non-zero value if you later add L2 regularization
        
        # Use different learning rate for pretrained layers:
        if id(param) in encoder_params:
            lr = pre_train_lr
        
        # If the model has CRF layers, adjust their lr:
        if crf_params and id(param) in crf_params:
            lr = CONFIG['lr'] * 10
        
        params.append({'params': param, 'lr': lr, 'weight_decay': weight_decay})
        warmup_params.append({'params': param, 'lr': 0 if id(param) in encoder_params else lr,
                                'weight_decay': weight_decay})
    
    if warmup:
        return warmup_params
    
    params = sorted(params, key=lambda x: x['lr'], reverse=True)
    return params


## ----------------------
# Training and Evaluation Functions
# ----------------------
def train_epoch(model, optimizer, data, epoch_num=0, max_step=-1):
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=-1)
    sampler = RandomSampler(data)
    dataloader = DataLoader(
        data,
        batch_size=CONFIG['batch_size'],
        sampler=sampler,
        num_workers=0  # you can use multiprocessing.cpu_count() if desired
    )
    tq_train = tqdm(total=len(dataloader), position=1)
    accumulation_steps = CONFIG['accumulation_steps']

    # Initialize variables to accumulate the raw loss and count batches.
    total_loss = 0.0
    num_batches = 0

    for batch_id, batch_data in enumerate(dataloader):
        batch_data = [x.to(model.device()) for x in batch_data]
        sentences = batch_data[0]
        speaker_ids = batch_data[1]
        emotion_idxs = batch_data[2]
        mask = batch_data[3]
        last_turns = batch_data[4]
        outputs = model(sentences, mask, speaker_ids, last_turns, emotion_idxs)
        loss = outputs
        tq_train.set_description('loss is {:.2f}'.format(loss.item()))
        tq_train.update()

        # Compute the raw loss before dividing by accumulation_steps.
        raw_loss = loss.item() * accumulation_steps
        total_loss += raw_loss
        num_batches += 1

        loss = loss / accumulation_steps
        loss.backward()

        if batch_id % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            # Optionally, you may call torch.cuda.empty_cache() here.

    tq_train.close()

    # Compute average loss per batch.
    average_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return average_loss


def test(model, data):
    pred_list = []
    y_true_list = []
    model.eval()
    sampler = SequentialSampler(data)
    dataloader = DataLoader(
        data,
        batch_size=CONFIG['batch_size'],
        sampler=sampler,
        num_workers=0,  # multiprocessing.cpu_count()
    )
    tq_test = tqdm(total=len(dataloader), desc="testing", position=2)
    
    for batch_id, batch_data in enumerate(dataloader):
        # Send all batch tensors to the proper device
        batch_data = [x.to(model.device()) for x in batch_data]
        sentences = batch_data[0]
        speaker_ids = batch_data[1]
        # emotion_idxs is assumed to be a tensor of shape (batch, turns)
        emotion_idxs = batch_data[2].cpu().numpy().tolist()
        mask = batch_data[3]
        last_turns = batch_data[4]
        
        # If using BaselineModel, process utterances individually.
        if CONFIG['model_type'].lower() == 'baseline':
            B, T, L = sentences.shape  # batch size, number of turns, sequence length
            # Flatten the dialogues to process each utterance independently
            sentences_flat = sentences.view(-1, L)
            # Compute token-level attention mask from sentences using the pad value
            mask_flat = (sentences_flat != CONFIG['pad_value']).long()
            # Call baseline forward with the computed mask
            logits = model(sentences_flat, mask_flat)
            # Predict class labels using argmax (shape: (B*T,))
            predictions = torch.argmax(logits, dim=1)
            # Reshape predictions back to (B, T)
            outputs = predictions.view(B, T)
        else:
            # For CRFModel, follow the original interface.
            outputs = model(sentences, mask, speaker_ids, last_turns)
        
        # Collect predictions and true labels from each dialogue turn.
        for batch_idx in range(mask.shape[0]):
            for seq_idx in range(mask.shape[1]):
                if mask[batch_idx][seq_idx].item():
                    # Convert prediction tensor to a CPU scalar
                    pred_val = outputs[batch_idx][seq_idx].cpu().item() if torch.is_tensor(outputs[batch_idx][seq_idx]) else outputs[batch_idx][seq_idx]
                    pred_list.append(pred_val)
                    y_true_list.append(emotion_idxs[batch_idx][seq_idx])
        tq_test.update()
    
    F1 = f1_score(y_true=y_true_list, y_pred=pred_list, average='weighted')
    model.train()
    return F1


## ----------------------
# Main Training Loop
# ----------------------
def train(model, train_data_path, dev_data_path, test_data_path):
    devset = load_builddataset(dev_data_path, use_query=CONFIG['use_query'])
    testset = load_builddataset(test_data_path, use_query=CONFIG['use_query'])
    trainset = load_builddataset(train_data_path, use_query=CONFIG['use_query'])

    # warmup
    optimizer = torch.optim.AdamW(get_paramsgroup(model, warmup=True))
    for epoch in range(CONFIG['wp']):
        train_epoch(model, optimizer, trainset, epoch_num=epoch)
        gc.collect()
        f1 = test(model, devset)
        gc.collect()
        print('f1 on dev @ warmup epoch {} is {:.4f}'.format(epoch+1, f1), flush=True)

    # train
    optimizer = torch.optim.AdamW(get_paramsgroup(model))
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    best_f1 = -1
    tq_epoch = tqdm(total=CONFIG['epochs'], position=0)
    
    # Define the checkpoint directory for the current task and create it if it doesn't exist.
    checkpoint_dir = os.path.join('models', CONFIG['task_name'])
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    best_f1 = -1
    best_dev = -1
    best_epoch = -1
    results_list = []
    tq_epoch = tqdm(total=CONFIG['epochs'], position=0)
    
    # Epoch training loop
    for epoch in range(CONFIG['epochs']):
        tq_epoch.set_description('training on epoch {}'.format(epoch+1))
        tq_epoch.update()
        
        # Run train_epoch and record average train loss returned.
        train_loss = train_epoch(model, optimizer, trainset, epoch_num=epoch)
        gc.collect()
        
        # Compute F1 on training, dev, and test sets.
        f1_train = test(model, trainset)
        f1_dev = test(model, devset)
        f1_test = test(model, testset)
        gc.collect()
        
        print('Epoch {}: Loss: {:.4f}, Train F1: {:.4f}, Dev F1: {:.4f}, Test F1: {:.4f}'.format(
              epoch+1, train_loss, f1_train, f1_dev, f1_test), flush=True)
        
        # Append results to the list for later logging.
        results_list.append({
            'epoch': epoch+1,
            'train_loss': train_loss,
            'f1_train': f1_train,
            'f1_dev': f1_dev,
            'f1_test': f1_test
        })
        
        # Update best dev score tracking.
        if f1_dev > best_dev:
            best_dev = f1_dev
            best_epoch = epoch+1
        
        # Save checkpoint if dev F1 improved.
        if f1_dev > best_f1:
            best_f1 = f1_dev
            if CONFIG['model_type'].lower() == "baseline":
                label = "baseline_qa" if CONFIG['use_query'] else "baseline"
            else:
                label = "crf_qa" if CONFIG['use_query'] else "crf"
            model_filename = os.path.join(
                checkpoint_dir,
                "{}_f1_{:.4f}_@epoch{}.pkl".format(label, best_f1, epoch+1)
            )
            torch.save(model, model_filename)
            
        # Step the learning rate scheduler if applicable.
        if lr_scheduler.get_last_lr()[0] > 1e-5:
            lr_scheduler.step()
        
    tq_epoch.close()
    
    # After training, log all metrics to a CSV file.
    os.makedirs('results', exist_ok=True)
    
    # Determine the label based on model type and use_query flag.
    if CONFIG['model_type'].lower() == "baseline":
        label = "baseline_qa" if CONFIG['use_query'] else "baseline"
    else:
        label = "crf_qa" if CONFIG['use_query'] else "crf"
        
    results_df = pd.DataFrame(results_list)
    result_filename = f"{CONFIG['task_name']}_{label}.csv"
    results_df.to_csv(os.path.join('results', result_filename), index=False)
    
    print(f"Best dev F1 of {best_dev:.4f} achieved at epoch {best_epoch}", flush=True)


## ----------------------
# Entry Point
# ----------------------
if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-te', '--test', action='store_true',
                        help='run test', default=False)
    parser.add_argument('-tr', '--train', action='store_true',
                        help='run train', default=False)
    parser.add_argument('-ft', '--finetune', action='store_true',
                        help='fine tune base the best model', default=False)
    parser.add_argument('-pr', '--print_error', action='store_true',
                        help='print error case', default=False)
    parser.add_argument('-bsz', '--batch', help='Batch_size',
                        required=False, default=CONFIG['batch_size'], type=int)
    parser.add_argument('-epochs', '--epochs', help='epochs',
                        required=False, default=CONFIG['epochs'], type=int)
    parser.add_argument('-lr', '--lr', help='learning rate',
                        required=False, default=CONFIG['lr'], type=float)
    parser.add_argument('-p_unk', '--p_unk', help='prob to generate unk speaker',
                        required=False, default=CONFIG['p_unk'], type=float)
    parser.add_argument('-ptmlr', '--ptm_lr', help='ptm learning rate',
                        required=False, default=CONFIG['ptmlr'], type=float)
    parser.add_argument('-tsk', '--task_name', default='meld', type=str)
    parser.add_argument('-wp', '--warm_up', default=CONFIG['wp'],
                        type=int, required=False)
    parser.add_argument('-dpt', '--dropout', default=CONFIG['dropout'],
                        type=float, required=False)
    parser.add_argument('-e_stop', '--eval_stop',
                        default=500, type=int, required=False)
    parser.add_argument('-bert_path', '--bert_path',
                        default=CONFIG['bert_path'], type=str, required=False)
    parser.add_argument('-data_path', '--data_path',
                        default=CONFIG['data_path'], type=str, required=False)
    parser.add_argument('-acc_step', '--accumulation_steps',
                        default=CONFIG['accumulation_steps'], type=int, required=False)
    parser.add_argument('-uq', '--use_query', default=True, type=str2bool)
    parser.add_argument('-mt', '--model_type', default='CRF', type=str)

    args = parser.parse_args()
    CONFIG['data_path'] = args.data_path
    CONFIG['lr'] = args.lr
    CONFIG['ptmlr'] = args.ptm_lr
    CONFIG['epochs'] = args.epochs
    CONFIG['bert_path'] = args.bert_path
    CONFIG['batch_size'] = args.batch
    CONFIG['dropout'] = args.dropout
    CONFIG['wp'] = args.warm_up
    CONFIG['p_unk'] = args.p_unk
    CONFIG['accumulation_steps'] = args.accumulation_steps
    CONFIG['task_name'] = args.task_name
    CONFIG['use_query'] = args.use_query
    CONFIG['model_type'] = args.model_type

    train_data_path = os.path.join(CONFIG['data_path'], 'meld_train.csv')
    test_data_path = os.path.join(CONFIG['data_path'], 'meld_test.csv')
    dev_data_path = os.path.join(CONFIG['data_path'], 'meld_dev.csv')
    if args.task_name =='iemocap':
        train_data_path = os.path.join(CONFIG['data_path'], 'iemocap_train.csv')
        test_data_path = os.path.join(CONFIG['data_path'], 'iemocap_test.csv')
        dev_data_path = os.path.join(CONFIG['data_path'], 'iemocap_dev.csv')
    
    speaker_vocab_dict_path = 'vocabs/meld/speaker_vocab.pkl'
    emotion_vocab_dict_path = 'vocabs/meld/emotion_vocab.pkl'
    if args.task_name == 'iemocap':
        speaker_vocab_dict_path = 'vocabs/iemocap/speaker_vocab.pkl'
        emotion_vocab_dict_path = 'vocabs/iemocap/emotion_vocab.pkl'

    # Create required directories
    os.makedirs('vocabs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    seed = 1024
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True

    # Build vocabularies from dataset and additional source
    if args.task_name =='iemocap':
        transcript = "iemocap_transcript.json"
    else:
        transcript = "friends_transcript.json"
    get_vocabs([train_data_path, dev_data_path, test_data_path],
               os.path.join('transcripts/', transcript))

    # Initialize model
    if args.model_type == 'baseline':
        model = BaselineModel(CONFIG)
    else:
        model = CRFModel(CONFIG)
    device = CONFIG['device']
    model.to(device)
    print('---config---')
    for k, v in CONFIG.items():
        print(k, '\t\t\t', v, flush=True)

    '''
    # Define the checkpoint directory and create it if it doesn't exist.
    checkpoint_dir = os.path.join('models', CONFIG['task_name'])
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Pre-load the test set (using the current use_query flag) so it's available for finetuning.
    testset = load_builddataset(test_data_path, use_query=CONFIG['use_query'])
    # Load the most recent checkpoint if fine-tuning is enabled
    if args.finetune:
        lst = os.listdir(checkpoint_dir)
        lst = list(filter(lambda item: item.endswith('.pkl'), lst))
        if len(lst) > 0:
            lst.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))
            model = torch.load(os.path.join(checkpoint_dir, lst[-1]))
            f1 = test(model, testset)
            print('best f1 on test is {:.4f}'.format(f1), flush=True)
        else:
            print("No checkpoint found!")
    '''
        
    # Start training process
    if args.train:
        train(model, train_data_path, dev_data_path, test_data_path)

    # Run testing on the test dataset
    if args.test:
        testset = load_builddataset(test_data_path)
        best_f1 = test(model, testset)
        print(best_f1)

